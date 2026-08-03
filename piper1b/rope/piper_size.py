"""TE vs Helion RoPE at the piper Qwen3-1B shapes.

piper 1B: dim=1024, n_heads=16, n_kv_heads=8, head_dim=64, max_seq_len=2048,
rope_theta=1e6. Harness defaults: batch 4, seq 1024, 4 microbatches.
Shapes below cover per-microbatch (b=1), dp-split (b=2), and full batch (b=4).

Interleaved A/B timing (drift-controlled), n=200 per shape.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/data/zejiaqi/tmp/torch_extensions")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data/zejiaqi/tmp/inductor_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/data/zejiaqi/tmp/triton_cache")
sys.path.insert(0, "/data/zejiaqi/torchtitan")

import statistics as pystats

import torch
from scipy import stats as st
from torch.utils.cpp_extension import load

from torchtitan.overrides.helion_rope import _helion_cossin_rope_fwd

DEV = "cuda"
N = 200
WARMUP = 30
MAX_SEQ = 2048
THETA = 1_000_000.0
ROPE_SOURCE = str(Path(__file__).with_name("te_rope_standalone.cu"))

te = load(
    name="te_rope_standalone",
    sources=[ROPE_SOURCE],
    build_directory="/data/zejiaqi/tmp/torch_extensions/te_rope",
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
    ],
    verbose=False,
)


def tables(head_dim):
    inv = 1.0 / (THETA ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(MAX_SEQ, dtype=torch.float32), inv)
    ang = torch.cat([ang, ang], -1)
    return (
        torch.cat([ang.cos(), ang.sin()], -1).to(DEV),
        ang.contiguous().to(DEV),
    )


def run_shape(b, s, nh, nkv, hd):
    cache, freqs = tables(hd)
    g = torch.Generator(device=DEV).manual_seed(0)
    q = torch.randn(b, s, nh, hd, device=DEV, dtype=torch.bfloat16, generator=g)
    k = torch.randn(b, s, nkv, hd, device=DEV, dtype=torch.bfloat16, generator=g)
    pos = (
        torch.arange(s, device=DEV, dtype=torch.int32)
        .unsqueeze(0)
        .expand(b, -1)
        .contiguous()
    )
    q_out, k_out = torch.empty_like(q), torch.empty_like(k)
    qk_bytes = 2 * (q.numel() + k.numel()) * 2  # read+write, bf16

    # correctness spot check at this head_dim (64 is a new specialization)
    hq, hk = _helion_cossin_rope_fwd(q, k, cache, pos)
    tq = te.forward(q, freqs, False)
    tk = te.forward(k, freqs, False)
    max_abs = max(
        (hq.float() - tq.float()).abs().max().item(),
        (hk.float() - tk.float()).abs().max().item(),
    )

    def cp():
        q_out.copy_(q)
        k_out.copy_(k)

    def hel():
        _helion_cossin_rope_fwd(q, k, cache, pos)

    def tef():
        te.forward(q, freqs, False)
        te.forward(k, freqs, False)

    for _ in range(WARMUP):
        cp()
        hel()
        tef()
    torch.cuda.synchronize()

    ev = [[torch.cuda.Event(enable_timing=True) for _ in range(4)] for _ in range(N)]
    for i in range(N):
        e0, e1, e2, e3 = ev[i]
        e0.record()
        cp()
        e1.record()
        hel()
        e2.record()
        tef()
        e3.record()
    torch.cuda.synchronize()

    cp_us = [a.elapsed_time(bb) * 1e3 for a, bb, _, _ in ev]
    hel_us = [bb.elapsed_time(c) * 1e3 for _, bb, c, _ in ev]
    te_us = [c.elapsed_time(d) * 1e3 for _, _, c, d in ev]

    med = lambda x: pystats.median(x)
    mwu = st.mannwhitneyu(te_us, hel_us, alternative="two-sided")
    wil = st.wilcoxon([t - h for h, t in zip(hel_us, te_us)])
    n_te_faster = sum(1 for h, t in zip(hel_us, te_us) if t < h)
    gb = lambda us: qk_bytes / (us * 1e-6) / 1e9

    print(
        f"\nshape (b={b}, s={s}, hq={nh}, hkv={nkv}, d={hd})   "
        f"q+k traffic {qk_bytes / 1e6:.2f} MB   max_abs(H vs TE)={max_abs:.1e}"
    )
    print(
        f"  copy floor : {med(cp_us):8.2f} us  ({gb(med(cp_us)):6.1f} GB/s)"
    )
    print(
        f"  Helion fwd : {med(hel_us):8.2f} us  ({gb(med(hel_us)):6.1f} GB/s)"
        f"  {med(hel_us) / med(cp_us):.3f}x floor   sd {pystats.stdev(hel_us):.2f}"
    )
    print(
        f"  TE fwd     : {med(te_us):8.2f} us  ({gb(med(te_us)):6.1f} GB/s)"
        f"  {med(te_us) / med(cp_us):.3f}x floor   sd {pystats.stdev(te_us):.2f}"
    )
    print(
        f"  TE/Helion = {med(te_us) / med(hel_us):.3f}x   TE faster in {n_te_faster}/{N}"
        f"   MWU p = {mwu.pvalue:.3g}   Wilcoxon p = {wil.pvalue:.3g}"
    )


print(f"GPU: {torch.cuda.get_device_name(0)}")
run_shape(1, 1024, 16, 8, 64)  # per-microbatch (default harness, 4 mb)
run_shape(2, 1024, 16, 8, 64)  # custom_order schedule (2 mb)
run_shape(4, 1024, 16, 8, 64)  # full default batch, no pp split
run_shape(4, 2048, 16, 8, 64)  # full batch at max_seq_len
