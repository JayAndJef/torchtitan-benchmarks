"""Statistical significance of the TE-vs-Helion RoPE gap.

Interleaved A/B design: each iteration times one Helion call and one TE
(q+k) call back to back, so slow drift (clocks, thermals) hits both arms
equally. Reports per-arm distributions, Welch t, Mann-Whitney U, Cohen's d,
and the paired per-iteration delta.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/data/zejiaqi/tmp/torch_extensions")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data/zejiaqi/tmp/inductor_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/data/zejiaqi/tmp/triton_cache")
sys.path.insert(0, "/data/zejiaqi/torchtitan")

import torch
from scipy import stats as st
from torch.utils.cpp_extension import load

from torchtitan.overrides.helion_rope import _helion_cossin_rope_fwd

DEV = "cuda"
N = 200
WARMUP = 30
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


def tables(head_dim, max_seq, theta=1_000_000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(max_seq, dtype=torch.float32), inv)
    ang = torch.cat([ang, ang], -1)
    return (
        torch.cat([ang.cos(), ang.sin()], -1).to(DEV),
        ang.contiguous().to(DEV),
    )


def run_shape(b, s, nh, nkv, hd):
    cache, freqs = tables(hd, s)
    g = torch.Generator(device=DEV).manual_seed(0)
    q = torch.randn(b, s, nh, hd, device=DEV, dtype=torch.bfloat16, generator=g)
    k = torch.randn(b, s, nkv, hd, device=DEV, dtype=torch.bfloat16, generator=g)
    pos = (
        torch.arange(s, device=DEV, dtype=torch.int32)
        .unsqueeze(0)
        .expand(b, -1)
        .contiguous()
    )

    def hel():
        _helion_cossin_rope_fwd(q, k, cache, pos)

    def tef():
        te.forward(q, freqs, False)
        te.forward(k, freqs, False)

    for _ in range(WARMUP):
        hel()
        tef()
    torch.cuda.synchronize()

    ev = [
        [torch.cuda.Event(enable_timing=True) for _ in range(3)] for _ in range(N)
    ]
    for i in range(N):
        e0, e1, e2 = ev[i]
        e0.record()
        hel()
        e1.record()
        tef()
        e2.record()
    torch.cuda.synchronize()

    hel_us = [e0.elapsed_time(e1) * 1e3 for e0, e1, _ in ev]
    te_us = [e1.elapsed_time(e2) * 1e3 for _, e1, e2 in ev]

    import statistics as py_st

    h_med, t_med = py_st.median(hel_us), py_st.median(te_us)
    h_mean, t_mean = py_st.mean(hel_us), py_st.mean(te_us)
    h_sd, t_sd = py_st.stdev(hel_us), py_st.stdev(te_us)
    deltas = [t - h for h, t in zip(hel_us, te_us)]
    n_te_faster = sum(1 for d in deltas if d < 0)

    welch = st.ttest_ind(te_us, hel_us, equal_var=False)
    mwu = st.mannwhitneyu(te_us, hel_us, alternative="two-sided")
    wil = st.wilcoxon(deltas)
    pooled_sd = ((h_sd**2 + t_sd**2) / 2) ** 0.5
    d = (t_mean - h_mean) / pooled_sd

    print(f"\nshape (b={b}, s={s}, hq={nh}, hkv={nkv}, d={hd})  n={N} interleaved")
    print(f"  Helion: mean {h_mean:8.2f} us  sd {h_sd:6.2f}  median {h_med:8.2f}")
    print(f"  TE:     mean {t_mean:8.2f} us  sd {t_sd:6.2f}  median {t_med:8.2f}")
    print(
        f"  delta:  mean {t_mean - h_mean:+8.2f} us  ({t_mean / h_mean:.4f}x)  "
        f"TE faster in {n_te_faster}/{N} iterations"
    )
    print(f"  Welch t p = {welch.pvalue:.3g}   MWU p = {mwu.pvalue:.3g}   "
          f"Wilcoxon(paired) p = {wil.pvalue:.3g}   Cohen's d = {d:.2f}")


print(f"GPU: {torch.cuda.get_device_name(0)}")
run_shape(8, 4096, 16, 8, 128)
run_shape(1, 8192, 32, 8, 128)
run_shape(4, 2048, 8, 8, 128)
