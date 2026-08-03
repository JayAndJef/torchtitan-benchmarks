"""Distinguish CPU-dispatch-bound from GPU-bound at the piper microbatch shape.

Times a burst of B back-to-back calls between two events and divides by B.
If per-call time collapses as B grows, the single-call number was dispatch-
starved (GPU idle waiting for the CPU); if it stays flat, the kernel itself
is that slow.
"""

import os
import statistics as pystats
import sys
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/data/zejiaqi/tmp/torch_extensions")
sys.path.insert(0, "/data/zejiaqi/torchtitan")

import torch
from torch.utils.cpp_extension import load

from torchtitan.overrides.helion_rope import _helion_cossin_rope_fwd

DEV = "cuda"
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

hd, s, b, nh, nkv = 64, 1024, 1, 16, 8
inv = 1.0 / (1e6 ** (torch.arange(0, hd, 2).float() / hd))
ang = torch.outer(torch.arange(2048, dtype=torch.float32), inv)
ang = torch.cat([ang, ang], -1)
cache = torch.cat([ang.cos(), ang.sin()], -1).to(DEV)
freqs = ang.contiguous().to(DEV)

g = torch.Generator(device=DEV).manual_seed(0)
q = torch.randn(b, s, nh, hd, device=DEV, dtype=torch.bfloat16, generator=g)
k = torch.randn(b, s, nkv, hd, device=DEV, dtype=torch.bfloat16, generator=g)
pos = torch.arange(s, device=DEV, dtype=torch.int32).unsqueeze(0).expand(b, -1).contiguous()


def hel():
    _helion_cossin_rope_fwd(q, k, cache, pos)


def tef():
    te.forward(q, freqs, False)
    te.forward(k, freqs, False)


def burst_us(fn, burst, iters=50):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(burst):
            fn()
        e1.record()
        torch.cuda.synchronize()
        out.append(e0.elapsed_time(e1) * 1e3 / burst)
    return pystats.median(out)


print(f"GPU: {torch.cuda.get_device_name(0)}  shape (1,1024,16/8,64)")
print(f"{'burst':>6} {'helion us/call':>15} {'TE us/call':>12}")
for burst in (1, 4, 16, 64):
    print(f"{burst:>6} {burst_us(hel, burst):>15.2f} {burst_us(tef, burst):>12.2f}")
