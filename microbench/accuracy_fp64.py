"""Unbiased accuracy check: both kernels vs an fp64 ground truth.

The main benchmark compared each kernel against a PyTorch reference built from
torchtitan's own fp32 cos/sin cache, which favours Helion by construction. Here
the ground truth is computed in float64 straight from the inverse-frequency
table, so neither kernel's cache convention is privileged.
"""

import os
import sys

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/data/zejiaqi/tmp/torch_extensions")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data/zejiaqi/tmp/inductor_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/data/zejiaqi/tmp/triton_cache")
sys.path.insert(0, "/data/zejiaqi/torchtitan")

import torch
from torch.utils.cpp_extension import load

from torchtitan.overrides.helion_rope import (
    _helion_cossin_rope_bwd,
    _helion_cossin_rope_fwd,
)

DEV = "cuda"
NO_BF16_GUARDS = [
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
]
te = load(
    name="te_rope_standalone",
    sources=["/data/zejiaqi/torchtitan-benchmarks/te_rope_standalone.cu"],
    build_directory="/data/zejiaqi/tmp/torch_extensions/te_rope",
    extra_cuda_cflags=["-O3", *NO_BF16_GUARDS],
    verbose=False,
)
te_fm = load(
    name="te_rope_standalone_fastmath",
    sources=["/data/zejiaqi/torchtitan-benchmarks/te_rope_standalone.cu"],
    build_directory="/data/zejiaqi/tmp/torch_extensions/te_rope_fm",
    extra_cuda_cflags=["-O3", "--use_fast_math", *NO_BF16_GUARDS],
    verbose=False,
)

B, S, NH, NKV, HD = 8, 4096, 16, 8, 128
BASE = 1_000_000.0

# ---- fp64 angle table (ground truth) ----
inv_freq64 = 1.0 / (
    BASE ** (torch.arange(0, HD, 2, dtype=torch.float64)[: HD // 2] / HD)
)
t64 = torch.arange(S, dtype=torch.float64)
freqs64 = torch.outer(t64, inv_freq64)
angles64 = torch.cat([freqs64, freqs64], dim=-1)  # (S, HD)
cos64 = angles64.cos().to(DEV)
sin64 = angles64.sin().to(DEV)

# ---- the two kernels' tables, both derived from the fp32 inv_freq ----
inv_freq = 1.0 / (BASE ** (torch.arange(0, HD, 2)[: HD // 2].float() / HD))
freqs = torch.outer(torch.arange(S, dtype=inv_freq.dtype), inv_freq).float()
angles = torch.cat([freqs, freqs], dim=-1)
tt_cache = torch.cat([angles.cos(), angles.sin()], dim=-1).to(DEV)  # torchtitan
te_freqs = angles.contiguous().to(DEV)  # TE: raw angles, sincosf in-kernel

g = torch.Generator(device=DEV).manual_seed(0)
q = torch.randn(B, S, NH, HD, device=DEV, dtype=torch.bfloat16, generator=g)
gq = torch.randn(B, S, NH, HD, device=DEV, dtype=torch.bfloat16, generator=g)
k = torch.randn(B, S, NKV, HD, device=DEV, dtype=torch.bfloat16, generator=g)
gk = torch.randn(B, S, NKV, HD, device=DEV, dtype=torch.bfloat16, generator=g)
pos = torch.arange(S, device=DEV, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()


def truth_fwd(x):
    xf = x.double()
    c = cos64.unsqueeze(0).unsqueeze(2)
    s = sin64.unsqueeze(0).unsqueeze(2)
    x1, x2 = xf[..., : HD // 2], xf[..., HD // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return xf * c + rot * s


def truth_bwd(x):
    xf = x.double()
    c = cos64.unsqueeze(0).unsqueeze(2)
    s = sin64.unsqueeze(0).unsqueeze(2)
    g1, g2 = xf[..., : HD // 2], xf[..., HD // 2 :]
    s1, s2 = s[..., : HD // 2], s[..., HD // 2 :]
    lo = g1 * c[..., : HD // 2] + g2 * s2
    hi = g2 * c[..., HD // 2 :] - g1 * s1
    return torch.cat((lo, hi), dim=-1)


def err(kernel_out, truth):
    """Error vs fp64 truth, expressed in bf16 ULPs of the true value."""
    d = (kernel_out.double() - truth).abs()
    # bf16 ulp of the true value: 2^(exponent-7)
    ulp = torch.ldexp(
        torch.ones_like(truth), torch.floor(torch.log2(truth.abs().clamp_min(1e-30))).long() - 7
    )
    u = d / ulp
    return d.max().item(), u.max().item(), u.mean().item(), (u > 0.5).float().mean().item()


print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"shape ({B}, {S}, {NH}, {HD}) bf16, theta={BASE:.0f}\n")

hq, _ = _helion_cossin_rope_fwd(q, k, tt_cache, pos)
tq = te.forward(q, te_freqs, False)
tqf = te_fm.forward(q, te_freqs, False)
T = truth_fwd(q)
print("FORWARD, error vs fp64 ground truth (units: bf16 ULP of the true value)")
print(f"  {'impl':<34} {'max_abs':>12} {'max_ulp':>9} {'mean_ulp':>9} {'frac>0.5ulp':>12}")
for name, out in [
    ("torchtitan Helion (fp32 cache)", hq),
    ("TE port (sincosf in kernel)", tq),
    ("TE port + --use_fast_math", tqf),
]:
    a, mx, mn, fr = err(out, T)
    print(f"  {name:<34} {a:>12.3e} {mx:>9.3f} {mn:>9.4f} {fr:>12.4%}")

print(f"\n  TE vs TE(--use_fast_math) bitwise equal: {torch.equal(tq, tqf)}"
      f"  (differing elems: {(tq != tqf).sum().item()})")

hgq, _ = _helion_cossin_rope_bwd(gq, gk, tt_cache, pos)
tgq = te.backward(gq, te_freqs, False)
Tb = truth_bwd(gq)
print("\nBACKWARD, error vs fp64 ground truth")
print(f"  {'impl':<34} {'max_abs':>12} {'max_ulp':>9} {'mean_ulp':>9} {'frac>0.5ulp':>12}")
for name, out in [
    ("torchtitan Helion (fp32 cache)", hgq),
    ("TE port (sincosf in kernel)", tgq),
]:
    a, mx, mn, fr = err(out, Tb)
    print(f"  {name:<34} {a:>12.3e} {mx:>9.3f} {mn:>9.4f} {fr:>12.4%}")

# How much of the difference is purely the cos/sin table?
te_cos_recomputed = torch.cos(te_freqs.double())
tt_cos = tt_cache[:, :HD].double()
print(
    f"\ncos table error vs fp64: torchtitan fp32 cache max={((tt_cos - cos64.double()).abs()).max():.3e}"
)
print(f"                          (TE computes cos in-kernel via sincosf on the same fp32 angles)")
print(
    f"fp32 angle-table error vs fp64 angles: max={((te_freqs.double() - angles64.to(DEV)).abs()).max():.3e}"
)
