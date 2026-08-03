"""Where does TE's per-launch bandwidth deficit come from?

ncu is unavailable on this box (ERR_NVGPUCTRPERM), so instead of counters this
uses a controlled ablation: the same TE kernel with the rotate-half partner load
removed. The output is deliberately wrong; only the time is meaningful.
"""

import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/data/zejiaqi/tmp/torch_extensions")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/data/zejiaqi/tmp/inductor_cache")
os.environ.setdefault("TRITON_CACHE_DIR", "/data/zejiaqi/tmp/triton_cache")
sys.path.insert(0, "/data/zejiaqi/torchtitan")

import torch
from torch.utils.cpp_extension import load

from torchtitan.overrides.helion_rope import _helion_cossin_rope_fwd

DEV = "cuda"
ROPE_SOURCE = str(Path(__file__).with_name("te_rope_standalone.cu"))
te = load(
    name="te_rope_ablate",
    sources=[ROPE_SOURCE],
    build_directory="/data/zejiaqi/tmp/torch_extensions/te_rope_ablate",
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
    ],
    verbose=False,
)


def time_median_us(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in zip(st, en))


print(f"GPU: {torch.cuda.get_device_name(0)}\n")
for B, S, NH, HD in [(8, 4096, 16, 128), (8, 4096, 8, 128)]:
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, HD, 2)[: HD // 2].float() / HD))
    freqs = torch.outer(torch.arange(S, dtype=inv_freq.dtype), inv_freq).float()
    angles = torch.cat([freqs, freqs], dim=-1)
    tt_cache = torch.cat([angles.cos(), angles.sin()], dim=-1).to(DEV)
    te_freqs = angles.contiguous().to(DEV)

    g = torch.Generator(device=DEV).manual_seed(0)
    x = torch.randn(B, S, NH, HD, device=DEV, dtype=torch.bfloat16, generator=g)
    out = torch.empty_like(x)
    nbytes = 2 * x.numel() * 2  # read + write

    t_copy = time_median_us(lambda: out.copy_(x))
    t_te = time_median_us(lambda: te.forward(x, te_freqs, False))
    t_abl = time_median_us(lambda: te.ablate_no_partner_load(x, te_freqs))

    def gb(us):
        return nbytes / (us * 1e-6) / 1e9

    print(f"shape ({B}, {S}, h={NH}, d={HD}) -- {nbytes / 1e6:.1f} MB read+write, 1 launch each")
    print(f"  {'variant':<52} {'us':>9} {'GB/s':>8} {'x copy':>8}")
    print(f"  {'copy_ (vectorized 16B ld/st, 1 ld + 1 st per elem)':<52}"
          f" {t_copy:>9.2f} {gb(t_copy):>8.1f} {1.0:>8.3f}")
    print(f"  {'TE verbatim (scalar 2B, 2 ld + 1 st per elem)':<52}"
          f" {t_te:>9.2f} {gb(t_te):>8.1f} {t_te / t_copy:>8.3f}")
    print(f"  {'TE ABLATED: partner load removed (scalar 2B, 1 ld)':<52}"
          f" {t_abl:>9.2f} {gb(t_abl):>8.1f} {t_abl / t_copy:>8.3f}")
    print(f"  -> cost attributable to the 2nd (partner) load: "
          f"{t_te - t_abl:+.2f} us ({(t_te / t_abl - 1) * 100:+.1f}%)")
    print(f"  -> residual gap of the 1-load scalar kernel vs copy_: "
          f"{t_abl - t_copy:+.2f} us ({(t_abl / t_copy - 1) * 100:+.1f}%)\n")
    del x, out
    torch.cuda.empty_cache()
