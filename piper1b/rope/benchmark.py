"""Head-to-head: TransformerEngine fused RoPE kernel vs torchtitan's Helion RoPE.

Builds TE's fused_rope_{forward,backward}_kernel as a standalone torch CUDA
extension (device code copied verbatim, same launch config) and benchmarks it
against torchtitan.overrides.helion_rope on identical inputs.
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

from torchtitan.overrides.helion_rope import (
    _helion_cossin_rope_bwd,
    _helion_cossin_rope_fwd,
    _rope_complex_fwd,
    _run_tuned,
    _complex_fwd_config,
)

DEV = "cuda"
ROPE_SOURCE = str(Path(__file__).with_name("te_rope_standalone.cu"))
BUILD_DIR = "/data/zejiaqi/tmp/torch_extensions/te_rope"
os.makedirs(BUILD_DIR, exist_ok=True)

# torch's cpp_extension defines __CUDA_NO_BFLOAT16_{CONVERSIONS,OPERATORS}__,
# which TE's own build does not; TE's device code relies on the implicit
# __nv_bfloat16 <-> float conversions. Undefine them so the ported code compiles
# exactly as written upstream.
NO_BF16_GUARDS = [
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
]

te = load(
    name="te_rope_standalone",
    sources=[ROPE_SOURCE],
    build_directory=BUILD_DIR,
    extra_cuda_cflags=["-O3", *NO_BF16_GUARDS],
    verbose=False,
)

# Same code with --use_fast_math (turns sincosf into the fast intrinsic), to
# check whether TE's in-kernel trig has any measurable cost here.
FM_DIR = "/data/zejiaqi/tmp/torch_extensions/te_rope_fm"
os.makedirs(FM_DIR, exist_ok=True)
te_fm = load(
    name="te_rope_standalone_fastmath",
    sources=[ROPE_SOURCE],
    build_directory=FM_DIR,
    extra_cuda_cflags=["-O3", "--use_fast_math", *NO_BF16_GUARDS],
    verbose=False,
)


# ---------------------------------------------------------------------------
# Shared angle table -> both cache conventions
# ---------------------------------------------------------------------------
def build_tables(head_dim: int, max_seq: int, theta_base: float = 1_000_000.0):
    """Return (torchtitan cos/sin cache, TE freqs) from one inv_freq table."""
    inv_freq = 1.0 / (
        theta_base
        ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim)
    )
    t = torch.arange(max_seq, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq).float()  # (max_seq, head_dim // 2)
    angles = torch.cat([freqs, freqs], dim=-1)  # (max_seq, head_dim)

    # torchtitan CosSinRoPE.cache: (max_seq, 2 * head_dim) = cat([cos, sin], -1)
    tt_cache = torch.cat([angles.cos(), angles.sin()], dim=-1).to(DEV)
    # TE freqs: (max_seq, 1, 1, d2) fp32 raw ANGLES, d2 == head_dim
    te_freqs = angles.contiguous().to(DEV)

    # ComplexRoPE cache, for the optional interleaved cross-check.
    complex_cache = torch.polar(torch.ones_like(freqs), freqs).to(DEV)
    # TE interleaved freqs: angles repeated pairwise (see TE RotaryPositionEmbedding)
    te_freqs_interleaved = (
        torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1)
        .view(freqs.shape[0], -1)
        .contiguous()
        .to(DEV)
    )
    return tt_cache, te_freqs, complex_cache, te_freqs_interleaved


# ---------------------------------------------------------------------------
# Reference implementations (PyTorch), used to validate both kernels
# ---------------------------------------------------------------------------
def ref_cossin_fwd(x, tt_cache, positions):
    hd = x.shape[-1]
    c = tt_cache[positions]  # (b, s, 2*hd)
    cos = c[..., :hd].unsqueeze(2)
    sin = c[..., hd:].unsqueeze(2)
    xf = x.float()
    x1, x2 = xf[..., : hd // 2], xf[..., hd // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return (xf * cos + rot * sin).to(x.dtype)


def ref_cossin_bwd(g, tt_cache, positions):
    hd = g.shape[-1]
    c = tt_cache[positions]
    cos = c[..., :hd].unsqueeze(2)
    sin = c[..., hd:].unsqueeze(2)
    gf = g.float()
    # transpose of the forward rotation
    g1, g2 = gf[..., : hd // 2], gf[..., hd // 2 :]
    s1, s2 = sin[..., : hd // 2], sin[..., hd // 2 :]
    out_lo = g1 * cos[..., : hd // 2] + g2 * s2
    out_hi = g2 * cos[..., hd // 2 :] - g1 * s1
    return torch.cat((out_lo, out_hi), dim=-1).to(g.dtype)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_median_us(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))


def gbps(us, nbytes):
    return nbytes / (us * 1e-6) / 1e9


# ---------------------------------------------------------------------------
SHAPES = [
    (8, 4096, 16, 8, 128),
    (1, 8192, 32, 8, 128),
    (4, 2048, 8, 8, 128),
]


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}\n")

    rows = []
    for b, s, nh, nkv, hd in SHAPES:
        tag = f"({b}, {s}, q_h={nh}, k_h={nkv}, d={hd})"
        print("=" * 100)
        print(f"SHAPE {tag}")
        tt_cache, te_freqs, cplx_cache, te_freqs_il = build_tables(hd, s)

        g = torch.Generator(device=DEV).manual_seed(0)
        q = torch.randn(b, s, nh, hd, device=DEV, dtype=torch.bfloat16, generator=g)
        k = torch.randn(b, s, nkv, hd, device=DEV, dtype=torch.bfloat16, generator=g)
        gq = torch.randn(b, s, nh, hd, device=DEV, dtype=torch.bfloat16, generator=g)
        gk = torch.randn(b, s, nkv, hd, device=DEV, dtype=torch.bfloat16, generator=g)
        pos = (
            torch.arange(s, device=DEV, dtype=torch.int32)
            .unsqueeze(0)
            .expand(b, -1)
            .contiguous()
        )

        qb = q.numel() * 2
        kb = k.numel() * 2
        qk_bytes = 2 * (qb + kb)  # read + write, q and k
        q_only_bytes = 2 * qb
        k_only_bytes = 2 * kb
        print(f"  q+k traffic (read+write) = {qk_bytes / 1e6:.2f} MB")

        # ---------------- correctness ----------------
        hq, hk = _helion_cossin_rope_fwd(q, k, tt_cache, pos)
        tq = te.forward(q, te_freqs, False)
        tk = te.forward(k, te_freqs, False)
        rq = ref_cossin_fwd(q, tt_cache, pos.long())
        rk = ref_cossin_fwd(k, tt_cache, pos.long())

        def stats(a, bb):
            d = (a.float() - bb.float()).abs()
            den = bb.float().abs().clamp_min(1e-6)
            return d.max().item(), (d / den).max().item()

        f_he_te_q = stats(hq, tq)
        f_he_te_k = stats(hk, tk)
        f_he_ref_q = stats(hq, rq)
        f_te_ref_q = stats(tq, rq)
        bit_q = torch.equal(hq, tq)
        bit_k = torch.equal(hk, tk)
        nmis_q = (hq != tq).sum().item()
        nmis_k = (hk != tk).sum().item()

        print("  FORWARD correctness (rotate-half / non-interleaved):")
        print(
            f"    Helion vs TE   q: max_abs={f_he_te_q[0]:.3e} max_rel={f_he_te_q[1]:.3e}"
            f"   bitwise_equal={bit_q}  mismatching elems={nmis_q}/{hq.numel()}"
        )
        print(
            f"    Helion vs TE   k: max_abs={f_he_te_k[0]:.3e} max_rel={f_he_te_k[1]:.3e}"
            f"   bitwise_equal={bit_k}  mismatching elems={nmis_k}/{hk.numel()}"
        )
        print(
            f"    Helion vs fp32 ref q: max_abs={f_he_ref_q[0]:.3e} max_rel={f_he_ref_q[1]:.3e}"
        )
        print(
            f"    TE     vs fp32 ref q: max_abs={f_te_ref_q[0]:.3e} max_rel={f_te_ref_q[1]:.3e}"
        )

        hgq, hgk = _helion_cossin_rope_bwd(gq, gk, tt_cache, pos)
        tgq = te.backward(gq, te_freqs, False)
        tgk = te.backward(gk, te_freqs, False)
        rgq = ref_cossin_bwd(gq, tt_cache, pos.long())
        b_he_te_q = stats(hgq, tgq)
        b_he_te_k = stats(hgk, tgk)
        b_he_ref_q = stats(hgq, rgq)
        b_te_ref_q = stats(tgq, rgq)
        print("  BACKWARD correctness:")
        print(
            f"    Helion vs TE   q: max_abs={b_he_te_q[0]:.3e} max_rel={b_he_te_q[1]:.3e}"
            f"   bitwise_equal={torch.equal(hgq, tgq)}"
            f"  mismatching elems={(hgq != tgq).sum().item()}/{hgq.numel()}"
        )
        print(
            f"    Helion vs TE   k: max_abs={b_he_te_k[0]:.3e} max_rel={b_he_te_k[1]:.3e}"
            f"   bitwise_equal={torch.equal(hgk, tgk)}"
        )
        print(
            f"    Helion vs fp32 ref q: max_abs={b_he_ref_q[0]:.3e} max_rel={b_he_ref_q[1]:.3e}"
        )
        print(
            f"    TE     vs fp32 ref q: max_abs={b_te_ref_q[0]:.3e} max_rel={b_te_ref_q[1]:.3e}"
        )

        # interleaved TE vs ComplexRoPE helion kernel
        cplx_real = torch.view_as_real(cplx_cache).contiguous()
        cq, ck = _run_tuned(_rope_complex_fwd, _complex_fwd_config(q), q, k, cplx_real, pos)
        tq_il = te.forward(q, te_freqs_il, True)
        tk_il = te.forward(k, te_freqs_il, True)
        il_q = stats(cq, tq_il)
        il_k = stats(ck, tk_il)
        print("  INTERLEAVED forward (ComplexRoPE convention):")
        print(
            f"    HelionComplex vs TE(interleaved) q: max_abs={il_q[0]:.3e} max_rel={il_q[1]:.3e}"
            f"   bitwise_equal={torch.equal(cq, tq_il)}"
        )
        print(
            f"    HelionComplex vs TE(interleaved) k: max_abs={il_k[0]:.3e} max_rel={il_k[1]:.3e}"
            f"   bitwise_equal={torch.equal(ck, tk_il)}"
        )

        assert f_he_te_q[0] < 0.05 and f_he_te_k[0] < 0.05, "fwd mismatch"
        assert b_he_te_q[0] < 0.05 and b_he_te_k[0] < 0.05, "bwd mismatch"

        # ---------------- timing ----------------
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        t_copy_q = time_median_us(lambda: q_out.copy_(q))
        t_copy_k = time_median_us(lambda: k_out.copy_(k))

        def copy_both():
            q_out.copy_(q)
            k_out.copy_(k)

        t_copy = time_median_us(copy_both)

        t_hel_f = time_median_us(lambda: _helion_cossin_rope_fwd(q, k, tt_cache, pos))
        t_hel_b = time_median_us(lambda: _helion_cossin_rope_bwd(gq, gk, tt_cache, pos))

        t_te_fq = time_median_us(lambda: te.forward(q, te_freqs, False))
        t_te_fk = time_median_us(lambda: te.forward(k, te_freqs, False))

        def te_fwd_both():
            te.forward(q, te_freqs, False)
            te.forward(k, te_freqs, False)

        t_te_f = time_median_us(te_fwd_both)

        t_te_bq = time_median_us(lambda: te.backward(gq, te_freqs, False))
        t_te_bk = time_median_us(lambda: te.backward(gk, te_freqs, False))

        def te_bwd_both():
            te.backward(gq, te_freqs, False)
            te.backward(gk, te_freqs, False)

        t_te_b = time_median_us(te_bwd_both)

        # helion complex (interleaved) fwd, for the optional comparison
        t_hel_cplx = time_median_us(
            lambda: _run_tuned(
                _rope_complex_fwd, _complex_fwd_config(q), q, k, cplx_real, pos
            )
        )

        def te_il_both():
            te.forward(q, te_freqs_il, True)
            te.forward(k, te_freqs_il, True)

        t_te_il = time_median_us(te_il_both)

        def te_fm_both():
            te_fm.forward(q, te_freqs, False)
            te_fm.forward(k, te_freqs, False)

        t_te_fm = time_median_us(te_fm_both)

        floor = t_copy
        print(f"\n  {'kernel':<42} {'us':>10} {'GB/s':>9} {'x floor':>9}")
        print("  " + "-" * 74)

        def line(name, us, nbytes, base=None):
            base = floor if base is None else base
            print(f"  {name:<42} {us:>10.2f} {gbps(us, nbytes):>9.1f} {us / base:>9.3f}")
            rows.append((tag, name, us, gbps(us, nbytes), us / base))

        line("copy_ q only (1 launch)", t_copy_q, q_only_bytes)
        line("copy_ k only (1 launch)", t_copy_k, k_only_bytes)
        line("COPY FLOOR: copy_ q+k (2 launches)", t_copy, qk_bytes)
        print("  " + "-" * 74)
        line("Helion fwd  q+k (1 launch)", t_hel_f, qk_bytes)
        line("TE fwd  q only (1 launch)", t_te_fq, q_only_bytes)
        line("TE fwd  k only (1 launch)", t_te_fk, k_only_bytes)
        line("TE fwd  q+k (2 launches)", t_te_f, qk_bytes)
        line("TE fwd  q+k (2 launches, --use_fast_math)", t_te_fm, qk_bytes)
        print("  " + "-" * 74)
        line("Helion bwd  q+k (1 launch)", t_hel_b, qk_bytes)
        line("TE bwd  q only (1 launch)", t_te_bq, q_only_bytes)
        line("TE bwd  k only (1 launch)", t_te_bk, k_only_bytes)
        line("TE bwd  q+k (2 launches)", t_te_b, qk_bytes)
        print("  " + "-" * 74)
        line("HelionComplex fwd q+k (1 launch)", t_hel_cplx, qk_bytes)
        line("TE interleaved fwd q+k (2 launches)", t_te_il, qk_bytes)

        print(
            f"\n  VERDICT fwd: TE(q+k)/Helion = {t_te_f / t_hel_f:.3f}x "
            f"({'TE slower' if t_te_f > t_hel_f else 'TE faster'})"
        )
        print(
            f"  VERDICT bwd: TE(q+k)/Helion = {t_te_b / t_hel_b:.3f}x "
            f"({'TE slower' if t_te_b > t_hel_b else 'TE faster'})"
        )
        print()

        del q, k, gq, gk, hq, hk, tq, tk, rq, rk, hgq, hgk, tgq, tgk, rgq
        del q_out, k_out, cq, ck, tq_il, tk_il
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
