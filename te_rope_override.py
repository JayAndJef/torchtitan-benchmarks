"""Out-of-tree torchtitan override: RoPE via TransformerEngine's fused kernel.

Wraps the standalone verbatim port of TE's fused_rope_{forward,backward}_kernel
(te_rope_standalone.cu, next to this file) as torch.library custom ops (fake +
autograd registered) and swaps CosSinRoPE for a TE-backed module, the same way
torchtitan.overrides.helion_rope does for Helion.

BENCHMARK-ONLY SEMANTICS CAVEAT: TE's kernel has no per-token position gather.
It rotates row s of every sequence by angle(s) (optionally plus one scalar
offset per sequence). torchtitan passes per-token ``positions`` that reset at
document boundaries; this override IGNORES them. Performance is representative
of TE's design; losses will NOT match the stock or Helion arms whenever packed
documents make positions != arange. Do not use outside benchmarking.

Activation:
    PYTHONPATH=/data/zejiaqi/torchtitan-benchmarks torchtitan_train ... \
        --override.imports te_rope_override.te_rope

The CUDA extension builds on first import (needs a C++20-capable host compiler;
on this box: ``source /opt/rh/gcc-toolset-13/enable``). Build artifacts go under
``$TORCH_EXTENSIONS_DIR`` (default /data/zejiaqi/tmp/torch_extensions -- never
/tmp, which is always full on this box).
"""

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from torchtitan.config import derive, override
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.tools.logging import logger, warn_once

_EXT_DIR = Path(
    os.environ.get("TORCH_EXTENSIONS_DIR", "/data/zejiaqi/tmp/torch_extensions")
)
_BUILD_DIR = _EXT_DIR / "te_rope"
_BUILD_DIR.mkdir(parents=True, exist_ok=True)

_te = load(
    name="te_rope_standalone",
    sources=[str(Path(__file__).resolve().parent / "te_rope_standalone.cu")],
    build_directory=str(_BUILD_DIR),
    extra_cuda_cflags=[
        "-O3",
        # torch's cpp_extension defines these guards; TE's own build does not,
        # and TE device code relies on implicit __nv_bfloat16 <-> float
        # conversion, so undefine them to compile the verbatim kernels.
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
    ],
    verbose=False,
)


@torch.library.custom_op(
    "torchtitan_benchmarks::te_rope_fwd", mutates_args=(), device_types="cuda"
)
def _te_rope_fwd(
    xq: torch.Tensor, xk: torch.Tensor, angles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TE handles one tensor per launch; q and k are two launches.
    return (
        _te.forward(xq, angles, False),
        _te.forward(xk, angles, False),
    )


@_te_rope_fwd.register_fake
def _te_rope_fwd_fake(xq, xk, angles):
    return (
        torch.empty(xq.size(), device=xq.device, dtype=xq.dtype),
        torch.empty(xk.size(), device=xk.device, dtype=xk.dtype),
    )


@torch.library.custom_op(
    "torchtitan_benchmarks::te_rope_bwd", mutates_args=(), device_types="cuda"
)
def _te_rope_bwd(
    grad_xq: torch.Tensor, grad_xk: torch.Tensor, angles: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_xq = grad_xq.contiguous()
    grad_xk = grad_xk.contiguous()
    return (
        _te.backward(grad_xq, angles, False),
        _te.backward(grad_xk, angles, False),
    )


@_te_rope_bwd.register_fake
def _te_rope_bwd_fake(grad_xq, grad_xk, angles):
    return (
        torch.empty(grad_xq.size(), device=grad_xq.device, dtype=grad_xq.dtype),
        torch.empty(grad_xk.size(), device=grad_xk.device, dtype=grad_xk.dtype),
    )


def _te_setup_context(ctx, inputs, output) -> None:
    _, _, angles = inputs
    ctx.save_for_backward(angles)


def _te_backward(ctx, grad_xq_out, grad_xk_out):
    (angles,) = ctx.saved_tensors
    grad_xq, grad_xk = _te_rope_bwd(grad_xq_out, grad_xk_out, angles)
    return grad_xq, grad_xk, None


_te_rope_fwd.register_autograd(_te_backward, setup_context=_te_setup_context)


class TECosSinRoPE(CosSinRoPE):
    """CosSinRoPE with the rotation applied by the TE fused kernel.

    Keeps the stock ``cache`` (checkpoint contract, fallback path) and adds a
    raw fp32 angle table ``te_angles`` (max_seq, head_dim), which is what TE's
    kernel consumes (it computes sincosf in-kernel).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        pass

    def __init__(self, config: "TECosSinRoPE.Config"):
        super().__init__(config)
        self.register_buffer("te_angles", self._precompute_angles(), persistent=False)
        # The forward-path warnings are suppressed under torch.compile (Dynamo
        # cannot trace logging), so state the benchmark caveat once at build.
        logger.warning(
            "TECosSinRoPE BENCHMARK MODE: per-token positions are IGNORED by "
            "the TE kernel (rotates by sequence index). Losses will diverge "
            "from the stock/Helion arms on packed documents."
        )

    def _precompute_angles(self) -> torch.Tensor:
        cfg = self.config
        if cfg.scaling != "none":
            raise ValueError(
                "TECosSinRoPE benchmark override only supports scaling='none'"
            )
        inv_freq = 1.0 / (
            cfg.theta ** (torch.arange(0, cfg.dim, 2)[: cfg.dim // 2].float() / cfg.dim)
        )
        t = torch.arange(cfg.max_seq_len, dtype=inv_freq.dtype, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq).float()
        return torch.cat([freqs, freqs], dim=-1).contiguous()

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        super()._init_self_buffers(buffer_device=buffer_device)
        if buffer_device is None:
            buffer_device = self.te_angles.device
        with torch.device(buffer_device):
            self.te_angles = self._precompute_angles()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            isinstance(query, torch.distributed.tensor.DTensor)
            or query.dtype != torch.bfloat16
            or not query.is_cuda
            or query.ndim != 4
        ):
            # Dynamo cannot trace logging.Logger methods; guard so the module
            # stays fullgraph-compilable (the caveat is logged at __init__).
            if not torch.compiler.is_compiling():
                warn_once(
                    logger,
                    "TECosSinRoPE: unsupported inputs (need plain 4D CUDA bf16 "
                    "tensors); falling back to the PyTorch cos/sin RoPE.",
                )
            return super().forward(query, key, positions)
        return _te_rope_fwd(query.contiguous(), key.contiguous(), self.te_angles)


@override(
    target=CosSinRoPE.Config,
    exact=True,
    description="TransformerEngine fused RoPE (benchmark-only; ignores positions).",
)
def te_rope(cfg: CosSinRoPE.Config) -> TECosSinRoPE.Config:
    return derive(cfg, TECosSinRoPE.Config)
