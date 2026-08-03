"""Out-of-tree torchtitan override: RoPE via a TransformerEngine-derived kernel.

The TE block rotation is copied verbatim. A Piper-specific BSHD entry point
selects the frequency row from TorchTitan's per-token ``positions`` tensor so
packed-document position resets match the stock RoPE implementation. The
extension is exposed as torch.library custom ops with fake and autograd
registrations and swaps CosSinRoPE for the TE-backed module.

Activation:
    PYTHONPATH=/data/zejiaqi/torchtitan-benchmarks torchtitan_train ... \
        --override.imports piper1b.rope.te_rope_override.te_rope

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
from torchtitan.models.common.rope import _maybe_check_max_pos, CosSinRoPE
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
    xq: torch.Tensor,
    xk: torch.Tensor,
    angles: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # TE handles one tensor per launch; q and k are two launches.
    return (
        _te.forward_positions(xq, angles, positions, False),
        _te.forward_positions(xk, angles, positions, False),
    )


@_te_rope_fwd.register_fake
def _te_rope_fwd_fake(xq, xk, angles, positions):
    return (
        torch.empty(xq.size(), device=xq.device, dtype=xq.dtype),
        torch.empty(xk.size(), device=xk.device, dtype=xk.dtype),
    )


@torch.library.custom_op(
    "torchtitan_benchmarks::te_rope_bwd", mutates_args=(), device_types="cuda"
)
def _te_rope_bwd(
    grad_xq: torch.Tensor,
    grad_xk: torch.Tensor,
    angles: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_xq = grad_xq.contiguous()
    grad_xk = grad_xk.contiguous()
    return (
        _te.backward_positions(grad_xq, angles, positions, False),
        _te.backward_positions(grad_xk, angles, positions, False),
    )


@_te_rope_bwd.register_fake
def _te_rope_bwd_fake(grad_xq, grad_xk, angles, positions):
    return (
        torch.empty(grad_xq.size(), device=grad_xq.device, dtype=grad_xq.dtype),
        torch.empty(grad_xk.size(), device=grad_xk.device, dtype=grad_xk.dtype),
    )


def _te_setup_context(ctx, inputs, output) -> None:
    _, _, angles, positions = inputs
    ctx.save_for_backward(angles, positions)


def _te_backward(ctx, grad_xq_out, grad_xk_out):
    angles, positions = ctx.saved_tensors
    grad_xq, grad_xk = _te_rope_bwd(
        grad_xq_out,
        grad_xk_out,
        angles,
        positions,
    )
    return grad_xq, grad_xk, None, None


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
            or positions is None
            or positions.dtype != torch.int64
            or not positions.is_cuda
            or positions.ndim != 2
            or positions.shape[:2] != query.shape[:2]
        ):
            # Dynamo cannot trace logging.Logger methods; keep the fallback
            # warning out of compiled graphs.
            if not torch.compiler.is_compiling():
                warn_once(
                    logger,
                    "TECosSinRoPE: unsupported inputs (need plain 4D CUDA bf16 "
                    "tensors and CUDA int64 positions shaped [batch, seq]); "
                    "falling back to the PyTorch cos/sin RoPE.",
                )
            return super().forward(query, key, positions)
        _maybe_check_max_pos(positions, max_valid_pos=self.te_angles.shape[0] - 1)
        return _te_rope_fwd(
            query.contiguous(),
            key.contiguous(),
            self.te_angles,
            positions.contiguous(),
        )


@override(
    target=CosSinRoPE.Config,
    exact=True,
    description="TransformerEngine-derived fused RoPE with per-token positions.",
)
def te_rope(cfg: CosSinRoPE.Config) -> TECosSinRoPE.Config:
    return derive(cfg, TECosSinRoPE.Config)
