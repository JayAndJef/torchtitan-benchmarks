"""Torch's cuDNN fused attention on packed documents, as a TorchTitan override.

Torch's ``varlen_attn`` never sends causal or grouped-query attention to
cuDNN, so this module calls the raw aten cuDNN ops through a custom op pair.

Activation:
    --override.imports benchmarks.models.piper_qwen3.components.attention.cudnn_override.packed_cudnn_attention
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.config import derive, override
from torchtitan.models.common.attention import GQAttention, VarlenMetadata
from torchtitan.protocols.module import Module

from benchmarks.models.piper_qwen3.components.attention.packed import (
    MAX_DOCUMENTS,
    PackedGQAttention,
)


@torch.library.custom_op(
    "torchtitan_benchmarks::cudnn_gqa_fwd", mutates_args=(), device_types="cuda"
)
def cudnn_gqa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal attention over packed ``(T, N, H)`` documents: the output and the ``(N, T)`` log-sum-exp."""
    result = torch.ops.aten._cudnn_attention_forward(
        q, k, v, None, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
        True, 0.0, True, False, scale=scale,
    )
    return result[0], result[1]


@cudnn_gqa_fwd.register_fake
def _(q, k, v, cu_seqlens, max_seqlen, scale):
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        q.new_empty((q.size(1), q.size(0)), dtype=torch.float32),
    )


@torch.library.custom_op(
    "torchtitan_benchmarks::cudnn_gqa_bwd", mutates_args=(), device_types="cuda"
)
def cudnn_gqa_bwd(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The gradients of ``cudnn_gqa_fwd`` with respect to q, k and v."""
    # Dropout is 0, so the backward never reads the philox seed and offset.
    philox = torch.zeros((), dtype=torch.int64, device=q.device)
    dq, dk, dv = torch.ops.aten._cudnn_attention_backward(
        grad_out, q, k, v, out, lse, philox, philox, None,
        cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, 0.0, True, scale=scale,
    )
    return dq, dk, dv


@cudnn_gqa_bwd.register_fake
def _(grad_out, q, k, v, out, lse, cu_seqlens, max_seqlen, scale):
    return tuple(
        torch.empty_like(t, memory_format=torch.contiguous_format) for t in (q, k, v)
    )


def _setup_context(ctx, inputs, output):
    q, k, v, cu_seqlens, max_seqlen, scale = inputs
    ctx.save_for_backward(q, k, v, output[0], output[1], cu_seqlens)
    ctx.max_seqlen = max_seqlen
    ctx.scale = scale


def _backward(ctx, grad_out, grad_lse):
    q, k, v, out, lse, cu_seqlens = ctx.saved_tensors
    dq, dk, dv = cudnn_gqa_bwd(
        grad_out.contiguous(), q, k, v, out, lse, cu_seqlens, ctx.max_seqlen, ctx.scale
    )
    return dq, dk, dv, None, None, None


cudnn_gqa_fwd.register_autograd(_backward, setup_context=_setup_context)


class PackedCuDNNAttention(Module):
    """Causal grouped-query attention on packed documents through torch's cuDNN ops."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: "PackedCuDNNAttention.Config"):
        super().__init__()

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if not isinstance(attention_masks, VarlenMetadata):
            raise TypeError(
                "PackedCuDNNAttention needs a VarlenMetadata, but got "
                f"{type(attention_masks).__name__}"
            )
        if q_BLNH.dtype != torch.bfloat16:
            raise TypeError(f"PackedCuDNNAttention needs bf16, but got {q_BLNH.dtype}")
        B, L, _, H = q_BLNH.shape
        q_TNH, k_TNH, v_TNH = (t.reshape(B * L, -1, H) for t in (q_BLNH, k_BLNH, v_BLNH))
        with spmd.no_typecheck():
            out_TNH, _ = cudnn_gqa_fwd(
                q_TNH, k_TNH, v_TNH,
                attention_masks.cu_seq_q,
                attention_masks.max_q,
                H**-0.5 if scale is None else scale,
            )
        return out_TNH.view(B, L, -1, H)


@override(
    target=GQAttention.Config,
    exact=True,
    description="Torch's cuDNN fused attention on packed documents, offsets built per microbatch.",
)
def packed_cudnn_attention(cfg: GQAttention.Config) -> PackedGQAttention.Config:
    return derive(
        cfg,
        PackedGQAttention.Config,
        max_documents=MAX_DOCUMENTS,
        inner_attention=derive(cfg.inner_attention, PackedCuDNNAttention.Config),
    )
