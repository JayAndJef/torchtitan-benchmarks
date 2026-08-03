# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Derived from TorchTitan's torchtitan/overrides/fused_swiglu.py at commit
# b5eb9d92f5c5fee60d4a9bf7a91dc6f1d3f1c1a1. See LICENSE.torchtitan.

"""Fused grouped experts with a combined-layout SwiGLU gradient.

TorchTitan's fused grouped-expert projection produces interleaved gate/up
columns, then presents them to its activation custom op as two stride-2 views.
Its backward returns two contiguous gradients, so autograd must interleave and
duplicate them before the fused projection's grouped-MM backward consumers.

This derivative keeps the same fused weight and grouped matrix multiplications,
but makes the activation custom op consume and return the combined ``[R, 2F]``
layout directly. The Triton math is otherwise the same as TorchTitan's kernel.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl
from torch.distributed.tensor import DTensor

from torchtitan.config import derive, override
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.protocols.sharding import ShardingConfig


__all__ = [
    "CombinedSwiGLUFusedGroupedExperts",
    "combined_silu_and_mul_backward_kernel",
    "combined_silu_and_mul_forward_kernel",
    "combined_silu_and_mul_op",
    "piper_optimized_fused_grouped_experts",
]


_MAX_BLOCK_N = 2048
_BLOCK_M = 4


@triton.jit
def _combined_silu_and_mul_forward_kernel(
    gate_up,
    out,
    offsets,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    NUM_OFFSETS: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    GATE_UP_ROW_STRIDE: tl.constexpr,
    GATE_UP_COL_STRIDE: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    OUT_COL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Compute ``silu(gate) * up`` from interleaved gate/up columns."""
    row_start = tl.program_id(0) * BLOCK_M
    row_limit = NUM_ROWS
    if HAS_OFFSETS:
        row_limit = tl.load(offsets + NUM_OFFSETS - 1)
        if row_start >= row_limit:
            return

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < row_limit) & (cols[None, :] < NUM_COLS)
    combined_offsets = (
        rows[:, None] * GATE_UP_ROW_STRIDE
        + 2 * cols[None, :] * GATE_UP_COL_STRIDE
    )

    gate = tl.load(gate_up + combined_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(
        gate_up + combined_offsets + GATE_UP_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    silu = gate * tl.sigmoid(gate)
    tl.store(
        out + rows[:, None] * OUT_ROW_STRIDE + cols[None, :] * OUT_COL_STRIDE,
        silu * up,
        mask=mask,
    )


@triton.jit
def _combined_silu_and_mul_backward_kernel(
    grad_out,
    gate_up,
    grad_gate_up,
    offsets,
    NUM_ROWS: tl.constexpr,
    NUM_COLS: tl.constexpr,
    NUM_OFFSETS: tl.constexpr,
    HAS_OFFSETS: tl.constexpr,
    GRAD_OUT_ROW_STRIDE: tl.constexpr,
    GRAD_OUT_COL_STRIDE: tl.constexpr,
    GATE_UP_ROW_STRIDE: tl.constexpr,
    GATE_UP_COL_STRIDE: tl.constexpr,
    GRAD_GATE_UP_ROW_STRIDE: tl.constexpr,
    GRAD_GATE_UP_COL_STRIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Write gate and up gradients directly into their interleaved layout."""
    row_start = tl.program_id(0) * BLOCK_M
    row_limit = NUM_ROWS
    if HAS_OFFSETS:
        row_limit = tl.load(offsets + NUM_OFFSETS - 1)
        if row_start >= row_limit:
            return

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < row_limit) & (cols[None, :] < NUM_COLS)
    combined_offsets = (
        rows[:, None] * GATE_UP_ROW_STRIDE
        + 2 * cols[None, :] * GATE_UP_COL_STRIDE
    )

    grad = tl.load(
        grad_out
        + rows[:, None] * GRAD_OUT_ROW_STRIDE
        + cols[None, :] * GRAD_OUT_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(gate_up + combined_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(
        gate_up + combined_offsets + GATE_UP_COL_STRIDE,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    sigmoid = tl.sigmoid(gate)
    silu = gate * sigmoid
    silu_grad = sigmoid * (1.0 + gate * (1.0 - sigmoid))
    grad_combined_offsets = (
        rows[:, None] * GRAD_GATE_UP_ROW_STRIDE
        + 2 * cols[None, :] * GRAD_GATE_UP_COL_STRIDE
    )
    tl.store(
        grad_gate_up + grad_combined_offsets,
        grad * up * silu_grad,
        mask=mask,
    )
    tl.store(
        grad_gate_up + grad_combined_offsets + GRAD_GATE_UP_COL_STRIDE,
        grad * silu,
        mask=mask,
    )


def _activation_shape(gate_up: torch.Tensor) -> tuple[int, int]:
    if gate_up.ndim != 2:
        raise ValueError("gate_up must have shape [num_rows, 2 * hidden_dim]")
    if gate_up.shape[1] % 2:
        raise ValueError("gate_up must contain an even number of columns")
    return gate_up.shape[0], gate_up.shape[1] // 2


def combined_silu_and_mul_forward_kernel(
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SwiGLU directly from a combined interleaved projection."""
    if offsets is not None and offsets.numel() == 0:
        raise ValueError("offsets must be non-empty when provided")
    num_rows, num_cols = _activation_shape(gate_up)
    out = torch.empty(
        (num_rows, num_cols),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(num_cols))
    grid = (triton.cdiv(num_rows, _BLOCK_M), triton.cdiv(num_cols, block_n))
    _combined_silu_and_mul_forward_kernel[grid](
        gate_up,
        out,
        offsets if offsets is not None else gate_up,
        NUM_ROWS=num_rows,
        NUM_COLS=num_cols,
        NUM_OFFSETS=offsets.numel() if offsets is not None else 0,
        HAS_OFFSETS=offsets is not None,
        GATE_UP_ROW_STRIDE=gate_up.stride(0),
        GATE_UP_COL_STRIDE=gate_up.stride(1),
        OUT_ROW_STRIDE=out.stride(0),
        OUT_COL_STRIDE=out.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out


def combined_silu_and_mul_backward_kernel(
    grad_out: torch.Tensor,
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a single interleaved gradient for the combined projection."""
    if offsets is not None and offsets.numel() == 0:
        raise ValueError("offsets must be non-empty when provided")
    num_rows, num_cols = _activation_shape(gate_up)
    grad_gate_up = torch.empty_like(gate_up, memory_format=torch.contiguous_format)
    block_n = min(_MAX_BLOCK_N, triton.next_power_of_2(num_cols))
    grid = (triton.cdiv(num_rows, _BLOCK_M), triton.cdiv(num_cols, block_n))
    _combined_silu_and_mul_backward_kernel[grid](
        grad_out,
        gate_up,
        grad_gate_up,
        offsets if offsets is not None else gate_up,
        NUM_ROWS=num_rows,
        NUM_COLS=num_cols,
        NUM_OFFSETS=offsets.numel() if offsets is not None else 0,
        HAS_OFFSETS=offsets is not None,
        GRAD_OUT_ROW_STRIDE=grad_out.stride(0),
        GRAD_OUT_COL_STRIDE=grad_out.stride(1),
        GATE_UP_ROW_STRIDE=gate_up.stride(0),
        GATE_UP_COL_STRIDE=gate_up.stride(1),
        GRAD_GATE_UP_ROW_STRIDE=grad_gate_up.stride(0),
        GRAD_GATE_UP_COL_STRIDE=grad_gate_up.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return grad_gate_up


@torch.library.custom_op(
    "torchtitan_benchmarks::combined_silu_and_mul",
    mutates_args=(),
    device_types="cuda",
)
def combined_silu_and_mul_op(
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    return combined_silu_and_mul_forward_kernel(gate_up, offsets)


@combined_silu_and_mul_op.register_fake
def _combined_silu_and_mul_fake(
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    del offsets
    return torch.empty(
        (gate_up.shape[0], gate_up.shape[1] // 2),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )


@torch.library.custom_op(
    "torchtitan_benchmarks::combined_silu_and_mul_backward",
    mutates_args=(),
    device_types="cuda",
)
def _combined_silu_and_mul_backward_op(
    grad_out: torch.Tensor,
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    return combined_silu_and_mul_backward_kernel(
        grad_out.contiguous(),
        gate_up,
        offsets,
    )


@_combined_silu_and_mul_backward_op.register_fake
def _combined_silu_and_mul_backward_fake(
    grad_out: torch.Tensor,
    gate_up: torch.Tensor,
    offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    del grad_out, offsets
    return torch.empty_like(gate_up, memory_format=torch.contiguous_format)


def _setup_context(ctx, inputs, output) -> None:
    del output
    gate_up, offsets = inputs
    ctx.has_offsets = offsets is not None
    if offsets is None:
        ctx.save_for_backward(gate_up)
    else:
        ctx.save_for_backward(gate_up, offsets)


def _backward(ctx, grad_out: torch.Tensor):
    if ctx.has_offsets:
        gate_up, offsets = ctx.saved_tensors
    else:
        (gate_up,) = ctx.saved_tensors
        offsets = None
    grad_gate_up = _combined_silu_and_mul_backward_op(
        grad_out,
        gate_up,
        offsets,
    )
    return grad_gate_up, None


combined_silu_and_mul_op.register_autograd(_backward, setup_context=_setup_context)


def _make_fused_gate_up_init(
    gate_init: Callable,
    up_init: Callable,
) -> Callable:
    def _init(weight: torch.Tensor) -> None:
        gate_init(weight[:, :, 0, :])
        up_init(weight[:, :, 1, :])

    return _init


class CombinedSwiGLUFusedGroupedExperts(GroupedExperts):
    """Grouped experts whose activation gradient stays in combined layout."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        del self.w1_EFD
        del self.w3_EFD
        self.w13 = torch.nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim, 2, config.dim)
        )
        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w13, DTensor):
            w13 = self.w13.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
        else:
            w13 = self.w13
            w2_EDF = self.w2_EDF

        num_experts, hidden_dim, _, dim = w13.shape
        offsets_E = torch.cumsum(
            num_tokens_per_expert_E,
            dim=0,
            dtype=torch.int32,
        )
        w13_E_D_2F = w13.bfloat16().reshape(
            num_experts,
            hidden_dim * 2,
            dim,
        ).transpose(-2, -1)
        gate_up_R2F = torch._grouped_mm(
            x_RD.bfloat16(),
            w13_E_D_2F,
            offs=offsets_E,
        )
        h_RF = combined_silu_and_mul_op(gate_up_R2F, offsets_E)
        return torch._grouped_mm(
            h_RF,
            w2_EDF.bfloat16().transpose(-2, -1),
            offs=offsets_E,
        ).type_as(x_RD)

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        del module, local_metadata
        w13 = state_dict.pop(f"{prefix}w13")
        state_dict[f"{prefix}w1_EFD"] = w13[:, :, 0, :].contiguous()
        state_dict[f"{prefix}w3_EFD"] = w13[:, :, 1, :].contiguous()

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        del module, args
        w1_key = f"{prefix}w1_EFD"
        w3_key = f"{prefix}w3_EFD"
        if w1_key in state_dict and w3_key in state_dict:
            state_dict[f"{prefix}w13"] = torch.stack(
                [state_dict.pop(w1_key), state_dict.pop(w3_key)],
                dim=2,
            )


def _fused_param_init(param_init: dict | None) -> dict | None:
    if param_init is None:
        return None
    gate_init = param_init.get("w1_EFD")
    up_init = param_init.get("w3_EFD")
    fused = {
        name: init
        for name, init in param_init.items()
        if name not in ("w1_EFD", "w3_EFD")
    }
    if gate_init is not None and up_init is not None:
        fused["w13"] = _make_fused_gate_up_init(gate_init, up_init)
    return fused or None


def _fused_sharding(base: ShardingConfig) -> ShardingConfig:
    state = dict(base.state_shardings)
    gate_layout = state.pop("w1_EFD")
    state.pop("w3_EFD")
    state["w13"] = gate_layout
    return replace(base, state_shardings=state)


@override(
    target=GroupedExperts.Config,
    description=(
        "Fuse routed-expert gate/up projection and keep its activation gradient "
        "in combined layout."
    ),
)
def piper_optimized_fused_grouped_experts(
    cfg: GroupedExperts.Config,
) -> GroupedExperts.Config:
    if type(cfg) is not GroupedExperts.Config:
        return cfg

    fused = derive(
        cfg,
        CombinedSwiGLUFusedGroupedExperts.Config,
        param_init=_fused_param_init(cfg.param_init),
    )
    if cfg.sharding_config is not None:
        fused.sharding_config = _fused_sharding(cfg.sharding_config)
    return fused
