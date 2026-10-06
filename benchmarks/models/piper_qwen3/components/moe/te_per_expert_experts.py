"""The routed-expert GEMMs as one TE cuBLAS GEMM per expert, as Megatron's GroupedLinear runs them, as a TorchTitan override.

The GEMMs read the rows of each expert from a host tensor, which the host_count_dispatcher override
supplies. So this override needs that override and an expert-parallel degree above 1.

Activation:
    --override.imports benchmarks.models.piper_qwen3.components.moe.te_per_expert_experts.te_per_expert_experts
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
from transformer_engine.pytorch.module.base import (
    _2X_ACC_DGRAD,
    _2X_ACC_FPROP,
    _2X_ACC_WGRAD,
)

from torchtitan.config import derive, override
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.moe import GroupedExperts

TRACE_MARKER = "torchtitan_benchmarks::te_per_expert_mm"
"""The profiler range that each op opens, because the profiler records no event for a custom op."""


def _host_splits(x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor) -> list[int]:
    """The rows of each expert as a list; raise unless TE can read x, w and counts as they are."""
    if not (x.is_contiguous() and w.is_contiguous()):
        raise RuntimeError("te_per_expert_mm needs a contiguous x and w")
    if x.dtype != w.dtype:
        raise RuntimeError(f"te_per_expert_mm got x {x.dtype} and w {w.dtype}")
    if x.ndim != 2 or w.ndim != 3 or x.shape[1] != w.shape[2]:
        raise RuntimeError(
            f"te_per_expert_mm got x {tuple(x.shape)} and w {tuple(w.shape)}"
        )
    if (
        counts.device.type != "cpu"
        or counts.dtype != torch.int64
        or tuple(counts.shape) != (w.shape[0],)
    ):
        raise RuntimeError(
            f"te_per_expert_mm needs int64 host counts of shape ({w.shape[0]},), "
            f"got {counts.dtype} {tuple(counts.shape)} on {counts.device}"
        )
    splits = counts.tolist()
    if min(splits) < 0 or sum(splits) != x.shape[0]:
        raise RuntimeError(
            f"te_per_expert_mm got counts {splits} for {x.shape[0]} rows"
        )
    return splits


@torch.library.custom_op(
    "torchtitan_benchmarks::te_per_expert_mm", mutates_args=(), device_types="cuda"
)
def te_per_expert_mm(
    x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    """Rows of expert e of ``x`` (R, K) times ``w[e].T`` (K, N), as one (R, N) tensor."""
    with torch.profiler.record_function(TRACE_MARKER):
        splits = _host_splits(x, w, counts)
        y = x.new_empty((x.shape[0], w.shape[1]))
        if x.shape[0] == 0:
            return y
        num = w.shape[0]
        general_grouped_gemm(
            list(w.unbind(0)),
            list(torch.split(x, splits)),
            [y],
            [None] * num,
            x.dtype,
            single_output=True,
            m_splits=splits,
            use_split_accumulator=_2X_ACC_FPROP,
        )
        return y


@te_per_expert_mm.register_fake
def _te_per_expert_mm_fake(x, w, counts):
    return x.new_empty((x.shape[0], w.shape[1]))


@torch.library.custom_op(
    "torchtitan_benchmarks::te_per_expert_mm_backward",
    mutates_args=(),
    device_types="cuda",
)
def te_per_expert_mm_backward(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, counts: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gradients of ``te_per_expert_mm`` for x and w; the weight gradient of an empty expert is zero."""
    with torch.profiler.record_function(f"{TRACE_MARKER}_backward"):
        splits = _host_splits(x, w, counts)
        dy = dy.contiguous()
        if dy.dtype != x.dtype or tuple(dy.shape) != (x.shape[0], w.shape[1]):
            raise RuntimeError(
                f"te_per_expert_mm_backward got dy {dy.dtype} {tuple(dy.shape)} for "
                f"x {x.dtype} {tuple(x.shape)} and w {tuple(w.shape)}"
            )
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        dw = torch.empty_like(w, memory_format=torch.contiguous_format)
        if x.shape[0] == 0:
            return dx, dw.zero_()
        num = w.shape[0]
        none = [None] * num
        dy_parts = list(torch.split(dy, splits))
        general_grouped_gemm(
            list(w.unbind(0)),
            dy_parts,
            [dx],
            none,
            x.dtype,
            single_output=True,
            layout="NN",
            m_splits=splits,
            grad=True,
            use_split_accumulator=_2X_ACC_DGRAD,
        )
        general_grouped_gemm(
            list(torch.split(x, splits)),
            dy_parts,
            list(dw.unbind(0)),
            none,
            x.dtype,
            layout="NT",
            m_splits=splits,
            grad=True,
            accumulate=False,
            use_split_accumulator=_2X_ACC_WGRAD,
        )
        # TE can skip the GEMM of an empty expert, and leave its weight gradient unwritten.
        for e, rows in enumerate(splits):
            if rows == 0:
                dw[e].zero_()
        return dx, dw


@te_per_expert_mm_backward.register_fake
def _te_per_expert_mm_backward_fake(dy, x, w, counts):
    return x.new_empty(x.shape), w.new_empty(w.shape)


def _setup_context(ctx, inputs, output) -> None:
    x, w, counts = inputs
    ctx.save_for_backward(x, w, counts)


def _backward(ctx, dy):
    x, w, counts = ctx.saved_tensors
    dx, dw = te_per_expert_mm_backward(dy, x, w, counts)
    return dx, dw, None


te_per_expert_mm.register_autograd(_backward, setup_context=_setup_context)


class TEPerExpertExperts(GroupedExperts):
    """GroupedExperts with each GEMM as one TE cuBLAS GEMM per expert; the parameters do not change."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if get_spmd_backend() == "spmd_types":
            raise RuntimeError(
                "TEPerExpertExperts has no SPMD type rule for its custom op"
            )
        if num_tokens_per_expert_E.device.type != "cpu":
            raise RuntimeError(
                "TEPerExpertExperts needs the rows of each expert on the host, "
                f"and got them on {num_tokens_per_expert_E.device}; import the "
                "host_count_dispatcher override and run at ep > 1"
            )
        if isinstance(self.w1_EFD, DTensor):
            w1_EFD = self.w1_EFD.to_local()
            assert isinstance(self.w2_EDF, DTensor)
            w2_EDF = self.w2_EDF.to_local()
            assert isinstance(self.w3_EFD, DTensor)
            w3_EFD = self.w3_EFD.to_local()
        else:
            w1_EFD = self.w1_EFD
            w2_EDF = self.w2_EDF
            w3_EFD = self.w3_EFD

        x_RD_bf16 = x_RD.bfloat16().contiguous()
        h_RF = F.silu(
            te_per_expert_mm(
                x_RD_bf16, w1_EFD.bfloat16().contiguous(), num_tokens_per_expert_E
            )
        )
        h_RF = h_RF * te_per_expert_mm(
            x_RD_bf16, w3_EFD.bfloat16().contiguous(), num_tokens_per_expert_E
        )
        return te_per_expert_mm(
            h_RF.contiguous(), w2_EDF.bfloat16().contiguous(), num_tokens_per_expert_E
        ).type_as(x_RD)


@override(
    target=GroupedExperts.Config,
    exact=True,
    description="Routed-expert GEMMs as one TE cuBLAS GEMM per expert, rows from the host.",
)
def te_per_expert_experts(cfg: GroupedExperts.Config) -> TEPerExpertExperts.Config:
    return derive(cfg, TEPerExpertExperts.Config)
