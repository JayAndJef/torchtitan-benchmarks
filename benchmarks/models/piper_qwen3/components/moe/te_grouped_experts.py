"""The routed-expert GEMMs on TransformerEngine's cuBLASLt grouped GEMM, as a TorchTitan override.

The split sizes stay on the device, so each GEMM is one launch with no host sync.

Activation:
    --override.imports benchmarks.models.piper_qwen3.components.moe.te_grouped_experts.te_grouped_experts
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformer_engine.pytorch.cpp_extensions import (
    general_grouped_gemm_for_grouped_tensor,
)
from transformer_engine.pytorch.module.base import (
    _2X_ACC_DGRAD,
    _2X_ACC_FPROP,
    _2X_ACC_WGRAD,
)
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensorStorage

# transformer_engine.pytorch loads the extension, so this import comes after it.
import transformer_engine_torch as tex

from torchtitan.config import derive, override
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.moe import GroupedExperts

MIN_CUBLASLT_VERSION = 130400
"""The cuBLASLt version that TE's grouped GEMM asserts on Hopper."""

if tex.get_cublasLt_version() < MIN_CUBLASLT_VERSION:
    raise RuntimeError(
        f"TE grouped GEMM needs cuBLASLt >= {MIN_CUBLASLT_VERSION}, but this "
        f"process loads {tex.get_cublasLt_version()}; sync the cu132 torch pin, "
        "which brings nvidia-cublas 13.4"
    )


def _check_operands(x: torch.Tensor, w: torch.Tensor, splits: torch.Tensor) -> None:
    """Raise unless TE can read x, w and splits as they are."""
    if not (x.is_contiguous() and w.is_contiguous()):
        raise RuntimeError("te_grouped_mm needs a contiguous x and w")
    if x.dtype != w.dtype:
        raise RuntimeError(f"te_grouped_mm got x {x.dtype} and w {w.dtype}")
    if x.ndim != 2 or w.ndim != 3 or x.shape[1] != w.shape[2]:
        raise RuntimeError(f"te_grouped_mm got x {tuple(x.shape)} and w {tuple(w.shape)}")
    if splits.dtype != torch.int64 or tuple(splits.shape) != (w.shape[0],):
        raise RuntimeError(
            f"te_grouped_mm needs int64 splits of shape ({w.shape[0]},), got "
            f"{splits.dtype} {tuple(splits.shape)}"
        )


def _packed(
    data: torch.Tensor, splits: torch.Tensor, offsets: torch.Tensor
) -> GroupedTensorStorage:
    """``data`` (R, C) as E row groups of sizes ``splits``."""
    return GroupedTensorStorage(
        shape=tuple(data.shape),
        dtype=data.dtype,
        num_tensors=splits.shape[0],
        quantizer=None,
        data=data.reshape(-1),
        first_dims=splits,
        tensor_offsets=offsets * data.shape[1],
    )


def _stacked(data: torch.Tensor) -> GroupedTensorStorage:
    """``data`` (E, N, K) as E matrices of one shape."""
    num, rows, cols = data.shape
    return GroupedTensorStorage(
        shape=(num * rows, cols),
        dtype=data.dtype,
        num_tensors=num,
        shapes=[(rows, cols)] * num,
        quantizer=None,
        data=data.reshape(-1),
    )


@torch.library.custom_op(
    "torchtitan_benchmarks::te_grouped_mm", mutates_args=(), device_types="cuda"
)
def te_grouped_mm(x: torch.Tensor, w: torch.Tensor, splits: torch.Tensor) -> torch.Tensor:
    """Rows of expert e of ``x`` (R, K) times ``w[e].T`` (K, N), as one (R, N) tensor."""
    _check_operands(x, w, splits)
    y = x.new_empty((x.shape[0], w.shape[1]))
    if x.shape[0] == 0:
        return y
    offsets = tex.splits_to_offsets(splits, 1)
    general_grouped_gemm_for_grouped_tensor(
        _stacked(w),
        _packed(x, splits, offsets),
        _packed(y, splits, offsets),
        layout="TN",
        use_split_accumulator=_2X_ACC_FPROP,
    )
    return y


@te_grouped_mm.register_fake
def _te_grouped_mm_fake(x, w, splits):
    return x.new_empty((x.shape[0], w.shape[1]))


@torch.library.custom_op(
    "torchtitan_benchmarks::te_grouped_mm_backward",
    mutates_args=(),
    device_types="cuda",
)
def te_grouped_mm_backward(
    dy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, splits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The gradients of ``te_grouped_mm`` for x and w."""
    _check_operands(x, w, splits)
    dy = dy.contiguous()
    if dy.dtype != x.dtype or tuple(dy.shape) != (x.shape[0], w.shape[1]):
        raise RuntimeError(
            f"te_grouped_mm_backward got dy {dy.dtype} {tuple(dy.shape)} for "
            f"x {x.dtype} {tuple(x.shape)} and w {tuple(w.shape)}"
        )
    dx = torch.empty_like(x, memory_format=torch.contiguous_format)
    dw = torch.empty_like(w, memory_format=torch.contiguous_format)
    if x.shape[0] == 0:
        return dx, dw.zero_()
    offsets = tex.splits_to_offsets(splits, 1)
    grouped_dy = _packed(dy, splits, offsets)
    general_grouped_gemm_for_grouped_tensor(
        _stacked(w),
        grouped_dy,
        _packed(dx, splits, offsets),
        layout="NN",
        use_split_accumulator=_2X_ACC_DGRAD,
    )
    general_grouped_gemm_for_grouped_tensor(
        _packed(x, splits, offsets),
        grouped_dy,
        _stacked(dw),
        layout="NT",
        use_split_accumulator=_2X_ACC_WGRAD,
    )
    return dx, dw


@te_grouped_mm_backward.register_fake
def _te_grouped_mm_backward_fake(dy, x, w, splits):
    return x.new_empty(x.shape), w.new_empty(w.shape)


def _setup_context(ctx, inputs, output) -> None:
    x, w, splits = inputs
    ctx.save_for_backward(x, w, splits)


def _backward(ctx, dy):
    x, w, splits = ctx.saved_tensors
    dx, dw = te_grouped_mm_backward(dy, x, w, splits)
    return dx, dw, None


te_grouped_mm.register_autograd(_backward, setup_context=_setup_context)


class TEGroupedExperts(GroupedExperts):
    """GroupedExperts with each grouped GEMM on TE's cuBLASLt grouped GEMM; the parameters do not change."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        pass

    def __init__(self, config: "TEGroupedExperts.Config"):
        super().__init__(config)
        capability = torch.cuda.get_device_capability()
        if not (9, 0) <= capability <= (11, 0):
            raise RuntimeError(
                "TEGroupedExperts needs a compute capability from 9.0 to 11.0, "
                f"but the device has {capability}"
            )

    def forward(
        self,
        x_RD: torch.Tensor,
        num_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        if get_spmd_backend() == "spmd_types":
            raise RuntimeError(
                "TEGroupedExperts has no SPMD type rule for its custom op"
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

        splits_E = num_tokens_per_expert_E.to(torch.int64)
        x_RD_bf16 = x_RD.bfloat16().contiguous()
        h_RF = F.silu(
            te_grouped_mm(x_RD_bf16, w1_EFD.bfloat16().contiguous(), splits_E)
        )
        h_RF = h_RF * te_grouped_mm(
            x_RD_bf16, w3_EFD.bfloat16().contiguous(), splits_E
        )
        return te_grouped_mm(
            h_RF.contiguous(), w2_EDF.bfloat16().contiguous(), splits_E
        ).type_as(x_RD)


@override(
    target=GroupedExperts.Config,
    exact=True,
    description="Routed-expert GEMMs on TE's cuBLASLt grouped GEMM, split sizes on the device.",
)
def te_grouped_experts(cfg: GroupedExperts.Config) -> TEGroupedExperts.Config:
    return derive(cfg, TEGroupedExperts.Config)
