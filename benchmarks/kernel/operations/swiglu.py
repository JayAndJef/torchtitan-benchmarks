"""Arm builders for the ``swiglu`` kernel scenario.

Three arms at *whole grouped-expert layer* scope, not activation scope:
TorchTitan's ``GroupedExperts`` against the two Piper variants, which share a
fused w13 GEMM and differ only in the activation (a combined ``[R, 2F]``
custom Triton op versus plain ops left to Inductor). Measuring the layer is
the point -- the fused layout's win is a materialization the surrounding
GEMMs pay for or avoid, so an activation-only benchmark inverts the verdict.

All three are built by one helper from one fp32 state dict, loaded before the
cast to bf16, so the arms start from bit-identical weights. The two weight-
gradient accessors are the only asymmetry: the fused layout stores w1 and w3
interleaved in a single ``w13`` parameter, so ``_fused_weight_grads`` slices
that gradient back into the stock layout's names for the correctness gate to
compare against ``_stock_weight_grads``. Without that unpacking the gate
would silently compare nothing.

The scenario carries ``requires_balanced_routing``: ``swiglu_inputs`` hands
every expert an equal ``rows // num_experts`` slice, so a workload where
``batch * seq_len * top_k`` does not divide by ``num_experts`` builds rows the
counts do not cover. Both ``benchmarks.kernel.runner`` and
``benchmarks.kernel.engine.run`` refuse such a run rather than rounding; this
module assumes they did.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torchtitan.models.common.moe import GroupedExperts

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    WEIGHT_STD,
    _compile_module,
    _randn,
    _reset_grads,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu import (
    CombinedSwiGLUFusedGroupedExperts,
    InductorSwiGLUFusedGroupedExperts,
)
from benchmarks.models.piper_qwen3.shape import PiperShape


@dataclass
class SwigluInputs:
    x: torch.Tensor
    grad_out: torch.Tensor
    counts: torch.Tensor
    stock_state: dict[str, torch.Tensor]


def swiglu_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> SwigluInputs:
    rows = workload.batch * workload.seq_len * shape.top_k
    hidden = shape.moe_hidden_dim
    per_expert = rows // shape.num_experts
    counts = torch.full(
        (shape.num_experts,), per_expert, device=device, dtype=torch.int32
    )
    stock_state = {
        "w1_EFD": _randn(
            (shape.num_experts, hidden, shape.dim),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
        "w2_EDF": _randn(
            (shape.num_experts, shape.dim, hidden),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
        "w3_EFD": _randn(
            (shape.num_experts, hidden, shape.dim),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
    }
    return SwigluInputs(
        x=_randn((rows, shape.dim), device, generator),
        grad_out=_randn((rows, shape.dim), device, generator),
        counts=counts,
        stock_state=stock_state,
    )


def _swiglu_module_arm(
    name: str,
    module: nn.Module,
    inputs: SwigluInputs,
    weight_grads,
) -> BuiltArm:
    forward_leaf = inputs.x.clone().requires_grad_()
    backward_leaf = inputs.x.clone().requires_grad_()
    round_trip_leaf = inputs.x.clone().requires_grad_()
    check_leaf = inputs.x.clone().requires_grad_()
    retained = module(backward_leaf, inputs.counts)

    def forward():
        return module(forward_leaf, inputs.counts)

    def backward() -> None:
        _reset_grads(backward_leaf, module)
        torch.autograd.backward(retained, inputs.grad_out, retain_graph=True)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, module)
        out = module(round_trip_leaf, inputs.counts)
        torch.autograd.backward(out, inputs.grad_out)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, module)
        out = module(check_leaf, inputs.counts)
        torch.autograd.backward(out, inputs.grad_out)
        grads = weight_grads(module)
        return {"out": out.detach(), "x_grad": check_leaf.grad, **grads}

    return BuiltArm(
        name=name,
        calls={
            "forward": forward,
            "backward": backward,
            "forward_backward": forward_backward,
        },
        correctness_outputs=correctness_outputs,
    )


def _stock_weight_grads(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        "w1_grad": module.w1_EFD.grad,
        "w2_grad": module.w2_EDF.grad,
        "w3_grad": module.w3_EFD.grad,
    }


def _fused_weight_grads(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        "w1_grad": module.w13.grad[:, :, 0, :],
        "w2_grad": module.w2_EDF.grad,
        "w3_grad": module.w13.grad[:, :, 1, :],
    }


def _build_swiglu_module(config_cls, shape: PiperShape, inputs: SwigluInputs):
    module = config_cls.Config(
        dim=shape.dim,
        hidden_dim=shape.moe_hidden_dim,
        num_experts=shape.num_experts,
    ).build()
    module.to(inputs.x.device)
    module.load_state_dict(inputs.stock_state)
    module.to(torch.bfloat16)
    return _compile_module(module)


def build_swiglu_baseline(
    shape: PiperShape, workload: KernelWorkload, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(GroupedExperts, shape, inputs)
    return _swiglu_module_arm(
        "baseline", module, inputs, _stock_weight_grads
    )


def build_swiglu_piper_optimized_triton(
    shape: PiperShape, workload: KernelWorkload, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(
        CombinedSwiGLUFusedGroupedExperts, shape, inputs
    )
    return _swiglu_module_arm(
        "piper_optimized_triton", module, inputs, _fused_weight_grads
    )


def build_swiglu_piper_optimized_inductor(
    shape: PiperShape, workload: KernelWorkload, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(
        InductorSwiGLUFusedGroupedExperts, shape, inputs
    )
    return _swiglu_module_arm(
        "piper_optimized_inductor", module, inputs, _fused_weight_grads
    )
