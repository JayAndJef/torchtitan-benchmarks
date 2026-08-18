"""Arm builders for the ``qkv`` kernel scenario.

Two arms: TorchTitan's ``QKVLinear`` (a Q GEMM and a fused KV GEMM) against
``FusedQKVLinear`` (one wqkv GEMM plus a split). Both are built from the
*same* three-weight state dict, and the fused module's own state-dict merge
hook is what turns ``wq``/``wk``/``wv`` into a single ``wqkv`` -- the arms
therefore start from bit-identical weights by construction rather than by a
transposition written here, and ``tests`` treats that hook as part of the
fork's contract.

That is also why the scenario carries an informational ``bitwise`` gate on
the outputs alongside the enforced ``rel_l2`` ones: with identical weights
the two arms *should* agree exactly, and for a while they did, until compiled
GEMM epilogues broke bit-identity. The check still records the difference
without failing the run, so a change that restores or further degrades exact
agreement is visible in results.json instead of invisible.

``qkv_reference`` quantizes each fp32 weight to bf16 before promoting to
fp64, so the fp64 truth is the truth for *the weights the arms actually
hold*; comparing against an fp64 projection of the unrounded weights would
charge both arms for the input cast.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common import FusedQKVLinear, Linear, QKVLinear

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    WEIGHT_STD,
    _compile_module,
    _randn,
    _reset_grads,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.shape import PiperShape


@dataclass
class QkvInputs:
    x: torch.Tensor
    gq: torch.Tensor
    gk: torch.Tensor
    gv: torch.Tensor
    weight_state: dict[str, torch.Tensor]


def qkv_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> QkvInputs:
    batch, seq = workload.batch, workload.seq_len
    q_out = shape.n_heads * shape.head_dim
    kv_out = shape.n_kv_heads * shape.head_dim
    weight_state = {
        "wq.weight": _randn(
            (q_out, shape.dim), device, generator, torch.float32, WEIGHT_STD
        ),
        "wk.weight": _randn(
            (kv_out, shape.dim), device, generator, torch.float32, WEIGHT_STD
        ),
        "wv.weight": _randn(
            (kv_out, shape.dim), device, generator, torch.float32, WEIGHT_STD
        ),
    }
    return QkvInputs(
        x=_randn((batch, seq, shape.dim), device, generator),
        gq=_randn(
            (batch, seq, shape.n_heads, shape.head_dim), device, generator
        ),
        gk=_randn(
            (batch, seq, shape.n_kv_heads, shape.head_dim), device, generator
        ),
        gv=_randn(
            (batch, seq, shape.n_kv_heads, shape.head_dim), device, generator
        ),
        weight_state=weight_state,
    )


def qkv_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: QkvInputs
) -> dict[str, torch.Tensor]:
    batch, seq = workload.batch, workload.seq_len
    x64 = inputs.x.double()

    def project(weight: torch.Tensor, heads: int) -> torch.Tensor:
        quantized = weight.to(torch.bfloat16).double()
        return F.linear(x64, quantized).view(
            batch, seq, heads, shape.head_dim
        )

    return {
        "q_out": project(inputs.weight_state["wq.weight"], shape.n_heads),
        "k_out": project(inputs.weight_state["wk.weight"], shape.n_kv_heads),
        "v_out": project(inputs.weight_state["wv.weight"], shape.n_kv_heads),
    }


def _qkv_arm(name: str, module: nn.Module, inputs: QkvInputs) -> BuiltArm:
    forward_leaf = inputs.x.clone().requires_grad_()
    backward_leaf = inputs.x.clone().requires_grad_()
    round_trip_leaf = inputs.x.clone().requires_grad_()
    check_leaf = inputs.x.clone().requires_grad_()
    grads = (inputs.gq, inputs.gk, inputs.gv)
    retained = module(backward_leaf)

    def forward():
        return module(forward_leaf)

    def backward() -> None:
        _reset_grads(backward_leaf, module)
        torch.autograd.backward(retained, grads, retain_graph=True)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, module)
        torch.autograd.backward(module(round_trip_leaf), grads)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, module)
        q_out, k_out, v_out = module(check_leaf)
        torch.autograd.backward((q_out, k_out, v_out), grads)
        return {
            "q_out": q_out.detach(),
            "k_out": k_out.detach(),
            "v_out": v_out.detach(),
            "x_grad": check_leaf.grad,
        }

    return BuiltArm(
        name=name,
        calls={
            "forward": forward,
            "backward": backward,
            "forward_backward": forward_backward,
        },
        correctness_outputs=correctness_outputs,
    )


def _finalize_qkv(module: nn.Module, inputs: QkvInputs) -> nn.Module:
    module.to(inputs.x.device)
    module.load_state_dict(inputs.weight_state)
    module.to(torch.bfloat16)
    return _compile_module(module)


def build_qkv_baseline(
    shape: PiperShape, workload: KernelWorkload, inputs: QkvInputs
) -> BuiltArm:
    module = QKVLinear.Config(
        head_dim=shape.head_dim,
        wq=Linear.Config(
            in_features=shape.dim, out_features=shape.n_heads * shape.head_dim
        ),
        wkv=Linear.Config(
            in_features=shape.dim,
            out_features=shape.n_kv_heads * shape.head_dim,
        ),
    ).build()
    return _qkv_arm("baseline", _finalize_qkv(module, inputs), inputs)


def build_qkv_fused_qkv(
    shape: PiperShape, workload: KernelWorkload, inputs: QkvInputs
) -> BuiltArm:
    module = FusedQKVLinear.Config(
        head_dim=shape.head_dim,
        n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads,
        wqkv=Linear.Config(
            in_features=shape.dim, out_features=shape.qkv_out_features
        ),
    ).build()
    return _qkv_arm("fused_qkv", _finalize_qkv(module, inputs), inputs)
