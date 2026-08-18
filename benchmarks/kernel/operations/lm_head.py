"""Arm builders for the ``lm_head`` kernel scenario.

Four loss strategies over one ``[B, L, D]`` hidden state and one
``[vocab, D]`` weight: full logits plus TorchTitan's ``CrossEntropyLoss``,
torch's ``linear_cross_entropy`` (which never materializes the logits), the
vendored TE Triton CE, and the Piper rework of it. Peak memory is the
secondary metric here precisely because that is where the strategies differ
most: at vocab 151936 the logit tensor is the largest allocation in the step.

``forward_backward`` is the only mode, and that is a constraint rather than a
choice: ``FusedLinearCrossEntropyLoss`` runs its backward inside ``__call__``,
so there is no point at which forward has finished and backward has not. The
other three arms are restricted to the same mode to stay comparable.

Every loss is built with ``_loss_compile()`` -- the production
``CompileConfig(components=["loss"])`` -- so the arms face the same Inductor
treatment they face end-to-end. ``fused_linear_ce`` is the one arm that owns
the LM head instead of receiving logits (torchtitan's ``LossWithLMHead``
protocol), which is why it builds its own ``nn.Linear``, copies the shared
weight into it, and reads the gradient back off ``lm_head.weight`` rather
than off a leaf tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import _randn
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.shape import PiperShape


def _loss_compile():
    """The production ``CompileConfig(components=["loss"])``, built on demand.

    A module-scope constant would need ``torchtitan.config`` at module scope,
    which the deferred-import rule forbids. Every builder calls this instead,
    so all four arms still face one treatment.
    """
    from torchtitan.config import CompileConfig

    return CompileConfig(enable=True, components=["loss"])


@dataclass
class LmHeadInputs:
    hidden: torch.Tensor
    weight: torch.Tensor
    labels: torch.Tensor
    valid_tokens: float


def lm_head_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> LmHeadInputs:
    batch, seq = workload.batch, workload.seq_len
    weight = _randn(
        (shape.vocab_size, shape.dim),
        device,
        generator,
        torch.bfloat16,
        1.0 / shape.dim**0.5,
    )
    labels = torch.randint(
        shape.vocab_size,
        (batch, seq),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )
    return LmHeadInputs(
        hidden=_randn((batch, seq, shape.dim), device, generator),
        weight=weight,
        labels=labels,
        valid_tokens=float(batch * seq),
    )


def _lm_head_arm(name: str, inputs: LmHeadInputs, loss_call) -> BuiltArm:
    hidden = inputs.hidden.clone().requires_grad_()
    weight = inputs.weight.clone().requires_grad_()

    def forward_backward():
        hidden.grad = None
        weight.grad = None
        loss = loss_call(hidden, weight)
        loss.backward()
        return loss

    def correctness_outputs() -> dict[str, torch.Tensor]:
        loss = forward_backward()
        return {
            "loss": loss.detach().float(),
            "hidden_grad": hidden.grad,
            "weight_grad": weight.grad,
        }

    return BuiltArm(
        name=name,
        calls={"forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
    )


def build_lm_head_baseline(
    shape: PiperShape, workload: KernelWorkload, inputs: LmHeadInputs
) -> BuiltArm:
    from torchtitan.components.loss import CrossEntropyLoss

    loss_obj = CrossEntropyLoss.Config(
        global_vocab_size=shape.vocab_size
    ).build(compile_config=_loss_compile())

    def loss_call(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss, _ = loss_obj(logits, inputs.labels, inputs.valid_tokens)
        return loss

    return _lm_head_arm("baseline", inputs, loss_call)


def build_lm_head_fused_linear_ce(
    shape: PiperShape, workload: KernelWorkload, inputs: LmHeadInputs
) -> BuiltArm:
    from benchmarks.models.piper_qwen3.components.lm_head.losses import (
        FusedLinearCrossEntropyLoss,
    )

    loss_obj = FusedLinearCrossEntropyLoss.Config(
        batch_chunk_size=None, chunking_method=None
    ).build(compile_config=_loss_compile())
    hidden = inputs.hidden.clone().requires_grad_()
    lm_head = nn.Linear(
        shape.dim, shape.vocab_size, bias=False, device=hidden.device
    ).to(torch.bfloat16)
    with torch.no_grad():
        lm_head.weight.copy_(inputs.weight)
    loss_obj.set_lm_head(lm_head)

    def forward_backward():
        hidden.grad = None
        lm_head.weight.grad = None
        loss, _ = loss_obj(hidden, inputs.labels, inputs.valid_tokens)
        loss.backward()
        return loss

    def correctness_outputs() -> dict[str, torch.Tensor]:
        loss = forward_backward()
        return {
            "loss": loss.detach().float(),
            "hidden_grad": hidden.grad,
            "weight_grad": lm_head.weight.grad,
        }

    return BuiltArm(
        name="fused_linear_ce",
        calls={"forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
    )


def build_lm_head_te_fused_ce(
    shape: PiperShape, workload: KernelWorkload, inputs: LmHeadInputs
) -> BuiltArm:
    from benchmarks.models.piper_qwen3.components.lm_head.losses import (
        TECrossEntropyLoss,
    )

    loss_obj = TECrossEntropyLoss.Config().build(compile_config=_loss_compile())

    def loss_call(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss, _ = loss_obj(logits, inputs.labels, inputs.valid_tokens)
        return loss

    return _lm_head_arm("te_fused_ce", inputs, loss_call)


def build_lm_head_piper_optimized_te_ce(
    shape: PiperShape, workload: KernelWorkload, inputs: LmHeadInputs
) -> BuiltArm:
    from benchmarks.models.piper_qwen3.components.lm_head.losses import (
        PiperOptimizedCrossEntropyLoss,
    )

    loss_obj = PiperOptimizedCrossEntropyLoss.Config().build(
        compile_config=_loss_compile()
    )

    def loss_call(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss, _ = loss_obj(logits, inputs.labels, inputs.valid_tokens)
        return loss

    return _lm_head_arm("piper_optimized_te_ce", inputs, loss_call)
