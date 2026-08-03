"""Benchmark-only LM-head loss implementations for piper-1B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from torchtitan.components.loss import BaseLoss, IGNORE_INDEX
from torchtitan.config import CompileConfig

from piper1b.lm_head.te_cross_entropy import parallel_cross_entropy


class TECrossEntropyLoss(BaseLoss):
    """TransformerEngine fused CE with TorchTitan token normalization."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        self.fn = self._loss_sum
        self._maybe_compile(compile_config)

    @staticmethod
    def _loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return parallel_cross_entropy(
            logits,
            labels,
            reduce_loss=False,
            ignore_idx=IGNORE_INDEX,
        ).sum()


class FusedLinearCrossEntropyLoss(BaseLoss):
    """Full-token PyTorch linear-CE for the single-GPU benchmark."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        batch_chunk_size: int | None = None
        chunking_method: str | None = None

    def __init__(
        self,
        config: Config,
        *,
        compile_config: CompileConfig | None = None,
    ):
        self.lm_head: nn.Module | None = None
        self.options = torch.nn.LinearCrossEntropyOptions(
            batch_chunk_size=config.batch_chunk_size,
            chunking_method=config.chunking_method,
        )

    def set_lm_head(self, lm_head: nn.Module) -> None:
        """Provide the LM head skipped by the model forward."""
        self.lm_head = lm_head

    def __call__(
        self,
        pred: torch.Tensor,
        labels: torch.Tensor,
        global_valid_tokens: float | None = None,
        **loss_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if loss_inputs:
            raise ValueError(
                "FusedLinearCrossEntropyLoss does not accept extra loss inputs"
            )
        if isinstance(pred, DTensor) or isinstance(labels, DTensor):
            raise ValueError(
                "FusedLinearCrossEntropyLoss is benchmark-only and does not support DTensor"
            )

        lm_head = self.lm_head
        assert lm_head is not None, "Set lm_head before calling fused linear CE"
        if getattr(lm_head, "bias", None) is not None:
            raise ValueError("FusedLinearCrossEntropyLoss requires a bias-free lm_head")

        hidden_states = pred
        requires_grad = hidden_states.requires_grad
        hidden_leaf = hidden_states.detach().requires_grad_(requires_grad)
        fsdp_enabled = isinstance(lm_head, FSDPModule)
        if fsdp_enabled:
            lm_head.unshard()

        loss = F.linear_cross_entropy(
            hidden_leaf.flatten(0, 1),
            lm_head.weight,
            labels.flatten(0, 1),
            reduction="sum",
            ignore_index=IGNORE_INDEX,
            options=self.options,
        )
        if global_valid_tokens is not None:
            loss = loss / global_valid_tokens
        total_loss = loss.detach()

        if not requires_grad:
            if fsdp_enabled:
                lm_head.reshard()
            return total_loss, {}

        loss.backward()
        assert hidden_leaf.grad is not None
        accumulated_grad = hidden_leaf.grad.to(hidden_states.dtype)
        if fsdp_enabled:
            lm_head.reshard()

        return (
            _LMHeadGradientBackprop.apply(
                hidden_states,
                accumulated_grad,
                total_loss,
            ),
            {},
        )


class _LMHeadGradientBackprop(torch.autograd.Function):
    """Connect a separately computed LM-head gradient to the decoder graph."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        hidden_gradient: torch.Tensor,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(hidden_gradient)
        return loss.detach()

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        (hidden_gradient,) = ctx.saved_tensors
        return hidden_gradient * grad_output, None, None


__all__ = ["FusedLinearCrossEntropyLoss", "TECrossEntropyLoss"]
