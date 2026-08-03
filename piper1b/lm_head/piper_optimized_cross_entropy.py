# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Single-GPU Piper optimization of TransformerEngine cross entropy."""

from __future__ import annotations

from functools import reduce
from operator import mul

import torch
import triton
import triton.language as tl

from piper1b.lm_head.te_common_cross_entropy import online_softmax_kernel

MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def piper_optimized_cross_entropy_kernel(
    X_ptr,
    X_stride,
    grad_input_ptr,
    grad_input_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    m_d_X_y_ptr,
    m_d_X_y_stride,
    ignore_idx,
    n_cols,
    gradient_scale,
    label_smoothing: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute FP32 loss math and write a normalized input-dtype gradient."""

    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride
    grad_input_ptr += program_id * grad_input_stride
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    if y == ignore_idx:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(grad_input_ptr + offsets, 0.0, mask=offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride
    m_d_X_y_ptr += program_id * 3 * m_d_X_y_stride
    m = tl.load(m_d_X_y_ptr)
    d = tl.load(m_d_X_y_ptr + m_d_X_y_stride)
    x_y = tl.load(m_d_X_y_ptr + 2 * m_d_X_y_stride)

    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols
    target_adjustment = 1.0 - label_smoothing

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets, mask=mask, other=float("-inf")).to(
            tl.float32
        )
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(mask, -eps * x, 0.0))

        gradient = tl.exp(x - m) / d - eps
        gradient = tl.where(offsets == y, gradient - target_adjustment, gradient)
        gradient *= gradient_scale
        tl.store(grad_input_ptr + offsets, gradient, mask=mask)

    loss = -(x_y - m - tl.log(d))
    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1.0 - label_smoothing) + smooth_loss
    tl.store(loss_ptr, loss)


def cross_entropy_forward(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gradient_scale: float,
    ignore_idx: int,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized scalar loss and its precomputed logits gradient."""

    batch_size, seq_len, vocab_size = logits.shape
    num_rows = batch_size * seq_len
    assert reduce(mul, list(target.size())) == num_rows

    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(vocab_size))
    loss_1d = torch.zeros(num_rows, dtype=torch.float32, device=logits.device)
    m_d_x_y = torch.zeros(num_rows * 3, dtype=torch.float32, device=logits.device)
    num_non_ignore = torch.zeros(1, dtype=torch.int64, device=logits.device)

    if logits.stride(-1) != 1 or logits.stride(-2) != logits.shape[-1]:
        logits = logits.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    # Match the logits dtype at the linear-backward boundary. For Piper this is
    # BF16; FP32 remains available for numerical validation.
    grad_input = torch.empty_like(logits)

    online_softmax_kernel[(num_rows,)](
        X_ptr=logits,
        X_stride=logits.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        m_d_X_y_ptr=m_d_x_y,
        m_d_X_y_stride=m_d_x_y.stride(-1),
        rank=0,
        n_cols=vocab_size,
        ignore_idx=ignore_idx,
        n_non_ignore=num_non_ignore,
        BLOCK_SIZE=block_size,
        num_warps=32,
    )
    piper_optimized_cross_entropy_kernel[(num_rows,)](
        X_ptr=logits,
        X_stride=logits.stride(-2),
        grad_input_ptr=grad_input,
        grad_input_stride=grad_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(-1),
        m_d_X_y_ptr=m_d_x_y,
        m_d_X_y_stride=m_d_x_y.stride(-1),
        ignore_idx=ignore_idx,
        n_cols=vocab_size,
        gradient_scale=gradient_scale,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=block_size,
        num_warps=32,
    )
    return loss_1d.sum() * gradient_scale, grad_input


class PiperOptimizedCrossEntropyFunction(torch.autograd.Function):
    """Autograd handoff for a terminal, internally normalized Piper loss."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        target: torch.Tensor,
        gradient_scale: float,
        ignore_idx: int,
    ) -> torch.Tensor:
        loss, grad_input = cross_entropy_forward(
            logits,
            target,
            gradient_scale=gradient_scale,
            ignore_idx=ignore_idx,
        )
        ctx.save_for_backward(grad_input.detach())
        return loss

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        (grad_input,) = ctx.saved_tensors
        # Piper calls backward directly on this already-normalized terminal
        # loss, so grad_output is one. Supporting arbitrary downstream scaling
        # would require another full-gradient multiplication pass.
        return grad_input, None, None, None


def piper_optimized_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    gradient_scale: float,
    ignore_idx: int = -100,
) -> torch.Tensor:
    """Compute Piper's terminal normalized CE without an FP32 gradient buffer."""

    return PiperOptimizedCrossEntropyFunction.apply(
        logits,
        target,
        gradient_scale,
        ignore_idx,
    )


__all__ = ["piper_optimized_cross_entropy"]
