"""The shim that makes Megatron print one step record per rank and step.

This module imports torch and megatron inside its functions alone.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from benchmarks.e2e.engines.megatron_stock.driver.markers import (
    H100_CLASS_BF16_PEAK_FLOPS,
    TRAINING_LOG_HEAD,
)
from benchmarks.e2e.engines.megatron_stock.steps import step_record


def tokens_per_second(
    local_tokens_per_step: int,
    elapsed_seconds: float,
    pipeline_degree: int,
) -> int:
    """The tokens per second of one device: the tokens of this rank divided by the time and the pipeline degree."""
    if pipeline_degree < 1:
        raise ValueError(f"pipeline degree {pipeline_degree} must be >= 1")
    return round(local_tokens_per_step / (elapsed_seconds * pipeline_degree))


def loss_value(loss_dict: dict) -> float | None:
    """The loss in ``loss_dict``, or ``None`` when this rank holds no loss; a loss that is not one number raises."""
    for key, value in loss_dict.items():
        if "loss" not in key:
            continue
        try:
            return float(value)
        except (TypeError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"megatron's loss entry {key!r} is {value!r}, not one number; "
                "the step record cannot state a loss"
            ) from error
    return None


def install_step_log_shim(
    *,
    local_tokens_per_step: int,
    pipeline_degree: int,
    num_flops_per_token: int,
) -> Callable[[], None]:
    """Wrap Megatron's ``training_log`` so that each rank prints one step record, and return the function that removes the wrap.

    The clock starts before ``pretrain`` builds the model, so step 1 holds the build time.
    """
    import inspect

    import torch

    import megatron.training.training as megatron_training

    original = megatron_training.training_log
    head = tuple(inspect.signature(original).parameters)[
        : len(TRAINING_LOG_HEAD)
    ]
    if head != TRAINING_LOG_HEAD:
        raise RuntimeError(
            "megatron's training_log takes "
            f"{head} where this driver forwards {TRAINING_LOG_HEAD}; the "
            "step record would hold the wrong value in a field, so the "
            "shim refuses to install"
        )
    state = {"last": time.perf_counter()}

    def replacement(
        loss_dict: dict,
        total_loss_dict: dict,
        learning_rate: Any,
        iteration: int,
        loss_scale: Any,
        report_memory_flag: Any,
        skipped_iter: Any,
        grad_norm: Any,
        *rest: Any,
        **keywords: Any,
    ) -> Any:
        result = original(
            loss_dict,
            total_loss_dict,
            learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            *rest,
            **keywords,
        )
        now = time.perf_counter()
        elapsed = now - state["last"]
        state["last"] = now
        reserved = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        tps = tokens_per_second(
            local_tokens_per_step, elapsed, pipeline_degree
        )
        tflops = num_flops_per_token * tps / 1e12
        print(
            step_record(
                step=iteration,
                tokens_per_second=tps,
                peak_memory_gib=reserved / 2**30,
                loss=broadcast_pipeline_loss(loss_value(loss_dict)),
                # Megatron states no norm on a skipped step, and a skipped step fails the arm.
                grad_norm=float("nan") if grad_norm is None else float(grad_norm),
                tflops=tflops,
                mfu=100 * tflops / (H100_CLASS_BF16_PEAK_FLOPS / 1e12),
            ),
            flush=True,
        )
        return result

    megatron_training.training_log = replacement

    def uninstall() -> None:
        megatron_training.training_log = original

    return uninstall


def broadcast_pipeline_loss(loss: "Any") -> "Any":
    """The loss of the last pipeline stage, broadcast to each rank of the pipeline."""
    import torch

    if not torch.distributed.is_initialized():
        return loss

    from megatron.core import mpu

    group = mpu.get_pipeline_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return loss
    ranks = torch.distributed.get_process_group_ranks(group)
    source = ranks[-1]
    device = torch.cuda.current_device()
    payload = torch.tensor(
        [0.0 if loss is None else float(loss), 0.0 if loss is None else 1.0],
        dtype=torch.float32,
        device=device,
    )
    torch.distributed.broadcast(payload, source, group=group)
    return float(payload[0]) if float(payload[1]) > 0 else None
