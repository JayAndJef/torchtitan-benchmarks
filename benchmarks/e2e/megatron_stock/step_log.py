"""The step line the evaluation parses, and the shim that prints it.

Megatron's own ``training_log`` writes a format ``benchmarks.e2e.results``
does not read. ``install_step_log_shim`` wraps it, so every rank prints one
line per step in the grammar both engines share, and Megatron's own output
is untouched beside it.

Every torch and megatron import sits inside a function. Importing this
module therefore costs neither, which is what lets the parent read the
grammar on a host with no Megatron-LM checkout.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from benchmarks.e2e.megatron_stock.markers import (
    H100_CLASS_BF16_PEAK_FLOPS,
    STEP_LINE,
    STEP_LINE_NO_LOSS,
    TRAINING_LOG_HEAD,
)


def tokens_per_second(
    local_tokens_per_step: int,
    elapsed_seconds: float,
    pipeline_degree: int,
) -> int:
    """Tokens per second PER DEVICE, which is the published figure.

    The same definition TorchTitan uses: one rank's own token count
    divided by ``cp * tp * pp``. The ranks of one
    pipeline share a batch; each data-parallel rank reads a batch of its
    own, so the data-parallel degree is absent from the divisor.
    """
    if pipeline_degree < 1:
        raise ValueError(f"pipeline degree {pipeline_degree} must be >= 1")
    return round(local_tokens_per_step / (elapsed_seconds * pipeline_degree))


def loss_value(loss_dict: dict) -> float | None:
    """The batch loss this rank holds, or None where it holds none.

    Only the last pipeline stage computes a loss, and this driver
    broadcasts nothing, so a middle stage returns None and prints no loss
    field. ``benchmarks/e2e/results.py`` reads the trajectory from one rank
    and ``loss_visible_rank`` picks a last-stage rank.
    """
    for key, value in loss_dict.items():
        if "loss" not in key:
            continue
        try:
            return float(value)
        except Exception:
            # A value that is not one number is not a loss. Megatron's own
            # entry is a 0-dim tensor, so this is a guard and not a path.
            return None
    return None


def step_log_line(
    *,
    step: int,
    loss: float | None,
    grad_norm: float | None,
    memory_bytes: int,
    device_total_bytes: int,
    tps: int,
    tflops: float,
    mfu: float,
) -> str:
    """One step line, in the shape ``STEP_METRICS`` parses.

    A pure function of numbers, so a test reads the format without a GPU.
    ``grad_norm`` is None on a skipped step, and prints as ``nan`` -- which
    is what ``GRAD_NORM_METRIC`` already accepts.
    """
    fields = {
        "step": step,
        "grad_norm": float("nan") if grad_norm is None else grad_norm,
        "memory": memory_bytes / 2**30,
        "percent": 100 * memory_bytes / device_total_bytes,
        "tps": tps,
        "tflops": tflops,
        "mfu": mfu,
    }
    if loss is None:
        return STEP_LINE_NO_LOSS.format(**fields)
    return STEP_LINE.format(loss=loss, **fields)


def install_step_log_shim(
    *,
    local_tokens_per_step: int,
    pipeline_degree: int,
    num_flops_per_token: int,
) -> Callable[[], None]:
    """Make Megatron print the step line the harness parses.

    Megatron's own ``training_log`` prints an ``iteration ... elapsed time
    per iteration (ms)`` line, on the last rank only, in a format
    ``benchmarks/e2e/results.py``'s three regexes do not match. Without a
    step line this arm publishes no throughput, no peak memory and no loss
    trajectory, so ``evaluate`` has nothing to compare.

    **The plan for this scenario does not name this shim.** It is added
    because the arm is otherwise unmeasurable. It prints one extra line per
    rank per step and changes nothing Megatron computes.

    The loss is broadcast from the last pipeline stage, so every rank
    prints the real number rather than an absent field. ``training_log``
    runs on every rank once per iteration, so the collective is safe: every
    member of the pipeline group reaches it the same number of times.

    **Step 1 carries the setup.** The clock starts when this function runs,
    which is before ``pretrain()`` builds the model, so step 1's rate and
    its peak memory include the build. ``results.py``'s ``stable_tps``
    excludes step 1 of every cycle by construction, and peak memory is a
    maximum over the run, so the published throughput is unaffected and the
    published memory is the run peak rather than a step peak.

    The device's total memory is read on the first step, not here.
    ``pretrain()`` binds this rank's device, and reading the property
    earlier would build a CUDA context on device 0 from every rank.

    Returns a callable that puts Megatron's own ``training_log`` back.
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
            "step line would print the wrong value in a column, so the "
            "shim refuses to install"
        )
    state: dict[str, Any] = {"last": time.perf_counter(), "total": None}

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
        if state["total"] is None:
            state["total"] = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
        reserved = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        tps = tokens_per_second(
            local_tokens_per_step, elapsed, pipeline_degree
        )
        tflops = num_flops_per_token * tps / 1e12
        print(
            step_log_line(
                step=iteration,
                loss=broadcast_pipeline_loss(loss_value(loss_dict)),
                grad_norm=None if grad_norm is None else float(grad_norm),
                memory_bytes=reserved,
                device_total_bytes=state["total"],
                tps=tps,
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
    """The last pipeline stage's loss, on every rank.

    Only the last stage computes one, and stock Megatron leaves the other
    stages with an empty ``loss_dict``. A rank with no loss would print no
    loss field, and ``loss_visible_rank`` -- TorchTitan's own arithmetic,
    ``(world_size // pp) * (pp - 1)`` -- does not always name a last-stage
    rank of Megatron's own layout. Broadcasting removes the question: every
    rank prints the same real number.

    The value the last stage holds is already the mean over the
    data-parallel group, because ``train_step`` all-reduces it there. So
    this broadcast alone makes the printed loss mean what TorchTitan's
    ``global_avg_loss`` means.

    Returns ``None`` when no rank held a loss, which is not a state a
    training step reaches.
    """
    import torch

    if not torch.distributed.is_initialized():
        return loss

    from megatron.core import mpu

    group = mpu.get_pipeline_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return loss
    # The last stage is the source. Its global rank is the last entry of
    # this rank's own pipeline group, which every rank of that group agrees
    # on, so no rank has to guess.
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
