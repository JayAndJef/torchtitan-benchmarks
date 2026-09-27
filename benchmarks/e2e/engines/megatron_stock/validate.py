"""The log lines that prove a stock Megatron arm ran the treatment it declares."""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.e2e.engines.api import Arm, DataSpec, RunSpec
from benchmarks.e2e.engines.megatron_stock.flags import (
    DATA_PARALLEL_WRAPPERS,
    PP_SCHEDULE,
    SHARDING_STRATEGIES,
    data_parallel_optimizer,
    data_parallel_overlap,
    grad_reduce_in_fp32,
    microbatch_geometry,
)
from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.validation import ValidationProfile, validate_against_profile


MEGATRON_STOCK_PROFILE = ValidationProfile(
    completion_marker="Training completed",
    compile_marker=None,
    failure_markers=(),
    ac_line=None,
    pipelined_pattern=re.compile(
        r"Megatron-LM stock parallelism: dp=\d+ pp=(?!1\b)\d+"
    ),
    data_parallel_pattern=re.compile(
        r"Megatron-LM stock parallelism: dp=(?!1\b)\d+"
        r"|Megatron-LM stock data parallel:"
    ),
)
"""The stock Megatron log lines that the shared rules read."""


def mesh_markers(
    spec: ParallelismSpec,
    data: DataSpec,
    precision: str,
    extra_flags: tuple[str, ...],
) -> tuple[str, ...]:
    """The mesh line and the data-parallel line that the driver prints for ``spec``."""
    _, microbatches, _ = microbatch_geometry(data, spec)
    markers = [
        f"Megatron-LM stock parallelism: dp={spec.dp} pp={spec.pp} "
        f"ep={spec.ep} schedule={PP_SCHEDULE} microbatches={microbatches} "
        f"stages={spec.pp}"
    ]
    if spec.dp > 1:
        markers.append(
            "Megatron-LM stock data parallel: "
            f"{DATA_PARALLEL_WRAPPERS[spec.zero]} over "
            f"{spec.dp} ranks (overlap_grad_reduce="
            f"{data_parallel_overlap(extra_flags)}, "
            f"grad_reduce_in_fp32={grad_reduce_in_fp32(precision)}, "
            f"sharding_strategy={SHARDING_STRATEGIES[spec.zero]}, "
            f"expert_parallel={spec.ep}, "
            f"optimizer={data_parallel_optimizer(spec.zero)})"
        )
    return tuple(markers)


def p2p_markers(spec: ParallelismSpec, p2p_sync: str) -> tuple[str, ...]:
    """The p2p line that the driver prints above one pipeline rank."""
    if spec.pp == 1:
        return ()
    return (
        "Megatron-LM stock p2p: batch_p2p_comm=True "
        f"batch_p2p_sync={p2p_sync == 'on'}",
    )


def nan_guard_markers(nan_guard: str) -> tuple[str, ...]:
    """The NaN-guard line that the driver prints on every rank."""
    return (
        "Megatron-LM stock nan guard: "
        f"check_for_nan_in_loss_and_grad={nan_guard == 'on'}",
    )


def precision_markers(precision: str) -> tuple[str, ...]:
    """The training-loop line and its four precision fields, each field a marker of its own."""
    lean = precision == "lean"
    dtype = "torch.bfloat16" if lean else "torch.float32"
    return (
        "Megatron-LM stock training loop (",
        f"use_precision_aware_optimizer={lean}",
        f"main_grads_dtype={dtype}",
        f"exp_avg_dtype={dtype}",
        f"exp_avg_sq_dtype={dtype}",
    )


def validate_outputs(run: RunSpec, arm: Arm, arm_dir: Path, log_path: Path) -> None:
    """Raise ``RuntimeError`` when the log or the traces of a stock Megatron arm refuse its numbers."""
    config = arm.config
    spec = run.parallelism
    required_lines = {}
    if spec.world_size > 1:
        required_lines["parallelism"] = mesh_markers(
            spec, run.data, config.precision, config.extra_flags
        ) + p2p_markers(spec, config.p2p_sync)
    required_lines["megatron nan guard"] = nan_guard_markers(config.nan_guard)
    required_lines["megatron precision"] = precision_markers(config.precision)
    validate_against_profile(
        run,
        arm.name,
        arm_dir,
        log_path,
        profile=MEGATRON_STOCK_PROFILE,
        required_lines=required_lines,
        trace_kernel_markers=config.trace_kernel_markers,
    )
