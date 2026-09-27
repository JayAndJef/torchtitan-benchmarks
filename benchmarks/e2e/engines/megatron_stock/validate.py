"""The stock Megatron rules that an arm's logs and traces must pass: the mesh detail, the three treatments and the profiler traces."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from benchmarks.artifacts.layout import trace_files_by_rank
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
from benchmarks.e2e.validation import (
    ALL_REDUCE_MARKER,
    count_trace_windows,
    trace_contains,
)


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


def required_lines(run: RunSpec, arm: Arm) -> dict[str, tuple[str, ...]]:
    """The lines that every rank must print, by the treatment that each one proves."""
    config = arm.config
    spec = run.parallelism
    lines = {}
    if spec.world_size > 1:
        lines["parallelism"] = mesh_markers(
            spec, run.data, config.precision, config.extra_flags
        ) + p2p_markers(spec, config.p2p_sync)
    lines["megatron nan guard"] = nan_guard_markers(config.nan_guard)
    lines["megatron precision"] = precision_markers(config.precision)
    return lines


def trace_refusals(run: RunSpec, arm: Arm, arm_dir: Path) -> list[str]:
    """The trace rules that a profiled arm breaks: the windows of each rank, the kernel markers and the all-reduce."""
    if not run.profile:
        return []
    spec = run.parallelism
    windows = count_trace_windows(arm_dir)
    ranks = range(spec.world_size)
    refusals = []
    if spec.world_size > 1:
        refusals.extend(
            f"rank {rank} wrote no trace under {arm_dir}"
            for rank in ranks
            if rank not in windows
        )
        refusals.extend(
            f"rank {rank} wrote traces under {arm_dir}, and the run declares "
            f"{spec.world_size} ranks"
            for rank in windows
            if rank >= spec.world_size
        )
    minimum = run.window.min_windows
    for rank, count in (windows or {0: 0}).items():
        if count < minimum:
            where = f"for rank {rank} under" if len(windows) > 1 else "under"
            refusals.append(
                f"expected at least {minimum} profiler windows, found {count} "
                f"{where} {arm_dir}"
            )
    traces = trace_files_by_rank(arm_dir)
    every_trace = [path for paths in traces.values() for path in paths]
    # A pipeline stage can lack a marker kernel, so the markers read every rank as one set.
    refusals.extend(
        f"marker kernel {marker!r} absent from profiler traces"
        for marker in arm.config.trace_kernel_markers
        if not any(trace_contains(path, marker) for path in every_trace)
    )
    if spec.dp > 1:
        refusals.extend(
            f"dp {spec.dp} was requested and rank {rank}'s profiler traces "
            f"under {arm_dir} carry no {ALL_REDUCE_MARKER!r}"
            for rank in ranks
            if not any(
                trace_contains(path, ALL_REDUCE_MARKER)
                for path in traces.get(rank, ())
            )
        )
    return refusals


def validate_outputs(
    run: RunSpec, arm: Arm, arm_dir: Path, rank_logs: Mapping[int, str]
) -> list[str]:
    """Every stock Megatron rule that the arm's logs and traces break."""
    required = required_lines(run, arm)
    refusals = [
        f"the requested {treatment} did not apply; the engine never logged "
        f"{marker!r} on rank {rank}"
        for rank, log in sorted(rank_logs.items())
        for treatment, markers in required.items()
        for marker in markers
        if marker not in log
    ]
    return refusals + trace_refusals(run, arm, arm_dir)
