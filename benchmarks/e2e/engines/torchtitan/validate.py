"""The TorchTitan rules that an arm's logs and traces must pass: the treatments it declares, and its profiler traces."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from benchmarks.artifacts.layout import trace_files_by_rank
from benchmarks.e2e.engines.api import Arm, CompileMode, DataSpec, RunSpec
from benchmarks.e2e.engines.torchtitan.mesh import SCHEDULES, titan_mesh
from benchmarks.e2e.parallelism import PP_SCHEDULES, ParallelismSpec, n_microbatches
from benchmarks.e2e.validation import (
    ALL_REDUCE_MARKER,
    count_trace_windows,
    trace_contains,
)


COMPILE_LINE = "with torch.compile"
"""The line that proves whole-block ``torch.compile``."""

SAC_LINE = "Applied SelectiveAC activation checkpointing"
"""The line that proves selective activation checkpointing."""

FALLBACK_LINES = ("falling back to the PyTorch",)
"""The phrases that mean a silent fallback to a slower path."""

OVERRIDE_LINE = re.compile(r"\[Override\]")


def mesh_markers(spec: ParallelismSpec, data: DataSpec) -> tuple[str, ...]:
    """The lines that TorchTitan and the parallelize plug-in log about the mesh of ``spec``."""
    replicate, shard = titan_mesh(spec)
    markers = [
        f"Building device mesh with parallelism: pp={spec.pp}, "
        f"dp_replicate={replicate}, dp_shard={shard}, cp=1, tp=1, "
        f"ep={spec.ep}"
    ]
    if replicate * shard > 1:
        # plugins/parallelize.py prints this line, and a test pins the two.
        markers.append(
            "piper1b data parallel: fully_shard applied "
            f"(dp_replicate={replicate}, dp_shard={shard})"
        )
    if spec.pp > 1:
        schedule = PP_SCHEDULES[spec.pp_schedule]
        microbatches = n_microbatches(spec, local_batch_size=data.local_batch_size)
        markers.append(
            f"Using pipeline schedule {SCHEDULES[spec.pp_schedule].titan_name} "
            f"with {microbatches} microbatches and "
            f"{spec.pp * schedule.stages_per_rank} stages"
        )
    return tuple(markers)


def required_lines(run: RunSpec) -> dict[str, tuple[str, ...]]:
    """The lines that every rank must print, by the treatment that each one proves."""
    if run.parallelism.world_size == 1:
        return {}
    return {"parallelism": mesh_markers(run.parallelism, run.data)}


def _log_refusals(run: RunSpec, arm: Arm, rank: int, log: str) -> list[str]:
    """The rules that one rank's log breaks."""
    config = arm.config
    where = f"on rank {rank}"
    refusals = []
    compiled = COMPILE_LINE in log
    if config.compile is CompileMode.TORCH and not compiled:
        refusals.append(
            f"the arm asks for torch.compile and the engine did not apply it {where}"
        )
    if config.compile is CompileMode.NONE and compiled:
        refusals.append(f"the arm runs eager and the engine compiled the model {where}")
    sac_applied = SAC_LINE in log
    if run.ac_mode == "sac" and not sac_applied:
        refusals.append(
            f"ac mode 'sac' requested but SelectiveAC was not applied {where}"
        )
    if run.ac_mode == "none" and sac_applied:
        refusals.append(f"ac mode 'none' requested but SelectiveAC was applied {where}")
    if config.overrides_per_block:
        expected = config.overrides_per_block * run.shape.n_layers
        found = len(OVERRIDE_LINE.findall(log))
        if found != expected:
            refusals.append(
                f"expected {expected} override applications, found {found} {where}"
            )
        refusals.extend(
            f"override {target!r} did not apply {where}"
            for target in config.override_imports
            if f"[Override] {target}:" not in log
        )
    refusals.extend(
        f"silent fallback marker {marker!r} found in the log {where}"
        for marker in FALLBACK_LINES
        if marker in log
    )
    refusals.extend(
        f"the requested {treatment} did not apply; the engine never logged "
        f"{marker!r} {where}"
        for treatment, markers in required_lines(run).items()
        for marker in markers
        if marker not in log
    )
    return refusals


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
    """Every TorchTitan rule that the arm's logs and traces break."""
    refusals = [
        refusal
        for rank, log in sorted(rank_logs.items())
        for refusal in _log_refusals(run, arm, rank, log)
    ]
    return refusals + trace_refusals(run, arm, arm_dir)
