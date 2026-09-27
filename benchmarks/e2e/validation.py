"""The helpers that an engine's validation may call, and ``validate_arm``, which gates an arm before the harness publishes it."""

from __future__ import annotations

import gzip
from pathlib import Path

from benchmarks.artifacts.layout import logs_by_rank, trace_files_by_rank
from benchmarks.e2e.engines.api import Arm, Engine, RunSpec
from benchmarks.e2e.evidence import check_evidence, non_finite_refusals, rank_steps


ALL_REDUCE_MARKER = "ncclDevKernel_AllReduce"
"""The NCCL kernel name that proves a rank reduced its gradients."""


def count_trace_windows(arm_dir: Path) -> dict[int, int]:
    """The profiler windows that each rank wrote under ``arm_dir``; a rank that wrote none holds no key."""
    return {rank: len(paths) for rank, paths in trace_files_by_rank(arm_dir).items()}


def trace_contains(trace_path: Path, marker: str) -> bool:
    """Whether the gzipped trace holds ``marker``; an unreadable trace holds nothing."""
    try:
        with gzip.open(trace_path, "rt", errors="replace") as trace_file:
            overlap = ""
            while chunk := trace_file.read(1024 * 1024):
                text = overlap + chunk
                if marker in text:
                    return True
                overlap = text[-len(marker) :] if marker else ""
            return False
    except OSError:
        return False


def one_value(rank: int, fact: str, values: set) -> object | None:
    """The one value of ``fact`` that a rank's log states, or ``None``; two values raise ``RuntimeError``."""
    if len(values) > 1:
        raise RuntimeError(
            f"rank {rank} states two values of its {fact}: "
            + ", ".join(sorted(str(value) for value in values))
        )
    return next(iter(values), None)


def trace_refusals(
    run: RunSpec, arm_dir: Path, kernel_markers: tuple[str, ...]
) -> list[str]:
    """The trace rules that a profiled arm breaks: the windows of each rank, the kernel markers and the all-reduce."""
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
        for marker in kernel_markers
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


def validate_arm(
    run: RunSpec, arm: Arm, engine: Engine, arm_dir: Path, log_path: Path
) -> None:
    """Raise one ``RuntimeError`` that lists every harness fact and every engine rule that the arm breaks."""
    if not log_path.is_file():
        raise RuntimeError(f"{arm.name}: training log is missing: {log_path}")
    rank_logs = logs_by_rank(log_path.read_text(errors="replace"))
    evidence = {
        rank: engine.read_evidence(rank, text) for rank, text in rank_logs.items()
    }
    failures = check_evidence(run, evidence)
    steps = {}
    for rank, text in sorted(rank_logs.items()):
        try:
            steps.update(rank_steps(engine, {rank: text}))
        except ValueError as error:
            failures.append(str(error))
    failures.extend(non_finite_refusals(steps))
    failures.extend(engine.validate(run, arm, arm_dir, rank_logs))
    if failures:
        raise RuntimeError(
            f"{arm.name}: " + "; ".join(failures) + f"; see {log_path}"
        )
