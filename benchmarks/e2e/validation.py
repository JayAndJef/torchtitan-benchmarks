"""The helpers that an engine's validation may call, and ``validate_arm``, which gates an arm before the harness publishes it."""

from __future__ import annotations

import gzip
import re
from pathlib import Path

from benchmarks.artifacts.layout import logs_by_rank, trace_files_by_rank
from benchmarks.e2e.engines.api import Arm, Engine, RunSpec
from benchmarks.e2e.evidence import check_evidence, non_finite_refusals


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


def find_lines(text: str, pattern: re.Pattern[str]) -> list[re.Match[str]]:
    """The match of ``pattern`` on each line of ``text`` that it matches, in log order."""
    return [
        match
        for line in text.splitlines()
        if (match := pattern.search(line)) is not None
    ]


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
    failures = [
        *check_evidence(run, evidence),
        *non_finite_refusals(rank_logs),
        *engine.validate(run, arm, arm_dir, rank_logs),
    ]
    if failures:
        raise RuntimeError(
            f"{arm.name}: " + "; ".join(failures) + f"; see {log_path}"
        )
