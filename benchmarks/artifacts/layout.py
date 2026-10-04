"""Where the files of a run go on disk, and the one JSON writer."""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from benchmarks.execution.paths import BENCH_DIR

if TYPE_CHECKING:
    from benchmarks.e2e.schema import Scenario


_LOG_LINE_RANK = re.compile(r"^\[rank(\d+)\]:")
"""The rank prefix that torchrun puts on each line that a rank writes."""

TRACE_FILE_GLOB = "rank*_trace.json.gz"
"""The file name of the trace of one profiler window of one rank."""

_TRACE_FILE_NAME = re.compile(r"\Arank(\d+)_trace\.json\.gz\Z")


def rank_of_trace(path: Path) -> int | None:
    """The rank that wrote the trace ``path``; ``None`` when the file name names no rank."""
    match = _TRACE_FILE_NAME.match(Path(path).name)
    return int(match.group(1)) if match else None


def trace_files(arm_dir: Path) -> list[Path]:
    """Every trace file of the arm, from every rank."""
    return sorted(
        arm_dir.glob(f"profiling/traces*/iteration_*/{TRACE_FILE_GLOB}")
    )


def trace_files_by_rank(arm_dir: Path) -> dict[int, list[Path]]:
    """The trace files of the arm, by the rank that wrote them, in rank order."""
    by_rank: dict[int, list[Path]] = {}
    for path in trace_files(arm_dir):
        rank = rank_of_trace(path)
        if rank is None:
            raise ValueError(
                f"{path}: matched the trace glob {TRACE_FILE_GLOB!r} but does "
                "not name a rank; a trace file is 'rank<n>_trace.json.gz'"
            )
        by_rank.setdefault(rank, []).append(path)
    return {rank: by_rank[rank] for rank in sorted(by_rank)}


def logs_by_rank(text: str) -> dict[int, str]:
    """The text that each rank wrote to one arm log, without NUL bytes; a log that names fewer than two ranks comes back whole, under the rank it names or under 0."""
    text = text.replace("\x00", "")
    by_rank: dict[int, list[str]] = {}
    for line in text.splitlines(keepends=True):
        match = _LOG_LINE_RANK.match(line)
        if match:
            by_rank.setdefault(int(match.group(1)), []).append(
                line[match.end() :]
            )
    if len(by_rank) < 2:
        return {next(iter(by_rank), 0): text}
    return {rank: "".join(by_rank[rank]) for rank in sorted(by_rank)}


def atomic_write_json(path: Path, value: Any) -> None:
    """Write ``value`` as strict JSON through a temporary file, so that a killed process leaves the previous file whole."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def archive_incomplete_arm(out_dir: Path, arm_name: str) -> Path | None:
    """Move the arm directory and the log of a failed attempt under ``attempts/<timestamp>/<arm>/``; ``None`` when the arm left no file."""
    arm_dir = out_dir / arm_name
    log_path = out_dir / f"{arm_name}.log"
    if not arm_dir.exists() and not log_path.exists():
        return None

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = out_dir / "attempts" / timestamp / arm_name
    suffix = 1
    while archive.exists():
        archive = out_dir / "attempts" / f"{timestamp}-{suffix}" / arm_name
        suffix += 1
    archive.mkdir(parents=True)
    if arm_dir.exists():
        shutil.move(str(arm_dir), str(archive / "artifacts"))
    if log_path.exists():
        shutil.move(str(log_path), str(archive / log_path.name))
    return archive


def run_timestamp() -> str:
    """The UTC time stamp that names the output directory of one run."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_output_dir(
    scenario: Scenario,
    hardware: str,
    requested: Path | None,
    environment: Mapping[str, str],
    timestamp: str | None = None,
    occurrence: int = 1,
) -> Path:
    """The output directory: ``--out``, else ``OUT``, else a new directory under ``out/``."""
    if requested is not None:
        return requested.expanduser().resolve()
    if env_out := environment.get("OUT"):
        return Path(env_out).expanduser().resolve()
    directory = scenario.name if occurrence == 1 else f"{scenario.name}-run{occurrence}"
    return BENCH_DIR / "out" / (timestamp or run_timestamp()) / directory / hardware
