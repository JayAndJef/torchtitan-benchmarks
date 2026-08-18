"""Where a run's files go on disk, and the one way they are written.

The engine-neutral half of ``artifacts/``: nothing here knows what a scenario
or an arm *is*, only where its output lands and what a trace file is called.
That is an import fact as much as a tidiness one. ``benchmarks.kernel.
results.schema`` and ``benchmarks.kernel.runner`` need ``atomic_write_json``
and ``run_timestamp`` and nothing else, and taking them from ``manifests.py``
dragged ``benchmarks.e2e.registry`` -- every scenario, arm and workload
declaration in the repository -- into the kernel system's import graph for a
four-line JSON writer. ``benchmarks.e2e.validation`` and
``tools/collect_matrix.py`` are in the same position for ``trace_files``.

``atomic_write_json`` is the only place anything under ``benchmarks/``
writes JSON -- manifests, run state, and both systems' ``results.json`` all
go through it. It
writes a sibling ``.tmp`` and renames, so a run killed mid-write leaves the
previous file intact rather than a truncated one -- which matters most for
``run_state.json``, rewritten on every arm transition -- and it passes
``allow_nan=False``, so a NaN metric fails loudly at write time instead of
producing a file no strict parser will read back.

``archive_incomplete_arm`` is the counterpart on the destructive side. A
resume never overwrites a failed arm's artifacts; it moves the arm directory
and its log under ``attempts/<timestamp>/<arm>/`` first, so the evidence of
why an arm failed survives the retry that replaces it. The numbered-suffix
loop is not paranoia: an arm that dies during launch fails inside the same
second as its predecessor.

``run_timestamp`` is public because ``benchmarks.cli.e2e`` generates one
stamp and hands it to every scenario of a ``run-all --all-scenarios`` sweep,
which is what groups them under a single ``out/<timestamp>/``.
``_default_output_dir`` takes a ``Scenario`` only to read its ``.name``, so
the annotation lives under ``TYPE_CHECKING`` and this module keeps no
runtime dependency on ``benchmarks.e2e`` at all.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from benchmarks.execution.paths import BENCH_DIR

if TYPE_CHECKING:
    from benchmarks.e2e.registry import Scenario


def trace_files(arm_dir: Path) -> list[Path]:
    return sorted(arm_dir.glob("profiling/traces*/iteration_*/rank0_trace.json.gz"))


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def archive_incomplete_arm(out_dir: Path, arm_name: str) -> Path | None:
    """Move incomplete artifacts aside so retrying never destroys evidence."""
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
    """Directory-safe UTC stamp; shared across a multi-scenario sweep."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_output_dir(
    scenario: Scenario,
    hardware: str,
    requested: Path | None,
    environment: Mapping[str, str],
    timestamp: str | None = None,
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    if env_out := environment.get("OUT"):
        return Path(env_out).expanduser().resolve()
    return BENCH_DIR / "out" / (timestamp or run_timestamp()) / scenario.name / hardware
