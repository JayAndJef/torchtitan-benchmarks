"""``run_state.json``: the per-arm progress ledger a resume reads.

A separate file from the manifest, and a separate module from
``manifests.py``, because it answers a different question and changes at a
different rate. The manifest records *what a run is* -- scenario, arms,
commands, the three global axes, the model shape, provenance -- and has
reached schema 9 doing it. This records *how far the run got*, and is still
at schema 1: nothing about "pending / running / completed / failed, plus an
attempt count and a UTC stamp per transition" has needed to change since it
was written. Splitting them means a manifest schema bump no longer touches
the file whose schema is not bumping.

Two of its properties are what make ``run --resume`` safe rather than
merely convenient. Every transition is rewritten through
``atomic_write_json``, so a run interrupted between arms leaves a readable
ledger and not a truncated one; and ``update_run_state`` increments
``attempts`` on entry to ``running`` rather than on failure, so an arm killed
by ``SIGKILL`` -- which never gets to report anything -- is still counted.

This is also the one module in ``artifacts/`` with no runtime dependency on
``benchmarks.e2e`` whatsoever. ``Arm`` appears in two signatures and is used
for ``arm.name`` alone, so the import sits under ``TYPE_CHECKING``; see
``manifests.py`` for the runtime edge that cannot be avoided so cheaply.

``record_evaluation_status`` deliberately does not take the arm definitions
the other three do, and returns silently when there is no state file.
Evaluation runs after the arms are finished and is reachable from
``benchmarks.cli.e2e``'s ``evaluate <out_dir>`` against a directory whose
scenario the current process never loaded -- including one written by an
older schema -- so it reads and rewrites the ledger as plain JSON.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.artifacts.layout import atomic_write_json

if TYPE_CHECKING:
    from benchmarks.e2e.schema import Arm


STATE_SCHEMA_VERSION = 1


def initial_run_state(arms: tuple[Arm, ...]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "pending",
        "arms": {
            arm.name: {"status": "pending", "attempts": 0} for arm in arms
        },
    }


def load_run_state(out_dir: Path, arms: tuple[Arm, ...]) -> dict[str, Any]:
    path = out_dir / "run_state.json"
    if not path.exists():
        return initial_run_state(arms)
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read run state {path}: {error}") from error


def update_run_state(
    out_dir: Path,
    state: dict[str, Any],
    *,
    arm_name: str | None = None,
    status: str,
    error: str | None = None,
) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ")
    if arm_name is None:
        state["status"] = status
        state[f"{status}_at"] = now
    else:
        arm_state = state["arms"][arm_name]
        arm_state["status"] = status
        arm_state[f"{status}_at"] = now
        if status == "running":
            arm_state["attempts"] = int(arm_state.get("attempts", 0)) + 1
        if error is not None:
            arm_state["error"] = error
        elif "error" in arm_state:
            del arm_state["error"]
    atomic_write_json(out_dir / "run_state.json", state)


def record_evaluation_status(
    out_dir: Path, *, completed: bool, error: str | None = None
) -> None:
    """Record automatic evaluation without requiring arm definitions."""
    path = out_dir / "run_state.json"
    if not path.exists():
        return
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as state_error:
        raise ValueError(
            f"cannot read run state {path}: {state_error}"
        ) from state_error
    now = dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ")
    status = "completed" if completed else "evaluation_failed"
    state["status"] = status
    state["evaluation"] = {"status": status, f"{status}_at": now}
    if error is not None:
        state["evaluation"]["error"] = error
    atomic_write_json(path, state)
