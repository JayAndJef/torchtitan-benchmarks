"""``run_state.json``: the status and the attempt count of each arm, which a resume reads."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.artifacts.layout import atomic_write_json

if TYPE_CHECKING:
    from benchmarks.e2e.engines.api import Arm


STATE_SCHEMA_VERSION = 1


def initial_run_state(arms: tuple[Arm, ...]) -> dict[str, Any]:
    """The run state of a new run, with every arm pending."""
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "status": "pending",
        "arms": {
            arm.name: {"status": "pending", "attempts": 0} for arm in arms
        },
    }


def load_run_state(out_dir: Path, arms: tuple[Arm, ...]) -> dict[str, Any]:
    """The run state in ``out_dir``, or a new one when the directory holds none."""
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
    """Set the status of the run or of one arm, and rewrite the file; an arm that starts ``running`` counts one more attempt."""
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
    """Record the evaluation status in the run state of ``out_dir``; a directory without a run state is left as it is."""
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
