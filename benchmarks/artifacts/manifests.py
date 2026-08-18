"""Benchmark output layout, manifests, and resumable state.

**Structural edge, pending resolution.** This module imports from
``benchmarks.e2e.registry`` at runtime, which inverts the intuitive layering
(``artifacts/`` looks lower-level than ``e2e/`` and is not). It is
*structural* rather than incidental: a manifest is a serialization of a run's
scenario and arms, so the builders here need those types by construction. Two
names cross at runtime, ``Workload`` (``_resume_workload`` reconstructs and
revalidates it from recorded JSON) and ``EXECUTION_MODEL`` (a manifest
self-description field); ``Scenario`` and ``Arm`` are annotation-only and are
imported under ``TYPE_CHECKING``, so they cost nothing at runtime.

Be precise about what does and does not cycle. At *module* granularity there
is no cycle: ``e2e/registry.py`` imports nothing from ``artifacts/``. At
*package* granularity there is one, because ``e2e/runner.py``,
``e2e/results.py`` and ``e2e/validation.py`` all import from this module. So
``artifacts/`` and ``e2e/`` are mutually dependent as packages and only the
module-level ordering keeps imports resolvable. The third name this module
takes from ``e2e`` makes the point sharpest: ``RunRequest`` comes from
``e2e/runner.py``, which imports this module at runtime, so that pair *would*
be a genuine module-level cycle -- it is legal only because it is confined to
``TYPE_CHECKING``. Anything moved out of that block must be re-checked. The
candidate resolution
is to move the e2e-shaped manifest builders into ``e2e/`` and leave
``artifacts/`` holding only engine-neutral pieces (``atomic_write_json``,
schema-checked ``load_manifest``, layout helpers, ``summaries.py``); the
cheapest partial step is to relocate ``EXECUTION_MODEL`` into this module,
whose ``manifest_data`` is its only consumer repo-wide, which deletes one of
the two outright and leaves ``Workload`` as the single genuinely structural
runtime edge. Neither is done here: this is a mechanical move, and the
redesign is deferred to a follow-up commit rather than made permanent.

A **second and unrelated** ``e2e`` edge exists here: ``load_run`` imports
``PIPER_1B_REGIONS`` to infer regions for schema-<8 manifests. That
contradicts the rule that missing historical fields stay explicitly unknown
rather than being inferred from current declarations, and it is a known
defect. It is preserved byte-identical here because fixing it is a behavior
change; when it is fixed, this edge disappears on its own.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from benchmarks.e2e.registry import EXECUTION_MODEL, PIPER_1B_REGIONS, Workload
from benchmarks.execution.environment import BENCH_DIR
from benchmarks.models.piper_qwen3.shape import PIPER_SHAPES
from benchmarks.traces.schema import Region

if TYPE_CHECKING:
    from benchmarks.e2e.registry import Arm, Scenario
    from benchmarks.e2e.runner import RunRequest


MANIFEST_SCHEMA_VERSION = 9
STATE_SCHEMA_VERSION = 1


def trace_files(arm_dir: Path) -> list[Path]:
    return sorted(arm_dir.glob("profiling/traces*/iteration_*/rank0_trace.json.gz"))


def manifest_data(
    scenario: Scenario,
    selected_arms: tuple[Arm, ...],
    commands: dict[str, list[str]],
    hardware: str,
    metadata: dict[str, str],
    extra_args: list[str] | tuple[str, ...],
    compile_mode: str,
    ac_mode: str,
    model_size: str = "normal",
) -> dict[str, Any]:
    shape = PIPER_SHAPES[model_size]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scenario": scenario.name,
        "description": scenario.description,
        "hardware": hardware,
        "hardware_metadata": metadata,
        "workload": asdict(scenario.workload),
        "regions": [asdict(region) for region in scenario.regions],
        "arms": [asdict(arm) for arm in scenario.arms],
        "selected_arms": [arm.name for arm in selected_arms],
        "commands": commands,
        "extra_torchtitan_args": list(extra_args),
        "compile_mode": compile_mode,
        "ac_mode": ac_mode,
        "model_size": model_size,
        "model_shape": shape.describe(seq_len=scenario.workload.seq_len),
        "execution_model": EXECUTION_MODEL,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_manifest(
    out_dir: Path,
    scenario: Scenario,
    selected_arms: tuple[Arm, ...],
    commands: dict[str, list[str]],
    hardware: str,
    metadata: dict[str, str],
    extra_args: list[str] | tuple[str, ...],
    compile_mode: str,
    ac_mode: str,
    model_size: str = "normal",
) -> None:
    atomic_write_json(
        out_dir / "manifest.json",
        manifest_data(
            scenario,
            selected_arms,
            commands,
            hardware,
            metadata,
            extra_args,
            compile_mode,
            ac_mode,
            model_size,
        ),
    )


def load_manifest(out_dir: Path) -> dict[str, Any]:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"manifest is missing: {manifest_path}")
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {manifest_path}: {error}") from error


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


def _resume_mismatches(
    manifest: dict,
    scenario: Scenario,
    arms: tuple[Arm, ...],
    hardware: str,
    metadata: dict[str, str],
    extra_args: tuple[str, ...],
    compile_mode: str,
    ac_mode: str,
    model_size: str,
) -> list[str]:
    expected = {
        "scenario": scenario.name,
        "workload": asdict(scenario.workload),
        "selected_arms": [arm.name for arm in arms],
        "hardware": hardware,
        "extra_torchtitan_args": list(extra_args),
        "compile_mode": compile_mode,
        "ac_mode": ac_mode,
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    # Defaulted lookup rather than a generic entry: schema <= 8 output
    # directories predate the axis and are still resumable as "normal".
    if str(manifest.get("model_size", "normal")) != model_size:
        mismatches.append("model_size")
    existing_metadata = manifest.get("hardware_metadata", {})
    for key in (
        "nvidia_smi",
        "cpu_pinning",
        "torchtitan_git_rev",
        "benchmarks_git_rev",
        "megatron_git_rev",
    ):
        if existing_metadata.get(key) != metadata.get(key):
            mismatches.append(f"hardware_metadata.{key}")
    return mismatches


def _resume_workload(
    manifest: dict,
    request: RunRequest,
    environment: Mapping[str, str],
) -> Workload:
    """Hydrate unspecified settings and reject explicit resume conflicts."""
    try:
        workload = Workload(**manifest["workload"])
    except (KeyError, TypeError) as error:
        raise ValueError("resume manifest has an invalid workload") from error

    requested_values = {
        "seq_len": request.seq_len
        if request.seq_len is not None
        else environment.get("SEQ"),
        "steps": request.steps
        if request.steps is not None
        else environment.get("STEPS"),
        "local_batch_size": request.batch
        if request.batch is not None
        else environment.get("BATCH"),
    }
    conflicts = [
        name
        for name, value in requested_values.items()
        if value is not None and int(value) != getattr(workload, name)
    ]
    if conflicts:
        raise ValueError(
            "resume request conflicts with the recorded workload: "
            + ", ".join(conflicts)
        )
    return workload


def load_run(
    out_dir: Path, arms_override: list[str] | tuple[str, ...] | None
) -> tuple[dict[str, Any], list[str], tuple[Region, ...], list[str]]:
    """Load current and legacy manifests without hiding compatibility warnings."""
    warnings: list[str] = []
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not manifest:
        warnings.append(f"no manifest.json under {out_dir}; assuming piper-1B regions")
    manifest_regions = manifest.get("regions", [])
    if manifest_regions and not all("phase" in region for region in manifest_regions):
        warnings.append(
            f"{manifest_path} predates phase-based regions "
            f"(schema {manifest.get('schema_version')}); assuming piper-1B regions"
        )
        manifest_regions = []
    regions = tuple(Region(**region) for region in manifest_regions)
    if not regions and int(manifest.get("schema_version") or 0) < 8:
        # Pre-schema-8 manifests never declared empty regions on purpose;
        # from schema 8 on, an empty list is an honest declaration (the
        # megatron scenario) and must not be second-guessed.
        regions = PIPER_1B_REGIONS
    arms = list(
        arms_override
        or manifest.get("selected_arms")
        or [arm["name"] for arm in manifest.get("arms", [])]
        or ["baseline", "helion", "te"]
    )
    return manifest, arms, regions, warnings
