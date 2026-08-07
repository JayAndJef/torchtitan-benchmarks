"""Benchmark output layout, manifests, validation, and resumable state."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.profile_regions import pooled_window_metrics
from benchmarks.scenarios import Arm, Region, Scenario, Workload


MANIFEST_SCHEMA_VERSION = 7
STATE_SCHEMA_VERSION = 1

# How the training process executes the model. Constant since schema 7:
# plain bf16 params on one GPU, no FSDP wrapper, no fp32 masters
# (piper1b.parallelize). Recorded so a manifest self-describes without a
# git-rev lookup; earlier schemas ran under FSDP2 mixed precision.
EXECUTION_MODEL = "single-gpu-plain-bf16-no-fsdp"

# torch.compile modes selectable per run. The runner forwards the mode as
# --compile.mode, which the TorchTitan fork applies to each block's
# torch.compile.
COMPILE_MODES = (
    "default",
    "reduce-overhead",
    "max-autotune-no-cudagraphs",
    "max-autotune",
)
CUDAGRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})


def trace_files(arm_dir: Path) -> list[Path]:
    return sorted(arm_dir.glob("profiling/traces*/iteration_*/rank0_trace.json.gz"))


def _trace_contains(trace_path: Path, marker: str) -> bool:
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


def validate_arm(
    arm: Arm,
    arm_dir: Path,
    log_path: Path,
    workload: Workload,
    *,
    regions: tuple[Region, ...] = (),
    compile_mode: str = "default",
) -> None:
    """Reject partial or wrongly configured runs before analysis."""
    if not log_path.is_file():
        raise RuntimeError(f"{arm.name}: training log is missing: {log_path}")
    log = log_path.read_text(errors="replace")
    if "Training completed" not in log:
        raise RuntimeError(f"{arm.name}: training did not complete; see {log_path}")
    # TorchTitan's apply_compile reports the mode it applied to the blocks.
    if f"with torch.compile (mode={compile_mode})" not in log:
        raise RuntimeError(
            f"{arm.name}: compile mode {compile_mode!r} did not apply; "
            f"see {log_path}"
        )
    if arm.expected_override_count:
        override_count = len(re.findall(r"\[Override\]", log))
        if override_count != arm.expected_override_count:
            raise RuntimeError(
                f"{arm.name}: expected {arm.expected_override_count} override "
                "applications, "
                f"found {override_count}; see {log_path}"
            )
        for override_import in arm.override_imports:
            if f"[Override] {override_import}:" not in log:
                raise RuntimeError(
                    f"{arm.name}: override {override_import!r} did not apply; "
                    f"see {log_path}"
                )
    if "falling back to the PyTorch" in log:
        raise RuntimeError(f"{arm.name}: override fell back to PyTorch; see {log_path}")

    traces = trace_files(arm_dir)
    if len(traces) < workload.min_trace_windows:
        raise RuntimeError(
            f"{arm.name}: expected at least {workload.min_trace_windows} "
            "profiler windows, "
            f"found {len(traces)} under {arm_dir}"
        )
    for marker in arm.trace_kernel_markers:
        if not any(_trace_contains(path, marker) for path in traces):
            raise RuntimeError(
                f"{arm.name}: marker kernel {marker!r} absent from profiler traces"
            )
    if compile_mode in CUDAGRAPH_COMPILE_MODES and not any(
        _trace_contains(path, "cudaGraphLaunch") for path in traces
    ):
        raise RuntimeError(
            f"{arm.name}: compile mode {compile_mode!r} enables CUDA graphs but "
            f"no cudaGraphLaunch appears in the profiler traces under {arm_dir}"
        )
    if regions:
        try:
            pooled_window_metrics(traces, regions)
        except ValueError as error:
            raise RuntimeError(
                f"{arm.name}: profiler traces failed structural validation: {error}"
            ) from error


def manifest_data(
    scenario: Scenario,
    selected_arms: tuple[Arm, ...],
    commands: dict[str, list[str]],
    hardware: str,
    metadata: dict[str, str],
    extra_args: list[str] | tuple[str, ...],
    compile_mode: str,
) -> dict[str, Any]:
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
