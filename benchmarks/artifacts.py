"""Benchmark output layout, manifests, validation, and resumable state."""

from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from benchmarks.profile_regions import pooled_window_metrics
from benchmarks.scenarios import Arm, Region, Scenario, Workload
from piper1b.model_shape import PIPER_SHAPES


MANIFEST_SCHEMA_VERSION = 9
STATE_SCHEMA_VERSION = 1

# How the training process executes the model. Constant since schema 7:
# plain bf16 params on one GPU, no FSDP wrapper, no fp32 masters
# (piper1b.parallelize). Recorded so a manifest self-describes without a
# git-rev lookup; earlier schemas ran under FSDP2 mixed precision.
EXECUTION_MODEL = "single-gpu-plain-bf16-no-fsdp"

# Engine-neutral compile modes selectable per run. "cuda-graph" replaced the
# torch-level name "reduce-overhead" in schema 8; TORCH_COMPILE_MODE maps it
# back to the --compile.mode value the TorchTitan fork applies per block.
# The two max-autotune modes were removed in schema 8 after the full matrix
# showed them to be GPU-time regressions at these shapes (see
# reports/20260807-mode-matrix-plain-bf16.md); schema <= 7 manifests may
# still record them and the old reduce-overhead name.
COMPILE_MODES = ("default", "cuda-graph")
TORCH_COMPILE_MODE = {"default": "default", "cuda-graph": "reduce-overhead"}
CUDAGRAPH_COMPILE_MODES = frozenset({"cuda-graph"})

# Activation checkpointing modes selectable per run (schema 8). "sac" is
# TorchTitan's per-op SelectiveAC (the historical treatment, implied by
# schema <= 7 manifests); "none" disables checkpointing entirely, delivered
# to TorchTitan as the tyro subcommand token "activation-checkpoint:none".
AC_MODES = ("sac", "none")
_SAC_APPLIED_LINE = "Applied SelectiveAC activation checkpointing"

# Model shapes selectable per run (schema 9), the third global run axis
# alongside compile_mode and ac_mode. Numbers are only comparable within one
# size; --resume refuses to mix them. Schema <= 8 manifests imply "normal".
MODEL_SIZES = tuple(PIPER_SHAPES)


@dataclass(frozen=True)
class ValidationProfile:
    """Engine-specific pieces of validate_arm, selected by Arm.validation.

    The engine-neutral rules (trace-window count, kernel markers,
    cudaGraphLaunch under cuda-graph mode, override counting when declared)
    are shared; these fields carry what differs: the completion marker, the
    log line that proves the requested mode actually applied, the phrases
    that mean a silent fallback, and whether the SelectiveAC line and the
    compiled-region structure are expected at all.
    """

    completion_marker: str
    mode_line: Callable[[str], str]
    failure_markers: tuple[str, ...]
    check_ac_line: bool
    check_regions: bool


VALIDATION_PROFILES = {
    "torchtitan": ValidationProfile(
        completion_marker="Training completed",
        # apply_compile logs the torch-level mode name.
        mode_line=lambda mode: (
            f"with torch.compile (mode={TORCH_COMPILE_MODE[mode]})"
        ),
        failure_markers=("falling back to the PyTorch",),
        check_ac_line=True,
        check_regions=True,
    ),
    "megatron": ValidationProfile(
        completion_marker="Training completed",
        # megatron_baseline.train.MODE_LINE; the trailing comma pins the
        # mode token without pinning which graph implementation ran.
        mode_line=lambda mode: f"Megatron-LM training loop (mode={mode},",
        failure_markers=(),
        check_ac_line=False,
        check_regions=False,
    ),
}


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
    ac_mode: str = "sac",
    model_size: str = "normal",
) -> None:
    """Reject partial or wrongly configured runs before analysis."""
    profile = VALIDATION_PROFILES[arm.validation]
    shape = PIPER_SHAPES[model_size]
    if not log_path.is_file():
        raise RuntimeError(f"{arm.name}: training log is missing: {log_path}")
    log = log_path.read_text(errors="replace")
    if profile.completion_marker not in log:
        raise RuntimeError(f"{arm.name}: training did not complete; see {log_path}")
    # The engine reports which mode it actually applied.
    if profile.mode_line(compile_mode) not in log:
        raise RuntimeError(
            f"{arm.name}: compile mode {compile_mode!r} did not apply; "
            f"see {log_path}"
        )
    if profile.check_ac_line:
        # The AC policy logs its application; its presence must match the
        # requested mode or the run measured the wrong recompute treatment.
        sac_applied = _SAC_APPLIED_LINE in log
        if ac_mode == "sac" and not sac_applied:
            raise RuntimeError(
                f"{arm.name}: ac mode 'sac' requested but SelectiveAC was not "
                f"applied; see {log_path}"
            )
        if ac_mode == "none" and sac_applied:
            raise RuntimeError(
                f"{arm.name}: ac mode 'none' requested but SelectiveAC was "
                f"applied; see {log_path}"
            )
    # Both engines print this line; without the check a run whose --config
    # or --model-size silently fell back to another shape would pass every
    # other rule and be published under the wrong size.
    size_marker = f"size: {shape.param_count:,} total parameters"
    if size_marker not in log:
        raise RuntimeError(
            f"{arm.name}: model size {model_size!r} "
            f"({shape.param_count:,} parameters) did not apply; see {log_path}"
        )
    if arm.overrides_per_block:
        expected_overrides = arm.overrides_per_block * shape.n_layers
        override_count = len(re.findall(r"\[Override\]", log))
        if override_count != expected_overrides:
            raise RuntimeError(
                f"{arm.name}: expected {expected_overrides} override "
                "applications, "
                f"found {override_count}; see {log_path}"
            )
        for override_import in arm.override_imports:
            if f"[Override] {override_import}:" not in log:
                raise RuntimeError(
                    f"{arm.name}: override {override_import!r} did not apply; "
                    f"see {log_path}"
                )
    for marker in profile.failure_markers:
        if marker in log:
            raise RuntimeError(
                f"{arm.name}: silent fallback marker {marker!r} found in the "
                f"log; see {log_path}"
            )

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
    if regions and profile.check_regions:
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
