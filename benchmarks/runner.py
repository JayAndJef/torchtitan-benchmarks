"""Orchestrate declarative TorchTitan benchmark scenarios."""

from __future__ import annotations

import datetime as dt
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

from benchmarks.artifacts import (
    archive_incomplete_arm,
    initial_run_state,
    load_manifest,
    load_run_state,
    update_run_state,
    validate_arm,
    write_manifest,
)
from benchmarks.runtime import (
    BENCH_DIR,
    RuntimePaths,
    add_compiler_environment,
    command_for_arm,
    hardware_metadata,
    resolve_cpu_pinning,
    runtime_environment,
)
from benchmarks.scenarios import Arm, Scenario, Workload, scenario_by_name


@dataclass(frozen=True)
class RunRequest:
    """User-selected inputs for one benchmark execution."""

    gpu: str
    scenario_name: str | None = "piper1b_rope"
    arm_name: str | None = None
    hardware: str = "auto"
    out_dir: Path | None = None
    resume_dir: Path | None = None
    seq_len: int | None = None
    steps: int | None = None
    batch: int | None = None
    extra_args: tuple[str, ...] | None = None
    timestamp: str | None = None
    cache_root: Path | None = None
    compiler_env: Path | None = None


@dataclass(frozen=True)
class RunResult:
    out_dir: Path
    scenario: Scenario
    selected_arms: tuple[Arm, ...]
    resumed: bool


@dataclass(frozen=True)
class RunEvent:
    kind: str
    message: str
    arm_name: str | None = None


EventHandler = Callable[[RunEvent], None]
ProcessRunner = Callable[..., subprocess.CompletedProcess]


def _emit(
    handler: EventHandler | None,
    kind: str,
    message: str,
    arm_name: str | None = None,
) -> None:
    if handler is not None:
        handler(RunEvent(kind=kind, message=message, arm_name=arm_name))


def workload_with_overrides(
    scenario: Scenario,
    *,
    seq_len: int | None = None,
    steps: int | None = None,
    batch: int | None = None,
    environment: Mapping[str, str] | None = None,
) -> Workload:
    """Apply portable size overrides without changing scenario arms."""
    environment = environment or os.environ
    workload = scenario.workload
    resolved_seq_len = seq_len if seq_len is not None else environment.get("SEQ")
    resolved_steps = steps if steps is not None else environment.get("STEPS")
    resolved_batch = batch if batch is not None else environment.get("BATCH")
    if resolved_seq_len is not None:
        workload = replace(workload, seq_len=int(resolved_seq_len))
    if resolved_steps is not None:
        workload = replace(workload, steps=int(resolved_steps))
    if resolved_batch is not None:
        workload = replace(workload, local_batch_size=int(resolved_batch))
    minimum_steps = workload.profile_freq * workload.min_trace_windows
    if workload.steps < minimum_steps:
        raise ValueError(
            f"steps ({workload.steps}) must be at least {minimum_steps} to collect "
            f"{workload.min_trace_windows} profiler windows"
        )
    return workload


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
) -> list[str]:
    expected = {
        "scenario": scenario.name,
        "workload": asdict(scenario.workload),
        "selected_arms": [arm.name for arm in arms],
        "hardware": hardware,
        "extra_torchtitan_args": list(extra_args),
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    existing_metadata = manifest.get("hardware_metadata", {})
    for key in (
        "nvidia_smi",
        "cpu_pinning",
        "torchtitan_git_rev",
        "benchmarks_git_rev",
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


def _resolve_run(
    request: RunRequest,
    environment: Mapping[str, str],
) -> tuple[
    RuntimePaths,
    Scenario,
    tuple[Arm, ...],
    str,
    dict[str, str],
    Path,
    dict[str, list[str]],
    bool,
]:
    paths = RuntimePaths.resolve(
        cache_root=request.cache_root,
        compiler_env=request.compiler_env,
        environment=environment,
    )
    resumed = request.resume_dir is not None
    existing_manifest = None
    if resumed:
        if request.out_dir is not None:
            raise ValueError("--out cannot be combined with --resume")
        resume_dir = request.resume_dir.expanduser().resolve()
        existing_manifest = load_manifest(resume_dir)
        manifest_scenario = existing_manifest.get("scenario")
        if request.scenario_name and request.scenario_name != manifest_scenario:
            raise ValueError(
                f"resume manifest uses scenario {manifest_scenario!r}, not "
                f"{request.scenario_name!r}"
            )
        scenario_name = manifest_scenario
    else:
        resume_dir = None
        scenario_name = request.scenario_name or "piper1b_rope"

    scenario = scenario_by_name(str(scenario_name))
    if existing_manifest is not None:
        workload = _resume_workload(existing_manifest, request, environment)
        recorded_extra_args = tuple(
            existing_manifest.get("extra_torchtitan_args", ())
        )
        extra_args = (
            recorded_extra_args
            if request.extra_args is None
            else request.extra_args
        )
    else:
        workload = workload_with_overrides(
            scenario,
            seq_len=request.seq_len,
            steps=request.steps,
            batch=request.batch,
            environment=environment,
        )
        extra_args = request.extra_args or ()
    scenario = replace(scenario, workload=workload)
    arms = (scenario.arm(request.arm_name),) if request.arm_name else scenario.arms

    requested_hardware = request.hardware
    if existing_manifest is not None and requested_hardware == "auto":
        requested_hardware = str(existing_manifest.get("hardware", "auto"))
    hardware, metadata = hardware_metadata(paths, request.gpu, requested_hardware)
    pinning = resolve_cpu_pinning(request.gpu)
    metadata = {**metadata, "cpu_pinning": pinning.description}
    out_dir = resume_dir or _default_output_dir(
        scenario, hardware, request.out_dir, environment, request.timestamp
    )
    commands = {
        arm.name: list(pinning.prefix)
        + command_for_arm(scenario.workload, arm, out_dir / arm.name, extra_args)
        for arm in arms
    }

    if existing_manifest is not None:
        mismatches = _resume_mismatches(
            existing_manifest,
            scenario,
            arms,
            hardware,
            metadata,
            extra_args,
        )
        if mismatches:
            raise ValueError(
                "resume request does not match the existing manifest: "
                + ", ".join(mismatches)
            )
    return paths, scenario, arms, hardware, metadata, out_dir, commands, resumed


def execute_run(
    request: RunRequest,
    *,
    event_handler: EventHandler | None = None,
    process_runner: ProcessRunner = subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> RunResult:
    """Execute and validate the selected arms, preserving resumable state."""
    host_environment = dict(environment or os.environ)
    (
        paths,
        scenario,
        arms,
        hardware,
        metadata,
        out_dir,
        commands,
        resumed,
    ) = _resolve_run(request, host_environment)

    if resumed:
        state = load_run_state(out_dir, arms)
        update_run_state(out_dir, state, status="running")
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        write_manifest(
            out_dir,
            scenario,
            arms,
            commands,
            hardware,
            metadata,
            request.extra_args or (),
        )
        state = initial_run_state(arms)
        update_run_state(out_dir, state, status="running")

    _emit(event_handler, "summary", f"GPU (PCI index): {request.gpu}")
    _emit(event_handler, "summary", metadata["nvidia_smi"])
    _emit(event_handler, "summary", f"cpu pinning: {metadata['cpu_pinning']}")
    _emit(
        event_handler,
        "summary",
        f"scenario: {scenario.name}   hardware: {hardware}",
    )
    _emit(
        event_handler,
        "summary",
        f"arms: {' '.join(arm.name for arm in arms)}",
    )
    _emit(event_handler, "summary", f"output: {out_dir}")

    base_environment = runtime_environment(
        paths, request.gpu, environment=host_environment
    )
    for arm in arms:
        arm_dir = out_dir / arm.name
        log_path = out_dir / f"{arm.name}.log"
        if resumed:
            try:
                validate_arm(
                    arm,
                    arm_dir,
                    log_path,
                    scenario.workload,
                    regions=scenario.regions,
                )
            except RuntimeError:
                archive = archive_incomplete_arm(out_dir, arm.name)
                if archive is not None:
                    _emit(
                        event_handler,
                        "archive",
                        f"{arm.name}: archived incomplete attempt at {archive}",
                        arm.name,
                    )
            else:
                update_run_state(
                    out_dir, state, arm_name=arm.name, status="completed"
                )
                _emit(
                    event_handler,
                    "skip",
                    f"{arm.name}: already validated; skipping",
                    arm.name,
                )
                continue

        command = commands[arm.name]
        _emit(event_handler, "arm", f"=== arm: {arm.name} ===", arm.name)
        _emit(event_handler, "command", shlex.join(command), arm.name)
        update_run_state(out_dir, state, arm_name=arm.name, status="running")
        try:
            arm_environment = base_environment
            if arm.requires_gcc_toolset:
                arm_environment = add_compiler_environment(
                    base_environment, paths.compiler_env
                )
            with log_path.open("w") as log:
                log.write(
                    f"# scenario={scenario.name} arm={arm.name} "
                    f"gpu_pci_index={request.gpu} "
                    f"{dt.datetime.now(dt.timezone.utc):%FT%TZ}\n"
                )
                log.write(metadata["nvidia_smi"] + "\n")
                log.flush()
                completed = process_runner(
                    command,
                    cwd=paths.titan_dir,
                    env=arm_environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"{arm.name}: training exited with {completed.returncode}; "
                    f"see {log_path}"
                )
            validate_arm(
                arm,
                arm_dir,
                log_path,
                scenario.workload,
                regions=scenario.regions,
            )
        except (Exception, KeyboardInterrupt) as error:
            update_run_state(
                out_dir,
                state,
                arm_name=arm.name,
                status="failed",
                error=str(error),
            )
            update_run_state(out_dir, state, status="failed")
            raise

        update_run_state(out_dir, state, arm_name=arm.name, status="completed")
        _emit(event_handler, "validated", f"{arm.name}: validated", arm.name)

    update_run_state(out_dir, state, status="arms_completed")
    return RunResult(out_dir, scenario, arms, resumed)
