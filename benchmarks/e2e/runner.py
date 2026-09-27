"""Resolve a run request, check it, start each arm and validate each arm's outputs."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import shlex
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from benchmarks.artifacts.layout import _default_output_dir, archive_incomplete_arm
from benchmarks.artifacts.manifests import (
    ArmRecord,
    config_json,
    host_mismatches,
    load_manifest,
    run_record,
    write_manifest,
)
from benchmarks.artifacts.run_state import (
    initial_run_state,
    load_run_state,
    update_run_state,
)
from benchmarks.e2e.axes import RunRequest
from benchmarks.e2e.checks import check_run, run_warnings
from benchmarks.e2e.engines.api import Arm, DataSpec, Launch, RunSpec
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.e2e.overrides import Override, apply_overrides
from benchmarks.e2e.parallelism import TRIVIAL_SPEC
from benchmarks.e2e.registry import (
    DEFAULT_AC_MODE,
    DEFAULT_MODEL_SIZE,
    DEFAULT_PROFILE,
    DEFAULT_WARMUP_STEPS,
    SCENARIOS,
    SEED,
    scenario_by_name,
)
from benchmarks.e2e.schema import Scenario
from benchmarks.e2e.validation import validate_arm
from benchmarks.execution.affinity import CpuPinning, resolve_cpu_pinning
from benchmarks.execution.devices import parse_devices
from benchmarks.execution.environment import (
    add_compiler_environment,
    runtime_environment,
)
from benchmarks.execution.events import EventHandler, ProcessRunner, _emit
from benchmarks.execution.launcher import (
    build_command,
    command_line,
    environment_delta,
    pinning_record,
)
from benchmarks.execution.paths import RuntimePaths
from benchmarks.execution.provenance import hardware_metadata
from benchmarks.models.piper_qwen3.shape import shape_by_name


@dataclass(frozen=True)
class RunResult:
    out_dir: Path
    scenario: Scenario
    selected_arms: tuple[Arm, ...]
    resumed: bool


def select_arms(scenario: Scenario, names: tuple[str, ...]) -> tuple[Arm, ...]:
    """The ordered arm subset that ``names`` asks for; every arm when it is empty."""
    if not names:
        return scenario.arms

    duplicates = tuple(
        name for index, name in enumerate(names) if name in names[:index]
    )
    if duplicates:
        raise ValueError(
            "--arm repeats "
            + ", ".join(repr(name) for name in dict.fromkeys(duplicates))
        )

    available = {arm.name: arm for arm in scenario.arms}
    unknown = tuple(name for name in names if name not in available)
    if unknown:
        raise ValueError(
            f"scenario {scenario.name!r} has no arm(s) "
            + ", ".join(repr(name) for name in unknown)
            + f". Available: {', '.join(available)}"
        )
    return tuple(available[name] for name in names)


def data_with_overrides(
    data: DataSpec,
    *,
    seq_len: int | None = None,
    steps: int | None = None,
    batch: int | None = None,
    environment: Mapping[str, str],
) -> DataSpec:
    """``data`` with each size that a flag or ``SEQ``, ``STEPS`` or ``BATCH`` gives."""
    for field, flag, variable in (
        ("seq_len", seq_len, "SEQ"),
        ("steps", steps, "STEPS"),
        ("local_batch_size", batch, "BATCH"),
    ):
        value = flag if flag is not None else environment.get(variable)
        if value is not None:
            data = replace(data, **{field: int(value)})
    return data


def _resumed_arms(
    arms: tuple[Arm, ...],
    overrides: tuple[Override, ...],
    recorded: Mapping[str, Arm],
) -> tuple[Arm, ...]:
    """The arms of a resume: each recorded field, except the fields that ``--set`` names."""
    named = {(override.arm, override.field) for override in overrides}
    result = []
    for arm in arms:
        before = recorded.get(arm.name)
        if before is None or type(before.config) is not type(arm.config):
            result.append(arm)
            continue
        values = {
            field.name: getattr(
                arm.config if (arm.name, field.name) in named else before.config,
                field.name,
            )
            for field in dataclasses.fields(arm.config)
        }
        result.append(replace(arm, config=type(arm.config)(**values)))
    return tuple(result)


@dataclass(frozen=True)
class ResolvedRun:
    """One checked run: every question that the request left open has an answer."""

    paths: RuntimePaths
    scenario: Scenario
    run: RunSpec
    arms: tuple[Arm, ...]
    """The arms in the order the operator asked for, with the ``--set`` overrides applied."""
    hardware: str
    metadata: dict[str, str]
    """The provenance block, with the host's CPU pinning."""
    out_dir: Path
    launches: dict[str, Launch]
    pinning: CpuPinning
    records: dict[str, ArmRecord]
    """The manifest record of each arm."""
    resumed: bool

    @property
    def commands(self) -> dict[str, list[str]]:
        """The argv of each arm."""
        return {name: list(record.command) for name, record in self.records.items()}


def _resolve_run(
    request: RunRequest,
    environment: Mapping[str, str],
    *,
    event_handler: EventHandler | None = None,
) -> ResolvedRun:
    """Resolve and check one request; the result is a run that can start."""
    requested = request.axes
    paths = RuntimePaths.resolve(
        cache_root=request.cache_root,
        compiler_env=request.compiler_env,
        environment=environment,
    )
    resumed_manifest: dict[str, Any] | None = None
    recorded = None
    resume_dir = None
    if request.resume_dir is not None:
        if request.out_dir is not None:
            raise ValueError("--out cannot be combined with --resume")
        resume_dir = request.resume_dir.expanduser().resolve()
        resumed_manifest = load_manifest(resume_dir)
        recorded = run_record(resumed_manifest, str(resume_dir / "manifest.json"))
        if request.scenario_name and request.scenario_name != recorded.scenario:
            raise ValueError(
                f"resume manifest uses scenario {recorded.scenario!r}, not "
                f"{request.scenario_name!r}"
            )
        scenario_name = recorded.scenario
    elif request.scenario_name is None:
        raise ValueError(
            "no scenario requested, and there is no default. Pass "
            f"--scenario. Available scenarios: {', '.join(SCENARIOS)}"
        )
    else:
        scenario_name = request.scenario_name
    scenario = scenario_by_name(scenario_name)

    base = scenario if recorded is None else recorded.run
    data = data_with_overrides(
        base.data,
        seq_len=request.seq_len,
        steps=request.steps,
        batch=request.batch,
        environment=environment,
    )
    if recorded is None:
        ac_mode = requested.ac_mode or DEFAULT_AC_MODE
        model_size = requested.model_size or DEFAULT_MODEL_SIZE
        profile = DEFAULT_PROFILE if requested.profile is None else requested.profile
        warmup_steps = requested.warmup_steps
        seed: int | None = SEED
    else:
        ac_mode = requested.ac_mode or recorded.run.ac_mode
        model_size = requested.model_size or recorded.run.shape.name
        profile = recorded.run.profile if requested.profile is None else requested.profile
        warmup_steps = requested.warmup_steps
        if warmup_steps is None and profile == recorded.run.profile:
            warmup_steps = recorded.run.warmup_steps
        seed = recorded.run.seed
    if profile and warmup_steps is not None:
        raise ValueError(
            "--warmup-steps applies only without --profile, and this run "
            f"asks for profile on with {warmup_steps} warmup step(s); the "
            "profiler schedule decides the samples of a profiled run"
        )
    if not profile and warmup_steps is None:
        warmup_steps = DEFAULT_WARMUP_STEPS

    arms = apply_overrides(select_arms(scenario, request.arm_names), request.overrides)
    if recorded is not None:
        arms = _resumed_arms(
            arms,
            request.overrides,
            {record.arm.name: record.arm for record in recorded.arms},
        )
    run = RunSpec(
        shape=shape_by_name(model_size),
        data=data,
        parallelism=requested.parallelism or TRIVIAL_SPEC,
        ac_mode=ac_mode,
        profile=profile,
        window=scenario.window if recorded is None else recorded.run.window,
        warmup_steps=warmup_steps,
        seed=seed,
    )
    check_run(
        run,
        scenario,
        arms,
        device_count=len(parse_devices(request.gpu)),
        resumed=resumed_manifest,
    )
    # Printed before the host probe, so the operator reads them before the run claims a GPU.
    for warning in run_warnings(run, arms):
        _emit(event_handler, "summary", f"WARNING: {warning}")

    requested_hardware = request.hardware
    if recorded is not None and requested_hardware == "auto":
        requested_hardware = recorded.hardware
    hardware, metadata = hardware_metadata(paths, request.gpu, requested_hardware)
    pinning = resolve_cpu_pinning(request.gpu)
    metadata = {**metadata, "cpu_pinning": pinning.description}
    if resumed_manifest is not None:
        mismatches = host_mismatches(
            resumed_manifest, hardware=hardware, metadata=metadata
        )
        if mismatches:
            raise ValueError(
                "resume request does not match the existing manifest: "
                + ", ".join(mismatches)
            )
    out_dir = resume_dir or _default_output_dir(
        scenario,
        hardware,
        request.out_dir,
        environment,
        request.timestamp,
        request.occurrence,
    )
    world_size = run.parallelism.world_size
    launches = {
        arm.name: engine_for(arm).launch(run, arm, out_dir / arm.name)
        for arm in arms
    }
    records = {
        arm.name: ArmRecord(
            arm=arm,
            command=command_line(
                launches[arm.name], world_size=world_size, pinning=pinning
            ),
            env_delta=environment_delta(
                launches[arm.name], world_size=world_size, gpu=request.gpu
            ),
            cpu_pinning=pinning_record(launches[arm.name], pinning),
            execution_model=engine_for(arm).execution_model(run, arm),
        )
        for arm in arms
    }
    return ResolvedRun(
        paths=paths,
        scenario=scenario,
        run=run,
        arms=arms,
        hardware=hardware,
        metadata=metadata,
        out_dir=out_dir,
        launches=launches,
        pinning=pinning,
        records=records,
        resumed=recorded is not None,
    )


def _banner(resolved: ResolvedRun, gpu: str) -> list[str]:
    """The summary lines that a run prints before its first arm."""
    run = resolved.run
    spec = run.parallelism
    lines = [
        f"GPU (PCI index): {gpu}",
        resolved.metadata["nvidia_smi"],
        f"cpu pinning: {resolved.metadata['cpu_pinning']}",
        f"scenario: {resolved.scenario.name}   hardware: {resolved.hardware}",
        f"arms: {' '.join(arm.name for arm in resolved.arms)}",
        f"ac mode: {run.ac_mode}",
        f"model size: {run.shape.name}",
        f"parallelism: dp {spec.dp} x pp {spec.pp} (ep {spec.ep}, world size "
        f"{spec.world_size}, zero {spec.zero})",
        f"profile: {'on' if run.profile else 'off'}",
    ]
    if run.warmup_steps is not None:
        lines.append(f"warmup steps: {run.warmup_steps}")
    for arm in resolved.arms:
        lines.append(
            f"config {arm.name}: {engine_for(arm).name} "
            f"{json.dumps(config_json(arm.config))}"
        )
    lines.append(f"output: {resolved.out_dir}")
    return lines


def execute_run(
    request: RunRequest,
    *,
    event_handler: EventHandler | None = None,
    process_runner: ProcessRunner = subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> RunResult:
    """Start and validate the selected arms, and keep a state file that a resume reads."""
    host_environment = dict(environment or os.environ)
    resolved = _resolve_run(request, host_environment, event_handler=event_handler)
    arms = resolved.arms
    out_dir = resolved.out_dir

    if resolved.resumed:
        state = load_run_state(out_dir, arms)
        update_run_state(out_dir, state, status="running")
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        write_manifest(
            out_dir,
            scenario=resolved.scenario,
            hardware=resolved.hardware,
            metadata=resolved.metadata,
            run=resolved.run,
            arms=tuple(resolved.records[arm.name] for arm in arms),
        )
        state = initial_run_state(arms)
        update_run_state(out_dir, state, status="running")

    for line in _banner(resolved, request.gpu):
        _emit(event_handler, "summary", line)

    base_environment = runtime_environment(
        resolved.paths, environment=host_environment
    )
    for arm in arms:
        arm_dir = out_dir / arm.name
        log_path = out_dir / f"{arm.name}.log"
        if resolved.resumed:
            try:
                validate_arm(resolved.run, arm, engine_for(arm), arm_dir, log_path)
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

        command = resolved.commands[arm.name]
        _emit(event_handler, "arm", f"=== arm: {arm.name} ===", arm.name)
        _emit(event_handler, "command", shlex.join(command), arm.name)
        update_run_state(out_dir, state, arm_name=arm.name, status="running")
        try:
            launch = resolved.launches[arm.name]
            arm_environment = base_environment
            if launch.host_compiler:
                arm_environment = add_compiler_environment(
                    base_environment, resolved.paths.compiler_env
                )
            launched = build_command(
                launch,
                world_size=resolved.run.parallelism.world_size,
                gpu=request.gpu,
                pinning=resolved.pinning,
                base_env=arm_environment,
            )
            with log_path.open("w") as log:
                log.write(
                    f"# scenario={resolved.scenario.name} arm={arm.name} "
                    f"gpu_pci_index={request.gpu} "
                    f"{dt.datetime.now(dt.timezone.utc):%FT%TZ}\n"
                )
                log.write(resolved.metadata["nvidia_smi"] + "\n")
                log.flush()
                completed = process_runner(
                    list(launched.argv),
                    cwd=launched.cwd,
                    env=dict(launched.env),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode:
                raise RuntimeError(
                    f"{arm.name}: training exited with {completed.returncode}; "
                    f"see {log_path}"
                )
            validate_arm(resolved.run, arm, engine_for(arm), arm_dir, log_path)
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
    return RunResult(out_dir, resolved.scenario, arms, resolved.resumed)
