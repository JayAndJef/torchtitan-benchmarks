"""Orchestrate declarative TorchTitan benchmark scenarios."""

from __future__ import annotations

import datetime as dt
import os
import shlex
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

from benchmarks.artifacts.layout import _default_output_dir, archive_incomplete_arm
from benchmarks.artifacts.manifests import (
    _resume_mismatches,
    _resume_workload,
    load_manifest,
    write_manifest,
)
from benchmarks.artifacts.run_state import (
    initial_run_state,
    load_run_state,
    update_run_state,
)
from benchmarks.e2e.axes import RunAxes, RunRequest
from benchmarks.e2e.engines import command_for_arm
from benchmarks.e2e.parallelism import (
    MEGATRON_ENGINES,
    PP_SCHEDULES,
    TRIVIAL_SPEC,
    zero_warnings,
    validate_parallelism,
)
from benchmarks.e2e.registry import (
    AC_MODES,
    DEFAULT_AC_MODE,
    DEFAULT_MEGATRON_NAN_GUARD,
    DEFAULT_MEGATRON_P2P_SYNC,
    DEFAULT_MEGATRON_PRECISION,
    DEFAULT_MODEL_SIZE,
    DEFAULT_PROFILE,
    DEFAULT_WARMUP_STEPS,
    MEGATRON_NAN_GUARD_MODES,
    MEGATRON_P2P_SYNC_MODES,
    MEGATRON_PRECISION_MODES,
    SCENARIOS,
    scenario_by_name,
)
from benchmarks.e2e.schema import Arm, Scenario, Workload
from benchmarks.e2e.validation import validate_arm
from benchmarks.execution.affinity import resolve_cpu_pinning
from benchmarks.execution.devices import parse_devices
from benchmarks.execution.environment import (
    add_compiler_environment,
    runtime_environment,
)
from benchmarks.execution.events import EventHandler, ProcessRunner, _emit
from benchmarks.execution.paths import RuntimePaths
from benchmarks.execution.provenance import hardware_metadata
from benchmarks.models.piper_qwen3.shape import (
    canonical_size_name,
    MODEL_SIZE_CHOICES,
    PIPER_SHAPES,
)


@dataclass(frozen=True)
class RunResult:
    out_dir: Path
    scenario: Scenario
    selected_arms: tuple[Arm, ...]
    resumed: bool


def select_arms(scenario: Scenario, names: tuple[str, ...]) -> tuple[Arm, ...]:
    """Resolve an ordered arm subset, rejecting ambiguous requests early."""
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


def workload_with_overrides(
    scenario: Scenario,
    *,
    seq_len: int | None = None,
    steps: int | None = None,
    batch: int | None = None,
    environment: Mapping[str, str] | None = None,
    profile: bool,
    warmup_steps: int | None,
) -> Workload:
    """Apply portable size overrides without changing scenario arms.

    ``profile`` decides which step floor applies: a profiled run needs
    ``profile_freq * min_trace_windows`` steps, and an unprofiled one needs
    more than ``warmup_steps``.
    """
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
    if profile:
        minimum_steps = workload.profile_freq * workload.min_trace_windows
        if workload.steps < minimum_steps:
            raise ValueError(
                f"steps ({workload.steps}) must be at least {minimum_steps} to collect "
                f"{workload.min_trace_windows} profiler windows"
            )
    elif warmup_steps is not None and workload.steps <= warmup_steps:
        raise ValueError(
            f"steps ({workload.steps}) must be more than the "
            f"{warmup_steps} warmup step(s); a run that measures no step "
            "publishes no throughput"
        )
    return workload


@dataclass(frozen=True)
class ResolvedRun:
    """One run, with every question the request left open answered.

    ``_resolve_run`` builds this record and does every refusal on the way:
    the scenario name, the arm subset, the mesh, the three megatron axes
    and the resume comparison. Whatever it returns is startable, so the
    caller reads fields instead of repeating checks.

    Attributes:
        paths: The resolved repository, cache and compiler-env locations.
        scenario: The scenario, with the size overrides already applied to
            its workload.
        arms: The arms this run starts, in the order the operator asked
            for.
        hardware: The provenance label of the output directory.
        metadata: The provenance block, including the CPU pinning.
        out_dir: Where the run writes.
        commands: One argv per arm name.
        axes: The eight global run axes, resolved.
        torchtitan_args: The ``--torchtitan-arg`` tokens, resolved.
        megatron_args: The ``--megatron-arg`` tokens, resolved.
        resumed: Whether the run continues a recorded directory.
    """

    paths: RuntimePaths
    scenario: Scenario
    arms: tuple[Arm, ...]
    hardware: str
    metadata: dict[str, str]
    out_dir: Path
    commands: dict[str, list[str]]
    axes: RunAxes
    resumed: bool
    torchtitan_args: tuple[str, ...]
    megatron_args: tuple[str, ...]


def _resolve_run(
    request: RunRequest,
    environment: Mapping[str, str],
    *,
    event_handler: EventHandler | None = None,
) -> ResolvedRun:
    """Resolve one request into the arms, the run axes and one argv per arm.

    Whatever this returns is startable: the scenario, the arm subset, the
    mesh, the three megatron axes and the resume comparison are all checked
    on the way.
    """
    requested = request.axes
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
        if request.scenario_name is None:
            raise ValueError(
                "no scenario requested, and there is no default. Pass "
                "--scenario. Available scenarios: "
                f"{', '.join(SCENARIOS)}"
            )
        scenario_name = request.scenario_name

    scenario = scenario_by_name(str(scenario_name))
    if existing_manifest is not None:
        workload = _resume_workload(existing_manifest, request, environment)
        torchtitan_args = (
            tuple(existing_manifest["extra_torchtitan_args"])
            if request.torchtitan_args is None
            else request.torchtitan_args
        )
        megatron_args = (
            tuple(existing_manifest["extra_megatron_args"])
            if request.megatron_args is None
            else request.megatron_args
        )
        ac_mode = (
            str(existing_manifest["ac_mode"])
            if requested.ac_mode is None
            else requested.ac_mode
        )
        model_size = (
            str(existing_manifest["model_size"])
            if requested.model_size is None
            else requested.model_size
        )
        megatron_p2p_sync = (
            str(existing_manifest["megatron_p2p_sync"])
            if requested.megatron_p2p_sync is None
            else requested.megatron_p2p_sync
        )
        megatron_nan_guard = (
            str(existing_manifest["megatron_nan_guard"])
            if requested.megatron_nan_guard is None
            else requested.megatron_nan_guard
        )
        megatron_precision = (
            str(existing_manifest["megatron_precision"])
            if requested.megatron_precision is None
            else requested.megatron_precision
        )
        profile = (
            bool(existing_manifest["profile"])
            if requested.profile is None
            else requested.profile
        )
        recorded_warmup = existing_manifest["warmup_steps"]
        # A request wins; the mismatch check below refuses a disagreement.
        warmup_steps = (
            requested.warmup_steps
            if requested.warmup_steps is not None
            else (None if recorded_warmup is None else int(recorded_warmup))
        )
    else:
        profile = (
            DEFAULT_PROFILE if requested.profile is None else requested.profile
        )
        # None under a profiled run, where the schedule decides the samples.
        warmup_steps = (
            None
            if profile
            else (
                DEFAULT_WARMUP_STEPS
                if requested.warmup_steps is None
                else requested.warmup_steps
            )
        )
        workload = workload_with_overrides(
            scenario,
            seq_len=request.seq_len,
            steps=request.steps,
            batch=request.batch,
            environment=environment,
            profile=profile,
            warmup_steps=warmup_steps,
        )
        torchtitan_args = request.torchtitan_args or ()
        megatron_args = request.megatron_args or ()
        ac_mode = requested.ac_mode or DEFAULT_AC_MODE
        model_size = requested.model_size or DEFAULT_MODEL_SIZE
        megatron_p2p_sync = (
            requested.megatron_p2p_sync or DEFAULT_MEGATRON_P2P_SYNC
        )
        megatron_nan_guard = (
            requested.megatron_nan_guard or DEFAULT_MEGATRON_NAN_GUARD
        )
        megatron_precision = (
            requested.megatron_precision or DEFAULT_MEGATRON_PRECISION
        )
    if megatron_p2p_sync not in MEGATRON_P2P_SYNC_MODES:
        raise ValueError(
            f"unknown megatron p2p sync {megatron_p2p_sync!r}. Available: "
            f"{', '.join(MEGATRON_P2P_SYNC_MODES)}"
        )
    if megatron_nan_guard not in MEGATRON_NAN_GUARD_MODES:
        raise ValueError(
            f"unknown megatron nan guard {megatron_nan_guard!r}. Available: "
            f"{', '.join(MEGATRON_NAN_GUARD_MODES)}"
        )
    if megatron_precision not in MEGATRON_PRECISION_MODES:
        raise ValueError(
            f"unknown megatron precision {megatron_precision!r}. Available: "
            f"{', '.join(MEGATRON_PRECISION_MODES)}"
        )
    if ac_mode not in AC_MODES:
        raise ValueError(
            f"unknown ac mode {ac_mode!r}. Available: {', '.join(AC_MODES)}"
        )
    # Resolved once here, so nothing downstream sees a retired alias.
    model_size = canonical_size_name(model_size)
    if model_size not in PIPER_SHAPES:
        raise ValueError(
            f"unknown model size {model_size!r}. "
            f"Available: {', '.join(MODEL_SIZE_CHOICES)}"
        )
    shape = PIPER_SHAPES[model_size]
    scenario = replace(scenario, workload=workload)
    arms = select_arms(scenario, request.arm_names)
    if ac_mode not in scenario.supported_ac_modes:
        raise ValueError(
            f"scenario {scenario.name!r} does not support ac mode {ac_mode!r} "
            f"(supported: {', '.join(scenario.supported_ac_modes)})"
        )

    # Checked before any host probe, against the arms this run really starts.
    parallelism = requested.parallelism or TRIVIAL_SPEC
    devices = parse_devices(request.gpu)
    validate_parallelism(
        parallelism,
        shape=shape,
        workload=workload,
        engines={arm.engine for arm in arms},
        device_count=len(devices),
    )
    # Three schedules raise on a compiled stage module, and compile is an
    # arm property, so the spec alone cannot answer this.
    schedule = (
        PP_SCHEDULES.get(parallelism.pp_schedule)
        if parallelism.pp_schedule is not None
        else None
    )
    if schedule is not None and schedule.requires_uncompiled:
        compiled_arms = [arm.name for arm in arms if arm.compile == "torch"]
        if compiled_arms:
            raise ValueError(
                f"pipeline schedule {schedule.name!r} raises on a compiled "
                f"stage module, and {', '.join(compiled_arms)} asks for "
                "torch.compile; select the eager arms alone, or choose "
                "another --pp-schedule"
            )
    # Two legal meshes a reader can misread. Printed before the host probe,
    # so the operator reads them before the run claims a GPU.
    for warning in zero_warnings(
        parallelism, engines=[arm.engine for arm in arms]
    ):
        _emit(event_handler, "summary", f"WARNING: {warning}")

    # The literal ``on``, never the default: it is the value that asks for
    # a synchronize, so it is the value a mesh without messages cannot honor.
    if megatron_p2p_sync == "on":
        if parallelism.pp == 1:
            raise ValueError(
                f"--megatron-p2p-sync {megatron_p2p_sync!r} was requested at "
                "pp 1, where there is no pipeline message to synchronize; "
                "the manifest would record a treatment the run did not have"
            )
        if not any(arm.engine in MEGATRON_ENGINES for arm in arms):
            raise ValueError(
                f"--megatron-p2p-sync {megatron_p2p_sync!r} reaches no arm "
                f"of this run: {', '.join(arm.name for arm in arms)} run on "
                "TorchTitan, which sends no pipeline message through "
                "Megatron; select a megatron arm, or leave the option at "
                f"{DEFAULT_MEGATRON_P2P_SYNC!r}"
            )

    # Through the same helper the skip pre-pass reads, so both state one reason.
    refusal = megatron_nan_guard_refusal(arms, megatron_nan_guard)
    if refusal is not None:
        raise ValueError(refusal)

    # Through the same helper the skip pre-pass reads, so both state one reason.
    refusal = megatron_precision_refusal(
        arms, megatron_precision, parallelism.zero
    )
    if refusal is not None:
        raise ValueError(refusal)

    refusal = passthrough_refusal(arms, torchtitan_args, megatron_args)
    if refusal is not None:
        raise ValueError(refusal)

    # Every axis is answered, so one record carries them from here on.
    axes = RunAxes(
        ac_mode=ac_mode,
        model_size=model_size,
        parallelism=parallelism,
        megatron_p2p_sync=megatron_p2p_sync,
        megatron_nan_guard=megatron_nan_guard,
        megatron_precision=megatron_precision,
        profile=profile,
        warmup_steps=warmup_steps,
    )

    requested_hardware = request.hardware
    if existing_manifest is not None and requested_hardware == "auto":
        requested_hardware = str(existing_manifest.get("hardware", "auto"))
    hardware, metadata = hardware_metadata(paths, request.gpu, requested_hardware)
    pinning = resolve_cpu_pinning(request.gpu)
    metadata = {**metadata, "cpu_pinning": pinning.description}
    out_dir = resume_dir or _default_output_dir(
        scenario,
        hardware,
        request.out_dir,
        environment,
        request.timestamp,
        request.occurrence,
    )
    commands = {
        arm.name: list(pinning.prefix)
        + command_for_arm(
            scenario.workload,
            arm,
            out_dir / arm.name,
            megatron_args if arm.engine in MEGATRON_ENGINES else torchtitan_args,
            axes.ac_mode,
            model_size=axes.model_size,
            parallelism=axes.parallelism,
            megatron_p2p_sync=axes.megatron_p2p_sync,
            megatron_nan_guard=axes.megatron_nan_guard,
            megatron_precision=axes.megatron_precision,
            profile=axes.profile,
        )
        for arm in arms
    }

    if existing_manifest is not None:
        mismatches = _resume_mismatches(
            existing_manifest,
            scenario,
            arms,
            hardware,
            metadata,
            torchtitan_args,
            megatron_args=megatron_args,
            axes=axes,
        )
        if mismatches:
            raise ValueError(
                "resume request does not match the existing manifest: "
                + ", ".join(mismatches)
            )
    return ResolvedRun(
        paths=paths,
        scenario=scenario,
        arms=arms,
        hardware=hardware,
        metadata=metadata,
        out_dir=out_dir,
        commands=commands,
        axes=axes,
        resumed=resumed,
        torchtitan_args=torchtitan_args,
        megatron_args=megatron_args,
    )


def passthrough_refusal(
    arms: Iterable[Arm],
    torchtitan_args: tuple[str, ...],
    megatron_args: tuple[str, ...],
) -> str | None:
    """Why a passthrough list cannot reach ``arms``, or ``None``."""
    arms = tuple(arms)
    names = ", ".join(arm.name for arm in arms)
    if torchtitan_args and not any(
        arm.engine not in MEGATRON_ENGINES for arm in arms
    ):
        return (
            f"--torchtitan-arg reaches no arm of this run: {names} run on "
            "Megatron; select a TorchTitan arm, or omit the option"
        )
    if megatron_args and not any(
        arm.engine in MEGATRON_ENGINES for arm in arms
    ):
        return (
            f"--megatron-arg reaches no arm of this run: {names} run on "
            "TorchTitan; select the stock megatron arm, or omit the option"
        )
    return None


def megatron_precision_refusal(
    arms: Iterable[Arm], megatron_precision: str, zero: int
) -> str | None:
    """Why ``--megatron-precision lean`` cannot reach ``arms``, or ``None``.

    Two refusals, each naming its repair. A run with no stock megatron arm
    gives the value nothing to reach, and ``lean`` under ``zero 0`` asks
    Megatron for a precision-aware optimizer without the distributed
    optimizer it asserts. Refused here, the operator reads the repair
    parent-side rather than minutes into a subprocess.
    """
    if megatron_precision == DEFAULT_MEGATRON_PRECISION:
        return None
    arms = tuple(arms)
    if not any(arm.engine in MEGATRON_ENGINES for arm in arms):
        return (
            f"--megatron-precision {megatron_precision!r} reaches no arm of "
            f"this run: {', '.join(arm.name for arm in arms)} run on "
            "TorchTitan, which holds its own bf16 optimizer states; select "
            "the stock megatron arm, or leave the option at "
            f"{DEFAULT_MEGATRON_PRECISION!r}"
        )
    if zero == 0:
        return (
            f"--megatron-precision {megatron_precision!r} needs "
            "--zero 1: Megatron asserts use_distributed_optimizer under "
            "--use-precision-aware-optimizer, and the zero level is the "
            "one owner of that flag"
        )
    return None


def megatron_nan_guard_refusal(
    arms: Iterable[Arm], megatron_nan_guard: str
) -> str | None:
    """Why ``--megatron-nan-guard on`` cannot reach ``arms``, or ``None``.

    One refusal, naming its repair. A run with no stock megatron arm
    gives the value nothing to reach, which is the ``--megatron-p2p-sync``
    refusal with a smaller engine set. ``_resolve_run`` raises the
    string, and the skip pre-pass prints it and skips the
    scenario.

    The gate reads the literal ``on`` and never the axis default. ``on``
    asks stock Megatron to keep its own check, so a run with no stock
    megatron arm has nothing to ask. ``off`` is refused nowhere: it is the
    default here, and a TorchTitan arm's argv is untouched under either
    value.
    """
    if megatron_nan_guard != "on":
        return None
    arms = tuple(arms)
    if not any(arm.engine in MEGATRON_ENGINES for arm in arms):
        return (
            f"--megatron-nan-guard {megatron_nan_guard!r} reaches no arm of "
            f"this run: {', '.join(arm.name for arm in arms)} run on "
            "TorchTitan, which has no Megatron NaN guard; select the stock "
            f"megatron arm, or leave the option at {DEFAULT_MEGATRON_NAN_GUARD!r}"
        )
    return None


def execute_run(
    request: RunRequest,
    *,
    event_handler: EventHandler | None = None,
    process_runner: ProcessRunner = subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> RunResult:
    """Execute and validate the selected arms, preserving resumable state."""
    host_environment = dict(environment or os.environ)
    resolved = _resolve_run(
        request, host_environment, event_handler=event_handler
    )
    axes = resolved.axes
    # Two names for what the loop below reads on nearly every line.
    arms = resolved.arms
    out_dir = resolved.out_dir

    if resolved.resumed:
        state = load_run_state(out_dir, arms)
        update_run_state(out_dir, state, status="running")
    else:
        out_dir.mkdir(parents=True, exist_ok=False)
        write_manifest(
            out_dir,
            resolved.scenario,
            arms,
            resolved.commands,
            resolved.hardware,
            resolved.metadata,
            resolved.torchtitan_args,
            megatron_args=resolved.megatron_args,
            axes=axes,
        )
        state = initial_run_state(arms)
        update_run_state(out_dir, state, status="running")

    _emit(event_handler, "summary", f"GPU (PCI index): {request.gpu}")
    _emit(event_handler, "summary", resolved.metadata["nvidia_smi"])
    _emit(
        event_handler,
        "summary",
        f"cpu pinning: {resolved.metadata['cpu_pinning']}",
    )
    _emit(
        event_handler,
        "summary",
        f"scenario: {resolved.scenario.name}   hardware: {resolved.hardware}",
    )
    _emit(
        event_handler,
        "summary",
        f"arms: {' '.join(arm.name for arm in arms)}",
    )
    _emit(event_handler, "summary", f"ac mode: {axes.ac_mode}")
    _emit(event_handler, "summary", f"model size: {axes.model_size}")
    _emit(
        event_handler,
        "summary",
        f"parallelism: dp {axes.parallelism.dp} x pp {axes.parallelism.pp} "
        f"(ep {axes.parallelism.ep}, world size {axes.parallelism.world_size}, "
        f"zero {axes.parallelism.zero})",
    )
    _emit(
        event_handler, "summary", f"megatron p2p sync: {axes.megatron_p2p_sync}"
    )
    _emit(
        event_handler,
        "summary",
        f"megatron nan guard: {axes.megatron_nan_guard}",
    )
    _emit(
        event_handler,
        "summary",
        f"megatron precision: {axes.megatron_precision}",
    )
    _emit(
        event_handler, "summary", f"profile: {'on' if axes.profile else 'off'}"
    )
    if axes.warmup_steps is not None:
        _emit(event_handler, "summary", f"warmup steps: {axes.warmup_steps}")
    _emit(event_handler, "summary", f"output: {out_dir}")

    base_environment = runtime_environment(
        resolved.paths,
        request.gpu,
        environment=host_environment,
        world_size=axes.parallelism.world_size,
    )
    for arm in arms:
        arm_dir = out_dir / arm.name
        log_path = out_dir / f"{arm.name}.log"
        if resolved.resumed:
            try:
                validate_arm(
                    arm,
                    arm_dir,
                    log_path,
                    resolved.scenario.workload,
                    ac_mode=axes.ac_mode,
                    model_size=axes.model_size,
                    parallelism=axes.parallelism,
                    megatron_p2p_sync=axes.megatron_p2p_sync,
                    megatron_nan_guard=axes.megatron_nan_guard,
                    megatron_precision=axes.megatron_precision,
                    profile=axes.profile,
                    megatron_args=resolved.megatron_args,
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

        command = resolved.commands[arm.name]
        _emit(event_handler, "arm", f"=== arm: {arm.name} ===", arm.name)
        _emit(event_handler, "command", shlex.join(command), arm.name)
        update_run_state(out_dir, state, arm_name=arm.name, status="running")
        try:
            arm_environment = base_environment
            if arm.requires_gcc_toolset:
                arm_environment = add_compiler_environment(
                    base_environment, resolved.paths.compiler_env
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
                    command,
                    cwd=resolved.paths.titan_dir,
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
                resolved.scenario.workload,
                ac_mode=axes.ac_mode,
                model_size=axes.model_size,
                parallelism=axes.parallelism,
                megatron_p2p_sync=axes.megatron_p2p_sync,
                megatron_nan_guard=axes.megatron_nan_guard,
                megatron_precision=axes.megatron_precision,
                profile=axes.profile,
                megatron_args=resolved.megatron_args,
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
    return RunResult(out_dir, resolved.scenario, arms, resolved.resumed)
