"""Orchestrate declarative TorchTitan benchmark scenarios."""

from __future__ import annotations

import datetime as dt
import os
import shlex
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

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
from benchmarks.e2e.launch import command_for_arm
from benchmarks.e2e.parallelism import (
    MEGATRON_LAUNCHERS,
    ParallelismSpec,
    TRIVIAL_SPEC,
    validate_parallelism,
)
from benchmarks.e2e.registry import (
    AC_MODES,
    COMPILE_MODES,
    DEFAULT_AC_MODE,
    DEFAULT_COMPILE_MODE,
    DEFAULT_MEGATRON_P2P_SYNC,
    DEFAULT_MODEL_SIZE,
    MEGATRON_P2P_SYNC_MODES,
    SCENARIOS,
    UNCOMPILED_COMPILE_MODES,
    Arm,
    Scenario,
    Workload,
    piper_block_regions,
    scenario_by_name,
)
from benchmarks.e2e.validation import validate_arm
from benchmarks.execution.affinity import resolve_cpu_pinning
from benchmarks.execution.devices import parse_devices
from benchmarks.execution.environment import (
    add_compiler_environment,
    refuse_megatron_fsdp_connection_limit,
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
class RunRequest:
    """User-selected inputs for one benchmark execution."""

    # The ``<gpu>`` positional, kept exactly as the operator typed it. It
    # names a device *set* -- ``parse_devices`` splits it -- but the string
    # itself is never rewritten: roughly one hundred manifests under ``out/``
    # record it as ``hardware_metadata.requested_gpu``, and
    # ``CUDA_VISIBLE_DEVICES`` is set from the same value.
    gpu: str
    # No default scenario. ``None`` means "not requested", which only a resume
    # may leave unanswered: the recorded manifest names the scenario there. A
    # default could only be reached by an omission, and would then measure one
    # scenario under whatever label the operator assumed, which is a wrong
    # result rather than a missing one. ``_resolve_run`` refuses it otherwise.
    scenario_name: str | None = None
    # Empty means every scenario arm. A non-empty tuple is an ordered subset,
    # matching repeated ``run --arm NAME`` options exactly.
    arm_names: tuple[str, ...] = ()
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
    # None means "not requested": a resume then inherits the recorded mode,
    # while an explicit value is checked against the manifest.
    compile_mode: str | None = None
    ac_mode: str | None = None
    model_size: str | None = None
    # The fourth global run axis. ``None`` means "not requested" and resolves
    # to ``TRIVIAL_SPEC``, exactly as the three above resolve to their own
    # defaults.
    #
    # **A resume does not inherit it, and that is not an oversight.** The
    # three axes above are single strings, so a resume can read one back and
    # rebuild the run from it. A spec is five fields that together decide
    # every arm's command line, and ``--resume`` compares no command line --
    # so a reconstruction that dropped one field would relaunch the arms
    # differently and the gate would not see it. Omitting the flags on a
    # resume therefore asks for the trivial spec, which matches a
    # single-GPU directory and is refused against any other. The stage that
    # first runs a parallel job may add inheritance, with the round trip
    # under test.
    parallelism: ParallelismSpec | None = None
    # The Megatron pipeline p2p sync treatment. ``None`` means "not
    # requested": a resume inherits the recorded value, and a fresh run
    # takes ``on``, exactly as ``compile_mode`` does. It is not a field of
    # ``parallelism``, because it is a treatment of the pipeline messages
    # the way ``compile_mode`` is a treatment of the blocks, and
    # ``execution_model`` names degrees rather than mechanisms.
    megatron_p2p_sync: str | None = None


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
    str,
    str,
    str,
    ParallelismSpec,
    bool,
    str,
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
        recorded_extra_args = tuple(
            existing_manifest.get("extra_torchtitan_args", ())
        )
        extra_args = (
            recorded_extra_args
            if request.extra_args is None
            else request.extra_args
        )
        compile_mode = (
            str(existing_manifest.get("compile_mode", DEFAULT_COMPILE_MODE))
            if request.compile_mode is None
            else request.compile_mode
        )
        # Schema <= 7 manifests imply the historical SAC treatment.
        ac_mode = (
            str(existing_manifest.get("ac_mode", DEFAULT_AC_MODE))
            if request.ac_mode is None
            else request.ac_mode
        )
        # Schema <= 8 manifests predate the model-size axis.
        model_size = (
            str(existing_manifest.get("model_size", DEFAULT_MODEL_SIZE))
            if request.model_size is None
            else request.model_size
        )
        # Schema <= 12 manifests predate the p2p axis, and every one of
        # them ran stock Megatron's own sync.
        megatron_p2p_sync = (
            str(
                existing_manifest.get(
                    "megatron_p2p_sync", DEFAULT_MEGATRON_P2P_SYNC
                )
            )
            if request.megatron_p2p_sync is None
            else request.megatron_p2p_sync
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
        compile_mode = request.compile_mode or DEFAULT_COMPILE_MODE
        ac_mode = request.ac_mode or DEFAULT_AC_MODE
        model_size = request.model_size or DEFAULT_MODEL_SIZE
        megatron_p2p_sync = (
            request.megatron_p2p_sync or DEFAULT_MEGATRON_P2P_SYNC
        )
    if megatron_p2p_sync not in MEGATRON_P2P_SYNC_MODES:
        raise ValueError(
            f"unknown megatron p2p sync {megatron_p2p_sync!r}. Available: "
            f"{', '.join(MEGATRON_P2P_SYNC_MODES)}"
        )
    if compile_mode not in COMPILE_MODES:
        raise ValueError(
            f"unknown compile mode {compile_mode!r} (schema <= 7 manifests "
            f"recorded torch-level names; those runs cannot be resumed here). "
            f"Available: {', '.join(COMPILE_MODES)}"
        )
    if ac_mode not in AC_MODES:
        raise ValueError(
            f"unknown ac mode {ac_mode!r}. Available: {', '.join(AC_MODES)}"
        )
    # Resolved once, here: everything downstream -- the manifest record, the
    # --config-arg the training command carries, the resume comparison -- must
    # see the canonical name rather than a retired alias.
    model_size = canonical_size_name(model_size)
    if model_size not in PIPER_SHAPES:
        raise ValueError(
            f"unknown model size {model_size!r}. "
            f"Available: {', '.join(MODEL_SIZE_CHOICES)}"
        )
    shape = PIPER_SHAPES[model_size]
    scenario = replace(scenario, workload=workload)
    arms = select_arms(scenario, request.arm_names)
    # A scenario's supported modes describe a run of its complete roster. An
    # explicit subset may narrow away the engine that cannot honor a mode. The
    # only such treatment today is ``none``: it removes TorchTitan's whole-
    # block compile and has no truthful meaning for Megatron. Keep the subset
    # exception as narrow as the treatment itself, so a future scenario mode
    # does not become legal merely because ``--arm`` was present.
    selected_titan_only_none = (
        bool(request.arm_names)
        and compile_mode in UNCOMPILED_COMPILE_MODES
        and all(arm.launcher == "torchtitan" for arm in arms)
    )
    if (
        compile_mode not in scenario.supported_compile_modes
        and not selected_titan_only_none
    ):
        raise ValueError(
            f"scenario {scenario.name!r} does not support compile mode "
            f"{compile_mode!r} (supported: "
            f"{', '.join(scenario.supported_compile_modes)})"
        )
    if ac_mode not in scenario.supported_ac_modes:
        raise ValueError(
            f"scenario {scenario.name!r} does not support ac mode {ac_mode!r} "
            f"(supported: {', '.join(scenario.supported_ac_modes)})"
        )

    # The fourth global axis, resolved and checked before any host probe.
    # ``engines`` is the launcher set of the arms this run will really start,
    # so the Megatron restrictions follow the arms rather than a scenario
    # name. ``run --arm NAME`` narrows that set on purpose: a run of one
    # titan arm has no megatron opponent to match, and refusing it for the
    # sake of an arm nobody asked for would refuse a legal run.
    parallelism = request.parallelism or TRIVIAL_SPEC
    devices = parse_devices(request.gpu)
    validate_parallelism(
        parallelism,
        shape=shape,
        workload=workload,
        compile_mode=compile_mode,
        engines={arm.launcher for arm in arms},
        device_count=len(devices),
    )
    # Both engines start a second rank now, so the blanket refusal that stood
    # here is gone. What refuses an unimplemented mesh is the seventeen rules
    # above plus the engines themselves: ``parallelize_piper1b`` refuses a
    # tensor or context degree, a dropped shard-degree flag and a mesh that
    # replicates and shards at once, and the Megatron driver refuses a
    # schedule it does not implement. Each failure lands on the module that
    # owns the missing work.

    # The p2p sync treatment, refused parent-side for two reasons that each
    # name their own cause. Without a pipeline there is no message to
    # synchronize, so the field is inert and the manifest would record a
    # treatment the run did not have. Without a megatron arm the value
    # reaches nothing: TorchTitan sends no pipeline message through
    # Megatron. This is the ``--compile-mode none`` exception the other way
    # round -- that mode needs every selected arm to be TorchTitan, this
    # value needs at least one not to be -- and ``run --arm`` narrows the
    # engine set on purpose, so a megatron-only subset passes.
    if megatron_p2p_sync != DEFAULT_MEGATRON_P2P_SYNC:
        if parallelism.pp == 1:
            raise ValueError(
                f"--megatron-p2p-sync {megatron_p2p_sync!r} was requested at "
                "pp 1, where there is no pipeline message to synchronize; "
                "the manifest would record a treatment the run did not have"
            )
        if not any(arm.launcher in MEGATRON_LAUNCHERS for arm in arms):
            raise ValueError(
                f"--megatron-p2p-sync {megatron_p2p_sync!r} reaches no arm "
                f"of this run: {', '.join(arm.name for arm in arms)} run on "
                "TorchTitan, which sends no pipeline message through "
                "Megatron; select a megatron arm, or leave the option at "
                f"{DEFAULT_MEGATRON_P2P_SYNC!r}"
            )

    if scenario.regions:
        # A regioned scenario declares the per-block regions of the model it
        # actually runs. Three runs declare none instead. A shape whose block
        # graph is not structurally identifiable declares none (see
        # piper_block_regions), an uncompiled run declares none because it
        # emits no compiled-graph annotations at all, and a pipelined run
        # declares none because no rank holds every block -- the same honest
        # reason the megatron scenario declares none. Validation rule 7 then
        # guards none of the three, and arm rules 8, 10, 11 and 12 do.
        #
        # The pipeline condition is about the invocation count that IS the
        # region's identity. ``piper_block_regions`` asks for
        # ``n_layers * profiler_active`` invocations per window, and a rank
        # of a pipeline holds ``n_layers / pp`` blocks and runs each of them
        # once per microbatch. So the count a rank really reaches is
        # ``(n_layers / pp) * n_microbatches * profiler_active``, which is a
        # different number and is not the same number on every rank under an
        # interleaved schedule. Deriving a per-rank count instead would be
        # rule 7 rewritten rather than rule 7 applied. Real pipelined traces
        # exist, but no trace analysis has established a unique per-rank
        # invocation identity that could replace this declaration safely.
        regions_apply = (
            shape.supports_block_regions
            and compile_mode not in UNCOMPILED_COMPILE_MODES
            and parallelism.pp == 1
        )
        scenario = replace(
            scenario,
            regions=(
                piper_block_regions(
                    n_layers=shape.n_layers,
                    profiler_active=workload.profiler_active,
                )
                if regions_apply
                else ()
            ),
        )

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
        + command_for_arm(
            scenario.workload,
            arm,
            out_dir / arm.name,
            extra_args,
            compile_mode,
            ac_mode,
            model_size=model_size,
            parallelism=parallelism,
            megatron_p2p_sync=megatron_p2p_sync,
        )
        for arm in arms
    }

    # A host precondition, read off the argv this run will really start
    # rather than re-derived from the spec. Megatron refuses to parse its
    # own arguments under --use-megatron-fsdp when the shell sets
    # CUDA_DEVICE_MAX_CONNECTIONS=1, and the child inherits this shell.
    # Checked here, before the output directory exists, so the refusal
    # costs nothing and names its own repair.
    refuse_megatron_fsdp_connection_limit(
        environment,
        megatron_fsdp=any(
            "--use-megatron-fsdp" in command for command in commands.values()
        ),
    )

    if existing_manifest is not None:
        mismatches = _resume_mismatches(
            existing_manifest,
            scenario,
            arms,
            hardware,
            metadata,
            extra_args,
            compile_mode,
            ac_mode,
            model_size,
            parallelism=parallelism,
            megatron_p2p_sync=megatron_p2p_sync,
        )
        if mismatches:
            raise ValueError(
                "resume request does not match the existing manifest: "
                + ", ".join(mismatches)
            )
    return (
        paths,
        scenario,
        arms,
        hardware,
        metadata,
        out_dir,
        commands,
        compile_mode,
        ac_mode,
        model_size,
        parallelism,
        resumed,
        megatron_p2p_sync,
    )


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
        compile_mode,
        ac_mode,
        model_size,
        parallelism,
        resumed,
        megatron_p2p_sync,
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
            compile_mode,
            ac_mode,
            model_size,
            parallelism=parallelism,
            megatron_p2p_sync=megatron_p2p_sync,
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
    _emit(event_handler, "summary", f"compile mode: {compile_mode}")
    _emit(event_handler, "summary", f"ac mode: {ac_mode}")
    _emit(event_handler, "summary", f"model size: {model_size}")
    _emit(
        event_handler,
        "summary",
        f"parallelism: dp {parallelism.dp} x pp {parallelism.pp} "
        f"(ep {parallelism.ep}, world size {parallelism.world_size}, "
        f"dense sharding {parallelism.dense_sharding})",
    )
    _emit(event_handler, "summary", f"megatron p2p sync: {megatron_p2p_sync}")
    _emit(event_handler, "summary", f"output: {out_dir}")

    base_environment = runtime_environment(
        paths,
        request.gpu,
        environment=host_environment,
        world_size=parallelism.world_size,
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
                    compile_mode=compile_mode,
                    ac_mode=ac_mode,
                    model_size=model_size,
                    parallelism=parallelism,
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
                compile_mode=compile_mode,
                ac_mode=ac_mode,
                model_size=model_size,
                parallelism=parallelism,
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
