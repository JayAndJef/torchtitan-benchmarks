"""Orchestrate kernel-isolation benchmark scenarios from the torch-free CLI.

Mirrors ``e2e/runner.py``: the parent resolves provenance and pinning, writes
the manifest, spawns pinned GPU workers so CUDA device selection, NUMA
binding and the TE build's compiler environment all apply to the measuring
process, and assembles the results. Scenarios are independent short runs, so
unlike the e2e sweep the parent continues past a failing scenario and reports
all outcomes.

**One process per pass, not one per scenario.** A scenario is a correctness
worker followed by ``replicates x arms`` timing workers, each building a
single arm. That is what keeps one arm's dependencies out of another arm's
interpreter -- FA3 and TransformerEngine cannot share a process at all (a
cuDNN soname collision, see CLAUDE.md) -- and it is why the parent, not a
worker, writes ``results.json``: only the parent sees every fragment.

**The sweep is replicate-major.** One replicate spawns every arm once, in
declaration order, and the replicate is repeated. Drift is therefore shared
across arms rather than charged to whichever arm happened to be running, and
the anchor sits in every replicate slot.

**Workers run strictly sequentially.** A shared GPU invalidates timings
(CLAUDE.md operating rules), so the parent never has two in flight.

A crashed timing worker does not abort the sweep: it is reported the moment
it happens, and the merge decides what the scenario can still say. Losing a
non-anchor arm costs that arm; losing the anchor fails the scenario, because
every comparison is a ratio against it.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any, Mapping, Sequence

from benchmarks.artifacts.layout import atomic_write_json, run_timestamp
from benchmarks.execution.affinity import resolve_cpu_pinning
from benchmarks.execution.environment import (
    add_compiler_environment,
    runtime_environment,
)
from benchmarks.execution.events import EventHandler, _emit
from benchmarks.execution.paths import BENCH_DIR, RuntimePaths
from benchmarks.execution.provenance import hardware_metadata
from benchmarks.kernel.registry import kernel_scenario_by_name
from benchmarks.kernel.results.merge import merge_kernel_fragments
from benchmarks.kernel.results.schema import (
    KernelScenarioResult,
    write_kernel_results,
)
from benchmarks.kernel.schema import (
    KernelScenario,
    KernelWorkload,
    resolve_shape_and_workload,
    routing_divides_evenly,
    shape_summary,
)
from benchmarks.models.piper_qwen3.shape import PiperShape


# 2: the single flat shape record was replaced by model_size + model_shape
# (the same describe() the e2e manifest records) plus the workload.
#
# 3: n/warmup counted round-robin cycles and were replaced by the
# burst-timing parameters, matching the results schema. This file is
# write-only provenance -- no loader reads it -- so the bump is labelling.
#
# 4: a scenario is no longer one worker invocation, so the single "command"
# became "commands", one argv per pass. Renamed rather than redefined: a
# reader of the old field would take the correctness worker's argv for the
# whole run.
KERNEL_MANIFEST_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class KernelRunRequest:
    gpu: str
    scenario_names: tuple[str, ...]
    replicates: int = 5
    samples_per_replicate: int = 40
    burst_k: int = 16
    warmup_calls: int = 30
    burst: bool = False
    model_size: str = "normal"
    batch: int | None = None
    seq_len: int | None = None
    max_seq_len: int | None = None
    seed: int = 0
    hardware: str = "auto"
    out_dir: Path | None = None
    timestamp: str | None = None
    cache_root: Path | None = None
    compiler_env: Path | None = None


@dataclass(frozen=True)
class KernelScenarioOutcome:
    scenario: str
    out_dir: Path
    result: KernelScenarioResult | None
    correctness_failed: bool = False
    error: str | None = None
    # (arm, replicate) pairs whose worker did not produce a fragment. The
    # scenario still reports whatever the survivors measured, but it exits
    # nonzero: a partial roster published as a whole one is the failure this
    # harness exists to prevent.
    failed_passes: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        # A ``failed`` arm in the results counts too, and it is not covered by
        # ``failed_passes``: an arm whose workers all wrote a fragment and
        # measured nothing in it never lost a pass. The scenario still
        # published a short roster, so it still exits nonzero.
        arms = self.result.arms.values() if self.result is not None else ()
        return (
            self.correctness_failed
            or self.error is not None
            or bool(self.failed_passes)
            or any(arm.status == "failed" for arm in arms)
        )


def worker_command(
    scenario_name: str,
    fragment: Path,
    request: KernelRunRequest,
    prefix: Sequence[str],
    *,
    mode: str,
    arm: str | None = None,
    replicate: int | None = None,
    skip_arms: Sequence[str] = (),
) -> list[str]:
    """The argv for one worker pass. Deterministic, so the manifest can list
    every command the run will issue before the first one starts."""
    command = list(prefix) + [
        sys.executable,
        "-m",
        "benchmarks.kernel.worker",
        "--scenario",
        scenario_name,
        "--mode",
        mode,
        "--fragment",
        str(fragment),
        "--replicates",
        str(request.replicates),
        "--samples-per-replicate",
        str(request.samples_per_replicate),
        "--burst-k",
        str(request.burst_k),
        "--warmup-calls",
        str(request.warmup_calls),
        "--seed",
        str(request.seed),
        # Unconditional, unlike the overrides below: the run always has a
        # model size, and the worker must not fall back to its own default.
        "--model-size",
        request.model_size,
    ]
    if arm is not None:
        command.extend(("--arm", arm))
    if replicate is not None:
        command.extend(("--replicate", str(replicate)))
    for name in skip_arms:
        command.extend(("--skip-arm", name))
    if request.burst:
        command.append("--burst")
    if request.batch is not None:
        command.extend(("--batch", str(request.batch)))
    if request.seq_len is not None:
        command.extend(("--seq-len", str(request.seq_len)))
    if request.max_seq_len is not None:
        command.extend(("--max-seq-len", str(request.max_seq_len)))
    return command


def fragment_path(fragments_dir: Path, arm: str, replicate: int) -> Path:
    return fragments_dir / f"timing__{arm}__r{replicate}.json"


def resolve_arm_skips(
    scenario: KernelScenario, *, compiler_unavailable: str | None
) -> dict[str, str]:
    """Which arms this host cannot run, and why, keyed by arm name.

    **Requirements belong to the arm, not to the scenario.** Without a C++20
    host compiler, rope loses ``te`` and still measures ``baseline``,
    ``helion`` and ``copy_floor``. The scenario-level ``requires_gcc_toolset``
    is an OR across arms, so using it to decide cost all four; it keeps its
    one honest use, which is asking whether anything here needs the compiler
    at all.

    **The set is closed over correctness references.** An arm whose reference
    is skipped is skipped too. The alternative is to time an arm that nothing
    checked, which is the silent wrongness the gates exist to prevent. No
    scenario produces the case today -- ``te`` is a referrer, never a
    reference -- so the closure is a guard against the roster growing into
    it.
    """
    skipped: dict[str, str] = {}
    if compiler_unavailable is not None:
        for arm in scenario.arms:
            if arm.requires_gcc_toolset:
                skipped[arm.name] = compiler_unavailable
    while True:
        grew = False
        for arm in scenario.arms:
            if arm.name in skipped:
                continue
            for check in arm.correctness:
                if check.reference in skipped:
                    skipped[arm.name] = (
                        f"its correctness reference {check.reference!r} is "
                        f"skipped: {skipped[check.reference]}"
                    )
                    grew = True
                    break
        if not grew:
            return skipped


def planned_commands(
    scenario: KernelScenario,
    request: KernelRunRequest,
    fragments_dir: Path,
    prefix: Sequence[str],
    skipped: Mapping[str, str] = MappingProxyType({}),
) -> list[list[str]]:
    """Every worker argv this scenario will issue, in the order it issues it.

    Replicate-major, matching the sweep: one replicate spawns every arm once
    in declaration order, and the replicate repeats. A skipped arm is spawned
    in neither pass, and the manifest therefore lists what the run really
    does rather than what a fully-equipped host would have done.
    """
    commands = [
        worker_command(
            scenario.name,
            fragments_dir / "correctness.json",
            request,
            prefix,
            mode="correctness",
            skip_arms=[arm.name for arm in scenario.arms if arm.name in skipped],
        )
    ]
    for replicate in range(request.replicates):
        for arm in scenario.arms:
            if arm.name in skipped:
                continue
            commands.append(
                worker_command(
                    scenario.name,
                    fragment_path(fragments_dir, arm.name, replicate),
                    request,
                    prefix,
                    mode="timing",
                    arm=arm.name,
                    replicate=replicate,
                )
            )
    return commands


def kernel_manifest_data(
    scenario: KernelScenario,
    shape: PiperShape,
    workload: KernelWorkload,
    request: KernelRunRequest,
    commands: list[list[str]],
    hardware: str,
    metadata: dict[str, str],
) -> dict:
    return {
        "schema_version": KERNEL_MANIFEST_SCHEMA_VERSION,
        "kind": "kernel",
        "scenario": scenario.name,
        "description": scenario.description,
        "model_size": request.model_size,
        # The same record the e2e manifest writes, so both systems state
        # model identity identically.
        "model_shape": shape.describe(seq_len=workload.seq_len),
        "workload": asdict(workload),
        "shapes": shape_summary(scenario.name, shape, workload),
        "arms": [asdict(arm) for arm in scenario.arms],
        "baseline_arm": scenario.baseline_arm,
        "replicates": request.replicates,
        "samples_per_replicate": request.samples_per_replicate,
        "burst_k": request.burst_k,
        "warmup_calls": request.warmup_calls,
        "burst": request.burst,
        "seed": request.seed,
        "commands": commands,
        "hardware": hardware,
        "hardware_metadata": metadata,
        "created_at": dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ"),
    }


def _log_tail(log_path: Path, lines: int = 12) -> str:
    try:
        content = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(content[-lines:])


def _read_fragment(path: Path) -> dict[str, Any] | None:
    """A worker's fragment, or None when it never wrote one."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def execute_kernel_run(
    request: KernelRunRequest,
    *,
    event_handler: EventHandler | None = None,
    process_runner=subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> tuple[KernelScenarioOutcome, ...]:
    host_environment = dict(environment or os.environ)
    paths = RuntimePaths.resolve(
        cache_root=request.cache_root,
        compiler_env=request.compiler_env,
        environment=host_environment,
    )
    shape, workload = resolve_shape_and_workload(
        model_size=request.model_size,
        batch=request.batch,
        seq_len=request.seq_len,
        max_seq_len=request.max_seq_len,
    )
    hardware, metadata = hardware_metadata(paths, request.gpu, request.hardware)
    pinning = resolve_cpu_pinning(request.gpu)
    metadata = {**metadata, "cpu_pinning": pinning.description}
    timestamp = request.timestamp or run_timestamp()
    base_environment = runtime_environment(
        paths, request.gpu, environment=host_environment
    )

    _emit(event_handler, "summary", f"GPU (PCI index): {request.gpu}")
    _emit(event_handler, "summary", metadata["nvidia_smi"])
    _emit(event_handler, "summary", f"cpu pinning: {pinning.description}")

    # Resolved once per run, not once per scenario and certainly not once per
    # worker: add_compiler_environment shells out to bash, and the answer
    # cannot change between two scenarios of the same run.
    compiler_environment = base_environment
    compiler_unavailable: str | None = None
    if any(
        kernel_scenario_by_name(name).requires_gcc_toolset
        for name in request.scenario_names
    ):
        if paths.compiler_env is None:
            compiler_unavailable = (
                "needs a C++20 host compiler for the TE build; set "
                "--compiler-env/BENCH_COMPILER_ENV"
            )
        else:
            try:
                compiler_environment = add_compiler_environment(
                    base_environment, paths.compiler_env
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                compiler_unavailable = (
                    f"cannot prepare the compiler environment: {error}"
                )
        if compiler_unavailable is not None:
            _emit(
                event_handler,
                "summary",
                f"compiler environment unavailable: {compiler_unavailable}",
            )

    outcomes = []
    for name in request.scenario_names:
        scenario = kernel_scenario_by_name(name)
        out_dir = request.out_dir or (
            BENCH_DIR / "out" / timestamp / "kernels" / name / hardware
        )
        out_dir = out_dir.expanduser().resolve()
        _emit(event_handler, "arm", f"=== kernel scenario: {name} ===")

        skipped = resolve_arm_skips(
            scenario, compiler_unavailable=compiler_unavailable
        )
        if scenario.baseline_arm in skipped:
            # The anchor carries every ratio, so losing it is the one skip
            # that costs the scenario rather than an arm.
            outcome = KernelScenarioOutcome(
                scenario=name,
                out_dir=out_dir,
                result=None,
                error=(
                    f"{name}: the anchor arm {scenario.baseline_arm!r} "
                    f"{skipped[scenario.baseline_arm]}"
                ),
            )
            _emit(event_handler, "error", f"ERROR {outcome.error}")
            outcomes.append(outcome)
            continue
        for arm_name, reason in skipped.items():
            _emit(event_handler, "summary", f"skipping {name}/{arm_name}: {reason}")

        if scenario.requires_balanced_routing and not routing_divides_evenly(
            shape, workload
        ):
            # Loudly skipped rather than quietly rounded: an unbalanced split
            # would silently measure a different workload per expert.
            rows = workload.batch * workload.seq_len * shape.top_k
            outcome = KernelScenarioOutcome(
                scenario=name,
                out_dir=out_dir,
                result=None,
                error=(
                    f"{name}: {rows} routed rows (batch {workload.batch} x "
                    f"seq {workload.seq_len} x top_k {shape.top_k}) do not "
                    f"divide evenly among {shape.num_experts} experts"
                ),
            )
            _emit(event_handler, "error", f"ERROR {outcome.error}")
            outcomes.append(outcome)
            continue

        out_dir.mkdir(parents=True, exist_ok=False)
        fragments_dir = out_dir / "fragments"
        fragments_dir.mkdir()
        commands = planned_commands(
            scenario, request, fragments_dir, pinning.prefix, skipped
        )

        atomic_write_json(
            out_dir / "manifest.json",
            kernel_manifest_data(
                scenario, shape, workload, request, commands, hardware, metadata
            ),
        )
        scenario_environment = (
            compiler_environment
            if scenario.requires_gcc_toolset
            else base_environment
        )

        log_path = out_dir / "kernel_bench.log"
        # One log for the whole scenario: every worker of every pass appends
        # to it, so a crashed arm's traceback sits in sequence with the passes
        # around it rather than in a file per process.
        with log_path.open("w") as log:

            def spawn(command: list[str], log: IO[str] = log) -> int:
                _emit(event_handler, "command", " ".join(command))
                completed = process_runner(
                    command,
                    cwd=paths.bench_dir,
                    env=scenario_environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                # The log stays open for the whole scenario, so a tail read
                # after a crash would otherwise see an empty buffer.
                log.flush()
                return completed.returncode

            # Gates first. A failed gate means no timing is worth taking, so
            # the timing workers never launch -- unlike the single-process
            # predecessor, which measured an arm it already knew was wrong.
            correctness_code = spawn(commands[0])
            correctness = _read_fragment(fragments_dir / "correctness.json")
            if correctness is None:
                outcome = KernelScenarioOutcome(
                    scenario=name,
                    out_dir=out_dir,
                    result=None,
                    error=(
                        f"{name}: the correctness worker exited with "
                        f"{correctness_code} and wrote no fragment; log "
                        f"tail:\n{_log_tail(log_path)}"
                    ),
                )
                _emit(event_handler, "error", f"ERROR {outcome.error}")
                outcomes.append(outcome)
                continue

            # The exit code and the fragment are two independent statements of
            # one verdict, and either one failing is a failure. The worker
            # writes the fragment *before* it computes the code, so a process
            # that dies in that window -- a signal, an OSError on the report,
            # an OOM kill -- leaves "all_passed": false beside a code that is
            # not 3. Reading the code alone published a failed gate as a pass,
            # with a full set of timings under it.
            gates_failed = correctness_code == 3 or not correctness.get(
                "all_passed"
            )
            timings: list[dict[str, Any]] = []
            failed_passes: list[str] = []
            if gates_failed:
                _emit(
                    event_handler,
                    "error",
                    f"ERROR {name}: correctness gates failed; no arm is "
                    "timed. See the report below",
                )
            else:
                for command in commands[1:]:
                    arm = command[command.index("--arm") + 1]
                    replicate = int(command[command.index("--replicate") + 1])
                    code = spawn(command)
                    fragment = _read_fragment(
                        fragment_path(fragments_dir, arm, replicate)
                    )
                    if fragment is None:
                        label = f"{arm} r{replicate}"
                        failed_passes.append(label)
                        # Reported as it happens rather than at the end: the
                        # sweep continues, and an operator watching a long run
                        # should not learn about the first crash last.
                        _emit(
                            event_handler,
                            "error",
                            f"ERROR {name}: timing worker {label} exited "
                            f"with {code} and wrote no fragment",
                        )
                        continue
                    if code != 0:
                        # The samples stand -- the worker writes the fragment
                        # before it returns -- so this costs the arm nothing.
                        # It is still said out loud: the code reports a death
                        # after the write, and a discarded exit code is how a
                        # silent failure starts.
                        _emit(
                            event_handler,
                            "error",
                            f"WARNING {name}: timing worker {arm} r{replicate} "
                            f"wrote its fragment and then exited with {code}",
                        )
                    timings.append(fragment)

        result = None
        error = None
        try:
            result = merge_kernel_fragments(
                scenario=scenario,
                shape=shape,
                workload=workload,
                hardware=hardware,
                replicates=request.replicates,
                samples_per_replicate=request.samples_per_replicate,
                burst_k=request.burst_k,
                warmup_calls=request.warmup_calls,
                seed=request.seed,
                correctness=correctness,
                timings=timings,
                timings_ran=not gates_failed,
                skipped=skipped,
            )
        except (ValueError, KeyError) as merge_error:
            error = f"{name}: {merge_error}"
        else:
            write_kernel_results(result, out_dir / "results.json")

        outcome = KernelScenarioOutcome(
            scenario=name,
            out_dir=out_dir,
            result=result,
            correctness_failed=gates_failed,
            error=error,
            failed_passes=tuple(failed_passes),
        )
        # Report a failure the moment it happens. Scenarios continue past one
        # another, so holding this until the end would show a first-scenario
        # failure only after every later scenario had run.
        if outcome.error:
            _emit(event_handler, "error", f"ERROR {outcome.error}")
        outcomes.append(outcome)
    return tuple(outcomes)
