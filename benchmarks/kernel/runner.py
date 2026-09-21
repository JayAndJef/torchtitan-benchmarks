"""Orchestrate kernel-isolation measurements from the torch-free CLI.

A run measures two kinds of unit. A scenario cuts the model at one boundary
and ranks the implementations there. A span fuses across a cut, so its claim
is the span against the sum of the scenarios it replaces.
``measurement_plan`` puts every enclosed scenario ahead of its span, and each
scenario once, so both sides of the claim share one request's hardware,
shape, workload, seed, ``burst_k``, replicate count and NUMA pinning.

The parent resolves provenance and pinning, writes the manifest, spawns the
pinned GPU workers and assembles the results. Each timing worker builds one
arm, which keeps one arm's dependencies out of another arm's interpreter,
and only the parent sees every fragment, so only the parent writes
``results.json``. Scenarios are independent short runs, so the parent
continues past a failing one and reports every outcome.

The parent runs the workers strictly one at a time, because a shared GPU
invalidates a timing. A crashed timing worker costs its own arm. A crashed
anchor fails the scenario, because every comparison is a ratio against the
anchor.
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
from benchmarks.kernel.results.merge import (
    MeasuredScenario,
    merge_kernel_fragments,
    merge_kernel_span_fragments,
)
from benchmarks.kernel.results.schema import (
    KernelScenarioResult,
    KernelSpanResult,
    write_kernel_results,
)
from benchmarks.kernel.schema import (
    DEFAULT_MODEL_SIZE,
    KernelScenario,
    KernelSpan,
    KernelWorkload,
    resolve_shape_and_workload,
    resolve_symbol,
    routing_divides_evenly,
    shape_summary,
    timing_fragment_path,
)
from benchmarks.kernel.spans import kernel_span_by_name
from benchmarks.models.piper_qwen3.shape import (
    canonical_size_name,
    PiperShape,
)


KERNEL_MANIFEST_SCHEMA_VERSION = 7
"""The schema version of the manifest this runner writes.

No loader reads the file, so each bump labels a change rather than
reinterprets one. Schema 7 tells the two unit kinds apart: "kind" holds
``kernel_scenario`` or ``kernel_span``, and the file gains "unit_kind",
"span_scenarios" and "parts".
"""


@dataclass(frozen=True)
class KernelRunRequest:
    """One kernel-bench invocation.

    Attributes:
        arm_names: The operator's ``--arm`` choice, empty for every arm. It
            names arms of one scenario, and what it leaves out is skipped
            for a reason that says so. ``resolve_arm_skips`` refuses a
            selection it cannot honour.
        span_names: The spans this run measures, and never a default. One
            span adds every scenario it replaces to the run, so a bare
            ``kernel-bench <gpu>`` asks for no span.
        replicates_per_process: How many consecutive replicates of one arm
            share a worker process. It is a methodology choice, not a
            workload one. At 1 each replicate is a fresh process, and every
            arm runs within seconds of every other, so drift that moves a
            whole replicate cancels in the per-replicate log-ratio. Above 1
            the replicates stop sampling process-to-process variation and
            the arms move apart in time, so the interval narrows and drift
            lands in the point estimate. The results file renames every
            degraded statistic. Use 1 for anything published. The one
            measurement of the trade gave an interval 26-48% narrower with
            no better point estimate, on a box at load average 28 to 81, so
            that figure records what was tried and may not be cited.
    """

    gpu: str
    scenario_names: tuple[str, ...]
    arm_names: tuple[str, ...] = ()
    span_names: tuple[str, ...] = ()
    replicates: int = 5
    replicates_per_process: int = 1
    samples_per_replicate: int = 40
    burst_k: int = 16
    warmup_calls: int = 30
    burst: bool = False
    model_size: str = DEFAULT_MODEL_SIZE
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
    """What one measurement unit produced.

    Attributes:
        scenario: The unit's name, whichever kind of unit it names.
        unit_kind: Which kind of unit that is.
        failed_passes: The (arm, replicate) pairs whose worker wrote no
            fragment. The scenario still reports what the survivors
            measured, and it still exits nonzero.
    """

    scenario: str
    out_dir: Path
    result: KernelScenarioResult | KernelSpanResult | None
    correctness_failed: bool = False
    error: str | None = None
    unit_kind: str = "scenario"
    failed_passes: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        # A ``failed`` arm lost no pass, and it still shortens the roster.
        arms = self.result.arms.values() if self.result is not None else ()
        return (
            self.correctness_failed
            or self.error is not None
            or bool(self.failed_passes)
            or any(arm.status == "failed" for arm in arms)
        )


@dataclass(frozen=True)
class MeasurementUnit:
    """One thing this run measures: a scenario, or a span.

    Attributes:
        measurement: What the workers build and time. It is a
            ``KernelScenario`` either way, so ``benchmarks.kernel.engine``
            never learns that spans exist.
        span: Set for a span alone. It holds the ordered range and what
            each arm replaces, which the merge needs.
    """

    measurement: KernelScenario
    span: KernelSpan | None = None

    @property
    def name(self) -> str:
        return self.measurement.name

    @property
    def kind(self) -> str:
        return "scenario" if self.span is None else "span"


def measurement_plan(request: KernelRunRequest) -> tuple[MeasurementUnit, ...]:
    """Every unit this run measures, in the order it measures them.

    Scenarios come first, because the merge takes a span's second total
    from the results of this same run. Each scenario appears once, so no cut
    is measured twice and no span sums an ambiguous number.
    """
    ordered: list[str] = list(request.scenario_names)
    for span_name in request.span_names:
        for scenario_name in kernel_span_by_name(span_name).scenarios:
            if scenario_name not in ordered:
                ordered.append(scenario_name)
    return tuple(
        [
            MeasurementUnit(measurement=kernel_scenario_by_name(name))
            for name in ordered
        ]
        + [
            MeasurementUnit(
                measurement=(span := kernel_span_by_name(name)).measurement,
                span=span,
            )
            for name in request.span_names
        ]
    )


def worker_command(
    unit: MeasurementUnit,
    fragments_dir: Path,
    request: KernelRunRequest,
    prefix: Sequence[str],
    *,
    mode: str,
    arm: str | None = None,
    replicate: int | None = None,
    replicate_count: int = 1,
    skip_arms: Sequence[str] = (),
) -> list[str]:
    """The argv for one worker pass.

    It is deterministic, so the manifest lists every command before the
    first one starts. A timing worker takes the fragments directory, because
    a batched one writes a file per replicate. A span is named with
    ``--span`` and never with ``--scenario``, so the recorded argv says
    which roster the name belongs to.
    """
    command = list(prefix) + [
        sys.executable,
        "-m",
        "benchmarks.kernel.worker",
        "--span" if unit.span is not None else "--scenario",
        unit.name,
        "--mode",
        mode,
    ]
    command += (
        ["--fragments-dir", str(fragments_dir)]
        if mode == "timing"
        else ["--fragment", str(fragments_dir / "correctness.json")]
    )
    command += [
        # Provenance alone. No worker pass reads it; the parent owns the order.
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
        # Always sent and canonical: no worker default, no retired alias.
        "--model-size",
        canonical_size_name(request.model_size),
    ]
    if arm is not None:
        command.extend(("--arm", arm))
    if replicate is not None:
        command.extend(("--replicate", str(replicate)))
    if replicate_count != 1:
        command.extend(("--replicate-count", str(replicate_count)))
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


def replicate_blocks(
    replicates: int, replicates_per_process: int
) -> tuple[tuple[int, int], ...]:
    """The (first replicate, count) pairs one arm's timing workers cover.

    The blocks tile ``range(replicates)`` in order, and only the last one is
    short. One replicate per process gives one block per replicate.
    """
    size = max(1, replicates_per_process)
    return tuple(
        (start, min(size, replicates - start))
        for start in range(0, replicates, size)
    )


def _selection_skips(
    scenario: KernelScenario, selected: Sequence[str]
) -> dict[str, str]:
    """The arms ``--arm`` leaves out, keyed by name, with the reason.

    The reason names the flag, so a reader tells the operator's choice from
    the host's capability. A selection this scenario cannot honour is
    refused, never repaired: an unknown name, a missing anchor arm and a
    missing correctness reference each raise, and each names what is
    missing.
    """
    roster = [arm.name for arm in scenario.arms]
    unknown = sorted(set(selected) - set(roster))
    if unknown:
        raise ValueError(
            f"{scenario.name}: --arm names no such arm: "
            f"{', '.join(unknown)}. Available arms: {', '.join(roster)}"
        )
    chosen = set(selected)
    if scenario.baseline_arm not in chosen:
        raise ValueError(
            f"{scenario.name}: --arm must include the anchor arm "
            f"{scenario.baseline_arm!r}; every comparison is a ratio against "
            f"it, so a selection without it publishes none"
        )
    for arm in scenario.arms:
        if arm.name not in chosen:
            continue
        for check in arm.correctness:
            # "fp64" is the scenario's own builder, so it is always present.
            if check.reference == "fp64" or check.reference in chosen:
                continue
            raise ValueError(
                f"{scenario.name}: --arm selected {arm.name!r} but not its "
                f"correctness reference {check.reference!r}; a gate needs "
                f"both sides at once, so add --arm {check.reference}"
            )
    measured = ", ".join(name for name in roster if name in chosen)
    return {
        name: f"--arm did not select it; this run measures {measured}"
        for name in roster
        if name not in chosen
    }


def resolve_arm_skips(
    scenario: KernelScenario | KernelSpan,
    *,
    compiler_unavailable: str | None,
    shape: PiperShape,
    workload: KernelWorkload,
    selected: Sequence[str] = (),
) -> dict[str, str]:
    """Which arms this run does not measure, and why, keyed by arm name.

    A requirement belongs to an arm, not to a scenario, so a missing C++20
    host compiler costs ``titan/te`` alone. ``requires_gcc_toolset`` is a
    property of the host. ``KernelArm.requirement`` is the other kind: a
    parent-side predicate the shape and the workload answer. Both run in the
    parent, before a GPU is claimed, so the arm is never built and nothing
    has to tell a requirement from a bug.

    The set is closed over correctness references, because an arm nothing
    checked must not be timed. ``selected`` is the operator's ``--arm``
    choice, and it resolves first, so an arm nobody asked for keeps that
    reason rather than a capability reason it never had to meet.
    """
    skipped: dict[str, str] = (
        _selection_skips(scenario, selected) if selected else {}
    )
    for arm in scenario.arms:
        if arm.name in skipped:
            continue
        if compiler_unavailable is not None and arm.requires_gcc_toolset:
            skipped[arm.name] = compiler_unavailable
            continue
        if arm.requirement is None:
            continue
        reason = resolve_symbol(arm.requirement)(shape, workload)
        if reason:
            skipped[arm.name] = reason
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


def timing_passes(
    scenario: KernelScenario | KernelSpan,
    request: KernelRunRequest,
    skipped: Mapping[str, str] = MappingProxyType({}),
) -> tuple[tuple[str, int, int], ...]:
    """The ``(arm, first replicate, count)`` of every timing pass, in order.

    The order is block-major: the outer loop walks the replicate blocks and
    the inner loop walks the arms, so every arm runs once before any arm
    runs again. At one replicate per process a block is one replicate, and
    drift that moves a whole replicate cancels in the ratio. A skipped arm
    appears in no pass.

    Two callers need this order, so one function states it. The manifest
    writer turns each triple into an argv, and the spawn loop needs the same
    triple back. Neither side may recover it by string surgery on an argv.
    """
    return tuple(
        (arm.name, first, count)
        for first, count in replicate_blocks(
            request.replicates, request.replicates_per_process
        )
        for arm in scenario.arms
        if arm.name not in skipped
    )


def planned_commands(
    unit: MeasurementUnit,
    request: KernelRunRequest,
    fragments_dir: Path,
    prefix: Sequence[str],
    skipped: Mapping[str, str] = MappingProxyType({}),
) -> list[list[str]]:
    """Every worker argv this unit will issue, in the order it issues it.

    The correctness pass comes first, then one argv per entry of
    ``timing_passes``. A span's passes are a scenario's passes.
    """
    scenario = unit.measurement
    return [
        worker_command(
            unit,
            fragments_dir,
            request,
            prefix,
            mode="correctness",
            skip_arms=[arm.name for arm in scenario.arms if arm.name in skipped],
        ),
        *(
            worker_command(
                unit,
                fragments_dir,
                request,
                prefix,
                mode="timing",
                arm=arm,
                replicate=first,
                replicate_count=count,
            )
            for arm, first, count in timing_passes(scenario, request, skipped)
        ),
    ]


def kernel_manifest_data(
    unit: MeasurementUnit,
    shape: PiperShape,
    workload: KernelWorkload,
    request: KernelRunRequest,
    commands: list[list[str]],
    hardware: str,
    metadata: dict[str, str],
    skipped: Mapping[str, str] = MappingProxyType({}),
) -> dict:
    scenario = unit.measurement
    span = unit.span
    return {
        "schema_version": KERNEL_MANIFEST_SCHEMA_VERSION,
        # Renamed at schema 7, because "kernel" named the family and a member.
        "kind": f"kernel_{unit.kind}",
        "unit_kind": unit.kind,
        # A span name is not a scenario name, so each kind has its own field.
        "scenario": None if span else scenario.name,
        "span": span.name if span else None,
        "description": scenario.description,
        # The range a span replaces. The prefix keeps it apart from "scenario".
        "span_scenarios": list(span.scenarios) if span else None,
        "parts": (
            {entry.arm: list(entry.parts) for entry in span.parts}
            if span
            else None
        ),
        # Canonical by construction; an alias would name a size nobody reads.
        "model_size": shape.name,
        # The same record the e2e manifest writes, so both state one identity.
        "model_shape": shape.describe(seq_len=workload.seq_len),
        "workload": asdict(workload),
        # No single entry describes a span, so a span records the whole range.
        "shapes": (
            {
                name: shape_summary(name, shape, workload)
                for name in span.scenarios
            }
            if span
            else shape_summary(scenario.name, shape, workload)
        ),
        "arms": [asdict(arm) for arm in scenario.arms],
        # "arms" holds the registry's roster, so a dropped arm needs a reason.
        "skipped_arms": dict(skipped),
        "baseline_arm": scenario.baseline_arm,
        "replicates": request.replicates,
        # At 1 the per-replicate ratios cancel drift; above 1 they do so less.
        "replicates_per_process": request.replicates_per_process,
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

    # Resolved once per run: add_compiler_environment shells out to bash.
    plan = measurement_plan(request)
    compiler_environment = base_environment
    compiler_unavailable: str | None = None
    if any(unit.measurement.requires_gcc_toolset for unit in plan):
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
    # The scenarios this run measured, which the span merges below sum.
    measured: dict[str, MeasuredScenario] = {}
    for unit in plan:
        scenario = unit.measurement
        name = unit.name
        # A span sits one directory deeper, so a shallow glob cannot pool it
        # with the scenarios. A recursive walk must read the "kind" value.
        root = BENCH_DIR / "out" / timestamp / "kernels"
        if unit.span is not None:
            root = root / "spans"
        out_dir = request.out_dir or (root / name / hardware)
        out_dir = out_dir.expanduser().resolve()
        _emit(event_handler, "arm", f"=== kernel {unit.kind}: {name} ===")

        skipped = resolve_arm_skips(
            scenario,
            compiler_unavailable=compiler_unavailable,
            shape=shape,
            workload=workload,
            selected=request.arm_names,
        )
        if scenario.baseline_arm in skipped:
            # The anchor carries every ratio, so its loss costs the scenario.
            outcome = KernelScenarioOutcome(
                scenario=name,
                out_dir=out_dir,
                result=None,
                unit_kind=unit.kind,
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
            # Skipped loudly: an uneven split changes the workload per expert.
            rows = workload.batch * workload.seq_len * shape.top_k
            outcome = KernelScenarioOutcome(
                scenario=name,
                out_dir=out_dir,
                result=None,
                unit_kind=unit.kind,
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
            unit, request, fragments_dir, pinning.prefix, skipped
        )

        atomic_write_json(
            out_dir / "manifest.json",
            kernel_manifest_data(
                unit,
                shape,
                workload,
                request,
                commands,
                hardware,
                metadata,
                skipped,
            ),
        )
        scenario_environment = (
            compiler_environment
            if scenario.requires_gcc_toolset
            else base_environment
        )

        log_path = out_dir / "kernel_bench.log"
        # One log for the whole scenario, so a traceback sits in pass order.
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
                # The log stays open, so a tail read needs this flush.
                log.flush()
                return completed.returncode

            # Gates first: a failed gate means no timing is worth taking.
            correctness_code = spawn(commands[0])
            correctness = _read_fragment(fragments_dir / "correctness.json")
            if correctness is None:
                outcome = KernelScenarioOutcome(
                    scenario=name,
                    out_dir=out_dir,
                    result=None,
                    unit_kind=unit.kind,
                    error=(
                        f"{name}: the correctness worker exited with "
                        f"{correctness_code} and wrote no fragment; log "
                        f"tail:\n{_log_tail(log_path)}"
                    ),
                )
                _emit(event_handler, "error", f"ERROR {outcome.error}")
                outcomes.append(outcome)
                continue

            # The code and the fragment state one verdict twice. Either one
            # failing is a failure; the code alone once passed a failed gate.
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
                # In lockstep with commands[1:]. strict=True, so a plan and a
                # run of different lengths raise instead of dropping a tail.
                passes = timing_passes(scenario, request, skipped)
                for (arm, first, count), command in zip(
                    passes, commands[1:], strict=True
                ):
                    code = spawn(command)
                    # One fragment per replicate, reported one by one: the
                    # merge counts replicates rather than workers.
                    missing = False
                    for replicate in range(first, first + count):
                        fragment = _read_fragment(
                            timing_fragment_path(fragments_dir, arm, replicate)
                        )
                        if fragment is None:
                            missing = True
                            label = f"{arm} r{replicate}"
                            failed_passes.append(label)
                            # Reported now, because the run continues.
                            _emit(
                                event_handler,
                                "error",
                                f"ERROR {name}: timing worker {label} exited "
                                f"with {code} and wrote no fragment",
                            )
                            continue
                        timings.append(fragment)
                    if missing:
                        continue
                    if code != 0:
                        # The samples stand, because the worker writes first.
                        # A discarded exit code starts a silent failure.
                        block = (
                            f"r{first}"
                            if count == 1
                            else f"r{first}-{first + count - 1}"
                        )
                        _emit(
                            event_handler,
                            "error",
                            f"WARNING {name}: timing worker {arm} {block} "
                            f"wrote its fragments and then exited with {code}",
                        )

        result = None
        error = None
        results_path = out_dir / "results.json"
        shared = dict(
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
            replicates_per_process=request.replicates_per_process,
        )
        try:
            if unit.span is not None:
                # ``measured`` holds this run's scenarios, which the plan put
                # ahead of the span, so the sum shares every run axis.
                result = merge_kernel_span_fragments(
                    span=unit.span, parts=measured, **shared
                )
            else:
                result = merge_kernel_fragments(scenario=scenario, **shared)
        except (ValueError, KeyError) as merge_error:
            error = f"{name}: {merge_error}"
        else:
            write_kernel_results(result, results_path)
            if unit.span is None:
                measured[name] = MeasuredScenario(
                    result=result, results_path=str(results_path)
                )

        outcome = KernelScenarioOutcome(
            scenario=name,
            out_dir=out_dir,
            result=result,
            correctness_failed=gates_failed,
            error=error,
            unit_kind=unit.kind,
            failed_passes=tuple(failed_passes),
        )
        # Reported now, because the later scenarios still have to run.
        if outcome.error:
            _emit(event_handler, "error", f"ERROR {outcome.error}")
        outcomes.append(outcome)
    return tuple(outcomes)
