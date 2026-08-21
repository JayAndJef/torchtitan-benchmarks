"""Orchestrate kernel-isolation measurements from the torch-free CLI.

A run measures two kinds of unit. A **scenario** cuts the model at one
boundary and ranks the implementations there. A **span** fuses across a cut,
so it is declared over an ordered scenario range and its claim is the span
against the sum of the scenarios it replaces. ``measurement_plan`` puts every
enclosed scenario ahead of its span and each scenario once, and the span
merge takes its second total from those results -- so both sides of the claim
share one request's hardware, shape, workload, seed, ``burst_k``, replicate
count and NUMA pinning by construction rather than by a re-check.

Mirrors ``e2e/runner.py``: the parent resolves provenance and pinning, writes
the manifest, spawns pinned GPU workers so CUDA device selection, NUMA
binding and the TE build's compiler environment all apply to the measuring
process, and assembles the results. Scenarios are independent short runs, so
unlike the e2e sweep the parent continues past a failing scenario and reports
all outcomes.

**One process per pass, not one per scenario.** A scenario is a correctness
worker followed by ``blocks x arms`` timing workers, each building a single
arm. A block is ``replicates_per_process`` consecutive replicates, so at the
default of 1 there are ``replicates x arms`` of them. That is what keeps one
arm's dependencies out of another arm's interpreter -- a build failure, a
leaked CUDA context or a JIT-built CUDA extension in one arm cannot reach
another -- and it is why the parent, not a worker, writes ``results.json``:
only the parent sees every fragment.

**The sweep is block-major.** A block is ``replicates_per_process``
consecutive replicates of one arm; the outer loop walks the blocks and the
inner loop walks the arms. At the default of one replicate per process a
block is one replicate and this is the replicate-major sweep it has always
been: drift is shared across arms rather than charged to whichever arm
happened to be running, and the anchor sits in every replicate slot. Raising
it trades that adjacency for the arm builds it stops repeating; the trade is
stated on ``KernelRunRequest``.

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
#
# 5: "skipped_arms" records the arms this host never launched, and why. The
# roster in "arms" is the registry's, so a manifest-only reader previously
# had to diff it against "commands" to learn that an arm was dropped -- and
# would read a schema-4 manifest written after this change identically to
# one written before it. This file is write-only provenance, so the bump is
# labelling, exactly as 3 was.
#
# 6: "replicates_per_process" records how many of an arm's replicates shared
# a worker. A schema-5 manifest was always 1 -- the field did not exist
# because the choice did not -- so its absence is unambiguous, and the bump
# is labelling again rather than a reinterpretation. The timing entries of
# "commands" also change shape here, from "--fragment <path>" to
# "--fragments-dir <dir>": a worker that measures a block writes a file per
# replicate, so it is handed the directory. Nothing reads the field, but a
# reader diffing a schema-5 manifest against a schema-6 one meets the change
# and this is where it is explained.
#
# 7: a run measures two kinds of unit. A **span** fuses across a scenario
# cut, so it is declared over an ordered scenario range and its claim is the
# span against the sum of the scenarios it replaces. The manifest gains
# "unit_kind", "span_scenarios" and "parts", and its "kind" value changes
# from "kernel" to "kernel_scenario" or "kernel_span" -- the same rename the
# results file makes at schema 7, and for the same reason: "kernel" named the
# family and one member of it at once. This file is write-only provenance
# with no loader, so the bump is labelling, exactly as 3, 5 and 6 were.
KERNEL_MANIFEST_SCHEMA_VERSION = 7


@dataclass(frozen=True)
class KernelRunRequest:
    """One kernel-bench invocation.

    ``replicates_per_process`` is the one field here that is a methodology
    choice rather than a workload one. It says how many consecutive
    replicates of an arm share a worker process, and it trades measurement
    cost against the property the replicate exists for.

    At ``1`` every replicate is a fresh process and the sweep is
    replicate-major: an arm's five replicates are spread across the run, each
    within seconds of every other arm's matching replicate, so drift that
    moves a whole replicate cancels in the per-replicate log-ratio the
    bootstrap runs on. That is what makes the CI an honest statement about
    the ratio.

    Above ``1`` an arm's replicates become consecutive measurements inside
    one process, separated by milliseconds rather than by a rebuild. Two
    things follow, and both are costs. The replicates stop sampling
    process-to-process variation, so the CI narrows without the underlying
    quantity having become better known -- measured at 26-48% narrower while
    the point estimate's round-to-round spread did not improve, **on a box at
    load average 28 to 81, on ``qkv`` alone, two arms, one of them the
    anchor**. That figure is the record of what was tried and may not be
    cited. And the arms move apart in time -- at the extreme, arm A's whole
    block runs, then arm B's -- so drift between the blocks lands in the point
    estimate instead of cancelling. The same measurement could only have
    resolved a shift in the point estimate larger than 4.2-6.3%, and
    ``backward`` moved -4.78%, so a bias of a few percent in the noisiest mode
    is not ruled out. The results file renames every degraded statistic
    accordingly; see ``benchmarks.kernel.results.merge``.

    Raise it to buy wall-clock, and say in the report that you did. Use 1 for
    anything published.
    """

    gpu: str
    scenario_names: tuple[str, ...]
    # The spans this run measures, and never a default. A span is compared
    # against the sum of the scenarios it replaces, and those scenarios are
    # measured in the same run -- so one span can drag six scenarios into a
    # run that asked for none of them. Making it opt-in keeps a bare
    # ``kernel-bench <gpu>`` meaning what it has always meant.
    span_names: tuple[str, ...] = ()
    replicates: int = 5
    replicates_per_process: int = 1
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
    """What one measurement unit produced.

    ``scenario`` is the unit's name and ``unit_kind`` says which kind of unit
    it names. The published artifacts keep the two apart by shape -- a span
    writes a ``kernel_span`` results file with a ``span`` field, and never a
    ``scenario`` one -- and this type is the in-memory hand-off that carries
    both to the reporter.
    """

    scenario: str
    out_dir: Path
    result: KernelScenarioResult | KernelSpanResult | None
    correctness_failed: bool = False
    error: str | None = None
    unit_kind: str = "scenario"
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


@dataclass(frozen=True)
class MeasurementUnit:
    """One thing this run measures: a scenario, or a span.

    ``measurement`` is what the workers build and time, and it is a
    ``KernelScenario`` either way -- a span composes one. That is what keeps
    ``benchmarks.kernel.engine`` free of any knowledge that spans exist.

    ``span`` is set only for a span, and it is what the merge needs on top of
    the measurement: the ordered range and what each arm replaces. The
    runner reads it for the output directory, the manifest, the worker argv
    and the merge, and for nothing else.
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

    **Scenarios first, then spans.** A span's claim is against the sum of the
    scenarios it replaces, and the merge takes that sum from the results of
    the same run -- so every enclosed scenario must have finished before the
    span is merged. Running them together is what makes the two sides share a
    hardware label, a shape, a workload, a seed, a ``burst_k``, a replicate
    count and a NUMA pinning by construction rather than by a re-check.

    **Each scenario appears once.** A scenario the operator asked for and a
    scenario two spans both enclose is still one scenario. Measuring it twice
    would spend the GPU on it twice and produce two different numbers for one
    thing, and nothing would say which of them a span summed.
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
    """The argv for one worker pass. Deterministic, so the manifest can list
    every command the run will issue before the first one starts.

    A timing worker is handed the fragments directory rather than a path,
    because a batched one writes a file per replicate. The correctness worker
    writes exactly one file and is still handed it by name.

    A span is named with ``--span`` and never with ``--scenario``. The worker
    resolves it through the span registry and measures ``span.measurement``,
    so the recorded argv says which roster the name belongs to and a reader
    of the manifest never has to guess.
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
        # Neither worker pass reads --replicates. The parent owns the sweep,
        # and a timing worker measures the block --replicate and
        # --replicate-count name, which is one replicate at the default.
        # It is forwarded as provenance: the manifest publishes this argv as
        # the record of the run, so each command states the whole request it
        # came from, and the worker's own default of 5 never stands in for a
        # count the operator chose.
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
    short. ``replicates_per_process=1`` gives one block per replicate, which
    is the historical sweep.
    """
    size = max(1, replicates_per_process)
    return tuple(
        (start, min(size, replicates - start))
        for start in range(0, replicates, size)
    )


def resolve_arm_skips(
    scenario: KernelScenario | KernelSpan,
    *,
    compiler_unavailable: str | None,
    shape: PiperShape,
    workload: KernelWorkload,
) -> dict[str, str]:
    """Which arms this host and this workload cannot run, and why.

    **Requirements belong to the arm, not to the scenario.** Without a C++20
    host compiler, rope loses ``titan/te`` and still measures its other four
    arms. The scenario-level ``requires_gcc_toolset`` is an OR across arms, so
    using it to decide cost every arm of the scenario; it keeps its one honest
    use, which is asking whether anything here needs the compiler at all.

    **Two kinds of requirement, and the second needs the workload.**
    ``requires_gcc_toolset`` is a property of the host, answerable before the
    shape is known. ``KernelArm.requirement`` is the other kind: a dotted
    path to a parent-side predicate, called with ``(shape, workload)``, that
    returns the reason this arm cannot run here or ``None``. An unfused
    attention arm whose dense score tensor grows with the square of the
    sequence length is the case that forced it, and it is why a sequence
    sweep was impossible before: one builder raising killed the whole
    scenario, because the correctness pass builds every arm in one process
    and catches nothing.

    **The probe runs in the parent, before a GPU is claimed.** That is what
    makes it a probe rather than a rescue: the arm is never built, so nothing
    has to decide whether an exception was a requirement or a bug. A builder
    that raises for an undeclared reason still fails the scenario, which is
    what it is.

    **The set is closed over correctness references.** An arm whose reference
    is skipped is skipped too. The alternative is to time an arm that nothing
    checked, which is the silent wrongness the gates exist to prevent. The
    compiler skip has never reached that closure -- ``titan/te`` is a
    referrer, never a reference -- but a workload skip does: an arm dropped
    at a long sequence takes with it anything gated against it.
    """
    skipped: dict[str, str] = {}
    for arm in scenario.arms:
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

    **Block-major**: the outer loop walks the replicate blocks and the inner
    loop walks the arms, so every arm is measured once before any arm is
    measured again. At ``replicates_per_process=1`` a block is one replicate
    and this is exactly the replicate-major sweep -- every arm is timed
    within seconds of every other, and drift that moves a whole replicate
    cancels in the ratio. At a larger value the blocks get longer and that
    adjacency coarsens; ``KernelRunRequest.replicates_per_process`` states
    what it costs.

    A skipped arm appears in no pass, and the manifest therefore lists what
    the run really does rather than what a fully-equipped host would have
    done.

    The sweep is described here, once, because two callers need it: the
    manifest writer turns each triple into an argv, and the spawn loop needs
    the same triple back to know which fragments to look for. The spawn loop
    used to recover it by parsing the argv it had just generated -- which
    tested ``"--replicate-count" in command`` over the whole list, so a
    *value* equal to that string would have matched as readily as the flag.
    A worker must not learn a convention by string surgery on what it was
    handed (``benchmarks.kernel.schema.timing_fragment_path``), and neither
    must the parent.
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

    The correctness pass first, then one argv per entry of
    ``timing_passes``, which is where the order is decided. A span's passes
    are a scenario's passes: it composes a ``KernelScenario`` and the workers
    measure that.
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
        # Renamed at schema 7, exactly as the results file renames it: two
        # kinds of unit now write a manifest, and "kernel" named the family
        # and one member of it at once.
        "kind": f"kernel_{unit.kind}",
        "unit_kind": unit.kind,
        # The unit's name, under the field that says what kind of name it
        # is. A span name is NOT a scenario name: a reader who finds one in
        # "scenario" goes to KERNEL_SCENARIOS to look it up and finds
        # nothing. The results file has kept these apart since schema 7 and
        # this file put a span name in "scenario" one directory over.
        "scenario": None if span else scenario.name,
        "span": span.name if span else None,
        "description": scenario.description,
        # The ordered range a span replaces, and what each of its arms
        # replaces in that range. Absent on a scenario, where there is no
        # range to name. Prefixed, unlike the results file's "scenarios",
        # because both kinds of unit share this one dict and an unprefixed
        # plural would sit next to "scenario" and read as its list form.
        "span_scenarios": list(span.scenarios) if span else None,
        "parts": (
            {entry.arm: list(entry.parts) for entry in span.parts}
            if span
            else None
        ),
        "model_size": request.model_size,
        # The same record the e2e manifest writes, so both systems state
        # model identity identically.
        "model_shape": shape.describe(seq_len=workload.seq_len),
        "workload": asdict(workload),
        # A span's inputs are the first cut's and its outputs are the last
        # cut's, so no single entry describes it and shape_summary knows
        # scenario names only. The union of the range is what a reader needs.
        "shapes": (
            {
                name: shape_summary(name, shape, workload)
                for name in span.scenarios
            }
            if span
            else shape_summary(scenario.name, shape, workload)
        ),
        "arms": [asdict(arm) for arm in scenario.arms],
        # "arms" is the registry's roster, so it names arms this host never
        # ran. The reason is recorded beside the name: a reader of the
        # manifest alone can then tell a missing arm from a dropped one,
        # without a diff of "arms" against "commands".
        "skipped_arms": dict(skipped),
        "baseline_arm": scenario.baseline_arm,
        "replicates": request.replicates,
        # How many of them shared a process. A reader comparing two runs
        # needs it: at 1 the sweep is replicate-major and the per-replicate
        # ratios cancel drift, and above 1 they do so less. It is not
        # derivable from "commands" without parsing every argv.
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

    # Resolved once per run, not once per scenario and certainly not once per
    # worker: add_compiler_environment shells out to bash, and the answer
    # cannot change between two scenarios of the same run.
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
    # Every scenario this run measured, for the span merges below. A span's
    # second total is the sum of its scenarios, and the plan guarantees they
    # ran first and in this same run.
    measured: dict[str, MeasuredScenario] = {}
    for unit in plan:
        scenario = unit.measurement
        name = unit.name
        # A span sits under its own directory, so a glob of
        # out/*/kernels/*/*/ over the scenarios does not sweep up a span
        # beside them: a span total and a scenario total answer different
        # questions and must not be pooled by a path pattern.
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
        )
        if scenario.baseline_arm in skipped:
            # The anchor carries every ratio, so losing it is the one skip
            # that costs the scenario rather than an arm.
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
            # Loudly skipped rather than quietly rounded: an unbalanced split
            # would silently measure a different workload per expert.
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
                # In lockstep with commands[1:], which planned_commands built
                # from this same sequence. The pass is read from the plan, not
                # parsed back out of the argv the plan produced.
                # strict, so a plan and a sweep of different lengths raise
                # rather than silently dropping the tail of the longer one.
                passes = timing_passes(scenario, request, skipped)
                for (arm, first, count), command in zip(
                    passes, commands[1:], strict=True
                ):
                    code = spawn(command)
                    # One worker, one fragment per replicate it covered. A
                    # batched worker that died mid-block leaves some of them
                    # missing, and each missing one is reported on its own:
                    # the merge counts replicates, not workers, and an arm
                    # short of one is incomplete however few processes lost
                    # it.
                    missing = False
                    for replicate in range(first, first + count):
                        fragment = _read_fragment(
                            timing_fragment_path(fragments_dir, arm, replicate)
                        )
                        if fragment is None:
                            missing = True
                            label = f"{arm} r{replicate}"
                            failed_passes.append(label)
                            # Reported as it happens rather than at the end:
                            # the sweep continues, and an operator watching a
                            # long run should not learn about the first crash
                            # last.
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
                        # The samples stand -- the worker writes its fragments
                        # before it returns -- so this costs the arm nothing.
                        # It is still said out loud: the code reports a death
                        # after the write, and a discarded exit code is how a
                        # silent failure starts.
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
                # ``measured`` holds the scenarios this same run produced.
                # The plan put every one of them ahead of the span, so the
                # sum is taken from results that share this run's hardware,
                # shape, workload, seed, burst_k and replicate count by
                # construction.
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
        # Report a failure the moment it happens. Scenarios continue past one
        # another, so holding this until the end would show a first-scenario
        # failure only after every later scenario had run.
        if outcome.error:
            _emit(event_handler, "error", f"ERROR {outcome.error}")
        outcomes.append(outcome)
    return tuple(outcomes)
