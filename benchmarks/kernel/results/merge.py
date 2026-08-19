"""Assemble one scenario's results from the fragments its workers wrote.

Per-arm process isolation means no single process ever holds the whole
scenario: one worker gates every arm for correctness, and each timing worker
times exactly one arm -- one replicate of it at the default, a block of
consecutive replicates above ``replicates_per_process`` 1, and never a second
arm at any value. Each writes a small JSON fragment, one per replicate. This
module is what turns that pile back into a ``KernelScenarioResult``.

It runs in the **parent**, and it is torch-free. That is not incidental --
the parent is the only process that sees every fragment, so it is the only
place the cross-arm work can happen: the floor ratios, the anchor
comparisons, and the completeness check below. ``engine.statistics`` is
already parent-side for the same reason.

**Anchor loss fails loudly.** Every comparison in the file is a ratio
against the scenario's anchor arm, so an anchor that did not produce a
complete replicate set leaves a table with no ratios in it -- which reads
like a scenario that declared no comparisons rather than one whose baseline
crashed. ``merge_kernel_fragments`` raises instead. A *non-anchor* arm that
is incomplete costs only itself: the arms that did measure are still
reported.

**An empty arm is a failed arm.** An arm may write every fragment it owes
and still carry no samples in any mode. That is not an ``ok`` arm with a
short table; it is an arm that measured nothing, so it takes ``failed`` and
a warning, and an *anchor* in that state raises exactly as anchor loss does.

**Every declared arm reaches the file, measured or not.** An arm carries
``status`` ``ok``, ``skipped`` or ``failed``, with the reason attached. An
arm this host could not run (no C++20 compiler for the TE build) and an arm
the registry never declared would otherwise read identically -- both simply
absent -- and a reader has no way to tell a short roster from a complete one.
The loud channel is still ``warnings``; the status is the machine-readable
half of the same statement.

**Replicate boundaries survive the round trip.** A fragment holds one
replicate's samples, and the merge orders them by replicate index rather
than by arrival, because ``engine.statistics`` estimates the ratio once per
replicate and bootstraps across them. Sorting by arrival would still produce
a plausible-looking file with the pairing silently scrambled.

Two fields are per-arm rather than per-replicate -- peak memory and the
``--burst`` ladder -- so replicate 0's worker measures them and later
replicates carry ``None``. The merge reads them from that fragment alone.
"""

from __future__ import annotations

from dataclasses import asdict
from statistics import median
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from benchmarks.artifacts.summaries import summarize
from benchmarks.kernel.engine.statistics import (
    KERNEL_SIGNIFICANCE_METHODOLOGY,
    kernel_comparison,
)
from benchmarks.kernel.results.schema import (
    ArmResult,
    CorrectnessResult,
    KernelScenarioResult,
    ModeResult,
)
from benchmarks.kernel.schema import (
    CORRECTNESS_FRAGMENT_KIND,
    KernelScenario,
    KernelWorkload,
    MODES,
    shape_summary,
    TIMING_FRAGMENT_KIND,
)
from benchmarks.models.piper_qwen3.shape import PiperShape


# An arm whose per-call time falls by more than this across the ladder's top
# two rungs has not amortized its dispatch cost away at the chosen ``k``. The
# value separates the measured rope arms (3.9-7.2%) from ``copy_floor``, the
# one arm there that is genuinely device-bound (1.9%).
BURST_RESIDUAL_FLAG = 0.02


# How the numbers were produced. The rationale for each entry lives in
# ``benchmarks.kernel.engine.measurement``'s docstring; recorded here because
# the parent is what assembles the file that publishes them.
KERNEL_MEASUREMENT_METHODOLOGY = {
    "measurand": "burst_amortized_per_call_time",
    "measurand_note": (
        "the per-call cost under back-to-back dispatch, not device time: a "
        "CUDA event pair measures an interval on the stream, so where the "
        "host cannot enqueue faster than the device drains, the interval "
        "holds the host stalls too. It equals device time only where the arm "
        "is device-bound. Run --burst; burst_residual is the fraction by "
        "which per-call time still falls across the ladder's top two rungs, "
        "and an arm above burst_residual_flag_threshold has a burst_k-"
        "dependent ratio that is not a kernel-speed claim. The test is "
        "one-sided: an arm below the threshold is NOT thereby device-bound, "
        "because a ladder can plateau at a dispatch cost bursting cannot "
        "amortize. burst_residual is a difference of two medians and "
        "carries their noise, so a value near the threshold is undecided."
    ),
    "burst_residual_flag_threshold": BURST_RESIDUAL_FLAG,
    "l2_flush": False,
    "l2_flush_rationale": (
        "no flush, and no equalization is claimed: one arm is timed at a "
        "time, so an arm whose working set fits in L2 benefits from bursting "
        "more than one whose does not"
    ),
    "gc_paused_during_timing": True,
    "first_burst_discarded": True,
    "units": "microseconds",
}


def _isolation(replicates_per_process: int) -> dict[str, Any]:
    """How the run mapped replicates onto processes, said in the file.

    Two arms never share a timing process, at any value: that is the split
    the cuDNN soname collision makes non-negotiable. What the value changes
    is whether an arm's replicates are independent processes, which is what
    the per-replicate log-ratio treats them as. A reader who does not know
    the value cannot tell a CI narrowed by evidence from one narrowed by
    removing a source of variation, so the file says it rather than leaving
    it in the manifest.
    """
    if replicates_per_process <= 1:
        return {
            "arm_isolation": "one_process_per_arm_replicate",
            "replicates_per_process": 1,
        }
    return {
        "arm_isolation": "one_process_per_arm_replicate_block",
        "replicates_per_process": replicates_per_process,
        "replicates_per_process_note": (
            "consecutive replicates of one arm shared a process, so they do "
            "not sample process-to-process variation and the bootstrap CI on "
            "the per-replicate log-ratios is narrower than a fresh-process "
            "sweep would give. The arms also move apart in time as the block "
            "grows, so drift between two arms' blocks lands in the point "
            "estimate instead of cancelling. One arm per process still holds. "
            "Every statistic derived from the per-replicate log-ratios is "
            "therefore published under a within_process_ name -- "
            "within_process_ratio_ci_low/high and "
            "within_process_replicate_ratio_spread -- and never under the "
            "honest one."
        ),
    }


# Every field ``bootstrap_ratio_ci`` derives from the per-replicate
# log-ratios, and therefore every field a shared process degrades. The
# interval is the obvious one. ``replicate_ratio_spread`` is the same
# quantity read a second way -- ``exp(max) - exp(min)`` over those same
# ratios -- so it is arguably the *most* degraded field in the row: it is
# literally the round-to-round spread the note below says does not improve.
# Naming them in one tuple is what keeps a later addition to
# ``bootstrap_ratio_ci`` from shipping under an honest name by omission.
_DEGRADED_BY_A_SHARED_PROCESS = (
    "ratio_ci_low",
    "ratio_ci_high",
    "replicate_ratio_spread",
)


def _rename_degraded_ci(row: dict[str, Any]) -> dict[str, Any]:
    """Publish the within-process statistics under their own field names.

    The bootstrap runs on per-replicate log-ratios, and above one replicate
    per process those replicates are consecutive measurements of one build in
    one interpreter rather than independent processes. The interval narrows
    while the point estimate's round-to-round spread does not improve, which
    makes it a lower bound on the true one.

    Each field is renamed rather than reinterpreted, exactly as the schema
    history renames every field whose meaning changes: anything reading
    ``ratio_ci_low`` gets nothing instead of a narrower number under the
    honest name, and a reader who wants the degraded statistic has to ask for
    it by a name that says what it is.

    A missing key raises, as ``_require_kind`` does above and for the same
    reason: ``bootstrap_ratio_ci`` returns all three on every path, so an
    absent one is a wiring bug rather than a data condition, and swallowing
    it would leave the honest name in the row.
    """
    for field_name in _DEGRADED_BY_A_SHARED_PROCESS:
        row[f"within_process_{field_name}"] = row.pop(field_name)
    return row


def _require_kind(fragment: dict[str, Any], expected: str) -> None:
    """A kind mismatch is a wiring bug, not a data condition, so it raises."""
    kind = fragment.get("kind")
    if kind != expected:
        raise ValueError(
            f"expected a {expected!r} fragment, got {kind!r}"
        )


def _by_arm(
    timings: Sequence[dict[str, Any]]
) -> dict[str, dict[int, dict[str, Any]]]:
    """Fragments indexed by arm name, then by replicate index."""
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for fragment in timings:
        _require_kind(fragment, TIMING_FRAGMENT_KIND)
        replicate = int(fragment["replicate"])
        arm = grouped.setdefault(str(fragment["arm"]), {})
        if replicate in arm:
            raise ValueError(
                f"{fragment['arm']}: two fragments for replicate {replicate}"
            )
        arm[replicate] = fragment
    return grouped


def _ordered_replicates(
    fragments: dict[int, dict[str, Any]], replicates: int
) -> list[dict[str, Any]] | None:
    """The full replicate sequence in index order, or None if incomplete."""
    if sorted(fragments) != list(range(replicates)):
        return None
    return [fragments[index] for index in range(replicates)]


def _mode_samples(
    ordered: list[dict[str, Any]]
) -> dict[str, list[list[float]]]:
    """Replicate-major samples per mode, in ``MODES`` order.

    A mode is kept only when every replicate measured it. A mode that
    appeared in some replicates and not others would carry a replicate
    sequence shorter than the arm's, and zipping it against the anchor's
    would silently pair replicate 2 with replicate 0.
    """
    samples: dict[str, list[list[float]]] = {}
    for mode in MODES:
        if all(fragment["modes"].get(mode) for fragment in ordered):
            samples[mode] = [
                [float(value) for value in fragment["modes"][mode]]
                for fragment in ordered
            ]
    return samples


def _burst_residual(
    ladders: dict[str, dict[str, float]] | None, mode: str
) -> float | None:
    """How far per-call time still falls across the ladder's top two rungs.

    This is the only evidence in the file that says whether a number is
    device time or dispatch cost. The primary pass fixes one ``k``, so it
    cannot tell the two apart on its own: a device-bound arm and a
    dispatch-bound arm both report a single per-call figure. An arm still
    falling at the top of the ladder has host stalls inside its measured
    interval, and its ratio against another arm moves with ``k``.

    Returns ``None`` when the ``--burst`` ladder did not run for this mode,
    which is the default. Absent evidence reads as absent, never as zero.
    """
    rungs = (ladders or {}).get(mode) or {}
    if len(rungs) < 2:
        return None
    sizes = sorted(rungs, key=int)
    penultimate, last = rungs[sizes[-2]], rungs[sizes[-1]]
    if last <= 0:
        return None
    return (penultimate - last) / last


def merge_kernel_fragments(
    *,
    scenario: KernelScenario,
    shape: PiperShape,
    workload: KernelWorkload,
    hardware: str,
    replicates: int,
    samples_per_replicate: int,
    burst_k: int,
    warmup_calls: int,
    seed: int,
    correctness: dict[str, Any],
    timings: Sequence[dict[str, Any]],
    timings_ran: bool = True,
    skipped: Mapping[str, str] = MappingProxyType({}),
    replicates_per_process: int = 1,
) -> KernelScenarioResult:
    """Build the scenario result from one correctness and N timing fragments.

    Raises ``ValueError`` when the anchor arm has no complete replicate set,
    when it was skipped, or when every replicate it did write is empty: every
    comparison is a ratio against it, so there is nothing coherent to write.

    ``timings_ran=False`` records a scenario whose gates failed, so no timing
    worker was ever launched. The file still carries the correctness rows --
    a failed gate is the result -- and ``replicates`` still records what was
    requested rather than the zero that ran, because the request is what the
    reader needs in order to repeat it.

    ``skipped`` maps an arm this host never launched to the reason. The arm
    still appears in the file, with ``status="skipped"``, because an absent
    arm and an undeclared arm read identically.
    """
    _require_kind(correctness, CORRECTNESS_FRAGMENT_KIND)
    if scenario.baseline_arm in skipped:
        raise ValueError(
            f"{scenario.name}: the anchor arm {scenario.baseline_arm!r} was "
            f"skipped ({skipped[scenario.baseline_arm]}). Every comparison "
            "is a ratio against it, so no results are written."
        )
    grouped = _by_arm(timings)
    warnings: list[str] = []
    if not timings_ran:
        warnings.append(
            "correctness gates failed, so no arm was timed; this file "
            "reports the gates only"
        )

    complete: dict[str, list[dict[str, Any]]] = {}
    lost: dict[str, str] = {}
    for arm in scenario.arms if timings_ran else ():
        if arm.name in skipped:
            continue
        ordered = _ordered_replicates(grouped.get(arm.name, {}), replicates)
        if ordered is None:
            found = len(grouped.get(arm.name, {}))
            message = (
                f"{found} of {replicates} replicates produced a fragment, so "
                "this arm carries no timings"
            )
            if arm.name == scenario.baseline_arm:
                raise ValueError(
                    f"{scenario.name}: the anchor arm {arm.name!r} produced "
                    f"{found} of {replicates} replicates. Every comparison is "
                    "a ratio against it, so no results are written rather "
                    "than a table with no ratios in it."
                )
            warnings.append(f"{arm.name}: {message}")
            lost[arm.name] = message
            continue
        complete[arm.name] = ordered

    samples = {
        name: _mode_samples(ordered) for name, ordered in complete.items()
    }
    # An arm every replicate of which arrived empty ran and measured nothing.
    # ``status`` is a dataclass default, so such an arm used to reach the file
    # as "ok" with no modes under it -- and the reporter tabulates mode by
    # mode, so it appeared in no table and in no unmeasured list either. It
    # simply vanished. ``--samples-per-replicate 0`` reaches this state, and
    # so does a declared mode the timing pass never times.
    unmeasured = {name for name, modes in samples.items() if not modes}
    if scenario.baseline_arm in unmeasured:
        raise ValueError(
            f"{scenario.name}: the anchor arm {scenario.baseline_arm!r} "
            "produced no samples in any declared mode. Every comparison is a "
            "ratio against it, so no results are written rather than a table "
            "with no ratios in it."
        )
    # Declared, not reported by the fragment: a floor is a property of the
    # arm the registry describes, and the x-floor column belongs to a reader
    # who has the registry and no GPU.
    floor_arms = {
        name for name in complete if scenario.arm(name).is_floor
    }
    floor_medians = {
        mode: median(value for replicate in modes[mode] for value in replicate)
        for name, modes in samples.items()
        if name in floor_arms
        for mode in modes
    }

    # Every declared arm, in declaration order, measured or not. An arm this
    # host could not run and an arm the registry never declared must not read
    # the same way, and at schema 3 both were simply absent.
    arm_results: dict[str, ArmResult] = {}
    for declaration in scenario.arms:
        name = declaration.name
        if name not in complete:
            if name in skipped:
                status, reason = "skipped", skipped[name]
            elif name in lost:
                status, reason = "failed", lost[name]
            else:
                status, reason = "skipped", (
                    "the correctness gates failed, so no arm was timed"
                )
            arm_results[name] = ArmResult(
                name=name, modes={}, status=status, status_reason=reason
            )
            continue
        if name in unmeasured:
            message = "produced no samples in any declared mode"
            warnings.append(f"{name}: {message}")
            arm_results[name] = ArmResult(
                name=name, modes={}, status="failed", status_reason=message
            )
            continue
        ordered = complete[name]
        first = ordered[0]
        bytes_moved = first["bytes_moved"]
        is_floor = name in floor_arms
        modes: dict[str, ModeResult] = {}
        for mode, replicate_samples in samples[name].items():
            pooled = [
                value for replicate in replicate_samples for value in replicate
            ]
            mode_median = median(pooled)
            derived: dict[str, float] = {}
            if bytes_moved and mode_median:
                derived["gbps"] = bytes_moved / (mode_median * 1e-6) / 1e9
            if mode in floor_medians and not is_floor and floor_medians[mode]:
                derived["x_floor"] = mode_median / floor_medians[mode]
            residual = _burst_residual(first["burst_us_per_call"], mode)
            if residual is not None:
                derived["burst_residual"] = residual
            modes[mode] = ModeResult(
                summary=summarize(pooled),
                replicates_us=tuple(
                    tuple(replicate) for replicate in replicate_samples
                ),
                derived=derived,
            )
        arm_results[name] = ArmResult(
            name=name,
            modes=modes,
            # Both measured once per arm, by replicate 0's worker.
            peak_memory_gib=first["peak_memory_gib"],
            burst_us_per_call=first["burst_us_per_call"],
        )

    # Which rows exist is declared by the scenario, never inferred here. A
    # scenario whose two sides are not a like-for-like cut says so by
    # declaring no pair, and this loop then writes no ratio -- where the
    # former per-arm ``compare_to`` could only redirect a row, not decline
    # one. An arm absent from ``samples`` was already warned about above, or
    # the gates failed and nothing was timed.
    comparisons: list[dict[str, Any]] = []
    for arm_name, opponent in scenario.comparison_pairs():
        if arm_name not in samples:
            continue
        if opponent not in samples:
            # No current scenario reaches this. All five derive their pairs,
            # so every opponent is the anchor, and an anchor missing from
            # ``samples`` raised above. The branch is here for a scenario that
            # declares ``comparisons`` explicitly and names a second arm as an
            # opponent: that arm can be skipped or lost while the anchor
            # survives, and one absent row must not cost the other rows.
            warnings.append(
                f"{arm_name}: opponent {opponent!r} is absent, so this arm "
                "carries no ratio"
            )
            continue
        for mode in samples[arm_name]:
            if mode not in samples[opponent]:
                continue
            row: dict[str, Any] = {
                "arm": arm_name,
                "opponent": opponent,
                "mode": mode,
            }
            row.update(
                kernel_comparison(samples[opponent][mode], samples[arm_name][mode])
            )
            if replicates_per_process > 1:
                _rename_degraded_ci(row)
            comparisons.append(row)

    return KernelScenarioResult(
        scenario=scenario.name,
        hardware=hardware,
        model_size=shape.name,
        model_shape=shape.describe(seq_len=workload.seq_len),
        workload=asdict(workload),
        shapes=shape_summary(scenario.name, shape, workload),
        replicates=replicates,
        samples_per_replicate=samples_per_replicate,
        burst_k=burst_k,
        warmup_calls=warmup_calls,
        seed=seed,
        arms=arm_results,
        comparisons=comparisons,
        correctness=[
            CorrectnessResult(**row) for row in correctness["rows"]
        ],
        all_correctness_passed=bool(correctness["all_passed"]),
        methodology={
            **KERNEL_SIGNIFICANCE_METHODOLOGY,
            **KERNEL_MEASUREMENT_METHODOLOGY,
            **_isolation(replicates_per_process),
        },
        environment=dict(correctness["environment"]),
        warnings=tuple(warnings),
    )
