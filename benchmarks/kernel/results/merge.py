"""Assemble one scenario's results from the fragments its workers wrote.

Per-arm process isolation means no single process ever holds the whole
scenario: one worker gates every arm for correctness, and one worker per
(arm, replicate) times exactly one arm. Each writes a small JSON fragment.
This module is what turns that pile back into a ``KernelScenarioResult``.

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
    KernelScenario,
    KernelWorkload,
    MODES,
    shape_summary,
)
from benchmarks.models.piper_qwen3.shape import PiperShape


# The two fragment shapes, named once. The worker writes them through
# ``benchmarks.kernel.engine.run``; this module reads them. A ``kind``
# mismatch is a wiring bug, not a data condition, so it raises.
CORRECTNESS_FRAGMENT_KIND = "kernel_correctness_fragment"
TIMING_FRAGMENT_KIND = "kernel_timing_fragment"

# How the numbers were produced. The rationale for each entry lives in
# ``benchmarks.kernel.engine.measurement``'s docstring; recorded here because
# the parent is what assembles the file that publishes them.
KERNEL_MEASUREMENT_METHODOLOGY = {
    "measurand": "burst_amortized_per_call_device_time",
    "arm_isolation": "one_process_per_arm_replicate",
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


def _require_kind(fragment: dict[str, Any], expected: str) -> None:
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
) -> KernelScenarioResult:
    """Build the scenario result from one correctness and N timing fragments.

    Raises ``ValueError`` when the anchor arm has no complete replicate set,
    or when it was skipped: every comparison is a ratio against it, so there
    is nothing coherent to write.

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
        },
        environment=dict(correctness["environment"]),
        warnings=tuple(warnings),
    )
