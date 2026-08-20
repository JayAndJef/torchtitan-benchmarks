"""Human-readable rendering of kernel-isolation results."""

from __future__ import annotations

import textwrap

from benchmarks.artifacts.summaries import _pvalue, _value
from benchmarks.kernel.results.merge import BURST_RESIDUAL_FLAG
from benchmarks.kernel.results.schema import KernelScenarioResult
from benchmarks.kernel.schema import MODES


def _ratio_ci(row: dict | None) -> str:
    """The bootstrap interval on per-replicate log-ratios, as ``[lo, hi]``.

    ``~`` marks a within-process interval. Above one replicate per process the
    merge publishes the interval under ``within_process_ratio_ci_*``, because
    those replicates are consecutive measurements of one build rather than
    independent processes and the interval is a lower bound. This function
    reads that name too, so the degradation reaches the printed table instead
    of showing as an empty column -- but it never prints a degraded interval
    unmarked.
    """
    if not row:
        return "-"
    low, high = row.get("ratio_ci_low"), row.get("ratio_ci_high")
    if low is not None and high is not None:
        return f"[{low:.4f},{high:.4f}]"
    low = row.get("within_process_ratio_ci_low")
    high = row.get("within_process_ratio_ci_high")
    if low is None or high is None:
        return "-"
    return f"~[{low:.4f},{high:.4f}]"


def _residual(derived: dict, width: int) -> str:
    """The ladder residual as a percentage, with ``!`` when it is flagged.

    ``!`` says the row above it is a dispatch comparison rather than a
    kernel comparison, so it sits in the timing table beside the ratio it
    qualifies rather than under the ladder further down.
    """
    value = derived.get("burst_residual")
    if value is None:
        return f"{'-':>{width}s}"
    mark = "!" if value > BURST_RESIDUAL_FLAG else " "
    return f"{value * 100:>{width - 1}.1f}{mark}"


def _isolation_lines(result: KernelScenarioResult) -> list[str]:
    """State how replicates were mapped onto processes, above the table.

    ``results.json`` has said this since the flag arrived, and a reader of the
    printed output learned nothing: the ``95% CI`` column kept its heading
    whether the interval was taken across processes or inside one. Printed
    always, never only in the degraded case -- a reader must not have to infer
    the honest state from the absence of a warning.
    """
    per_process = result.methodology.get("replicates_per_process", 1)
    isolation = result.methodology.get(
        "arm_isolation", "one_process_per_arm_replicate"
    )
    lines = [
        f"arm isolation: {isolation} "
        f"({per_process} replicate{'' if per_process == 1 else 's'} "
        "per process)"
    ]
    if per_process > 1:
        # No figure. The one that stood here -- 26-48% narrower -- was taken
        # on a box carrying load average 28 to 81, on qkv alone, two arms, one
        # of them the anchor. CLAUDE.md says none of that work's numbers may
        # be cited, and this line printed one as an operational fact on every
        # degraded run. The qualitative statement carries the whole message,
        # and the last line already says what to do about it.
        lines.extend(
            [
                "WARNING: an arm's replicates shared a process, so they do not"
                " sample process-to-process",
                "  variation. The interval in the '95% CI' column is marked"
                " '~' and is a WITHIN-PROCESS",
                "  interval -- a lower bound on the true one, narrower without"
                " the ratio having become",
                "  better known. Do not publish a number from this run.",
            ]
        )
    return lines


def _description_lines(result: KernelScenarioResult) -> list[str]:
    """The scenario's own description, printed above its tables.

    **A scenario that declares no cross-engine ratio has to be able to say so
    where the numbers are read.** ``description`` reached ``results.json`` and
    nothing else: this renderer never looked at it. That is harmless while
    every description is informational, and it stops being harmless the moment
    one of them carries a prohibition.

    ``expert_mlp`` is the first scenario in that class. Its two engines compute
    two different functions at the cut -- megatron applies the routing
    probabilities inside the experts and titan applies them in combine -- so it
    declares its ``comparisons`` explicitly and publishes no row between the
    engines. But the per-mode table still lists all eight arms in one column,
    four ``mcore/*`` above four ``titan/*``, each with a median. Two of those
    medians must not be divided, and before this the only place that said so
    was a JSON field nobody reading the table has open.

    Wrapped rather than printed raw, because a description that states a
    prohibition is a paragraph and not a label.
    """
    if not result.description:
        return []
    return ["", *textwrap.wrap(result.description.strip(), width=78), ""]


def render_kernel_results(result: KernelScenarioResult) -> str:
    """Render one kernel scenario: per-mode timings and correctness gates."""
    shapes = "  ".join(
        f"{name}={value}" for name, value in result.shapes.items()
    )
    lines = [
        f"== kernel scenario: {result.scenario}   "
        f"model size: {result.model_size}   hardware: {result.hardware} ==",
        f"shapes: {shapes}",
        f"{result.replicates} replicates x {result.samples_per_replicate} "
        f"samples, burst k={result.burst_k}, "
        f"warmup={result.warmup_calls} calls, seed={result.seed}",
        *_isolation_lines(result),
        *_description_lines(result),
    ]
    lines.extend(f"WARNING: {warning}" for warning in result.warnings)

    # Printed before the tables, so an arm missing from them is never read as
    # an arm the scenario does not declare.
    unmeasured = [arm for arm in result.arms.values() if arm.status != "ok"]
    if unmeasured:
        lines.append("")
        for arm in unmeasured:
            lines.append(
                f"  {arm.status.upper()} {arm.name:22s} {arm.status_reason}"
            )

    comparisons = {
        (row["arm"], row["mode"]): row for row in result.comparisons
    }
    # MODES, not a copy of it. A sixth mode added to the schema would
    # otherwise be measured, merged, written to results.json, and printed by
    # nothing -- the one failure this table cannot report about itself.
    for mode in MODES:
        arms = [
            name
            for name, arm in result.arms.items()
            if mode in arm.modes
        ]
        if not arms:
            continue
        lines.extend(
            [
                "",
                f"{mode}:",
                "  "
                + f"{'arm':22s} {'median us':>10s} {'sd':>8s} {'vs':>18s} "
                + f"{'ratio':>7s} {'95% CI':>17s} {'GB/s':>8s} "
                + f"{'x floor':>8s} {'resid %':>8s} "
                + f"{'Welch p':>9s} {'MWU p':>9s} {'d':>6s}",
            ]
        )
        for name in arms:
            mode_result = result.arms[name].modes[mode]
            row = comparisons.get((name, mode))
            derived = mode_result.derived
            lines.append(
                f"  {name:22s} "
                f"{mode_result.summary.median_us:10.2f} "
                f"{mode_result.summary.standard_deviation_us:8.2f} "
                f"{(row['opponent'] if row else '-'):>18s} "
                f"{_value(row['median_ratio'] if row else None, 7, 4)} "
                f"{_ratio_ci(row):>17s} "
                f"{_value(derived.get('gbps'), 8, 1)} "
                f"{_value(derived.get('x_floor'), 8, 2)} "
                f"{_residual(derived, 8)} "
                f"{_pvalue(row['welch_p'] if row else None, 9)} "
                f"{_pvalue(row['mwu_p'] if row else None, 9)} "
                f"{_value(row['cohens_d'] if row else None, 6, 2)}"
            )

    memory = {
        name: arm.peak_memory_gib
        for name, arm in result.arms.items()
        if arm.peak_memory_gib is not None
    }
    if memory:
        lines.extend(["", "peak memory (GiB, heaviest mode, isolated):"])
        for name, peak in memory.items():
            lines.append(f"  {name:22s} {peak:8.3f}")

    bursts = {
        name: arm.burst_us_per_call
        for name, arm in result.arms.items()
        if arm.burst_us_per_call
    }
    if bursts:
        modes_seen = sorted(
            {mode for ladders in bursts.values() for mode in ladders}
        )
        for mode in modes_seen:
            present = {
                name: ladders[mode]
                for name, ladders in bursts.items()
                if ladders.get(mode)
            }
            if not present:
                continue
            sizes = sorted(next(iter(present.values())), key=int)
            lines.extend(
                [
                    "",
                    f"burst dispatch diagnostic (us per call, {mode}):",
                    "  "
                    + f"{'arm':22s} "
                    + " ".join(f"{size:>9s}" for size in sizes),
                ]
            )
            for name, values in present.items():
                lines.append(
                    f"  {name:22s} "
                    + " ".join(f"{values[size]:9.2f}" for size in sizes)
                )
        lines.append(
            f"  The 'resid %' column above is the fall across the top two"
            f" rungs, and '!' marks it above {BURST_RESIDUAL_FLAG * 100:g}%."
        )
        lines.append(
            f"  A marked arm is still buying amortization at burst k="
            f"{result.burst_k}, so its ratio moves with k and is not a"
        )
        lines.append(
            "  kernel-speed claim. The test is one-sided: an unmarked arm is"
        )
        lines.append(
            "  NOT thereby device-bound. A ladder can plateau at a dispatch"
        )
        lines.append(
            "  cost that bursting cannot amortize, which is what the backward"
        )
        lines.append(
            "  modes here do. Only profiler-summed device time separates the"
        )
        lines.append(
            "  two, and nothing in this repo measures it. Treat a residual"
        )
        lines.append(
            "  within a few points of the threshold as undecided: it is a"
        )
        lines.append(
            "  difference of two medians and carries their noise."
        )

    lines.extend(["", "correctness:"])
    if not result.correctness:
        lines.append("  (no checks declared)")
    for row in result.correctness:
        if row.passed is None:
            verdict = "INFO"
        else:
            verdict = "PASS" if row.passed else "FAIL"
        limit = "-" if row.threshold is None else f"{row.threshold:g}"
        lines.append(
            f"  {verdict:4s} {row.arm:22s} {row.output:14s} vs "
            f"{row.reference:16s} {row.metric}={row.value:.4g} (limit {limit})"
        )
    lines.append(
        "  ALL CORRECTNESS PASSED"
        if result.all_correctness_passed
        else "  CORRECTNESS FAILURES PRESENT"
    )

    lines.extend(
        [
            "",
            "Each number is the per-call cost under back-to-back dispatch.",
            "It is not device time: where the host cannot keep the stream",
            "fed, the interval holds host stalls too. Run --burst and read",
            "the residual column; a flagged arm has a k-dependent ratio.",
            "The ratio is estimated once per replicate and",
            "bootstrapped across replicates, so the CI is the statistic to",
            "read; Welch, MWU and d describe the pooled sample distribution",
            "only, and their independence assumption is not met. These are",
            "isolated-kernel numbers on synthetic inputs; never present them",
            "as end-to-end training results.",
        ]
    )
    return "\n".join(lines)
