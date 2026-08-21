"""Human-readable rendering of kernel-isolation results."""

from __future__ import annotations

import textwrap

from benchmarks.artifacts.summaries import _pvalue, _value
from benchmarks.kernel.results.merge import BURST_RESIDUAL_FLAG
from benchmarks.kernel.results.schema import (
    KernelScenarioResult,
    KernelSpanResult,
)
from benchmarks.kernel.schema import MODES

# The arm-name column. Widened from 22 when the MoE scenarios arrived: the
# longest declared name is `mcore/no_bias_activation_fusion` at 31. One
# 28-character name, `mcore/no_bias_dropout_fusion`, already overflowed the
# old width, so rows holding it already pushed the rest of their line right.
# One constant rather than seven literals, so the next long name moves the
# column once. `benchmarks/kernel/registry.py` is the authority on the names;
# a test pins the width against it.
ARM_FIELD = 31


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


def _shape_lines(shapes: dict) -> list[str]:
    """The derived shapes, one line for a scenario and one per cut for a span.

    A span's ``shapes`` is keyed by enclosed scenario, because a span's inputs
    are the first cut's and its outputs are the last cut's, so no single
    entry describes it.
    """
    if shapes and all(isinstance(value, dict) for value in shapes.values()):
        return [
            "shapes ("
            + name
            + "): "
            + "  ".join(f"{key}={value}" for key, value in entry.items())
            for name, entry in shapes.items()
        ]
    return [
        "shapes: "
        + "  ".join(f"{name}={value}" for name, value in shapes.items())
    ]


def _common_lines(
    result: KernelScenarioResult | KernelSpanResult, headline: str
) -> list[str]:
    """Everything both kinds of results file print, in one order.

    A span's arms are arms, so its per-mode tables, its memory table, its
    burst ladder and its gates render exactly as a scenario's do. Only the
    headline above them and the parts table below them differ.
    """
    lines = [
        headline,
        *_shape_lines(result.shapes),
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
                f"  {arm.status.upper()} {arm.name:{ARM_FIELD}s} {arm.status_reason}"
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
                + f"{'arm':{ARM_FIELD}s} {'median us':>10s} {'sd':>8s} {'vs':>18s} "
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
                f"  {name:{ARM_FIELD}s} "
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
            lines.append(f"  {name:{ARM_FIELD}s} {peak:8.3f}")

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
                    + f"{'arm':{ARM_FIELD}s} "
                    + " ".join(f"{size:>9s}" for size in sizes),
                ]
            )
            for name, values in present.items():
                lines.append(
                    f"  {name:{ARM_FIELD}s} "
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
            f"  {verdict:4s} {row.arm:{ARM_FIELD}s} {row.output:14s} vs "
            f"{row.reference:16s} {row.metric}={row.value:.4g} (limit {limit})"
        )
    lines.append(
        "  ALL CORRECTNESS PASSED"
        if result.all_correctness_passed
        else "  CORRECTNESS FAILURES PRESENT"
    )

    return lines


def render_kernel_results(result: KernelScenarioResult) -> str:
    """Render one kernel scenario: per-mode timings and correctness gates."""
    lines = _common_lines(
        result,
        f"== kernel scenario: {result.scenario}   "
        f"model size: {result.model_size}   hardware: {result.hardware} ==",
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


def _parts_ci(row: dict) -> str:
    """The span-versus-parts interval, always marked.

    ``~`` says the same thing it says on a within-process interval: the
    number under it was not taken under the conditions the honest name
    promises. Here the span and its parts were measured in separate sweeps of
    one run, so replicate ``r`` of each shares an index but not a moment.
    There is no unmarked spelling of this column, because there is no
    condition under which a span and its parts are measured adjacently.
    """
    low = row.get("cross_sweep_ratio_ci_low")
    high = row.get("cross_sweep_ratio_ci_high")
    if low is None or high is None:
        return "-"
    return f"~[{low:.4f},{high:.4f}]"


def render_kernel_span_results(result: KernelSpanResult) -> str:
    """Render one span: its own timings, and the sum of what it replaces.

    The arm tables above are the span's **own** measurement. The parts table
    below is the other total, and the two are never printed in one table: a
    reader scanning a per-mode row must not be able to take a summed number
    for a measured one.

    The caption under the parts table states the dispatch bias, and it is
    printed on every span, unconditionally. The bias is systematic, it runs
    in one direction, and that direction is the one a span is written to
    support -- so a reader must meet it beside the number and not only in
    ``results.json``.
    """
    lines = _common_lines(
        result,
        f"== kernel span: {result.span}   "
        f"replaces: {' + '.join(result.scenarios)}   "
        f"model size: {result.model_size}   hardware: {result.hardware} ==",
    )

    lines.extend(
        [
            "",
            "against the sum of the scenarios it replaces:",
            "  "
            + f"{'arm':{ARM_FIELD}s} {'mode':>17s} {'span us':>10s} "
            + f"{'parts us':>10s} {'ratio':>7s} {'95% CI':>18s}  parts",
        ]
    )
    for row in result.parts_comparisons:
        lines.append(
            f"  {row['arm']:{ARM_FIELD}s} "
            f"{row['mode']:>17s} "
            f"{_value(row.get('span_median_us'), 10, 2)} "
            f"{_value(row.get('parts_median_us'), 10, 2)} "
            f"{_value(row.get('median_ratio'), 7, 4)} "
            f"{_parts_ci(row):>18s}  "
            + " + ".join(row.get("parts", ()))
        )
    if not result.parts_comparisons:
        lines.append("  (no arm carries a parts total)")

    lines.extend(
        [
            "",
            "The 'span us' column is measured. The 'parts us' column is a",
            "SUM: per replicate it adds each part arm's median in that",
            "replicate, and the column is the median of those sums.",
            "",
            f"BIAS, and it favours the span. The parts total pays one host",
            f"dispatch chain per enclosed scenario -- "
            f"{len(result.scenarios)} of them here -- and the",
            "span pays one. Roughly 85% of a kernel number in this repo is",
            "host dispatch, not device time, so the parts total carries",
            f"{len(result.scenarios) - 1} extra chain(s) that no fusion removed. "
            "The ratio is",
            "therefore SMALLER than fusion alone would make it, and the",
            "effect grows with the length of the range. It is not corrected",
            "for: nothing here measures the device time that would separate",
            "the two. Read a ratio below 1.0 as fusion PLUS the dispatch",
            "chains the harness stopped paying, never as fusion alone.",
            "",
            "The two sides were measured in separate sweeps of one run, so",
            "replicate r of each shares an index but not a moment: drift",
            "between the sweeps lands in the ratio instead of cancelling.",
            "The interval is marked '~' for that reason and is published as",
            "cross_sweep_ratio_ci_* in results.json, never under the honest",
            "name. No Welch, MWU or d is computed against a sum.",
            "",
            "Each number is the per-call cost under back-to-back dispatch.",
            "It is not device time: where the host cannot keep the stream",
            "fed, the interval holds host stalls too. Run --burst and read",
            "the residual column; a flagged arm has a k-dependent ratio.",
            "These are isolated-kernel numbers on synthetic inputs; never",
            "present them as end-to-end training results.",
        ]
    )
    return "\n".join(lines)
