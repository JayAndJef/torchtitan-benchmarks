"""Human-readable rendering of kernel-isolation results."""

from __future__ import annotations

from benchmarks.artifacts.summaries import _pvalue, _value
from benchmarks.kernel.results.merge import BURST_RESIDUAL_FLAG
from benchmarks.kernel.results.schema import KernelScenarioResult
from benchmarks.kernel.schema import MODES


def _ratio_ci(row: dict | None) -> str:
    """The bootstrap interval on per-replicate log-ratios, as ``[lo, hi]``."""
    if not row:
        return "-"
    low, high = row.get("ratio_ci_low"), row.get("ratio_ci_high")
    if low is None or high is None:
        return "-"
    return f"[{low:.4f},{high:.4f}]"


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
            f"  Per-call time still falling at the top means burst k="
            f"{result.burst_k} is too small for that arm. The 'resid %'"
        )
        lines.append(
            f"  column above is that fall, and '!' marks it above "
            f"{BURST_RESIDUAL_FLAG * 100:g}%: those rows compare dispatch"
        )
        lines.append(
            "  cost, not kernel speed, and their ratios move with burst k."
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
