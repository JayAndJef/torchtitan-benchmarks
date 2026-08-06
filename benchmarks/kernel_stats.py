"""Statistics for interleaved kernel-benchmark samples.

Unlike the e2e span diagnostics, interleaved kernel cycles are legitimate
paired repeats: every cycle runs all arms back to back, so clock and thermal
drift hit each arm equally and the per-cycle deltas are exchangeable. The
p-values here are therefore inferential for the run they came from.
"""

from __future__ import annotations

import statistics

from scipy import stats as scipy_stats

from benchmarks.metrics import describe


KERNEL_SIGNIFICANCE_METHODOLOGY = {
    "interpretation": "paired_interleaved_repeats",
    "sample_unit": "interleaved_timing_cycle",
    "independence_assumption_met": True,
    "note": (
        "Arms are timed round-robin within each cycle, so drift affects all "
        "arms equally and Wilcoxon operates on paired per-cycle deltas. "
        "These p-values are inferential for this run on this hardware; they "
        "say nothing about other shapes, GPUs, or torch versions."
    ),
}


def kernel_comparison(
    base_us: list[float], arm_us: list[float]
) -> dict[str, float | int | None]:
    """Compare one arm's samples against its opponent's."""
    n_base, mean_base, sd_base, median_base = describe(base_us)
    n_arm, mean_arm, sd_arm, median_arm = describe(arm_us)
    welch = scipy_stats.ttest_ind(base_us, arm_us, equal_var=False)
    mann_whitney = scipy_stats.mannwhitneyu(
        base_us, arm_us, alternative="two-sided"
    )
    wilcoxon_p = None
    if n_base == n_arm:
        deltas = [arm - base for base, arm in zip(base_us, arm_us)]
        if any(deltas):
            wilcoxon_p = float(
                scipy_stats.wilcoxon(deltas, alternative="two-sided").pvalue
            )
    pooled_sd = (
        (
            ((n_base - 1) * sd_base**2 + (n_arm - 1) * sd_arm**2)
            / (n_base + n_arm - 2)
        )
        ** 0.5
        if n_base + n_arm > 2
        else 0.0
    )
    faster = (
        statistics.mean(
            arm < base for base, arm in zip(base_us, arm_us)
        )
        if n_base == n_arm
        else None
    )
    return {
        "n_base": n_base,
        "n_arm": n_arm,
        "base_median_us": median_base,
        "arm_median_us": median_arm,
        "base_mean_us": mean_base,
        "arm_mean_us": mean_arm,
        "median_ratio": median_arm / median_base if median_base else None,
        "mean_ratio": mean_arm / mean_base if mean_base else None,
        "welch_p": float(welch.pvalue),
        "mwu_p": float(mann_whitney.pvalue),
        "wilcoxon_p": wilcoxon_p,
        "cohens_d": (mean_arm - mean_base) / pooled_sd if pooled_sd else 0.0,
        "arm_faster_fraction": faster,
    }
