"""Statistics for replicated kernel-benchmark samples.

The unit of repetition is the **replicate**, not the sample. One replicate
sweeps every arm of a scenario once, in declaration order, and the sweep is
repeated. Within a replicate an arm's samples are consecutive bursts of the
same closure, so they share that arm's cache and clock state and are not
independent of each other.

That distinction drives everything here:

* **The headline is a bootstrap CI on per-replicate log-ratios.** One ratio
  per replicate, from the two arms' medians inside it. Drift that moves a
  whole replicate cancels in the ratio, which is the property the old
  round-robin got from adjacency and this design gets from the sweep.
* **Welch and Mann-Whitney stay, and are labelled as what they now are.**
  They run on the pooled samples, where the independence assumption is
  **not** met -- consecutive bursts of one closure are correlated. They are
  within-run distribution diagnostics. Do not report them as evidence that
  one kernel is faster than another.
* **Wilcoxon is gone.** It needed the per-cycle pairing the round-robin
  provided. At the replicate level it cannot work either: the exact
  two-sided minimum p at n=5 is 0.0625, so it can never reject at any
  conventional level, and printing it would imply a test that cannot produce
  a result.
"""

from __future__ import annotations

import math
import statistics

import numpy as np
from scipy import stats as scipy_stats

from benchmarks.artifacts.summaries import describe


BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

KERNEL_SIGNIFICANCE_METHODOLOGY = {
    "interpretation": "replicated_sweeps",
    "sample_unit": "burst_amortized_call",
    "repetition_unit": "replicate",
    "independence_assumption_met": False,
    "primary_statistic": "bootstrap_ci_on_per_replicate_log_ratios",
    "note": (
        "One replicate sweeps every arm once, in declaration order, and the "
        "sweep is repeated. The ratio is estimated per replicate and "
        "bootstrapped across replicates, so drift that moves a whole "
        "replicate cancels. Welch and Mann-Whitney run on the pooled samples "
        "and are within-run distribution diagnostics only: consecutive "
        "bursts of one closure are correlated, so their independence "
        "assumption is not met. These numbers are inferential for this run "
        "on this hardware; they say nothing about other shapes, GPUs, or "
        "torch versions."
    ),
}


def _flatten(replicates: list[list[float]]) -> list[float]:
    return [value for replicate in replicates for value in replicate]


def _log_ratios(
    base: list[list[float]], arm: list[list[float]]
) -> list[float]:
    """One log-ratio per replicate the two arms both have."""
    ratios = []
    for base_replicate, arm_replicate in zip(base, arm):
        if not base_replicate or not arm_replicate:
            continue
        base_median = statistics.median(base_replicate)
        arm_median = statistics.median(arm_replicate)
        if base_median > 0 and arm_median > 0:
            ratios.append(math.log(arm_median / base_median))
    return ratios


def bootstrap_ratio_ci(
    log_ratios: list[float], confidence: float = 0.95
) -> dict[str, float | int | None]:
    """Percentile bootstrap over per-replicate log-ratios.

    Reported on the ratio scale. The seed is fixed so a results file is
    reproducible from its own raw samples.
    """
    n = len(log_ratios)
    if n == 0:
        return {
            "replicates": 0,
            "ratio": None,
            "ratio_ci_low": None,
            "ratio_ci_high": None,
            "replicate_ratio_spread": None,
        }
    point = math.exp(statistics.median(log_ratios))
    spread = (
        math.exp(max(log_ratios)) - math.exp(min(log_ratios)) if n > 1 else 0.0
    )
    if n < 2:
        return {
            "replicates": n,
            "ratio": point,
            "ratio_ci_low": None,
            "ratio_ci_high": None,
            "replicate_ratio_spread": spread,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(log_ratios, dtype=float)
    draws = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, n), replace=True)
    medians = np.median(draws, axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(medians, [tail, 1.0 - tail])
    return {
        "replicates": n,
        "ratio": point,
        "ratio_ci_low": float(np.exp(low)),
        "ratio_ci_high": float(np.exp(high)),
        "replicate_ratio_spread": spread,
    }


def kernel_comparison(
    base_replicates: list[list[float]], arm_replicates: list[list[float]]
) -> dict[str, float | int | None]:
    """Compare one arm's replicated samples against its opponent's."""
    base_us = _flatten(base_replicates)
    arm_us = _flatten(arm_replicates)
    n_base, mean_base, sd_base, median_base = describe(base_us)
    n_arm, mean_arm, sd_arm, median_arm = describe(arm_us)
    welch = scipy_stats.ttest_ind(base_us, arm_us, equal_var=False)
    mann_whitney = scipy_stats.mannwhitneyu(
        base_us, arm_us, alternative="two-sided"
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
    return {
        "n_base": n_base,
        "n_arm": n_arm,
        "base_median_us": median_base,
        "arm_median_us": median_arm,
        "base_mean_us": mean_base,
        "arm_mean_us": mean_arm,
        "median_ratio": median_arm / median_base if median_base else None,
        "mean_ratio": mean_arm / mean_base if mean_base else None,
        **bootstrap_ratio_ci(_log_ratios(base_replicates, arm_replicates)),
        "welch_p": float(welch.pvalue),
        "mwu_p": float(mann_whitney.pvalue),
        "cohens_d": (mean_arm - mean_base) / pooled_sd if pooled_sd else 0.0,
    }
