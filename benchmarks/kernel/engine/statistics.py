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


def span_comparison(
    part_replicate_medians: list[list[float]],
    span_replicates: list[list[float]],
) -> dict[str, float | int | None]:
    """A span against the sum of the scenarios it replaces. **Unpaired.**

    A span is an implementation that fuses across a scenario cut, so its
    claim is not an arm against an arm. One side of it is a **sum over
    scenarios** and has no arm and no samples of its own.

    **The replicate index does not pair a span with its parts, so this
    estimator does not pretend it does.** Inside one scenario the sweep is
    block-major, so replicate ``r`` of two arms runs within seconds of
    itself and drift that moves a replicate cancels in the ratio -- which is
    the whole reason ``kernel_comparison`` pairs. Across units there is no
    such adjacency: ``benchmarks.kernel.runner`` runs each unit to
    completion before the next starts, so a six-scenario span's replicate 0
    and the first scenario's replicate 0 are separated by every worker in
    between. Index ``r`` on one side and index ``r`` on the other share
    nothing but the number. Any permutation of the parts' indices would be
    as justified as the identity, so a paired estimate here is a choice
    among equals dressed as a measurement.

    So the point estimate is a **ratio of two medians**, each taken over its
    own side, and the interval is an **unpaired bootstrap**: the span's
    per-replicate medians and each part's per-replicate medians are
    resampled independently. It is published as ``unpaired_ratio_ci_*``,
    never under the honest name, because it cancels no drift and a reader
    must not set it beside a scenario's interval as though it were the same
    statistic.

    Two further absences, each deliberate.

    * **Welch, Mann-Whitney and Cohen's d.** The parts side is a handful of
      per-replicate medians against the span's pooled bursts, which are two
      hundred. A two-sample test between them is a diagnostic of nothing.
      Their absence is by construction here, so no later edit restores them
      by forgetting to delete them.
    * **A per-replicate ratio spread.** It described the spread of a
      quantity that no longer exists. Each side's own spread is recoverable
      from the file: the span's samples are in ``arms`` and each part's
      per-replicate medians are in ``parts``.

    The sum is of each part's **median**, and never of samples. Sample ``i``
    of one scenario and sample ``i`` of the next are unrelated bursts from
    different sweeps.

    A ratio below 1.0 says the span costs less than the sum of the cuts it
    replaces. Read it with the dispatch-chain bias
    (``benchmarks.kernel.results.merge``): the parts side pays one host
    dispatch chain per scenario and the span pays one.
    """
    span_values = [
        statistics.median(replicate)
        for replicate in span_replicates
        if replicate
    ]
    part_values = [values for values in part_replicate_medians if values]
    if not span_values or len(part_values) != len(part_replicate_medians):
        return {
            "n_replicates": 0,
            "span_median_us": None,
            "parts_median_us": None,
            "part_medians_us": [],
            "median_ratio": None,
            "unpaired_ratio_ci_low": None,
            "unpaired_ratio_ci_high": None,
        }
    span_median = statistics.median(span_values)
    part_medians = [statistics.median(values) for values in part_values]
    parts_median = sum(part_medians)
    return {
        "n_replicates": len(span_values),
        # Both sides read the same way: the median over replicates of that
        # replicate's own median. The parts side has no pooled samples to
        # take a pooled median over, so a pooled median on the span side
        # would compare two different estimators.
        "span_median_us": span_median,
        "parts_median_us": parts_median,
        # Every term of the sum, in the order ``parts`` names them, so the
        # total is auditable from the row itself.
        "part_medians_us": part_medians,
        "median_ratio": (
            span_median / parts_median if parts_median else None
        ),
        **_unpaired_ratio_ci(span_values, part_values),
    }


def _unpaired_ratio_ci(
    span_values: list[float],
    part_values: list[list[float]],
    confidence: float = 0.95,
) -> dict[str, float | None]:
    """Percentile bootstrap on span median / sum of part medians.

    Each side is resampled **independently**, because nothing pairs them.
    That is the difference from ``bootstrap_ratio_ci``, which resamples one
    list of per-replicate log-ratios and therefore assumes the pairing that
    produced them.

    The seed is fixed, so a results file is reproducible from its own raw
    samples.
    """
    if len(span_values) < 2 or any(len(v) < 2 for v in part_values):
        return {
            "unpaired_ratio_ci_low": None,
            "unpaired_ratio_ci_high": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    span = np.asarray(span_values, dtype=float)
    numerator = np.median(
        rng.choice(span, size=(BOOTSTRAP_RESAMPLES, span.size), replace=True),
        axis=1,
    )
    denominator = np.zeros(BOOTSTRAP_RESAMPLES, dtype=float)
    for values in part_values:
        part = np.asarray(values, dtype=float)
        denominator += np.median(
            rng.choice(
                part, size=(BOOTSTRAP_RESAMPLES, part.size), replace=True
            ),
            axis=1,
        )
    ratios = numerator[denominator > 0] / denominator[denominator > 0]
    if ratios.size == 0:
        return {
            "unpaired_ratio_ci_low": None,
            "unpaired_ratio_ci_high": None,
        }
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(ratios, [tail, 1.0 - tail])
    return {
        "unpaired_ratio_ci_low": float(low),
        "unpaired_ratio_ci_high": float(high),
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
