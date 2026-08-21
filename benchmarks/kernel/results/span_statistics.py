"""The span-versus-parts statistic, and why it is not the scenario one.

A **span** is an implementation that fuses across a scenario cut. Its claim is
the span against the **sum of the scenarios it replaces**, so one side of the
comparison is a sum over units and has no arm and no samples of its own.
``benchmarks.kernel.engine.statistics`` cannot answer that question, and this
module exists so that it does not have to try.

**It lives here rather than under ``engine/`` because it is parent-side work
about a parent-side object.** The engine measures one arm in one worker and
never sees a second unit, so a span is invisible to it: a worker is handed
``span.measurement``, which is a ``KernelScenario``. Everything that makes a
span a span -- the range, the parts, the second total, this statistic -- is
assembled by the parent in ``benchmarks.kernel.results.merge``, which is the
only caller here. Putting these functions under ``engine/`` would have made
"the engine gained no branch for spans" true of the import graph and false of
the directory.

Torch-free, like the merge it serves. It shares ``BOOTSTRAP_RESAMPLES`` and
``BOOTSTRAP_SEED`` with the scenario statistics rather than restating them, so
one run's two kinds of interval are drawn the same number of times from the
same seed.
"""

from __future__ import annotations

import statistics

import numpy as np

from benchmarks.kernel.engine.statistics import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
)


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
