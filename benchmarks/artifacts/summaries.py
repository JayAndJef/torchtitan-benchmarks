"""Sample summarization and the shared numeric column formatters.

Deliberately dependency-light: ``dataclasses`` and ``statistics`` only. Both
measurement systems summarize samples the same way and render them into the
same fixed-width columns, so these live below both rather than inside either
-- ``benchmarks.e2e.results`` and ``benchmarks.kernel.results.*`` each import
from here and neither imports from the other.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class SampleSummary:
    count: int
    mean_us: float
    median_us: float
    standard_deviation_us: float


def describe(values: list[float]) -> tuple[int, float, float, float]:
    """Return (n, mean, sd, median) for one sample."""
    count = len(values)
    if not count:
        raise ValueError("cannot summarize an empty sample")
    standard_deviation = statistics.stdev(values) if count > 1 else 0.0
    return (
        count,
        statistics.mean(values),
        standard_deviation,
        statistics.median(values),
    )


def summarize(values: list[float]) -> SampleSummary:
    count, mean, standard_deviation, median = describe(values)
    return SampleSummary(count, mean, median, standard_deviation)


def _value(value: float | None, width: int, precision: int = 1) -> str:
    if value is None:
        return f"{'n/a':>{width}s}"
    return f"{value:{width}.{precision}f}"


def _pvalue(value: float | None, width: int) -> str:
    """Significant-digit form; p-values span many orders of magnitude."""
    if value is None:
        return f"{'n/a':>{width}s}"
    return f"{value:{width}.3g}"
