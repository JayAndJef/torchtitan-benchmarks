"""The sample summary and the fixed-width number columns that both measurement systems print."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class SampleSummary:
    """The count, the mean, the median and the standard deviation of one sample, in microseconds."""

    count: int
    mean_us: float
    median_us: float
    standard_deviation_us: float


def describe(values: list[float]) -> tuple[int, float, float, float]:
    """The count, the mean, the standard deviation and the median of one sample."""
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
    """The summary of one sample."""
    count, mean, standard_deviation, median = describe(values)
    return SampleSummary(count, mean, median, standard_deviation)


def _value(value: float | None, width: int, precision: int = 1) -> str:
    """``value`` in a fixed-width column; ``None`` prints as ``n/a``."""
    if value is None:
        return f"{'n/a':>{width}s}"
    return f"{value:{width}.{precision}f}"


def _pvalue(value: float | None, width: int) -> str:
    """``value`` in three significant digits, because a p-value spans many orders of magnitude."""
    if value is None:
        return f"{'n/a':>{width}s}"
    return f"{value:{width}.3g}"
