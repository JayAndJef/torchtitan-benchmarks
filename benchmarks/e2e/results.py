"""The end-to-end evaluation: the figures that the step lines give, ``results.json`` and the printed report."""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.artifacts.layout import atomic_write_json, logs_by_rank
from benchmarks.artifacts.manifests import ArmRecord, load_run_record
from benchmarks.artifacts.summaries import _value
from benchmarks.e2e.checks import run_warnings
from benchmarks.e2e.engines.api import ProfileWindow
from benchmarks.e2e.evidence import (
    GRAD_NORM_METRIC,
    LOSS_METRIC,
    non_finite_refusals,
    trajectory,
)
from benchmarks.execution.affinity import is_pinned


STEP_METRICS = re.compile(
    r"step:\s*(\d+).*?memory:\s*([0-9.]+)GiB.*?tps:\s*([0-9,]+)"
)
@dataclass(frozen=True)
class StepMs:
    """The step cost of one rank, in milliseconds; ``p95`` is a nearest-rank value, so it is always a measured step."""

    mean: float | None
    median: float | None
    p95: float | None
    series: tuple[float, ...]


@dataclass(frozen=True)
class RankThroughput:
    """One rank's own figures, before any reduction across ranks."""

    rank: int
    stable_tokens_per_second: float | None
    stable_sample_count: int
    step_ms: StepMs


@dataclass(frozen=True)
class ArmResult:
    """One arm's published figures: the slowest rank's throughput and step cost, and the peak memory of every rank."""

    stable_tokens_per_second: float | None
    stable_sample_count: int
    peak_memory_gib: float | None
    step_ms: StepMs
    rank_reduction: str
    published_rank: int
    per_rank: tuple[RankThroughput, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Complete machine-readable result for one benchmark output directory."""

    output_dir: str
    scenario: str
    hardware: str
    arms: tuple[str, ...]
    results: dict[str, ArmResult]
    losses: dict[str, list[tuple[int, float]]]
    gradient_norms: dict[str, list[tuple[int, float]]]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": 6,
            "scenario": self.scenario,
            "hardware": self.hardware,
            "output_dir": self.output_dir,
            "arms": list(self.arms),
            "results": {
                arm: asdict(summary) for arm, summary in self.results.items()
            },
            "losses": {
                arm: [{"step": step, "value": value} for step, value in values]
                for arm, values in self.losses.items()
            },
            "gradient_norms": {
                arm: [{"step": step, "value": value} for step, value in values]
                for arm, values in self.gradient_norms.items()
            },
            "warnings": list(self.warnings),
        }
        return _json_safe(value)


def _json_safe(value: Any) -> Any:
    """``value`` as strict JSON: a non-finite float becomes null and a tuple becomes a list."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _log_by_rank(log_path: Path) -> dict[int, str]:
    """This arm's log, split into what each rank wrote. Empty when missing."""
    if not log_path.exists():
        return {}
    return logs_by_rank(log_path.read_text(errors="replace"))


def loss_visible_rank(*, world_size: int, pp: int) -> int:
    """The rank whose step line carries the run's real loss, for the ``1F1B`` and ``Interleaved1F1B`` schedules."""
    return (world_size // pp) * (pp - 1)


def losses(log_path: Path, *, rank: int = 0) -> list[tuple[int, float]]:
    return trajectory(_log_by_rank(log_path).get(rank, ""), LOSS_METRIC)


def refuse_non_finite_trajectories(arm: str, log_path: Path) -> None:
    """Raise ``ValueError`` when any rank's step lines carry a ``nan`` or an ``inf``."""
    refusals = non_finite_refusals(_log_by_rank(log_path))
    if refusals:
        raise ValueError(
            f"{arm}: {refusals[0]}; a run that diverged cannot publish a "
            f"throughput, so no results.json is written for it (see {log_path})"
        )


def grad_norms(log_path: Path, *, rank: int = 0) -> list[tuple[int, float]]:
    return trajectory(_log_by_rank(log_path).get(rank, ""), GRAD_NORM_METRIC)


def _rows(text: str) -> list[tuple[int, float, int]]:
    rows = []
    for line in text.splitlines():
        match = STEP_METRICS.search(line)
        if match:
            rows.append(
                (
                    int(match.group(1)),
                    float(match.group(2)),
                    int(match.group(3).replace(",", "")),
                )
            )
    return rows


def per_rank_training_metrics(
    log_path: Path,
) -> dict[int, list[tuple[int, float, int]]]:
    """The (step, peak memory GiB, tokens/s) rows of each rank."""
    return {
        rank: _rows(text) for rank, text in _log_by_rank(log_path).items()
    }


def training_metrics(log_path: Path) -> list[tuple[int, float, int]]:
    """Every rank's rows, pooled in rank order; a memory input and not a throughput input."""
    return [
        row
        for _, rows in sorted(per_rank_training_metrics(log_path).items())
        for row in rows
    ]


def stable_tps(
    rows: list[tuple[int, float, int]], window: ProfileWindow
) -> list[int]:
    """The tokens/s samples of a profiled run: the steps of each profiler cycle that carry no profiler cost."""
    wait = window.freq - window.warmup - window.active
    return [
        tps
        for step, _, tps in rows
        if 2 <= ((step - 1) % window.freq) + 1 <= wait
    ]


def measured_tps(
    rows: list[tuple[int, float, int]], warmup_steps: int
) -> list[int]:
    """The tokens/s samples of an unprofiled run: every step after the warmup."""
    return [tps for step, _, tps in rows if step > warmup_steps]


def _slowest_rank(per_rank: dict[int, float | None]) -> int:
    """The rank with the lowest throughput; a tie goes to the lower rank, and a rank with no sample sorts last."""
    if not per_rank:
        return 0
    measured = {
        rank: value for rank, value in per_rank.items() if value is not None
    }
    if not measured:
        return min(per_rank)
    return min(measured, key=lambda rank: (measured[rank], rank))


def _throughput_spread(per_rank: dict[int, float | None]) -> float | None:
    """max/min over the ranks that reported, or None below two of them."""
    values = [value for value in per_rank.values() if value]
    if len(values) < 2:
        return None
    return max(values) / min(values)


def _rank_throughput_summary(per_rank: dict[int, float | None]) -> str:
    return ", ".join(
        f"rank {rank} {value:,.0f}" if value else f"rank {rank} none"
        for rank, value in sorted(per_rank.items())
    )


def _nearest_rank_percentile(values: list[float], fraction: float) -> float:
    """The nearest-rank percentile: a measured value, never an interpolation."""
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered))
    return ordered[max(index, 1) - 1]


def step_ms(
    samples: list[int], *, tokens_per_step: int, pp: int
) -> StepMs:
    """The step costs of one rank's tokens/s samples; a sample of zero has no cost and is dropped."""
    series = tuple(
        1000.0 * tokens_per_step / (tps * pp) for tps in samples if tps > 0
    )
    if not series:
        return StepMs(mean=None, median=None, p95=None, series=())
    return StepMs(
        mean=statistics.fmean(series),
        median=statistics.median(series),
        p95=_nearest_rank_percentile(list(series), 0.95),
        series=series,
    )


def pinning_warnings(records: list[ArmRecord]) -> list[str]:
    """A warning when some arms run pinned and others run unpinned."""
    pinned = [record for record in records if is_pinned(record.cpu_pinning)]
    unpinned = [record for record in records if not is_pinned(record.cpu_pinning)]
    if not pinned or not unpinned:
        return []
    return [
        "the arms mix CPU pinning, so their numbers are not comparable: "
        + "; ".join(
            f"{record.arm.name}: {record.cpu_pinning}" for record in records
        )
    ]


def evaluate_run(
    out_dir: Path, arms_override: list[str] | tuple[str, ...] | None = None
) -> EvaluationResult:
    """Evaluate the end-to-end metrics of the selected arms."""
    out_dir = out_dir.resolve()
    record = load_run_record(out_dir)
    run = record.run
    arms = list(arms_override or [each.arm.name for each in record.arms])
    selected = [record.arm(name) for name in arms]
    warnings = [
        *run_warnings(run, tuple(each.arm for each in selected)),
        *pinning_warnings(selected),
    ]
    profile = run.profile
    world_size = run.parallelism.world_size
    raw_training = {
        arm: per_rank_training_metrics(out_dir / f"{arm}.log") for arm in arms
    }
    # The sample rule follows the axis the run was measured under.
    if profile:
        def _samples(rows: list[tuple[int, float, int]]) -> list[int]:
            return stable_tps(rows, run.window)
    else:
        warmup_steps = run.warmup_steps

        def _samples(rows: list[tuple[int, float, int]]) -> list[int]:
            return measured_tps(rows, warmup_steps)

    stable_samples = {
        arm: {rank: _samples(rows) for rank, rows in by_rank.items()}
        for arm, by_rank in raw_training.items()
    }
    throughput = {
        arm: {
            rank: statistics.median(samples) if samples else None
            for rank, samples in by_rank.items()
        }
        for arm, by_rank in stable_samples.items()
    }
    published_throughput_rank = {
        arm: _slowest_rank(by_rank) for arm, by_rank in throughput.items()
    }
    # One step's tokens, and the degree that divides its cost.
    tokens_per_step = run.data.local_batch_size * run.data.seq_len
    pp = run.parallelism.pp
    rank_step_ms = {
        arm: {
            rank: step_ms(samples, tokens_per_step=tokens_per_step, pp=pp)
            for rank, samples in by_rank.items()
        }
        for arm, by_rank in stable_samples.items()
    }
    results = {}
    for arm in arms:
        rank = published_throughput_rank[arm]
        median_tps = throughput[arm].get(rank)
        peak_memory = max(
            (
                memory
                for rows in raw_training[arm].values()
                for _, memory, _ in rows
            ),
            default=None,
        )
        results[arm] = ArmResult(
            stable_tokens_per_second=median_tps,
            stable_sample_count=len(stable_samples[arm].get(rank, ())),
            peak_memory_gib=peak_memory,
            step_ms=rank_step_ms[arm].get(
                rank, StepMs(mean=None, median=None, p95=None, series=())
            ),
            rank_reduction="min_over_ranks",
            published_rank=rank,
            per_rank=tuple(
                RankThroughput(
                    rank=each,
                    stable_tokens_per_second=throughput[arm][each],
                    stable_sample_count=len(stable_samples[arm][each]),
                    step_ms=rank_step_ms[arm][each],
                )
                for each in sorted(throughput[arm])
            ),
        )
        spread = _throughput_spread(throughput[arm])
        if spread is not None and spread > 1.15:
            warnings.append(
                f"{arm}: tokens/s varies {spread:.2f}x across ranks "
                f"({_rank_throughput_summary(throughput[arm])}); a schedule "
                "holds the ranks in step, so a spread this wide means one "
                "rank is starved or the ranks are not running one job"
            )

    # One rank's trajectory: under a pipeline split only one rank holds it.
    trajectory_rank = loss_visible_rank(world_size=world_size, pp=pp)
    # Refuse a diverged rank before anything is published.
    for arm in arms:
        refuse_non_finite_trajectories(arm, out_dir / f"{arm}.log")
    return EvaluationResult(
        output_dir=str(out_dir),
        scenario=record.scenario,
        hardware=record.hardware,
        arms=tuple(arms),
        results=results,
        losses={
            arm: losses(out_dir / f"{arm}.log", rank=trajectory_rank)
            for arm in arms
        },
        gradient_norms={
            arm: grad_norms(out_dir / f"{arm}.log", rank=trajectory_rank)
            for arm in arms
        },
        warnings=tuple(warnings),
    )


def write_results(result: EvaluationResult, path: Path | None = None) -> Path:
    destination = path or Path(result.output_dir) / "results.json"
    atomic_write_json(destination, result.to_dict())
    return destination


def _render_trajectory(values: list[tuple[int, float]], nonfinite_label: str) -> str:
    if not values:
        return "(no log)"
    picks = [values[0]] + [values[i] for i in (9, 19, 29, 39) if i < len(values)]
    rendered = "  ".join(f"s{step}:{value:.5f}" for step, value in picks)
    # Only a result that evaluate_run did not build reaches this label.
    if not all(math.isfinite(value) for _, value in values):
        rendered += f"   NON-FINITE {nonfinite_label}"
    return rendered


def render_evaluation(result: EvaluationResult) -> str:
    """The printed report of one evaluation: one row of absolute figures per arm, the trajectories and the warnings."""
    lines = [
        f"== {result.output_dir} ==",
        f"scenario: {result.scenario}   hardware: {result.hardware}",
        "",
        "benchmark summary:",
        "  "
        + f"{'arm':22s} {'tokens/s':>12s} {'n':>4s} "
        + f"{'step ms':>9s} {'p95 ms':>9s} {'peak GiB':>9s}",
    ]
    for arm in result.arms:
        summary = result.results[arm]
        lines.append(
            f"  {arm:22s} "
            f"{_value(summary.stable_tokens_per_second, 12)} "
            f"{summary.stable_sample_count:4d} "
            f"{_value(summary.step_ms.median, 9, 2)} "
            f"{_value(summary.step_ms.p95, 9, 2)} "
            f"{_value(summary.peak_memory_gib, 9, 2)}"
        )
    lines.append(
        "tokens/s is per device, taken at the slowest rank; 'step ms' is "
        "that rank's median step."
    )

    lines.extend(["", "loss trajectories (sanity check, not a measurement):"])
    for arm in result.arms:
        lines.append(f"  {arm:22s} {_render_trajectory(result.losses[arm], 'LOSS')}")
    lines.extend(
        ["", "gradient norm trajectories (sanity check, not a measurement):"]
    )
    for arm in result.arms:
        lines.append(
            f"  {arm:22s} "
            f"{_render_trajectory(result.gradient_norms[arm], 'GRAD NORM')}"
        )

    if result.warnings:
        lines.append("")
        lines.extend(f"WARNING: {warning}" for warning in result.warnings)

    return "\n".join(lines)
