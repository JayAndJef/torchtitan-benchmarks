"""The end-to-end evaluation: the figures of the step samples, ``results.json`` and the printed report."""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.artifacts.layout import atomic_write_json, logs_by_rank
from benchmarks.artifacts.manifests import ArmRecord, load_run_record
from benchmarks.artifacts.summaries import _value
from benchmarks.e2e.checks import run_warnings
from benchmarks.e2e.engines.api import Arm, ProfileWindow, StepRead, StepSample
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.e2e.evidence import non_finite_refusals, rank_steps
from benchmarks.execution.affinity import is_pinned
from benchmarks.execution.launcher import RANK_PREFIX


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
    extras: dict[str, dict[str, float]]
    """The engine's other figures, each the median over the published rank's samples, under the engine's name."""


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


RANK_LINE = re.compile(rf"^{RANK_PREFIX}")
"""A log line that starts with a rank prefix."""


def arm_steps(arm: Arm, log_path: Path) -> dict[int, StepRead]:
    """What the arm's engine reads from each rank in the arm's log; a missing log holds no rank."""
    if not log_path.exists():
        return {}
    return rank_steps(
        engine_for(arm), logs_by_rank(log_path.read_text(errors="replace"))
    )


def _log_line(text: str, rank: int, line: int) -> int:
    """The line number in the whole log of line ``line`` of the rank's text, as ``logs_by_rank`` splits it."""
    lines = text.replace("\x00", "").splitlines()
    prefixed = [
        (number, int(re.search(r"\d+", match.group()).group()))
        for number, content in enumerate(lines, start=1)
        if (match := RANK_LINE.match(content))
    ]
    if len({owner for _, owner in prefixed}) < 2:
        return line
    return [number for number, owner in prefixed if owner == rank][line - 1]


def dropped_line_warnings(
    arm: str, log_path: Path, reads: Mapping[int, StepRead]
) -> list[str]:
    """One warning for each step line that the engine dropped, with its line in the log."""
    dropped = [line for read in reads.values() for line in read.dropped]
    if not dropped:
        return []
    text = log_path.read_text(errors="replace")
    return [
        f"{arm}: rank {line.rank} step "
        f"{'unknown' if line.step is None else line.step}: a rank prefix "
        f"cut the step line at line {_log_line(text, line.rank, line.line)} "
        f"of {log_path.name}, so the evaluation drops that step"
        for line in dropped
    ]


def loss_visible_rank(*, world_size: int, pp: int) -> int:
    """The rank whose step line carries the run's real loss, for the ``1F1B`` and ``Interleaved1F1B`` schedules."""
    return (world_size // pp) * (pp - 1)


def trajectory(samples: Sequence[StepSample], metric: str) -> list[tuple[int, float]]:
    """The (step, value) pairs of ``metric``, which is ``loss`` or ``grad_norm``; a step without the value is absent."""
    return [
        (sample.step, value)
        for sample in samples
        if (value := getattr(sample, metric)) is not None
    ]


def refuse_non_finite_trajectories(
    arm: str, steps: Mapping[int, Sequence[StepSample]], log_path: Path
) -> None:
    """Raise ``ValueError`` when the loss or the gradient norm of any rank is a ``nan`` or an ``inf``."""
    refusals = non_finite_refusals(steps)
    if refusals:
        raise ValueError(
            f"{arm}: {refusals[0]}; a run that diverged cannot publish a "
            f"throughput, so no results.json is written for it (see {log_path})"
        )


def stable_samples(
    samples: Sequence[StepSample], window: ProfileWindow
) -> list[StepSample]:
    """The samples of a profiled run: the steps of each profiler cycle that carry no profiler cost."""
    wait = window.freq - window.warmup - window.active
    return [
        sample
        for sample in samples
        if 2 <= ((sample.step - 1) % window.freq) + 1 <= wait
    ]


def measured_samples(
    samples: Sequence[StepSample], warmup_steps: int
) -> list[StepSample]:
    """The samples of an unprofiled run: every step after the warmup."""
    return [sample for sample in samples if sample.step > warmup_steps]


def extras_medians(samples: Sequence[StepSample]) -> dict[str, float]:
    """The median of each extra figure over the samples that state it."""
    names = sorted({name for sample in samples for name in sample.extras})
    return {
        name: statistics.median(
            sample.extras[name] for sample in samples if name in sample.extras
        )
        for name in names
    }


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
    samples: list[float], *, tokens_per_step: int, pp: int
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
    world_size = run.parallelism.world_size
    reads = {
        arm: arm_steps(record.arm(arm).arm, out_dir / f"{arm}.log") for arm in arms
    }
    for arm in arms:
        warnings.extend(
            dropped_line_warnings(arm, out_dir / f"{arm}.log", reads[arm])
        )
    steps = {
        arm: {rank: list(read.samples) for rank, read in by_rank.items()}
        for arm, by_rank in reads.items()
    }
    if run.profile:
        def _samples(samples: list[StepSample]) -> list[StepSample]:
            return stable_samples(samples, run.window)
    else:
        warmup_steps = run.warmup_steps

        def _samples(samples: list[StepSample]) -> list[StepSample]:
            return measured_samples(samples, warmup_steps)

    sampled = {
        arm: {rank: _samples(samples) for rank, samples in by_rank.items()}
        for arm, by_rank in steps.items()
    }
    tps_samples = {
        arm: {
            rank: [sample.tokens_per_second for sample in samples]
            for rank, samples in by_rank.items()
        }
        for arm, by_rank in sampled.items()
    }
    throughput = {
        arm: {
            rank: statistics.median(samples) if samples else None
            for rank, samples in by_rank.items()
        }
        for arm, by_rank in tps_samples.items()
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
        for arm, by_rank in tps_samples.items()
    }
    results = {}
    for arm in arms:
        rank = published_throughput_rank[arm]
        median_tps = throughput[arm].get(rank)
        peak_memory = max(
            (
                sample.peak_memory_gib
                for samples in steps[arm].values()
                for sample in samples
            ),
            default=None,
        )
        results[arm] = ArmResult(
            stable_tokens_per_second=median_tps,
            stable_sample_count=len(tps_samples[arm].get(rank, ())),
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
                    stable_sample_count=len(tps_samples[arm][each]),
                    step_ms=rank_step_ms[arm][each],
                )
                for each in sorted(throughput[arm])
            ),
            extras={
                engine_for(record.arm(arm).arm).name: extras_medians(
                    sampled[arm].get(rank, ())
                )
            },
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
        refuse_non_finite_trajectories(arm, steps[arm], out_dir / f"{arm}.log")
    return EvaluationResult(
        output_dir=str(out_dir),
        scenario=record.scenario,
        hardware=record.hardware,
        arms=tuple(arms),
        results=results,
        losses={
            arm: trajectory(steps[arm].get(trajectory_rank, ()), "loss")
            for arm in arms
        },
        gradient_norms={
            arm: trajectory(steps[arm].get(trajectory_rank, ()), "grad_norm")
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
