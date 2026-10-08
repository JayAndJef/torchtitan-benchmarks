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
from benchmarks.e2e.engines.api import (
    Arm,
    ProfileWindow,
    RunSpec,
    StepRead,
    StepSample,
)
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.e2e.evidence import non_finite_refusals, rank_steps
from benchmarks.execution.affinity import is_pinned
from benchmarks.execution.launcher import RANK_PREFIX


RESULTS_SCHEMA_VERSION = 7
"""The schema of ``results.json``."""


@dataclass(frozen=True)
class StepFigures:
    """The figures of one sampled step of one rank."""

    step: int
    tokens_per_second: float
    step_ms: float
    peak_memory_gib: float
    extras: dict[str, float]


@dataclass(frozen=True)
class RankStatistic:
    """One rate of one rank over its sampled steps: the median, and the mean that ``rate_mean`` gives."""

    median: float
    mean: float


@dataclass(frozen=True)
class RankStepMs:
    """The step cost of one rank, in milliseconds; ``p95`` is a nearest-rank value, so it is always a measured step."""

    median: float
    mean: float
    p95: float


@dataclass(frozen=True)
class RankMemory:
    """The peak memory of one rank, in GiB: the maximum over every step, and the median and the mean over the sampled steps."""

    max: float
    median: float
    mean: float


@dataclass(frozen=True)
class RankResult:
    """One rank's own figures, before any reduction across ranks."""

    rank: int
    tokens_per_second: RankStatistic
    step_ms: RankStepMs
    peak_memory_gib: RankMemory
    extras: dict[str, RankStatistic]
    """The engine's other figures, under the names that the engine gives them."""
    steps: tuple[StepFigures, ...]
    """The sampled steps, in step order."""


@dataclass(frozen=True)
class Statistic:
    """One rate of one arm: the median and the mean, each at the rank with the lowest value."""

    median: float
    median_rank: int
    mean: float
    mean_rank: int


@dataclass(frozen=True)
class StepMsStatistic:
    """The step cost of one arm, in milliseconds; each statistic is at the rank with the highest value."""

    median: float
    median_rank: int
    mean: float
    mean_rank: int
    p95: float
    p95_rank: int


@dataclass(frozen=True)
class MemoryStatistic:
    """The peak memory of one arm, in GiB; each statistic is at the rank with the highest value."""

    max: float
    max_rank: int
    median: float
    median_rank: int
    mean: float
    mean_rank: int


@dataclass(frozen=True)
class ArmResult:
    """One arm's published figures: each statistic of each figure at its own worst rank, and every rank's own figures."""

    sample_count: int
    """The sampled steps of each rank; every rank holds the same steps."""
    tokens_per_second: Statistic
    step_ms: StepMsStatistic
    peak_memory_gib: MemoryStatistic
    extras: dict[str, dict[str, Statistic]]
    """The engine's other figures, under the engine's name; each statistic is at the rank with the lowest value."""
    rank_reduction: str
    per_rank: tuple[RankResult, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """The evaluation of one run directory, as ``results.json`` records it."""

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
            "schema_version": RESULTS_SCHEMA_VERSION,
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


SLOW_FIRST_STEP = 2
"""The first step that the profiled rule would take; it runs slower than the steps after it in every measured arm."""


def _stable_step(step: int, window: ProfileWindow) -> bool:
    """Whether the profiled rule takes ``step``."""
    wait = window.freq - window.warmup - window.active
    return 2 <= ((step - 1) % window.freq) + 1 <= wait and step != SLOW_FIRST_STEP


def stable_samples(
    samples: Sequence[StepSample], window: ProfileWindow
) -> list[StepSample]:
    """The samples of a profiled run: the steps of each profiler cycle that carry no profiler cost, without ``SLOW_FIRST_STEP``."""
    return [sample for sample in samples if _stable_step(sample.step, window)]


def measured_samples(
    samples: Sequence[StepSample], warmup_steps: int
) -> list[StepSample]:
    """The samples of an unprofiled run: every step after the warmup."""
    return [sample for sample in samples if sample.step > warmup_steps]


def run_samples(run: RunSpec, samples: Sequence[StepSample]) -> list[StepSample]:
    """The samples that the run's own rule takes: ``stable_samples`` when the run is profiled, else ``measured_samples``."""
    if run.profile:
        return stable_samples(samples, run.window)
    return measured_samples(samples, run.warmup_steps)


def sampled_steps(run: RunSpec) -> tuple[int, ...]:
    """The steps that the run's rule takes from a rank that logs every step, from step 1 to the run's last step."""
    if run.profile:
        return tuple(
            step
            for step in range(1, run.data.steps + 1)
            if _stable_step(step, run.window)
        )
    return tuple(range(run.warmup_steps + 1, run.data.steps + 1))


def lost_step_refusals(
    run: RunSpec, sampled: Mapping[int, Sequence[StepSample]]
) -> list[str]:
    """The first lost sampled step of each rank, or its first sampled step past the run; a rank absent from ``sampled`` logged none."""
    expected = sampled_steps(run)
    if not expected:
        return [f"the sample rule takes none of the run's {run.data.steps} steps"]
    refusals = []
    for rank in sorted(set(range(run.parallelism.world_size)) | set(sampled)):
        found = {sample.step for sample in sampled.get(rank, ())}
        lost = [step for step in expected if step not in found]
        extra = sorted(found.difference(expected))
        if lost:
            refusals.append(f"rank {rank} lacks sampled step {lost[0]}")
        elif extra:
            refusals.append(
                f"rank {rank} logs sampled step {extra[0]}, and the run has "
                f"{run.data.steps} steps"
            )
    return refusals


def refuse_lost_steps(
    arm: str, run: RunSpec, sampled: Mapping[int, Sequence[StepSample]], log_path: Path
) -> None:
    """Raise ``ValueError`` when a rank lacks a step that the sample rule takes, so that every rank holds the same sampled steps."""
    refusals = lost_step_refusals(run, sampled)
    if refusals:
        raise ValueError(
            f"{arm}: {'; '.join(refusals)}; every rank must log every sampled "
            f"step, so no results.json is written for it (see {log_path})"
        )


def rate_mean(rates: Sequence[float]) -> float:
    """The mean of per-step rates whose steps each do the same work: the total work over the total time, which is the harmonic mean.

    Each step of one rank holds the same token count, so the mean tokens/s
    of a rank is its total tokens over its total step time.
    """
    return statistics.harmonic_mean(rates)


def extras_statistics(samples: Sequence[StepSample]) -> dict[str, RankStatistic]:
    """The median and the ``rate_mean`` of each extra figure over the samples that state it.

    The extras of both engines are rates: TFLOPS and MFU are each the step's
    tokens/s times a constant, so the mean of a rate is its total over the
    total time.
    """
    names = sorted({name for sample in samples for name in sample.extras})
    figures = {}
    for name in names:
        values = [sample.extras[name] for sample in samples if name in sample.extras]
        figures[name] = RankStatistic(
            median=statistics.median(values), mean=rate_mean(values)
        )
    return figures


def _worst_rank(values: Mapping[int, float], *, highest: bool) -> int:
    """The rank with the highest value when ``highest`` is true, else the lowest; a tie goes to the lower rank."""
    sign = -1 if highest else 1
    return min(values, key=lambda rank: (sign * values[rank], rank))


def _worst(
    figures: Mapping[int, Any], name: str, *, highest: bool
) -> tuple[float, int]:
    """The worst value of the field ``name`` over the ranks' ``figures``, and its rank."""
    values = {rank: getattr(each, name) for rank, each in figures.items()}
    rank = _worst_rank(values, highest=highest)
    return values[rank], rank


def _statistic(figures: Mapping[int, RankStatistic]) -> Statistic:
    """The median and the mean of one rate over the ranks, each at the rank with the lowest value."""
    median, median_rank = _worst(figures, "median", highest=False)
    mean, mean_rank = _worst(figures, "mean", highest=False)
    return Statistic(
        median=median, median_rank=median_rank, mean=mean, mean_rank=mean_rank
    )


def _throughput_spread(per_rank: Mapping[int, float]) -> float | None:
    """The highest rank throughput over the lowest; ``None`` below two ranks."""
    if len(per_rank) < 2:
        return None
    return max(per_rank.values()) / min(per_rank.values())


def _rank_throughput_summary(per_rank: Mapping[int, float]) -> str:
    return ", ".join(
        f"rank {rank} {value:,.0f}" for rank, value in sorted(per_rank.items())
    )


def _nearest_rank_percentile(values: list[float], fraction: float) -> float:
    """The nearest-rank percentile: a measured value, never an interpolation."""
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered))
    return ordered[max(index, 1) - 1]


def _step_cost(tps: float, *, tokens_per_step: int, pp: int) -> float:
    """The milliseconds of one step at ``tps`` tokens/s per device."""
    return 1000.0 * tokens_per_step / (tps * pp)


def step_ms(
    samples: Sequence[float], *, tokens_per_step: int, pp: int
) -> RankStepMs:
    """The step costs of one rank's tokens/s samples: the median, the arithmetic mean and the p95."""
    series = [
        _step_cost(tps, tokens_per_step=tokens_per_step, pp=pp) for tps in samples
    ]
    return RankStepMs(
        median=statistics.median(series),
        mean=statistics.fmean(series),
        p95=_nearest_rank_percentile(series, 0.95),
    )


def rank_result(
    arm: str,
    rank: int,
    steps: Sequence[StepSample],
    sampled: Sequence[StepSample],
    *,
    tokens_per_step: int,
    pp: int,
) -> RankResult:
    """The figures of one rank: ``sampled`` gives every statistic, and ``steps`` gives the memory maximum; a sampled rate that is not positive and finite raises ``ValueError``."""
    for sample in sampled:
        for figure, value in (
            ("tokens/s", sample.tokens_per_second),
            *sample.extras.items(),
        ):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(
                    f"{arm}: rank {rank} logs {figure} {value} at sampled step "
                    f"{sample.step}; each rate of a sampled step must be "
                    "positive and finite"
                )
    rates = [sample.tokens_per_second for sample in sampled]
    memory = [sample.peak_memory_gib for sample in sampled]
    return RankResult(
        rank=rank,
        tokens_per_second=RankStatistic(
            median=statistics.median(rates), mean=rate_mean(rates)
        ),
        step_ms=step_ms(rates, tokens_per_step=tokens_per_step, pp=pp),
        peak_memory_gib=RankMemory(
            max=max(sample.peak_memory_gib for sample in steps),
            median=statistics.median(memory),
            mean=statistics.fmean(memory),
        ),
        extras=extras_statistics(sampled),
        steps=tuple(
            StepFigures(
                step=sample.step,
                tokens_per_second=sample.tokens_per_second,
                step_ms=_step_cost(
                    sample.tokens_per_second, tokens_per_step=tokens_per_step, pp=pp
                ),
                peak_memory_gib=sample.peak_memory_gib,
                extras=dict(sample.extras),
            )
            for sample in sampled
        ),
    )


def arm_result(
    per_rank: Sequence[RankResult], *, engine: str, sample_count: int
) -> ArmResult:
    """The published figures of one arm: the worst rank of each statistic, the lowest for a rate and the highest for a cost or a memory figure."""
    by_rank = {each.rank: each for each in per_rank}
    step_costs = {rank: each.step_ms for rank, each in by_rank.items()}
    ms_median, ms_median_rank = _worst(step_costs, "median", highest=True)
    ms_mean, ms_mean_rank = _worst(step_costs, "mean", highest=True)
    ms_p95, ms_p95_rank = _worst(step_costs, "p95", highest=True)
    memory = {rank: each.peak_memory_gib for rank, each in by_rank.items()}
    memory_max, memory_max_rank = _worst(memory, "max", highest=True)
    memory_median, memory_median_rank = _worst(memory, "median", highest=True)
    memory_mean, memory_mean_rank = _worst(memory, "mean", highest=True)
    extra_names = sorted({name for each in per_rank for name in each.extras})
    return ArmResult(
        sample_count=sample_count,
        tokens_per_second=_statistic(
            {rank: each.tokens_per_second for rank, each in by_rank.items()}
        ),
        step_ms=StepMsStatistic(
            median=ms_median,
            median_rank=ms_median_rank,
            mean=ms_mean,
            mean_rank=ms_mean_rank,
            p95=ms_p95,
            p95_rank=ms_p95_rank,
        ),
        peak_memory_gib=MemoryStatistic(
            max=memory_max,
            max_rank=memory_max_rank,
            median=memory_median,
            median_rank=memory_median_rank,
            mean=memory_mean,
            mean_rank=memory_mean_rank,
        ),
        extras={
            engine: {
                name: _statistic(
                    {
                        rank: each.extras[name]
                        for rank, each in by_rank.items()
                        if name in each.extras
                    }
                )
                for name in extra_names
            }
        },
        rank_reduction="slowest_rank_per_statistic",
        per_rank=tuple(per_rank),
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
    sampled = {
        arm: {rank: run_samples(run, samples) for rank, samples in by_rank.items()}
        for arm, by_rank in steps.items()
    }
    for arm in arms:
        refuse_lost_steps(arm, run, sampled[arm], out_dir / f"{arm}.log")
    tokens_per_step = run.data.local_batch_size * run.data.seq_len
    pp = run.parallelism.pp
    sample_count = len(sampled_steps(run))
    results = {}
    for arm in arms:
        per_rank = [
            rank_result(
                arm,
                rank,
                steps[arm][rank],
                samples,
                tokens_per_step=tokens_per_step,
                pp=pp,
            )
            for rank, samples in sorted(sampled[arm].items())
        ]
        results[arm] = arm_result(
            per_rank,
            engine=engine_for(record.arm(arm).arm).name,
            sample_count=sample_count,
        )
        medians = {each.rank: each.tokens_per_second.median for each in per_rank}
        spread = _throughput_spread(medians)
        if spread is not None and spread > 1.15:
            warnings.append(
                f"{arm}: median tokens/s varies {spread:.2f}x across ranks "
                f"({_rank_throughput_summary(medians)}); a schedule "
                "holds the ranks in step, so a spread this wide means one "
                "rank is starved or the ranks are not running one job"
            )

    # Under a pipeline, one rank alone holds the loss.
    trajectory_rank = loss_visible_rank(world_size=world_size, pp=pp)
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
    """Write ``result`` to ``path``, else to ``results.json`` in the run directory, and return the file."""
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
        + f"{'arm':22s} {'tokens/s':>12s} {'mean tok/s':>12s} {'n':>4s} "
        + f"{'step ms':>9s} {'mean ms':>9s} {'p95 ms':>9s} {'peak GiB':>9s}",
    ]
    for arm in result.arms:
        summary = result.results[arm]
        lines.append(
            f"  {arm:22s} "
            f"{_value(summary.tokens_per_second.median, 12)} "
            f"{_value(summary.tokens_per_second.mean, 12)} "
            f"{summary.sample_count:4d} "
            f"{_value(summary.step_ms.median, 9, 2)} "
            f"{_value(summary.step_ms.mean, 9, 2)} "
            f"{_value(summary.step_ms.p95, 9, 2)} "
            f"{_value(summary.peak_memory_gib.max, 9, 2)}"
        )
    lines.extend(
        [
            "tokens/s and 'step ms' are medians over the sampled steps. "
            "'mean tok/s' is the total tokens over the total time, and "
            "'mean ms' is the arithmetic mean of the step times.",
            "tokens/s is per device. Each statistic is taken at its own "
            "slowest rank; 'peak GiB' is the maximum over every step and rank.",
        ]
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
