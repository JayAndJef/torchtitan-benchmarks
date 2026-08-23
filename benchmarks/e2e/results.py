"""Extraction, comparison, and rendering of end-to-end run metrics.

Everything an evaluated output directory turns into: the log-derived
throughput and trajectory series, the trace-derived GPU-time and
compiled-region summaries, the machine-readable ``results.json`` payload, and
the human-readable report the CLI prints.

**How a run with more than one rank becomes one published number.** Each rank
is pooled on its own (``per_rank_pooled_metrics``), and the arm's step cost is
the **maximum** over ranks, never the mean. A parallel schedule locks the
ranks together at every step boundary, so the step is as long as the busiest
rank; a mean would report a step nobody ran, and it would move whenever an
idle rank got idler. The sum over ranks is recorded beside it, because that is
the total device work the mesh did, and the per-rank vector is recorded too,
because a slow rank is the thing a reader most needs to see.

The published figures other than the step cost -- regions, launch latency, the
collective split, the busy basis -- are the **busiest rank's own** figures,
not a per-field maximum. Mixing fields across ranks produces incoherent rows:
``other_kernel_ms_per_step`` **includes every collective**: it is
``kernel - regions``, and a collective is a kernel that lies outside every
declared region. So on a multi-rank run the ``other ms`` column holds the
NCCL wait as well as the non-region compute, and a reader who takes it for
compute reads a bubble as work. ``compute ms`` and ``nccl ms`` are the split;
read those two instead. For the same reason ``region_span - region_kernel``
is not purely host idle once a collective shares a region's stream.

``other_kernel_ms_per_step`` is ``kernel - regions``, and taking each side
from a different rank can make it negative. ``published_rank`` names the rank
every such field came from, and ``per_rank`` carries the rest.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scipy import stats as scipy_stats

from benchmarks.artifacts.layout import atomic_write_json, trace_files_by_rank
from benchmarks.artifacts.manifests import load_run
from benchmarks.artifacts.summaries import (
    SampleSummary,
    _value,
    describe,
    summarize,
)
from benchmarks.traces.extraction import PooledMetrics, per_rank_pooled_metrics
from benchmarks.traces.schema import Region


STEP_METRICS = re.compile(
    r"step:\s*(\d+).*?memory:\s*([0-9.]+)GiB.*?tps:\s*([0-9,]+)"
)
LOSS_METRIC = re.compile(r"step:\s*(\d+).*?loss:\s*([0-9.eE+-]+|nan|inf)")
GRAD_NORM_METRIC = re.compile(
    r"step:\s*(\d+).*?grad_norm:\s*([0-9.eE+-]+|nan|inf)"
)
SIGNIFICANCE_METHODOLOGY = {
    "interpretation": "invocation_distribution_diagnostic",
    "sample_unit": "compiled_region_invocation",
    "independence_assumption_met": False,
    "limitation": (
        "Invocations share training steps and layer structure within one run. "
        "The span_* Welch and Mann-Whitney p-values are distribution "
        "diagnostics, not inferential evidence from independent benchmark "
        "repetitions."
    ),
}


@dataclass(frozen=True)
class RegionSummary:
    """One compiled region measured two ways.

    ``span`` is first-kernel-to-last per invocation and includes idle gaps
    where the GPU waited on the host; ``kernel`` sums only the kernel
    durations inside the span and is immune to host speed.
    """

    span: SampleSummary
    kernel: SampleSummary


@dataclass(frozen=True)
class RankGpuTime:
    """One rank's per-step GPU time, before any reduction across ranks."""

    rank: int
    kernel_ms_per_step: float | None
    compute_ms_per_step: float | None
    collective_ms_per_step: float | None
    busy_kernel_ms_per_step: float | None
    wall_ms_per_step: float | None
    region_kernel_ms_per_step: float | None
    launch_latency_us: float | None
    windows: int


@dataclass(frozen=True)
class GpuTimeSummary:
    """Per-step GPU kernel time: the host-speed-immune cost of one arm.

    ``kernel_ms_per_step`` is the maximum over the arm's ranks, and every
    other scalar here is the figure that same rank reported. See this
    module's docstring for why the reduction is a maximum and not a mean.

    ``baseline_kernel_ratio`` therefore divides this arm's busiest rank by the
    baseline's busiest rank, and the two need not be the same rank index. That
    is the right comparison when the ranks hold equal work, and it compares
    two different model partitions when they do not -- so read it as a ratio
    of step costs, which is what a schedule that locks the ranks together
    makes it.
    """

    kernel_ms_per_step: float | None
    region_kernel_ms_per_step: float | None
    other_kernel_ms_per_step: float | None
    baseline_kernel_ratio: float | None
    launch_latency_us: float | None
    compute_ms_per_step: float | None
    collective_ms_per_step: float | None
    busy_kernel_ms_per_step: float | None
    wall_ms_per_step: float | None
    rank_reduction: str
    published_rank: int
    ranks: tuple[int, ...]
    kernel_ms_per_step_summed_over_ranks: float | None
    per_rank: tuple[RankGpuTime, ...]


@dataclass(frozen=True)
class TrainingSummary:
    stable_tokens_per_second: float | None
    stable_sample_count: int
    baseline_ratio: float | None
    peak_memory_gib: float | None


@dataclass(frozen=True)
class EvaluationResult:
    """Complete machine-readable result for one benchmark output directory."""

    output_dir: str
    scenario: str
    hardware: str
    arms: tuple[str, ...]
    trace_windows: dict[str, int]
    regions: dict[str, dict[str, RegionSummary]]
    gpu_time: dict[str, GpuTimeSummary]
    comparisons: dict[str, list[dict[str, float | int | str]]]
    training: dict[str, TrainingSummary]
    losses: dict[str, list[tuple[int, float]]]
    gradient_norms: dict[str, list[tuple[int, float]]]
    significance_methodology: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": 4,
            "output_dir": self.output_dir,
            "scenario": self.scenario,
            "hardware": self.hardware,
            "arms": list(self.arms),
            "trace_windows": self.trace_windows,
            "regions": {
                arm: {name: asdict(summary) for name, summary in values.items()}
                for arm, values in self.regions.items()
            },
            "gpu_time": {
                arm: asdict(summary) for arm, summary in self.gpu_time.items()
            },
            "comparisons": self.comparisons,
            "training": {
                arm: asdict(summary) for arm, summary in self.training.items()
            },
            "losses": {
                arm: [{"step": step, "value": value} for step, value in values]
                for arm, values in self.losses.items()
            },
            "gradient_norms": {
                arm: [{"step": step, "value": value} for step, value in values]
                for arm, values in self.gradient_norms.items()
            },
            "significance_methodology": self.significance_methodology,
            "warnings": list(self.warnings),
        }
        return _json_safe(value)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with null so results are strict JSON.

    Tuples become lists on the way through. ``asdict`` keeps a dataclass
    field's container type, so ``ranks`` and ``per_rank`` arrive here as
    tuples; leaving them that way would make ``to_dict()`` disagree with the
    file it writes, and a test comparing the two would have to know which.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def region_comparison(
    base: PooledMetrics,
    arm: PooledMetrics,
    regions: tuple[Region, ...],
) -> list[dict[str, float | int | str]]:
    """Compare invocation distributions without claiming independent samples.

    ``span_*`` columns cover the annotation spans (host-confounded);
    ``kernel_*`` columns cover the summed kernel time inside them. The
    significance diagnostics describe the span distributions only.

    Both sides are one rank's invocations -- each arm's published rank -- for
    the same reason the scalars are: pooling two ranks' invocations into one
    distribution describes no rank's work.
    """
    rows: list[dict[str, float | int | str]] = []
    for region in regions:
        base_spans = base.region_spans[region.name]
        arm_spans = arm.region_spans[region.name]
        n_base, span_mean_base, span_sd_base, span_median_base = describe(base_spans)
        n_arm, span_mean_arm, span_sd_arm, span_median_arm = describe(arm_spans)
        welch = scipy_stats.ttest_ind(base_spans, arm_spans, equal_var=False)
        mann_whitney = scipy_stats.mannwhitneyu(
            base_spans, arm_spans, alternative="two-sided"
        )
        pooled_sd = (
            (
                ((n_base - 1) * span_sd_base**2 + (n_arm - 1) * span_sd_arm**2)
                / (n_base + n_arm - 2)
            )
            ** 0.5
            if n_base + n_arm > 2
            else 0.0
        )
        kernel_mean_base = statistics.mean(base.region_kernel[region.name])
        kernel_mean_arm = statistics.mean(arm.region_kernel[region.name])
        rows.append(
            {
                "region": region.name,
                "n_base": n_base,
                "n_arm": n_arm,
                "base_kernel_mean_us": kernel_mean_base,
                "arm_kernel_mean_us": kernel_mean_arm,
                "kernel_delta_us": kernel_mean_arm - kernel_mean_base,
                "kernel_ratio": (
                    kernel_mean_arm / kernel_mean_base if kernel_mean_base else None
                ),
                "base_span_mean_us": span_mean_base,
                "base_span_median_us": span_median_base,
                "base_span_sd_us": span_sd_base,
                "arm_span_mean_us": span_mean_arm,
                "arm_span_median_us": span_median_arm,
                "arm_span_sd_us": span_sd_arm,
                "span_delta_us": span_mean_arm - span_mean_base,
                "span_ratio": (
                    span_mean_arm / span_mean_base if span_mean_base else None
                ),
                "span_welch_p": float(welch.pvalue),
                "span_mwu_p": float(mann_whitney.pvalue),
                "span_cohens_d": (
                    (span_mean_arm - span_mean_base) / pooled_sd if pooled_sd else 0.0
                ),
            }
        )
    return rows


def _trajectory(log_path: Path, pattern: re.Pattern[str]) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    values = []
    with log_path.open(errors="replace") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                values.append((int(match.group(1)), float(match.group(2))))
    return values


def losses(log_path: Path) -> list[tuple[int, float]]:
    return _trajectory(log_path, LOSS_METRIC)


def grad_norms(log_path: Path) -> list[tuple[int, float]]:
    return _trajectory(log_path, GRAD_NORM_METRIC)


def training_metrics(log_path: Path) -> list[tuple[int, float, int]]:
    """Return (step, peak-memory-GiB, tokens/s) rows from a training log."""
    if not log_path.exists():
        return []
    rows = []
    with log_path.open(errors="replace") as log_file:
        for line in log_file:
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


def stable_tps(
    rows: list[tuple[int, float, int]], workload: dict[str, Any]
) -> list[int]:
    """Select post-compile steps before each profiler warmup begins."""
    profile_freq = int(workload.get("profile_freq", 20))
    wait = profile_freq - int(workload.get("profiler_warmup", 5)) - int(
        workload.get("profiler_active", 5)
    )
    return [
        tps
        for step, _, tps in rows
        if 2 <= ((step - 1) % profile_freq) + 1 <= wait
    ]


def busiest_rank(per_rank: dict[int, PooledMetrics]) -> int:
    """The rank whose step costs the most GPU kernel time; ties go to the lowest.

    This is the rank the arm's published figures come from. It is a maximum,
    never a mean: the schedule holds the ranks in step, so the step is as long
    as its slowest participant.

    A rank whose windows carry no ``ProfilerStep`` annotation reports ``None``
    rather than a per-step cost, and a maximum cannot rank ``None`` against a
    number. Treating it as zero would silently drop that rank from the
    maximum, which is the failure this whole reduction exists to prevent -- so
    a mixture is refused, exactly as ``pooled_window_metrics`` refuses the
    same mixture between the windows of one rank. All-``None`` is not a
    mixture: it is a run nothing profiled, and it keeps today's answer.
    """
    measured = [
        rank
        for rank, pooled in per_rank.items()
        if pooled.kernel_ms_per_step is not None
    ]
    if measured and len(measured) != len(per_rank):
        stepless = sorted(set(per_rank) - set(measured))
        raise ValueError(
            f"ranks {stepless} carry no ProfilerStep events and ranks "
            f"{sorted(measured)} do; a maximum over that mixture would "
            "exclude the ranks it cannot measure, and one of them may be the "
            "busiest"
        )
    return min(
        per_rank,
        key=lambda rank: (-(per_rank[rank].kernel_ms_per_step or 0.0), rank),
    )


def _rank_gpu_time(rank: int, pooled: PooledMetrics) -> RankGpuTime:
    return RankGpuTime(
        rank=rank,
        kernel_ms_per_step=pooled.kernel_ms_per_step,
        compute_ms_per_step=pooled.compute_ms_per_step,
        collective_ms_per_step=pooled.collective_ms_per_step,
        busy_kernel_ms_per_step=pooled.busy_kernel_ms_per_step,
        wall_ms_per_step=pooled.wall_ms_per_step,
        region_kernel_ms_per_step=pooled.region_kernel_ms_per_step,
        launch_latency_us=pooled.launch_latency_us,
        windows=pooled.windows,
    )


def _summed_over_ranks(per_rank: dict[int, PooledMetrics]) -> float | None:
    """Total device work across the mesh, or ``None`` if any rank has none."""
    values = [pooled.kernel_ms_per_step for pooled in per_rank.values()]
    return None if any(value is None for value in values) else sum(values)


def evaluate_run(
    out_dir: Path, arms_override: list[str] | tuple[str, ...] | None = None
) -> EvaluationResult:
    """Evaluate profiler regions and end-to-end metrics for selected arms."""
    out_dir = out_dir.resolve()
    manifest, arms, declared_regions, warnings = load_run(out_dir, arms_override)
    if "baseline" not in arms:
        raise ValueError("comparison needs the baseline arm")

    per_rank: dict[str, dict[int, PooledMetrics]] = {}
    published_rank: dict[str, int] = {}
    pooled: dict[str, PooledMetrics] = {}
    trace_windows: dict[str, int] = {}
    for arm in arms:
        by_rank = trace_files_by_rank(out_dir / arm)
        if not by_rank:
            raise ValueError(f"no profiler traces under {out_dir / arm}")
        try:
            per_rank[arm] = per_rank_pooled_metrics(by_rank, declared_regions)
        except ValueError as error:
            raise ValueError(f"{arm}: {error}") from error
        try:
            published_rank[arm] = busiest_rank(per_rank[arm])
        except ValueError as error:
            raise ValueError(f"{arm}: {error}") from error
        pooled[arm] = per_rank[arm][published_rank[arm]]
        trace_windows[arm] = pooled[arm].windows

    regions = {
        arm: {
            region.name: RegionSummary(
                span=summarize(pooled[arm].region_spans[region.name]),
                kernel=summarize(pooled[arm].region_kernel[region.name]),
            )
            for region in declared_regions
        }
        for arm in arms
    }
    baseline_kernel_ms = pooled["baseline"].kernel_ms_per_step
    gpu_time = {}
    for arm in arms:
        kernel_ms = pooled[arm].kernel_ms_per_step
        region_ms = pooled[arm].region_kernel_ms_per_step
        gpu_time[arm] = GpuTimeSummary(
            kernel_ms_per_step=kernel_ms,
            region_kernel_ms_per_step=region_ms,
            other_kernel_ms_per_step=(
                kernel_ms - region_ms
                if kernel_ms is not None and region_ms is not None
                else None
            ),
            baseline_kernel_ratio=(
                kernel_ms / baseline_kernel_ms
                if kernel_ms is not None and baseline_kernel_ms
                else None
            ),
            launch_latency_us=pooled[arm].launch_latency_us,
            compute_ms_per_step=pooled[arm].compute_ms_per_step,
            collective_ms_per_step=pooled[arm].collective_ms_per_step,
            busy_kernel_ms_per_step=pooled[arm].busy_kernel_ms_per_step,
            wall_ms_per_step=pooled[arm].wall_ms_per_step,
            rank_reduction="max_over_ranks",
            published_rank=published_rank[arm],
            ranks=tuple(sorted(per_rank[arm])),
            kernel_ms_per_step_summed_over_ranks=_summed_over_ranks(per_rank[arm]),
            per_rank=tuple(
                _rank_gpu_time(rank, per_rank[arm][rank])
                for rank in sorted(per_rank[arm])
            ),
        )
    comparisons = {
        arm: region_comparison(pooled["baseline"], pooled[arm], declared_regions)
        for arm in arms
        if arm != "baseline"
    }
    latencies = {
        arm: summary.launch_latency_us
        for arm, summary in gpu_time.items()
        if summary.launch_latency_us
    }
    if len(latencies) == len(arms) and len(arms) > 1:
        slowest = max(latencies, key=latencies.get)
        fastest = min(latencies, key=latencies.get)
        spread = latencies[slowest] / latencies[fastest]
        if spread > 1.15:
            warnings.append(
                f"host launch latency varies {spread:.2f}x across arms "
                f"({fastest} {latencies[fastest]:.2f}us .. "
                f"{slowest} {latencies[slowest]:.2f}us); tokens/s and span "
                f"metrics are host-speed-confounded — compare kernel time"
            )

    workload = manifest.get("workload", {})
    raw_training = {
        arm: training_metrics(out_dir / f"{arm}.log") for arm in arms
    }
    stable_samples = {
        arm: stable_tps(rows, workload) for arm, rows in raw_training.items()
    }
    baseline_samples = stable_samples["baseline"]
    baseline_median = (
        statistics.median(baseline_samples) if baseline_samples else None
    )
    training = {}
    for arm in arms:
        samples = stable_samples[arm]
        median_tps = statistics.median(samples) if samples else None
        peak_memory = max(
            (memory for _, memory, _ in raw_training[arm]), default=None
        )
        ratio = (
            median_tps / baseline_median
            if median_tps is not None
            and baseline_median is not None
            and baseline_median != 0
            else None
        )
        training[arm] = TrainingSummary(
            stable_tokens_per_second=median_tps,
            stable_sample_count=len(samples),
            baseline_ratio=ratio,
            peak_memory_gib=peak_memory,
        )

    return EvaluationResult(
        output_dir=str(out_dir),
        scenario=manifest.get("scenario", "unknown"),
        hardware=manifest.get("hardware", "unknown"),
        arms=tuple(arms),
        trace_windows=trace_windows,
        regions=regions,
        gpu_time=gpu_time,
        comparisons=comparisons,
        training=training,
        losses={arm: losses(out_dir / f"{arm}.log") for arm in arms},
        gradient_norms={arm: grad_norms(out_dir / f"{arm}.log") for arm in arms},
        significance_methodology=SIGNIFICANCE_METHODOLOGY.copy(),
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
    if not all(math.isfinite(value) for _, value in values):
        rendered += f"   NON-FINITE {nonfinite_label}"
    return rendered


def _render_rank_split(result: EvaluationResult) -> list[str]:
    """Per-rank rows and the compute/collective split, when either applies.

    Printed only when the run has more than one rank or ran a collective.
    Every run recorded before this existed had one rank and no collective, so
    a single-GPU report is unchanged, character for character.
    """
    interesting = [
        arm
        for arm in result.arms
        if len(result.gpu_time[arm].ranks) > 1
        or (result.gpu_time[arm].collective_ms_per_step or 0.0) > 0.0
    ]
    if not interesting:
        return []
    lines = [
        "",
        "per-rank gpu time (published figure is the MAX over ranks, never the "
        "mean):",
        "  "
        + f"{'arm':22s} {'rank':>4s} {'kernel ms':>10s} {'compute ms':>11s} "
        + f"{'nccl ms':>8s} {'busy ms':>8s} {'wall ms':>8s} {'launch us':>10s}",
    ]
    for arm in result.arms:
        gpu = result.gpu_time[arm]
        for rank in gpu.per_rank:
            marker = "*" if rank.rank == gpu.published_rank else " "
            lines.append(
                f"  {arm:22s} {rank.rank:>3d}{marker} "
                f"{_value(rank.kernel_ms_per_step, 10, 2)} "
                f"{_value(rank.compute_ms_per_step, 11, 2)} "
                f"{_value(rank.collective_ms_per_step, 8, 2)} "
                f"{_value(rank.busy_kernel_ms_per_step, 8, 2)} "
                f"{_value(rank.wall_ms_per_step, 8, 2)} "
                f"{_value(rank.launch_latency_us, 10, 2)}"
            )
        lines.append(
            f"  {arm:22s} sum  "
            f"{_value(gpu.kernel_ms_per_step_summed_over_ranks, 10, 2)}"
        )
    lines.extend(
        [
            "* = the published rank. 'compute' excludes the collectives, "
            "because a blocking",
            "collective's duration includes waiting for a peer; 'busy' is the "
            "interval union,",
            "which the summed column double-counts across streams; "
            "wall - busy is the bubble.",
        ]
    )
    return lines


def render_evaluation(result: EvaluationResult) -> str:
    """Render the complete stable-throughput and compiled-region report."""
    lines = [
        f"== {result.output_dir} ==",
        f"scenario: {result.scenario}   hardware: {result.hardware}",
    ]
    lines.extend(f"WARNING: {warning}" for warning in result.warnings)

    lines.extend(
        [
            "",
            "benchmark summary:",
            "  "
            + f"{'arm':22s} {'stable tokens/s':>15s} {'n':>4s} {'ratio':>8s} "
            + f"{'peak GiB':>9s}",
        ]
    )
    for arm in result.arms:
        training = result.training[arm]
        lines.append(
            f"  {arm:22s} "
            f"{_value(training.stable_tokens_per_second, 15)} "
            f"{training.stable_sample_count:4d} "
            f"{_value(training.baseline_ratio, 8, 4)} "
            f"{_value(training.peak_memory_gib, 9, 2)}"
        )

    lines.extend(
        [
            "",
            "gpu kernel time (host-speed-immune; compare kernels with this):",
            "  "
            + f"{'arm':22s} {'kernel ms/step':>14s} {'vs base':>8s} "
            + f"{'regions ms':>11s} {'other ms':>9s} {'fwd kernel us':>14s} "
            + f"{'bwd kernel us':>14s} {'launch us':>10s}",
        ]
    )
    for arm in result.arms:
        gpu = result.gpu_time[arm]
        forward = result.regions[arm].get("forward_block")
        backward = result.regions[arm].get("backward_block")
        lines.append(
            f"  {arm:22s} "
            f"{_value(gpu.kernel_ms_per_step, 14, 2)} "
            f"{_value(gpu.baseline_kernel_ratio, 8, 4)} "
            f"{_value(gpu.region_kernel_ms_per_step, 11, 2)} "
            f"{_value(gpu.other_kernel_ms_per_step, 9, 2)} "
            f"{_value(forward.kernel.mean_us if forward else None, 14)} "
            f"{_value(backward.kernel.mean_us if backward else None, 14)} "
            f"{_value(gpu.launch_latency_us, 10, 2)}"
        )

    lines.extend(_render_rank_split(result))

    for arm in result.arms:
        if arm == "baseline":
            continue
        lines.extend(
            [
                "",
                f"compiled-region distributions, baseline vs {arm} "
                f"(pooled over {result.trace_windows['baseline']}+"
                f"{result.trace_windows[arm]} windows):",
                "  "
                + f"{'region':16s} {'n':>4} | {'base kern':>10} {'arm kern':>9} "
                + f"{'ratio':>7} | {'base span':>10} {'arm span':>9} "
                + f"{'ratio':>7} | {'diag Welch p':>12} {'diag MWU p':>10} "
                + f"{'d':>6}",
            ]
        )
        for row in result.comparisons[arm]:
            lines.append(
                f"  {row['region']:16s} {row['n_arm']:>4} | "
                f"{row['base_kernel_mean_us']:10.1f} "
                f"{row['arm_kernel_mean_us']:9.1f} "
                f"{_value(row['kernel_ratio'], 7, 4)} | "
                f"{row['base_span_mean_us']:10.1f} "
                f"{row['arm_span_mean_us']:9.1f} "
                f"{_value(row['span_ratio'], 7, 4)} | "
                f"{row['span_welch_p']:12.3g} {row['span_mwu_p']:10.3g} "
                f"{row['span_cohens_d']:6.2f}"
            )

    lines.extend(
        [
            "",
            "Significance limitation: pooled compiled-region invocations share",
            "training steps and layer structure. Welch/MWU p-values and Cohen's d",
            "describe span distributions; they are not inference from",
            "independent benchmark repetitions.",
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

    lines.extend(
        [
            "",
            "Note: region rows time whole compiled forward/backward blocks; they",
            "are not measurements of an individual generated Inductor kernel.",
            "Span times include idle gaps where the GPU waited on the host, so",
            "they move with host speed; kernel times count only GPU execution.",
        ]
    )
    return "\n".join(lines)
