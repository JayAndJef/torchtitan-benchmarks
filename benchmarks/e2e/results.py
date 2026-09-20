"""Extraction, comparison, and rendering of end-to-end run metrics.

Everything an evaluated output directory turns into: the log-derived
throughput and trajectory series, the trace-derived GPU-time summaries, the
machine-readable ``results.json`` payload, and the human-readable report the
CLI prints.

**How a run with more than one rank becomes one published number.** Each rank
is pooled on its own (``per_rank_pooled_metrics``), and the arm's step cost is
the **maximum** over ranks, never the mean. A parallel schedule locks the
ranks together at every step boundary, so the step is as long as the busiest
rank; a mean would report a step nobody ran, and it would move whenever an
idle rank got idler. The sum over ranks is recorded beside it, because that is
the total device work the mesh did, and the per-rank vector is recorded too,
because a slow rank is the thing a reader most needs to see.

The published figures other than the step cost -- launch latency, the
collective split, the busy basis -- are the **busiest rank's own** figures,
not a per-field maximum. Mixing fields across ranks produces incoherent
rows. ``published_rank`` names the rank every such field came from, and
``per_rank`` carries the rest.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.artifacts.layout import (
    atomic_write_json,
    logs_by_rank,
    trace_files_by_rank,
)
from benchmarks.artifacts.manifests import load_run
from benchmarks.artifacts.summaries import _value
from benchmarks.e2e.parallelism import (
    DEFAULT_DENSE_SHARDING,
    ParallelismSpec,
    dense_sharding_warnings,
)
from benchmarks.traces.extraction import PooledMetrics, per_rank_pooled_metrics


STEP_METRICS = re.compile(
    r"step:\s*(\d+).*?memory:\s*([0-9.]+)GiB.*?tps:\s*([0-9,]+)"
)
# The words before the digits, because the alternation is leftmost-first: a
# ``-inf`` read by the digit class alone yields ``-``, and ``float`` then
# raises where refuse_non_finite_trajectories should name the step.
LOSS_METRIC = re.compile(r"step:\s*(\d+).*?loss:\s*(nan|-?inf|[0-9.eE+-]+)")
GRAD_NORM_METRIC = re.compile(
    r"step:\s*(\d+).*?grad_norm:\s*(nan|-?inf|[0-9.eE+-]+)"
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
class RankGpuTime:
    """One rank's per-step GPU time, before any reduction across ranks."""

    rank: int
    kernel_ms_per_step: float | None
    compute_ms_per_step: float | None
    collective_ms_per_step: float | None
    busy_kernel_ms_per_step: float | None
    wall_ms_per_step: float | None
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
class RankThroughput:
    """One rank's own throughput, before any reduction across ranks."""

    rank: int
    stable_tokens_per_second: float | None
    stable_sample_count: int


@dataclass(frozen=True)
class TrainingSummary:
    """Tokens per second per device, and what the mesh did with them.

    ``stable_tokens_per_second`` is the **minimum** over ranks, which is the
    throughput twin of the maximum this module takes over each rank's kernel
    time: a schedule that locks the ranks together runs at the pace of the
    slowest one, and a mean would report a rate nobody achieved. At one rank
    it is that rank's own median, exactly as before.

    ``tokens_per_second_global`` is that figure times the world size. The
    relation holds under pipeline and data parallelism alike, because both
    engines divide a rank's own token count by ``cp * tp * pp`` and each
    data-parallel rank reads a batch of its own. The manifest records the
    definition in ``throughput_definition``.

    ``peak_memory_gib`` is the maximum over every rank, which needs no
    reduction rule: it is the most memory any device in the mesh held.
    """

    stable_tokens_per_second: float | None
    stable_sample_count: int
    baseline_ratio: float | None
    peak_memory_gib: float | None
    tokens_per_second_global: float | None
    rank_reduction: str
    published_rank: int
    ranks: tuple[int, ...]
    per_rank: tuple[RankThroughput, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Complete machine-readable result for one benchmark output directory."""

    output_dir: str
    scenario: str
    hardware: str
    arms: tuple[str, ...]
    trace_windows: dict[str, int]
    gpu_time: dict[str, GpuTimeSummary]
    comparisons: dict[str, list[dict[str, float | int | str]]]
    training: dict[str, TrainingSummary]
    losses: dict[str, list[tuple[int, float]]]
    gradient_norms: dict[str, list[tuple[int, float]]]
    significance_methodology: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": 5,
            "output_dir": self.output_dir,
            "scenario": self.scenario,
            "hardware": self.hardware,
            "arms": list(self.arms),
            "trace_windows": self.trace_windows,
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


def _log_by_rank(log_path: Path) -> dict[int, str]:
    """This arm's log, split into what each rank wrote. Empty when missing."""
    if not log_path.exists():
        return {}
    return logs_by_rank(log_path.read_text(errors="replace"))


def _trajectory(text: str, pattern: re.Pattern[str]) -> list[tuple[int, float]]:
    values = []
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            values.append((int(match.group(1)), float(match.group(2))))
    return values


def loss_visible_rank(*, world_size: int, pp: int) -> int:
    """The rank whose step line carries the run's real loss.

    TorchTitan computes the loss on the last pipeline stage and every rank
    calls ``MetricsProcessor.log``. A rank without that stage does not print
    a smaller or noisier loss -- it prints a **sentinel**: ``trainer.py``
    sets ``loss = torch.tensor([-1.0])`` there, and at ``dp 1`` that reaches
    the step line unreduced, so rank 0 of a pp2 run logs ``loss: -1.00000``.
    Pooling that with the real trajectory would not add noise, it would add
    a constant that is not a loss at all.

    **Above ``dp`` 1 the sentinel is not ``-1.0`` any more, and it is still
    not a loss.** ``trainer.py`` sums the loss over its ``loss`` mesh, which
    holds the data-parallel ranks of one pipeline column, so a rank without
    the last stage prints ``-dp``. The ranks that hold the last stage sum
    their own halves and print the true global average. So the rank this
    function names carries a real loss at every degree, and no other rank
    does.

    This is TorchTitan's own ``_get_metrics_rank`` arithmetic. The megatron
    driver satisfies it too, by a different route: it broadcasts the last
    stage's loss over the pipeline group and takes the mean over the
    data-parallel group, so every rank prints the real one and this rank is
    one of them.

    **Both engines' rank layouts were read, and the arithmetic holds on
    each.** TorchTitan unflattens its mesh as ``(pp, batch, cp, tp)``, so
    ``pp`` is the outermost axis. Megatron's ``RankGenerator`` runs
    ``order="tp-cp-ep-dp-pp"``, which puts ``pp`` outermost too: at ``dp 2,
    pp 2`` its pipeline groups are ``[[0, 2], [1, 3]]`` and its
    data-parallel groups are ``[[0, 1], [2, 3]]``, so rank 2 is the first
    rank of the last stage on both engines.

    At the trivial spec it is 0, which is the rank a single-GPU run has.

    **It is right for the two schedules this repo runs and not for every
    schedule.** ``ZBVZeroBubble`` returns the loss on rank 0, and TorchTitan
    special-cases it; parallelism rule 5 refuses that schedule for any run
    holding a megatron arm and this repo has never run one, so the case is
    recorded rather than handled.
    """
    return (world_size // pp) * (pp - 1)


def losses(log_path: Path, *, rank: int = 0) -> list[tuple[int, float]]:
    return _trajectory(_log_by_rank(log_path).get(rank, ""), LOSS_METRIC)


# The two trajectories a step line carries, by the name a failure prints.
_TRAJECTORY_METRICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("loss", LOSS_METRIC),
    ("grad_norm", GRAD_NORM_METRIC),
)


def refuse_non_finite_trajectories(arm: str, log_path: Path) -> None:
    """Fail an arm whose step lines carry a ``nan`` or an ``inf``.

    ``LOSS_METRIC`` and ``GRAD_NORM_METRIC`` accept both words on purpose,
    so a diverged run is read rather than dropped. Reading it is not
    publishing it: a tokens/s figure taken over steps whose loss is not a
    number is the throughput of a run that trained nothing, and every other
    figure in ``results.json`` would then sit under a healthy label.

    Every rank is read, not only the rank whose trajectory is published.
    No rank prints a non-finite value on purpose: TorchTitan's rank without
    the loss prints the ``-1.0`` sentinel, and the stock megatron driver
    omits the field. So a ``nan`` on any rank is a process that diverged,
    and the rule reads per rank the way the validation rules do.

    One printed ``nan`` is deliberate and still refused, and that is the
    safe direction. The stock driver's step shim prints ``grad_norm: nan``
    on a step Megatron skipped, which is a step that applied no update. No
    honest run here reaches it: the arm runs ``--bf16`` with no
    ``--loss-scale``, so ``get_megatron_optimizer`` builds no grad scaler,
    ``prepare_grads`` returns False, and ``train_step`` never sets
    ``skipped_iter``. A run that did skip a step trained fewer steps than
    it claims, and its throughput is not the throughput of the workload.

    This is the one non-finite check the harness owns. Stock Megatron
    carries its own, ``check_for_nan_in_loss_and_grad``, and it is
    Megatron's to turn off; TorchTitan and the tuned megatron driver carry
    none. The check therefore runs on every arm, whatever the engine's own
    guard did, and it names the rank and the first step that failed.
    """
    for rank, text in sorted(_log_by_rank(log_path).items()):
        for metric, pattern in _TRAJECTORY_METRICS:
            for step, value in _trajectory(text, pattern):
                if not math.isfinite(value):
                    raise ValueError(
                        f"{arm}: rank {rank} logged a non-finite {metric} at "
                        f"step {step} ({value}); a run that diverged cannot "
                        "publish a throughput, so no results.json is written "
                        f"for it (see {log_path})"
                    )


def grad_norms(log_path: Path, *, rank: int = 0) -> list[tuple[int, float]]:
    return _trajectory(_log_by_rank(log_path).get(rank, ""), GRAD_NORM_METRIC)


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
    """(step, peak-memory-GiB, tokens/s) rows, per the rank that printed them.

    Both engines print a step line on every rank, each carrying that rank's
    own throughput, so a run with two ranks writes two rows per step into one
    file. Pooling them would take a median over twice as many samples as the
    run has steps, and it would hide the case this split exists to show: one
    rank running slower than the rest.
    """
    return {
        rank: _rows(text) for rank, text in _log_by_rank(log_path).items()
    }


def training_metrics(log_path: Path) -> list[tuple[int, float, int]]:
    """Every rank's rows, pooled in rank order.

    Correct as a *memory* input, where the answer is a maximum over the
    whole mesh. **Not correct as a throughput input on more than one rank**:
    see ``per_rank_training_metrics``.
    """
    return [
        row
        for _, rows in sorted(per_rank_training_metrics(log_path).items())
        for row in rows
    ]


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


def _slowest_rank(per_rank: dict[int, float | None]) -> int:
    """The rank with the lowest throughput; ties and empties go to the lowest.

    The throughput twin of ``busiest_rank``: a schedule that locks the ranks
    together runs at the pace of its slowest participant. A rank whose log
    holds no stable sample sorts last rather than winning as a zero, because
    "no sample" is a measurement that did not happen and not a slow rank.
    """
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
    """Evaluate the end-to-end metrics of the selected arms."""
    out_dir = out_dir.resolve()
    manifest, arms, warnings = load_run(out_dir, arms_override)
    baseline = "baseline" if "baseline" in arms else None
    if baseline is None and len(arms) != 1:
        raise ValueError(
            "comparison needs the baseline arm; only a one-arm run can "
            "publish absolute metrics without it"
        )

    per_rank: dict[str, dict[int, PooledMetrics]] = {}
    published_rank: dict[str, int] = {}
    pooled: dict[str, PooledMetrics] = {}
    trace_windows: dict[str, int] = {}
    for arm in arms:
        by_rank = trace_files_by_rank(out_dir / arm)
        if not by_rank:
            raise ValueError(f"no profiler traces under {out_dir / arm}")
        try:
            per_rank[arm] = per_rank_pooled_metrics(by_rank)
        except ValueError as error:
            raise ValueError(f"{arm}: {error}") from error
        try:
            published_rank[arm] = busiest_rank(per_rank[arm])
        except ValueError as error:
            raise ValueError(f"{arm}: {error}") from error
        pooled[arm] = per_rank[arm][published_rank[arm]]
        trace_windows[arm] = pooled[arm].windows

    baseline_kernel_ms = (
        pooled[baseline].kernel_ms_per_step if baseline is not None else None
    )
    gpu_time = {}
    for arm in arms:
        kernel_ms = pooled[arm].kernel_ms_per_step
        gpu_time[arm] = GpuTimeSummary(
            kernel_ms_per_step=kernel_ms,
            baseline_kernel_ratio=(
                kernel_ms / baseline_kernel_ms
                if baseline is not None
                and kernel_ms is not None
                and baseline_kernel_ms
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
    # The ratio divides one rank of this arm by one rank of the baseline, and
    # each side names its own busiest rank. Under a pipeline split those two
    # rank indices hold different partitions of the model, so the ratio stops
    # being "the same work, two implementations". It is still the right
    # comparison of step costs -- the schedule holds the ranks together -- but
    # a reader of results.json holds no docstring, so the file says so.
    #
    # Captioned rather than pinned to one rank index. Pinning would divide two
    # ranks nobody chose for being busy, which is a different and weaker
    # figure, and it would move the ratio a single-GPU run has always
    # published the moment a run has two ranks.
    for arm in arms:
        if baseline is None or arm == baseline:
            continue
        if gpu_time[arm].published_rank != gpu_time[baseline].published_rank:
            warnings.append(
                f"{arm}: the 'vs base' ratio divides rank "
                f"{gpu_time[arm].published_rank} by baseline rank "
                f"{gpu_time[baseline].published_rank}; each side is its own "
                "busiest rank, so under a pipeline split the two hold "
                "different partitions of the model. Read it as a ratio of "
                "step costs, never as one component against itself"
            )
    # Compiled-region distributions were the only rows this ever held.
    comparisons: dict[str, list[dict[str, float | int | str]]] = {}
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
    # The declared mesh, read back from the manifest. Schema <= 9 directories
    # carry no record and every one of them ran on one GPU.
    recorded_parallelism = manifest.get("parallelism", {})
    world_size = int(recorded_parallelism.get("world_size", 1))
    # The two dense-sharding warnings reach the artifact as well as the
    # console. The runner says them when the run starts, and a reader of
    # results.json was not there. The file is what a report quotes.
    #
    # ``dense_sharding_warnings`` reads a spec, so the record becomes a spec
    # again. Only the fields that function reads are rebuilt: a record also
    # carries keys the spec derives for itself, such as ``world_size``.
    #
    # **A directory written before the rename records ``shard``, which this
    # axis no longer declares, so the rebuild refuses it.** Three published
    # cells are in that state. They predate the question these warnings ask,
    # so they earn none -- and evaluating them must not fail here.
    try:
        recorded_spec = ParallelismSpec(
            dp=int(recorded_parallelism.get("dp", 1)),
            pp=int(recorded_parallelism.get("pp", 1)),
            ep=int(recorded_parallelism.get("ep", 1)),
            dense_sharding=str(
                recorded_parallelism.get(
                    "dense_sharding", DEFAULT_DENSE_SHARDING
                )
            ),
        )
    except (TypeError, ValueError):
        recorded_spec = None
    if recorded_spec is not None:
        warnings.extend(dense_sharding_warnings(recorded_spec))
    raw_training = {
        arm: per_rank_training_metrics(out_dir / f"{arm}.log") for arm in arms
    }
    stable_samples = {
        arm: {
            rank: stable_tps(rows, workload) for rank, rows in by_rank.items()
        }
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
    baseline_median = (
        throughput[baseline].get(published_throughput_rank[baseline])
        if baseline is not None
        else None
    )
    training = {}
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
        ratio = (
            median_tps / baseline_median
            if baseline is not None
            and median_tps is not None
            and baseline_median is not None
            and baseline_median != 0
            else None
        )
        training[arm] = TrainingSummary(
            stable_tokens_per_second=median_tps,
            stable_sample_count=len(stable_samples[arm].get(rank, ())),
            baseline_ratio=ratio,
            peak_memory_gib=peak_memory,
            tokens_per_second_global=(
                median_tps * world_size if median_tps is not None else None
            ),
            rank_reduction="min_over_ranks",
            published_rank=rank,
            ranks=tuple(sorted(throughput[arm])),
            per_rank=tuple(
                RankThroughput(
                    rank=each,
                    stable_tokens_per_second=throughput[arm][each],
                    stable_sample_count=len(stable_samples[arm][each]),
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

    # The twin of the caption on baseline_kernel_ratio, for the same reason:
    # each side of the ratio names its own slowest rank, and under a pipeline
    # split those two rank indices hold different partitions of the model.
    for arm in arms:
        if baseline is None or arm == baseline:
            continue
        if (
            training[arm].published_rank
            != training[baseline].published_rank
        ):
            warnings.append(
                f"{arm}: the tokens/s 'ratio' divides rank "
                f"{training[arm].published_rank} by baseline rank "
                f"{training[baseline].published_rank}; each side is its "
                "own slowest rank, so under a pipeline split the two hold "
                "different partitions of the model"
            )

    # One rank's trajectory, not every rank's concatenated. Under a pipeline
    # split the loss lives on the last stage, and a rank without it still
    # prints a step line -- carrying TorchTitan's -1.0 sentinel.
    trajectory_rank = loss_visible_rank(
        world_size=world_size, pp=int(recorded_parallelism.get("pp", 1))
    )
    # Before anything is published. Every rank's lines, not only the
    # published rank's; see refuse_non_finite_trajectories.
    for arm in arms:
        refuse_non_finite_trajectories(arm, out_dir / f"{arm}.log")
    return EvaluationResult(
        output_dir=str(out_dir),
        scenario=manifest.get("scenario", "unknown"),
        hardware=manifest.get("hardware", "unknown"),
        arms=tuple(arms),
        trace_windows=trace_windows,
        gpu_time=gpu_time,
        comparisons=comparisons,
        training=training,
        losses={
            arm: losses(out_dir / f"{arm}.log", rank=trajectory_rank)
            for arm in arms
        },
        gradient_norms={
            arm: grad_norms(out_dir / f"{arm}.log", rank=trajectory_rank)
            for arm in arms
        },
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
    # evaluate_run refuses a non-finite trajectory before it builds a
    # result, so this label is reached only by a result built by hand.
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


def _render_rank_throughput(result: EvaluationResult) -> list[str]:
    """Each rank's own tokens/s, and the global figure the mesh reached.

    Printed only when a run holds more than one rank. Every run recorded
    before this existed held one, so a single-GPU report is unchanged,
    character for character.
    """
    if not any(len(result.training[arm].ranks) > 1 for arm in result.arms):
        return []
    lines = [
        "",
        "per-rank tokens/s (published figure is the MIN over ranks, never "
        "the mean):",
        "  " + f"{'arm':22s} {'rank':>4s} {'tokens/s':>12s} {'n':>4s}",
    ]
    for arm in result.arms:
        training = result.training[arm]
        for rank in training.per_rank:
            marker = "*" if rank.rank == training.published_rank else " "
            lines.append(
                f"  {arm:22s} {rank.rank:>3d}{marker} "
                f"{_value(rank.stable_tokens_per_second, 12)} "
                f"{rank.stable_sample_count:4d}"
            )
        lines.append(
            f"  {arm:22s} all  "
            f"{_value(training.tokens_per_second_global, 12)}"
        )
    lines.extend(
        [
            "* = the published rank. Every published tokens/s is per device, "
            "and the 'all' row",
            "is that figure times the world size. The manifest names the "
            "definition in throughput_definition.",
        ]
    )
    return lines


def render_evaluation(result: EvaluationResult) -> str:
    """Render the complete stable-throughput and GPU-time report."""
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
            + f"{'launch us':>10s}",
        ]
    )
    for arm in result.arms:
        gpu = result.gpu_time[arm]
        lines.append(
            f"  {arm:22s} "
            f"{_value(gpu.kernel_ms_per_step, 14, 2)} "
            f"{_value(gpu.baseline_kernel_ratio, 8, 4)} "
            f"{_value(gpu.launch_latency_us, 10, 2)}"
        )

    lines.extend(_render_rank_throughput(result))
    lines.extend(_render_rank_split(result))

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

    return "\n".join(lines)
