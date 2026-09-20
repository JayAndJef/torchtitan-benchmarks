"""Extraction and rendering of end-to-end run metrics.

Everything an evaluated output directory turns into: the log-derived
throughput and trajectory series, the machine-readable ``results.json``
payload, and the human-readable report the CLI prints.

**Evaluation reads the logs alone.** Both engines print every figure this
module publishes on a step line, under ``--profile`` and without it, so a
directory evaluates the same way in both modes. The traces stay a
validation input; no number here comes from one.

**How a run with more than one rank becomes one published number.** The
headline tokens/s is the **minimum** over ranks, never the mean: a parallel
schedule locks the ranks together at every step boundary, so the mesh runs
at the pace of its slowest rank and a mean would report a rate nobody
reached. ``published_rank`` names the rank the headline came from, and
``per_rank`` carries every rank's own figures beside it.

**Each arm reports absolute numbers.** No arm is a baseline and nothing is
a ratio: the three engine arms share no implementation, so a reader
compares two absolute rows rather than one derived number.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.artifacts.layout import atomic_write_json, logs_by_rank
from benchmarks.artifacts.manifests import load_run
from benchmarks.artifacts.summaries import _value
from benchmarks.e2e.schema import DEFAULT_ZERO, ParallelismSpec
from benchmarks.e2e.parallelism import zero_warnings


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


@dataclass(frozen=True)
class StepMs:
    """How long one training step took, in milliseconds.

    Derived from the throughput samples rather than measured beside them:
    a step line carries tokens per second, and one step moves a known
    number of tokens. ``series`` holds one value per sample, in step order,
    so a reader can see the spread the three statistics summarize.

    ``p95`` uses the **nearest-rank** method: the series is sorted and the
    value at 1-based index ``ceil(0.95 * n)`` is taken. It is therefore
    always a measured step and never an interpolation between two.
    """

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
    """One arm's published figures.

    ``stable_tokens_per_second`` is the **minimum** over ranks, and
    ``step_ms`` is that same rank's step cost. See this module's docstring
    for why the reduction is a minimum and not a mean. At one rank it is
    that rank's own median.

    ``peak_memory_gib`` is the maximum over every rank, which needs no
    reduction rule: it is the most memory any device in the mesh held.
    """

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
    """Select post-compile steps before each profiler warmup begins.

    The sample rule of a **profiled** run. The profiler costs host and GPU
    time on every warmup and active step, so the samples are the steps of
    each cycle that carry none: steps 2 to ``wait`` of every
    ``profile_freq``. Step 1 of each cycle is dropped as startup noise.
    """
    profile_freq = int(workload.get("profile_freq", 20))
    wait = profile_freq - int(workload.get("profiler_warmup", 5)) - int(
        workload.get("profiler_active", 5)
    )
    return [
        tps
        for step, _, tps in rows
        if 2 <= ((step - 1) % profile_freq) + 1 <= wait
    ]


def measured_tps(
    rows: list[tuple[int, float, int]], warmup_steps: int
) -> list[int]:
    """Select every step after the warmup.

    The sample rule of an **unprofiled** run, and it is a different figure
    from ``stable_tps`` rather than a wider reading of the same one. There
    is no profiler to sample around, so every step after the warmup is a
    sample: the count grows with ``--steps`` where the profiled rule's does
    not, and the median is taken over steps the profiled rule discards.
    Numbers are only comparable within one value of ``--profile``, and
    within one ``--warmup-steps``.

    The step numbers come from the engine's own log lines, which both
    engines number from 1.
    """
    return [tps for step, _, tps in rows if step > warmup_steps]


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


def _nearest_rank_percentile(values: list[float], fraction: float) -> float:
    """The nearest-rank percentile: a measured value, never an interpolation."""
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered))
    return ordered[max(index, 1) - 1]


def step_ms(
    samples: list[int], *, tokens_per_step: int, pp: int
) -> StepMs:
    """Turn one rank's throughput samples into its step costs.

    A sample says how many tokens per second the rank moved, and one step
    moves ``tokens_per_step`` of them, so the step took
    ``1000 * tokens_per_step / (tps * pp)`` milliseconds. The pipeline
    degree divides it because a pipeline stage holds a slice of the model
    and the step line counts the whole batch's tokens against it.

    A sample of zero is dropped rather than published. It describes a step
    with no measured rate, and its step cost is not a number.
    """
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


def evaluate_run(
    out_dir: Path, arms_override: list[str] | tuple[str, ...] | None = None
) -> EvaluationResult:
    """Evaluate the end-to-end metrics of the selected arms."""
    out_dir = out_dir.resolve()
    manifest, arms = load_run(out_dir, arms_override)
    warnings: list[str] = []

    # The run axis, read back from the manifest. It picks the sample rule
    # alone: every figure below comes from the step lines, which both modes
    # print.
    profile = bool(manifest["profile"])

    workload = manifest["workload"]
    # The declared mesh, read back from the manifest.
    recorded_parallelism = manifest["parallelism"]
    world_size = int(recorded_parallelism.get("world_size", 1))
    # The two ZeRO-level warnings reach the artifact as well as the
    # console. The runner says them when the run starts, and a reader of
    # results.json was not there. The file is what a report quotes.
    #
    # ``zero_warnings`` reads a spec, so the record becomes a spec
    # again. Only the fields that function reads are rebuilt: a record also
    # carries keys the spec derives for itself, such as ``world_size``.
    #
    # It also reads the engines the evaluated arms run on, because one
    # warning is about TorchTitan's FSDP2 alone. The manifest arm records
    # carry the engine, so the file states the same facts the console did.
    warnings.extend(
        zero_warnings(
            ParallelismSpec(
                dp=int(recorded_parallelism.get("dp", 1)),
                pp=int(recorded_parallelism.get("pp", 1)),
                ep=int(recorded_parallelism.get("ep", 1)),
                zero=int(recorded_parallelism.get("zero", DEFAULT_ZERO)),
            ),
            engines=[
                str(record.get("engine", ""))
                for record in manifest.get("arms", ())
                if record.get("name") in set(arms)
            ],
        )
    )
    raw_training = {
        arm: per_rank_training_metrics(out_dir / f"{arm}.log") for arm in arms
    }
    # The sample rule follows the axis the run was measured under. A
    # profiled run samples around its profiler windows; an unprofiled one
    # takes every step after its warmup. Choosing by the manifest is what
    # keeps a directory readable by the rule that produced it.
    if profile:
        def _samples(rows: list[tuple[int, float, int]]) -> list[int]:
            return stable_tps(rows, workload)
    else:
        warmup_steps = int(manifest["warmup_steps"])

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
    # One step's tokens, and the degree that divides its cost. Both come
    # from the manifest, so a directory reports the step cost of the run
    # that wrote it.
    tokens_per_step = int(workload["local_batch_size"]) * int(
        workload["seq_len"]
    )
    pp = int(recorded_parallelism.get("pp", 1))
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

    # One rank's trajectory, not every rank's concatenated. Under a pipeline
    # split the loss lives on the last stage, and a rank without it still
    # prints a step line -- carrying TorchTitan's -1.0 sentinel.
    trajectory_rank = loss_visible_rank(world_size=world_size, pp=pp)
    # Before anything is published. Every rank's lines, not only the
    # published rank's; see refuse_non_finite_trajectories.
    for arm in arms:
        refuse_non_finite_trajectories(arm, out_dir / f"{arm}.log")
    return EvaluationResult(
        output_dir=str(out_dir),
        scenario=manifest.get("scenario", "unknown"),
        hardware=manifest.get("hardware", "unknown"),
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
    # evaluate_run refuses a non-finite trajectory before it builds a
    # result, so this label is reached only by a result built by hand.
    if not all(math.isfinite(value) for _, value in values):
        rendered += f"   NON-FINITE {nonfinite_label}"
    return rendered


def render_evaluation(result: EvaluationResult) -> str:
    """Render one scenario's table, its trajectories and its warnings.

    One row per arm, and every figure is absolute. The arms of the
    ``engines`` scenario share no implementation, so a ratio between two of
    them would name a difference no component of either arm owns.
    """
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
