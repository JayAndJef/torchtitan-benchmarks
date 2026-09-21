"""Extract GPU-time measurements from profiler traces.

**Kernel time** is the summed duration of the GPU kernel/memcpy/memset
events in a window. It depends only on the GPU and is the comparable
measurement when host speed varies between runs.

Three further per-window totals exist because the summed kernel time alone
stops answering the question once a run holds more than one rank:

- ``collective_us``. An NCCL kernel carries ``cat: "kernel"``, so it joins
  the sum like any other. A *blocking* collective also waits for its peer,
  and that wait sits inside the kernel's own duration -- so counting it as
  compute absorbs the pipeline bubble as work and makes one rank's figure
  depend on another rank's speed. ``compute_ms_per_step`` is the total
  without them and is the value that compares to a single-GPU run.
- ``busy_kernel_us``. The summed basis double-counts whatever overlaps:
  megatron already runs five CUDA streams to titan's one, which overstates
  its sum by about 6.5%, and NCCL adds its own stream on top. ``busy_union``
  is the interval-union basis, and both are recorded rather than one being
  chosen.
- ``step_wall_us``. A bubble runs no kernel, so no kernel-derived total sees
  it. The profiler's own ``ProfilerStep#`` annotation carries the step's wall
  clock, and ``bubble = wall - busy`` is what explains a pipeline result.

**The NCCL name prefix is confirmed against a real multi-rank trace.** A
``pp 2`` megatron run of this harness puts 35 collective device kernels in
each rank's window, every one of them named ``ncclDevKernel_*`` and every one
categorised as a kernel. The constant below carries the counts and the three
kernel names. **What the split is confirmed for is a pipeline**: no
data-parallel run has been read, and a fourth collective name could appear
there. Re-check the constant before publishing a data-parallel split.

Pooling happens **per rank** and never across ranks. ``pooled_window_metrics``
pools one rank's windows; two ranks of input would give an arithmetic mean,
which is neither one rank's cost nor the step's, so it refuses such a call
outright. ``per_rank_pooled_metrics`` is the entry point a measurement uses.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from benchmarks.artifacts.layout import rank_of_trace


PROFILER_STEP_TAG = "ProfilerStep#"
LAUNCH_CATEGORIES = frozenset({"cuda_runtime", "cuda_driver"})
LAUNCH_PREFIXES = ("cudaLaunchKernel", "cuLaunchKernel")
KERNEL_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})
COLLECTIVE_KERNEL_PREFIXES = ("nccl", "ncclDevKernel")
"""What a communication kernel is called.

``ncclDevKernel`` is the form torch's profiler emits today and ``nccl``
already covers it. Both are named so a rename of the longer one is visible
here rather than a silent reclassification of every collective as compute.

Confirmed against a real multi-rank trace: a ``pp 2`` megatron run carries
35 collective device kernels per window on each rank, and every one begins
with ``ncclDevKernel_`` under ``cat: "kernel"``. Without this split a
pipeline's wait time lands in the compute total.

Warning: before you widen this tuple, re-scan the arm directories for a
device kernel whose name begins with ``nccl``. A false positive shrinks a
published number silently rather than mis-splitting a new one alone.
"""


def busy_union(intervals: Iterable[tuple[float, float]]) -> float:
    """Wall time during which at least one interval is open.

    The overlap-correct counterpart to summing durations. Two kernels on two
    streams that run at the same time cost the device one wall interval, not
    two, so the sum overstates any arm that uses more than one stream --
    megatron's five against titan's one already overstate it by about 6.5%,
    which was once enough to invert the sign of a comparison. Collectives run
    on a stream of their own, so every multi-rank run has the same problem.
    """
    total = 0.0
    current_start: float | None = None
    current_end = 0.0
    for start, end in sorted(intervals):
        if current_start is None:
            current_start, current_end = start, end
        elif start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        total += current_end - current_start
    return total


@dataclass(frozen=True)
class WindowMetrics:
    """GPU-time measurements extracted from one profiler window."""

    kernel_total_us: float
    profiled_steps: int
    launch_total_us: float
    launch_count: int
    collective_us: float
    busy_kernel_us: float
    step_wall_us: float
    step_wall_count: int


@dataclass(frozen=True)
class PooledMetrics:
    """One rank's window metrics, pooled across that rank's windows.

    Every field is a total over the windows of a **single** rank. Nothing
    here is ever a quantity averaged over ranks; see the module docstring.
    """

    windows: int
    kernel_total_us: float
    profiled_steps: int
    launch_total_us: float
    launch_count: int
    collective_us: float
    busy_kernel_us: float
    step_wall_us: float
    step_wall_count: int

    @property
    def kernel_ms_per_step(self) -> float | None:
        if not self.profiled_steps:
            return None
        return self.kernel_total_us / self.profiled_steps / 1000.0

    @property
    def collective_ms_per_step(self) -> float | None:
        """Communication kernels alone. A blocking one includes peer wait."""
        if not self.profiled_steps:
            return None
        return self.collective_us / self.profiled_steps / 1000.0

    @property
    def compute_ms_per_step(self) -> float | None:
        """Kernel time without the collectives: what compares to one GPU."""
        if not self.profiled_steps:
            return None
        return (
            (self.kernel_total_us - self.collective_us)
            / self.profiled_steps
            / 1000.0
        )

    @property
    def busy_kernel_ms_per_step(self) -> float | None:
        """The same device work on the interval-union basis, not the sum."""
        if not self.profiled_steps:
            return None
        return self.busy_kernel_us / self.profiled_steps / 1000.0

    @property
    def wall_ms_per_step(self) -> float | None:
        """The profiler step's own wall clock. ``wall - busy`` is the bubble.

        Averaged over the steps that carried a host annotation, not over
        ``profiled_steps``. The two agree on every trace either engine writes:
        a sweep of 367 arm directories under ``out/`` found ``ProfilerStep#``
        under ``user_annotation`` and ``gpu_user_annotation`` only, never
        ``cpu_op``, and found the two counts equal in every one.

        They can part in two ways, and the two answers differ. Where **no**
        step carries a host annotation this returns ``None``, because the
        honest answer is "no wall was recorded" rather than a total spread
        over steps that never reported one. Where **some** steps carry one and
        others do not, the extraction refuses the trace instead: this average
        and every other per-step figure would then run over different
        denominators, and ``wall - busy`` would subtract a two-step average
        from a one-step average and call the difference a bubble.
        """
        if not self.step_wall_count:
            return None
        return self.step_wall_us / self.step_wall_count / 1000.0

    @property
    def launch_latency_us(self) -> float | None:
        if not self.launch_count:
            return None
        return self.launch_total_us / self.launch_count


def _load_events(trace_path: Path) -> list[dict]:
    try:
        with gzip.open(trace_path, "rt") as trace_file:
            return json.load(trace_file)["traceEvents"]
    except (OSError, EOFError, ValueError, KeyError) as error:
        raise ValueError(f"{trace_path}: unreadable profiler trace: {error}") from error


def trace_window_metrics(trace_path: Path) -> WindowMetrics:
    """Extract one window's measurements."""
    events = _load_events(trace_path)

    # ``step_walls`` reads the host annotation alone, because the step's
    # wall clock is the host one.
    step_names: set[str] = set()
    step_walls: dict[str, float] = {}
    device_intervals: list[tuple[float, float]] = []
    kernel_total = 0.0
    collective_total = 0.0
    launch_total = 0.0
    launch_count = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        category = event.get("cat")
        start = event.get("ts", 0.0)
        end = start + event.get("dur", 0.0)
        if category in KERNEL_CATEGORIES:
            kernel_total += end - start
            device_intervals.append((start, end))
            if name.startswith(COLLECTIVE_KERNEL_PREFIXES):
                collective_total += end - start
        elif category in LAUNCH_CATEGORIES and name.startswith(LAUNCH_PREFIXES):
            launch_total += end - start
            launch_count += 1
        elif name.startswith(PROFILER_STEP_TAG):
            step_names.add(name)
            if category == "user_annotation":
                step_walls[name] = max(step_walls.get(name, 0.0), end - start)

    if 0 < len(step_walls) < len(step_names):
        raise ValueError(
            "some profiler steps carry a host annotation and others do not; "
            "the wall clock would average over a different step count than "
            "the kernel totals"
        )

    return WindowMetrics(
        kernel_total_us=kernel_total,
        profiled_steps=len(step_names),
        launch_total_us=launch_total,
        launch_count=launch_count,
        collective_us=collective_total,
        busy_kernel_us=busy_union(device_intervals),
        step_wall_us=sum(step_walls.values()),
        step_wall_count=len(step_walls),
    )


def _refuse_mixed_ranks(trace_paths: tuple[Path, ...]) -> None:
    """Refuse windows from more than one rank.

    Pooling is a sum over windows divided by a step count, so two ranks of
    input produce an arithmetic mean across ranks. That is neither one rank's
    cost -- which is what a per-step figure claims to be -- nor the step's
    total work, and it would pass every other check in this file silently.
    Names that carry no rank token are left alone: the synthetic traces the
    tests write are named for what they test.
    """
    ranks = {
        rank
        for path in trace_paths
        if (rank := rank_of_trace(path)) is not None
    }
    if len(ranks) > 1:
        raise ValueError(
            f"windows from {len(ranks)} ranks {sorted(ranks)} reached one "
            "pooling call; pooling them would average across ranks, which is "
            "neither one rank's cost nor the step's. Group with "
            "trace_files_by_rank and pool with per_rank_pooled_metrics"
        )


def pooled_window_metrics(trace_paths: Iterable[Path]) -> PooledMetrics:
    """Pool one rank's profiler windows, validating each window.

    Windows only. Every field of the result is a total over the windows of a
    single rank, and a call carrying two ranks' files is refused rather than
    averaged.
    """
    trace_paths = tuple(trace_paths)
    _refuse_mixed_ranks(trace_paths)
    windows = 0
    kernel_total = 0.0
    collective_total = 0.0
    busy_total = 0.0
    step_wall_total = 0.0
    step_wall_count = 0
    step_counts: list[int] = []
    launch_total = 0.0
    launch_count = 0
    for trace_path in trace_paths:
        window = trace_window_metrics(trace_path)
        windows += 1
        kernel_total += window.kernel_total_us
        collective_total += window.collective_us
        busy_total += window.busy_kernel_us
        step_wall_total += window.step_wall_us
        step_wall_count += window.step_wall_count
        step_counts.append(window.profiled_steps)
        launch_total += window.launch_total_us
        launch_count += window.launch_count
    if windows == 0:
        raise ValueError("no profiler trace windows supplied")
    if 0 < sum(1 for count in step_counts if count) < windows:
        raise ValueError(
            "some trace windows carry ProfilerStep events and others do not; "
            "per-step totals would be wrong for this mixture"
        )
    if 0 < step_wall_count < sum(step_counts):
        raise ValueError(
            "some trace windows name their steps on the host and others only "
            "on the device; the wall clock would average over a different "
            "step count than the kernel totals"
        )
    return PooledMetrics(
        windows=windows,
        kernel_total_us=kernel_total,
        profiled_steps=sum(step_counts),
        launch_total_us=launch_total,
        launch_count=launch_count,
        collective_us=collective_total,
        busy_kernel_us=busy_total,
        step_wall_us=step_wall_total,
        step_wall_count=step_wall_count,
    )


def per_rank_pooled_metrics(
    files_by_rank: Mapping[int, Iterable[Path]],
) -> dict[int, PooledMetrics]:
    """Pool each rank's windows separately, in rank order.

    The measurement entry point for anything that publishes a number. Ranks
    stay apart all the way to the point where the reduction across them is
    named, so no code path can turn them into a mean by accident. A
    single-GPU run gives one entry and its value is exactly what
    ``pooled_window_metrics`` returned before ranks existed.
    """
    if not files_by_rank:
        raise ValueError("no profiler trace windows supplied")
    pooled: dict[int, PooledMetrics] = {}
    for rank in sorted(files_by_rank):
        try:
            pooled[rank] = pooled_window_metrics(files_by_rank[rank])
        except ValueError as error:
            raise ValueError(f"rank {rank}: {error}") from error
    return pooled
