"""Extract GPU-time measurements from profiler traces.

Two kinds of measurement come out of each trace window:

- **Region spans**: the GPU span of whole compiled regions (``## Call
  CompiledFxGraph`` GPU annotations), first kernel to last. A span includes
  the idle gaps where the GPU waited on the host, so it moves with host
  dispatch speed.
- **Kernel time**: the summed durations of the GPU kernel/memcpy/memset
  events inside each region span, plus the per-window total across all
  kernels. Kernel time depends only on the GPU and is the comparable
  measurement when host speed varies between runs.

Whole compiled regions are used rather than individual generated kernels:
Inductor kernel names are unstable across torch versions and arms.

Regions are matched structurally, not by name or size. Each compiled graph's
direction is read from the CPU side of the trace — backward graph calls nest
inside ``CompiledFunctionBackward`` autograd frames — and the scenario's
declared per-window invocation count then picks the transformer-block graph
among same-phase partitions. The count doubles as a guard: if the compiler
partitions the model differently, no graph matches and extraction fails
instead of silently mislabeling a region.
"""

from __future__ import annotations

import gzip
import json
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from benchmarks.scenarios import Region


COMPILED_GRAPH_TAG = "## Call CompiledFxGraph"
BACKWARD_FRAME = "CompiledFunctionBackward"
PROFILER_STEP_TAG = "ProfilerStep#"
LAUNCH_CATEGORIES = frozenset({"cuda_runtime", "cuda_driver"})
LAUNCH_PREFIXES = ("cudaLaunchKernel", "cuLaunchKernel")
KERNEL_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})


@dataclass(frozen=True)
class WindowMetrics:
    """GPU-time measurements extracted from one profiler window."""

    region_spans: dict[str, list[float]]
    region_kernel: dict[str, list[float]]
    kernel_total_us: float
    profiled_steps: int
    launch_total_us: float
    launch_count: int


@dataclass(frozen=True)
class PooledMetrics:
    """Window metrics pooled across the run's profiler windows."""

    region_spans: dict[str, list[float]]
    region_kernel: dict[str, list[float]]
    windows: int
    kernel_total_us: float
    profiled_steps: int
    launch_total_us: float
    launch_count: int

    @property
    def kernel_ms_per_step(self) -> float | None:
        if not self.profiled_steps:
            return None
        return self.kernel_total_us / self.profiled_steps / 1000.0

    @property
    def region_kernel_ms_per_step(self) -> float | None:
        if not self.profiled_steps:
            return None
        total = sum(sum(values) for values in self.region_kernel.values())
        return total / self.profiled_steps / 1000.0

    @property
    def launch_latency_us(self) -> float | None:
        if not self.launch_count:
            return None
        return self.launch_total_us / self.launch_count


def _short_hash(graph: str) -> str:
    """Abbreviate '## Call CompiledFxGraph <hash> ##' for error messages."""
    parts = graph.split()
    return (parts[3] if len(parts) > 3 else graph)[:16]


def _load_events(trace_path: Path) -> list[dict]:
    try:
        with gzip.open(trace_path, "rt") as trace_file:
            return json.load(trace_file)["traceEvents"]
    except (OSError, EOFError, ValueError, KeyError) as error:
        raise ValueError(f"{trace_path}: unreadable profiler trace: {error}") from error


def _graph_phases(
    trace_path: Path,
    backward_frames: dict[int, list[tuple[float, float]]],
    cpu_calls: dict[str, list[tuple[int, float, float]]],
    gpu_graphs: dict[str, list],
) -> dict[str, str]:
    """Classify each compiled graph as forward or backward via autograd frames."""
    phases: dict[str, str] = {}
    for graph, calls in cpu_calls.items():
        inside = {
            any(
                start <= call_start and call_end <= end
                for start, end in backward_frames.get(tid, ())
            )
            for tid, call_start, call_end in calls
        }
        if len(inside) != 1:
            raise ValueError(
                f"{trace_path}: graph {_short_hash(graph)}... appears both "
                f"inside and outside {BACKWARD_FRAME} frames; cannot classify"
            )
        phases[graph] = "backward" if inside.pop() else "forward"
    for graph in gpu_graphs:
        if graph not in phases:
            raise ValueError(
                f"{trace_path}: graph {_short_hash(graph)}... has GPU spans "
                f"but no CPU-side annotations; the trace lacks the CPU-side "
                f"autograd context needed to classify forward vs backward"
            )
    return phases


def _kernel_time_within(
    spans: list[tuple[int, int, float, float]],
    stream_kernels: dict[tuple[int, int], tuple[list[float], list[float]]],
) -> list[float]:
    """Sum kernel durations inside each span, per invocation."""
    busy = []
    for pid, tid, start, end in spans:
        starts, ends = stream_kernels.get((pid, tid), ((), ()))
        total = 0.0
        index = bisect_left(starts, start)
        while index < len(starts) and starts[index] < end:
            if ends[index] <= end:
                total += ends[index] - starts[index]
            index += 1
        busy.append(total)
    return busy


def trace_window_metrics(
    trace_path: Path,
    regions: Iterable[Region],
) -> WindowMetrics:
    """Extract one window's measurements or fail on partition changes."""
    events = _load_events(trace_path)

    backward_frames: dict[int, list[tuple[float, float]]] = defaultdict(list)
    cpu_calls: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    gpu_spans: dict[str, list[tuple[int, int, float, float]]] = defaultdict(list)
    raw_kernels: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    step_names: set[str] = set()
    kernel_total = 0.0
    launch_total = 0.0
    launch_count = 0
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        category = event.get("cat")
        start = event.get("ts", 0.0)
        end = start + event.get("dur", 0.0)
        if category == "cpu_op" and name == BACKWARD_FRAME:
            backward_frames[event.get("tid")].append((start, end))
        elif category in KERNEL_CATEGORIES:
            kernel_total += end - start
            raw_kernels[(event.get("pid"), event.get("tid"))].append((start, end))
        elif category in LAUNCH_CATEGORIES and name.startswith(LAUNCH_PREFIXES):
            launch_total += end - start
            launch_count += 1
        elif name.startswith(PROFILER_STEP_TAG):
            step_names.add(name)
        elif name.startswith(COMPILED_GRAPH_TAG):
            if category == "user_annotation":
                cpu_calls[name].append((event.get("tid"), start, end))
            elif category == "gpu_user_annotation":
                gpu_spans[name].append(
                    (event.get("pid"), event.get("tid"), start, end)
                )

    stream_kernels = {}
    for stream, intervals in raw_kernels.items():
        intervals.sort()
        stream_kernels[stream] = (
            [interval[0] for interval in intervals],
            [interval[1] for interval in intervals],
        )

    phases = _graph_phases(trace_path, backward_frames, cpu_calls, gpu_spans)

    region_spans: dict[str, list[float]] = {}
    region_kernel: dict[str, list[float]] = {}
    for region in regions:
        candidates = [
            graph
            for graph, spans in gpu_spans.items()
            if phases[graph] == region.phase
            and len(spans) == region.invocations_per_window
        ]
        if len(candidates) != 1:
            inventory = ", ".join(
                f"{_short_hash(graph)}...({phases[graph]}, n={len(spans)})"
                for graph, spans in sorted(gpu_spans.items())
            ) or "none"
            raise ValueError(
                f"{trace_path}: expected exactly one {region.phase} compiled "
                f"graph with {region.invocations_per_window} invocations for "
                f"region {region.name!r}, found {len(candidates)}; the compiled "
                f"partitioning changed and the graph-to-region mapping is no "
                f"longer valid. Graphs in trace: {inventory}"
            )
        spans = gpu_spans[candidates[0]]
        region_spans[region.name] = [end - start for _, _, start, end in spans]
        region_kernel[region.name] = _kernel_time_within(spans, stream_kernels)

    return WindowMetrics(
        region_spans=region_spans,
        region_kernel=region_kernel,
        kernel_total_us=kernel_total,
        profiled_steps=len(step_names),
        launch_total_us=launch_total,
        launch_count=launch_count,
    )


def pooled_window_metrics(
    trace_paths: Iterable[Path],
    regions: Iterable[Region],
) -> PooledMetrics:
    """Pool measurements across profiler windows, validating each window."""
    regions = tuple(regions)
    spans: dict[str, list[float]] = {region.name: [] for region in regions}
    kernel: dict[str, list[float]] = {region.name: [] for region in regions}
    windows = 0
    kernel_total = 0.0
    step_counts: list[int] = []
    launch_total = 0.0
    launch_count = 0
    for trace_path in trace_paths:
        window = trace_window_metrics(trace_path, regions)
        windows += 1
        for name in spans:
            spans[name].extend(window.region_spans[name])
            kernel[name].extend(window.region_kernel[name])
        kernel_total += window.kernel_total_us
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
    return PooledMetrics(
        region_spans=spans,
        region_kernel=kernel,
        windows=windows,
        kernel_total_us=kernel_total,
        profiled_steps=sum(step_counts),
        launch_total_us=launch_total,
        launch_count=launch_count,
    )
