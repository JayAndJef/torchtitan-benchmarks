"""Extract declared compiled forward/backward regions from profiler traces.

The benchmark measurement is the GPU span of whole compiled regions
(``## Call CompiledFxGraph`` GPU annotations), not individual generated
kernels: Inductor kernel names are unstable across torch versions and arms.

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
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from benchmarks.scenarios import Region


COMPILED_GRAPH_TAG = "## Call CompiledFxGraph"
BACKWARD_FRAME = "CompiledFunctionBackward"


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
    gpu_graphs: dict[str, list[float]],
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


def trace_region_samples(
    trace_path: Path,
    regions: Iterable[Region],
) -> dict[str, list[float]]:
    """Return declared region samples from one trace or fail on partition changes."""
    events = _load_events(trace_path)

    backward_frames: dict[int, list[tuple[float, float]]] = defaultdict(list)
    cpu_calls: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    gpu_graphs: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        category = event.get("cat")
        start = event.get("ts", 0.0)
        end = start + event.get("dur", 0.0)
        if category == "cpu_op" and name == BACKWARD_FRAME:
            backward_frames[event.get("tid")].append((start, end))
        elif name.startswith(COMPILED_GRAPH_TAG):
            if category == "user_annotation":
                cpu_calls[name].append((event.get("tid"), start, end))
            elif category == "gpu_user_annotation":
                gpu_graphs[name].append(event.get("dur", 0.0))

    phases = _graph_phases(trace_path, backward_frames, cpu_calls, gpu_graphs)

    samples: dict[str, list[float]] = {}
    for region in regions:
        candidates = [
            graph
            for graph, values in gpu_graphs.items()
            if phases[graph] == region.phase
            and len(values) == region.invocations_per_window
        ]
        if len(candidates) != 1:
            inventory = ", ".join(
                f"{_short_hash(graph)}...({phases[graph]}, n={len(values)})"
                for graph, values in sorted(gpu_graphs.items())
            ) or "none"
            raise ValueError(
                f"{trace_path}: expected exactly one {region.phase} compiled "
                f"graph with {region.invocations_per_window} invocations for "
                f"region {region.name!r}, found {len(candidates)}; the compiled "
                f"partitioning changed and the graph-to-region mapping is no "
                f"longer valid. Graphs in trace: {inventory}"
            )
        samples[region.name] = gpu_graphs[candidates[0]]
    return samples


def pooled_region_samples(
    trace_paths: Iterable[Path],
    regions: Iterable[Region],
) -> dict[str, list[float]]:
    """Pool region samples across profiler windows, validating each window."""
    regions = tuple(regions)
    pooled: dict[str, list[float]] = {region.name: [] for region in regions}
    window_count = 0
    for trace_path in trace_paths:
        window_count += 1
        for name, values in trace_region_samples(trace_path, regions).items():
            pooled[name].extend(values)
    if window_count == 0:
        raise ValueError("no profiler trace windows supplied")
    return pooled
