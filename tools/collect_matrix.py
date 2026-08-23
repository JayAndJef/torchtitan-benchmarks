"""Merge a run_matrix.sh output tree into one machine-readable JSON.

One object per cell: the manifest's provenance/workload/model_shape/axes
joined to results.json's throughput, gpu_time, comparisons and warnings,
plus the supervisor's contamination verdict for that cell.

    .venv/bin/python tools/collect_matrix.py out/matrix-<utc> \\
        [--size huge] [--launch-counts] [--out huge-matrix.json]

--launch-counts re-reads the profiler traces to add each arm's kernel-launch
count, which results.json does not carry. It is slow (a full trace parse per
arm), so it is opt-in and normally used only on the cells a report tabulates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def cells(root: Path, size_filter: str | None):
    """Yield (size, ac, mode, scenario, cell_dir) for every complete cell.

    The size comes from the cell's own manifest, not from the directory name.
    A directory name is what the supervisor was asked for and a manifest
    records what the run actually was, and the two can differ by a retired
    alias: a tree laid out as ``normal/`` holds manifests that record ``1b``.
    ``--size`` is canonicalised for the same reason, so either spelling
    selects the same cells. The directory name is the fallback only for a
    manifest that records no size at all (schema <= 8).
    """
    from benchmarks.models.piper_qwen3.shape import canonical_size_name

    wanted = canonical_size_name(size_filter) if size_filter else None
    for manifest_path in sorted(root.glob("*/ac-*/*/*/manifest.json")):
        cell_dir = manifest_path.parent
        scenario = cell_dir.name
        mode = cell_dir.parent.name
        ac = cell_dir.parent.parent.name.removeprefix("ac-")
        try:
            recorded = json.loads(manifest_path.read_text()).get("model_size")
        except (OSError, json.JSONDecodeError):
            recorded = None
        size = canonical_size_name(
            str(recorded) if recorded else cell_dir.parent.parent.parent.name
        )
        if wanted and size != wanted:
            continue
        yield size, ac, mode, scenario, cell_dir


def launch_counts(cell_dir: Path, arms: list[str]) -> dict[str, float]:
    """Launches per step, per arm: the maximum over the arm's ranks.

    The maximum for the same reason ``results.json`` publishes a maximum --
    the ranks run in step, so the busiest one sets the cost. A mean over ranks
    would be a number no rank paid.

    A rank whose windows carry no ``ProfilerStep`` annotation has no per-step
    figure and is left out of the maximum. Dividing its launch count by 1
    instead would give it the whole window's launches as a "per step" value,
    which is the largest number in the set and would therefore win the
    maximum outright -- turning a rank the profiler never measured into the
    arm's published figure.
    """
    from benchmarks.artifacts.layout import trace_files_by_rank
    from benchmarks.traces.extraction import per_rank_pooled_metrics

    counts: dict[str, float] = {}
    for arm in arms:
        by_rank = trace_files_by_rank(cell_dir / arm)
        if not by_rank:
            continue
        try:
            pooled = per_rank_pooled_metrics(by_rank, ())
        except ValueError:
            continue
        per_step = [
            rank.launch_count / rank.profiled_steps
            for rank in pooled.values()
            if rank.profiled_steps
        ]
        if per_step:
            counts[arm] = max(per_step)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--size", default=None)
    parser.add_argument("--launch-counts", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    collected = []
    for size, ac, mode, scenario, cell_dir in cells(args.root, args.size):
        manifest = json.loads((cell_dir / "manifest.json").read_text())
        results_path = cell_dir / "results.json"
        results = (
            json.loads(results_path.read_text()) if results_path.exists() else None
        )
        marker = cell_dir.with_suffix(cell_dir.suffix + ".CONTAMINATED")
        watch = cell_dir.with_suffix(cell_dir.suffix + ".watch")
        entry = {
            "cell": f"{size}|{ac}|{mode}|{scenario}",
            "model_size": size,
            "ac_mode": ac,
            "compile_mode": mode,
            "scenario": scenario,
            "path": str(cell_dir),
            "contaminated": marker.exists(),
            "watchdog_flags": (
                watch.read_text().splitlines() if watch.exists() else []
            ),
            "schema_version": manifest.get("schema_version"),
            "workload": manifest.get("workload"),
            "model_shape": manifest.get("model_shape"),
            "regions": manifest.get("regions"),
            "execution_model": manifest.get("execution_model"),
            "hardware": manifest.get("hardware"),
            "hardware_metadata": manifest.get("hardware_metadata"),
            "arms": [arm["name"] for arm in manifest.get("arms", [])],
            "commands": manifest.get("commands"),
            "results": results,
        }
        if args.launch_counts and results is not None:
            entry["launch_count_per_step"] = launch_counts(
                cell_dir, results.get("arms", [])
            )
        collected.append(entry)

    payload = {
        "root": str(args.root),
        "cell_count": len(collected),
        "cells": collected,
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
        print(f"{len(collected)} cells -> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
