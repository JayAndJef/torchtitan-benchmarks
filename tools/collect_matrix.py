"""Merge the cells of a run_matrix.sh output tree into one table, one row per arm.

    .venv/bin/python tools/collect_matrix.py out/matrix-<utc> [--size 1b] [--json matrix.json]

The tool reads manifest schemas 18 and 19, and results schema 6 alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.artifacts.manifests import (
    RunRecord,
    config_json,
    load_run_record,
)
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.models.piper_qwen3.shape import canonical_size_name

RESULTS_SCHEMA_VERSION = 6
"""The one results schema this tool reads."""

MOVED_ASIDE = (".contaminated-", ".nomanifest-")
"""Directory-name marks that run_matrix.sh puts on an abandoned cell."""

COLUMNS = (
    ("cell", "cell"),
    ("scenario", "scenario"),
    ("arm", "arm"),
    ("model_size", "size"),
    ("ac_mode", "ac"),
    ("dp", "dp"),
    ("pp", "pp"),
    ("ep", "ep"),
    ("zero", "zero"),
    ("profile", "prof"),
    ("engine", "engine"),
    ("extra_flags", "extra flags"),
    ("stable_tokens_per_second", "tokens/s"),
    ("stable_sample_count", "n"),
    ("step_ms_median", "step ms"),
    ("step_ms_p95", "p95 ms"),
    ("peak_memory_gib", "peak GiB"),
    ("contaminated", "bad"),
)
"""The printed columns, as (row key, heading). The JSON rows carry more."""


def cell_dirs(root: Path):
    """Every cell directory under ``root``, in name order; a cell that run_matrix.sh moved aside is skipped."""
    for manifest_path in sorted(root.glob("*/manifest.json")):
        cell_dir = manifest_path.parent
        if any(mark in cell_dir.name for mark in MOVED_ASIDE):
            continue
        yield cell_dir


def cell_axes(record: RunRecord) -> dict[str, Any]:
    """The run-wide columns of one cell, read from its manifest."""
    run = record.run
    spec = run.parallelism
    return {
        "scenario": record.scenario,
        "model_size": run.shape.name,
        "ac_mode": run.ac_mode,
        "profile": run.profile,
        "dp": spec.dp,
        "pp": spec.pp,
        "ep": spec.ep,
        "zero": spec.zero,
        "parallelism": {
            "dp": spec.dp,
            "pp": spec.pp,
            "ep": spec.ep,
            "pp_schedule": spec.pp_schedule,
            "pp_microbatch_size": spec.pp_microbatch_size,
            "zero": spec.zero,
        },
    }


def load_results(cell_dir: Path) -> dict[str, Any] | None:
    """One cell's results, or ``None`` when the cell never finished."""
    results_path = cell_dir / "results.json"
    if not results_path.is_file():
        return None
    results = json.loads(results_path.read_text())
    found = results.get("schema_version")
    if found != RESULTS_SCHEMA_VERSION:
        raise ValueError(
            f"{results_path} records results schema {found!r}; this tool "
            f"reads schema {RESULTS_SCHEMA_VERSION} only"
        )
    return results


def cell_rows(cell_dir: Path) -> list[dict[str, Any]]:
    """One row per arm of one cell. Empty when the cell has no results."""
    record = load_run_record(cell_dir)
    results = load_results(cell_dir)
    if results is None:
        return []
    marker = cell_dir.with_name(cell_dir.name + ".CONTAMINATED")
    axes = cell_axes(record)
    rows = []
    for arm in results.get("arms", []):
        summary = (results.get("results") or {}).get(arm) or {}
        step_ms = summary.get("step_ms") or {}
        config = record.arm(arm).arm.config
        rows.append(
            {
                "cell": cell_dir.name,
                "path": str(cell_dir),
                "arm": arm,
                "contaminated": marker.exists(),
                **axes,
                "engine": engine_for(record.arm(arm).arm).name,
                "extra_flags": " ".join(config.extra_flags),
                "config": config_json(config),
                "stable_tokens_per_second": summary.get(
                    "stable_tokens_per_second"
                ),
                "stable_sample_count": summary.get("stable_sample_count"),
                "step_ms_median": step_ms.get("median"),
                "step_ms_p95": step_ms.get("p95"),
                "peak_memory_gib": summary.get("peak_memory_gib"),
            }
        )
    return rows


def collect(root: Path, size_filter: str | None) -> list[dict[str, Any]]:
    """Every arm row under ``root``, cells in name order, arms in run order."""
    wanted = canonical_size_name(size_filter) if size_filter else None
    rows = []
    for cell_dir in cell_dirs(root):
        for row in cell_rows(cell_dir):
            if wanted and row.get("model_size") != wanted:
                continue
            rows.append(row)
    return rows


def _cell_text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render(rows: list[dict[str, Any]]) -> str:
    """The rows as a plain-text table. A heading line even when empty."""
    headings = [heading for _, heading in COLUMNS]
    body = [[_cell_text(row.get(key)) for key, _ in COLUMNS] for row in rows]
    widths = [
        max(len(heading), *(len(line[index]) for line in body))
        if body
        else len(heading)
        for index, heading in enumerate(headings)
    ]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headings, widths)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for line in body:
        lines.append(
            "  ".join(v.ljust(w) for v, w in zip(line, widths)).rstrip()
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--size", default=None)
    parser.add_argument(
        "--json", dest="json_path", type=Path, default=None,
        help="Write the same rows to this path as JSON.",
    )
    args = parser.parse_args(argv)

    rows = collect(args.root, args.size)
    print(render(rows))
    if args.json_path:
        payload = {"root": str(args.root), "row_count": len(rows), "rows": rows}
        args.json_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\n{len(rows)} rows -> {args.json_path}")


if __name__ == "__main__":
    main()
