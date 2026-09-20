"""Merge a run_matrix.sh output tree into one table, one row per arm.

Each cell directory under the matrix root holds a ``manifest.json`` and a
``results.json``. The manifest names the axes the cell was run at and the
results carry the published figures, so one row joins the two:

    .venv/bin/python tools/collect_matrix.py out/matrix-<utc> \\
        [--size 1b] [--json matrix.json]

Only ``results.json`` is read for the figures. A profiler trace is written
under ``--profile`` alone, so a tool that needed one could not report an
ordinary run at all.

**The results schema is not negotiable.** Every column below is a field of
schema 6. A file another schema wrote is refused by name rather than read
with defaulted lookups, because a missing field would render as a blank
cell and read as a measurement that came out empty.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_SCHEMA_VERSION = 6
"""The one results schema this tool reads."""

MOVED_ASIDE = (".contaminated-", ".nomanifest-")
"""Directory-name marks that run_matrix.sh puts on an abandoned cell."""

COLUMNS = (
    ("cell", "cell"),
    ("arm", "arm"),
    ("model_size", "size"),
    ("ac_mode", "ac"),
    ("dp", "dp"),
    ("pp", "pp"),
    ("ep", "ep"),
    ("zero", "zero"),
    ("profile", "prof"),
    ("stable_tokens_per_second", "tokens/s"),
    ("stable_sample_count", "n"),
    ("step_ms_median", "step ms"),
    ("step_ms_p95", "p95 ms"),
    ("peak_memory_gib", "peak GiB"),
    ("contaminated", "bad"),
)
"""The printed columns, as (row key, heading). The JSON rows carry more."""


def cell_dirs(root: Path):
    """Yield every cell directory under ``root``, in name order.

    A cell is a directory with a manifest directly inside it. The
    supervisor moves a condemned cell aside under a marked name, and those
    are skipped: the numbers in them are the ones it decided not to keep.
    """
    for manifest_path in sorted(root.glob("*/manifest.json")):
        cell_dir = manifest_path.parent
        if any(mark in cell_dir.name for mark in MOVED_ASIDE):
            continue
        yield cell_dir


def cell_axes(manifest: dict[str, Any]) -> dict[str, Any]:
    """The axis columns of one cell, read from its manifest."""
    parallelism = manifest.get("parallelism") or {}
    return {
        "scenario": manifest.get("scenario"),
        "model_size": manifest.get("model_size"),
        "ac_mode": manifest.get("ac_mode"),
        "profile": manifest.get("profile"),
        "dp": parallelism.get("dp"),
        "pp": parallelism.get("pp"),
        "ep": parallelism.get("ep"),
        "zero": parallelism.get("zero"),
        "parallelism": parallelism,
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
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    results = load_results(cell_dir)
    if results is None:
        return []
    marker = cell_dir.with_name(cell_dir.name + ".CONTAMINATED")
    axes = cell_axes(manifest)
    rows = []
    for arm in results.get("arms", []):
        summary = (results.get("results") or {}).get(arm) or {}
        step_ms = summary.get("step_ms") or {}
        rows.append(
            {
                "cell": cell_dir.name,
                "path": str(cell_dir),
                "arm": arm,
                "contaminated": marker.exists(),
                **axes,
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
    from benchmarks.models.piper_qwen3.shape import canonical_size_name

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
