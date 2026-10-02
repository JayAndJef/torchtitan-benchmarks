"""Tests for ``tools/collect_matrix.py`` over a synthetic matrix tree of two cells that differ in one axis."""

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.registry import ENGINES
from tests.engine_helpers import configured, run_spec, write_run_manifest
from tests.test_import_boundaries import REPO_ROOT

_SPEC = importlib.util.spec_from_file_location(
    "collect_matrix", REPO_ROOT / "tools" / "collect_matrix.py"
)
collect_matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collect_matrix)


GOLDEN_RUN = REPO_ROOT / "tests" / "fixtures" / "golden" / "runs" / "dp1-1b"
"""A schema 18 run directory with a results file."""


def write_manifest(cell: Path, *, model_size: str, dp: int, zero: int) -> None:
    """Write the schema 20 manifest of a titan_compiled and megatron_stock cell."""
    write_run_manifest(
        cell,
        run_spec(
            model_size,
            ac_mode="none",
            profile=False,
            parallelism=ParallelismSpec(dp=dp, zero=zero),
        ),
        (
            ENGINES.arm("titan_compiled"),
            configured(
                ENGINES.arm("megatron_stock"),
                extra_flags=("--moe-permute-fusion",),
            ),
        ),
    )


def results(*, tokens_per_second: float) -> dict:
    """A schema-6 results payload with two arms."""
    return {
        "schema_version": 6,
        "scenario": "engines",
        "arms": ["titan_compiled", "megatron_stock"],
        "results": {
            "titan_compiled": {
                "stable_tokens_per_second": tokens_per_second,
                "stable_sample_count": 70,
                "peak_memory_gib": 41.5,
                "step_ms": {"mean": 100.0, "median": 99.0, "p95": 110.0},
            },
            "megatron_stock": {
                "stable_tokens_per_second": tokens_per_second / 2,
                "stable_sample_count": 68,
                "peak_memory_gib": 63.25,
                "step_ms": {"mean": 200.0, "median": 198.0, "p95": 220.0},
            },
        },
        "warnings": [],
    }


def write_cell(
    root: Path, name: str, *, model_size: str, dp: int, zero: int, results_data
) -> Path:
    cell = root / name
    write_manifest(cell, model_size=model_size, dp=dp, zero=zero)
    if results_data is not None:
        (cell / "results.json").write_text(json.dumps(results_data))
    return cell


class CollectTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        write_cell(
            self.root,
            "model-size-1b-ac-none",
            model_size="1b",
            dp=1,
            zero=0,
            results_data=results(tokens_per_second=12000.0),
        )
        write_cell(
            self.root,
            "model-size-1b-ac-none-dp-2-zero-1",
            model_size="1b",
            dp=2,
            zero=1,
            results_data=results(tokens_per_second=21000.0),
        )
        self.addCleanup(self._tmp.cleanup)

    def test_one_row_per_arm_of_every_cell(self) -> None:
        rows = collect_matrix.collect(self.root, None)
        self.assertEqual(
            [(row["cell"], row["arm"]) for row in rows],
            [
                ("model-size-1b-ac-none", "titan_compiled"),
                ("model-size-1b-ac-none", "megatron_stock"),
                ("model-size-1b-ac-none-dp-2-zero-1", "titan_compiled"),
                ("model-size-1b-ac-none-dp-2-zero-1", "megatron_stock"),
            ],
        )

    def test_row_carries_the_published_figures(self) -> None:
        rows = collect_matrix.collect(self.root, None)
        row = rows[0]
        self.assertEqual(row["stable_tokens_per_second"], 12000.0)
        self.assertEqual(row["stable_sample_count"], 70)
        self.assertEqual(row["step_ms_median"], 99.0)
        self.assertEqual(row["step_ms_p95"], 110.0)
        self.assertEqual(row["peak_memory_gib"], 41.5)
        self.assertFalse(row["contaminated"])

    def test_row_carries_the_cell_axes(self) -> None:
        rows = collect_matrix.collect(self.root, None)
        first, sharded = rows[0], rows[2]
        self.assertEqual(first["model_size"], "1b")
        self.assertEqual(first["ac_mode"], "none")
        self.assertIs(first["profile"], False)
        self.assertEqual((first["dp"], first["pp"], first["ep"]), (1, 1, 1))
        self.assertEqual(first["zero"], 0)
        self.assertEqual((sharded["dp"], sharded["zero"]), (2, 1))

    def test_row_carries_the_engine_and_the_arm_config(self) -> None:
        titan, megatron = collect_matrix.collect(self.root, None)[:2]
        self.assertEqual(
            (titan["engine"], megatron["engine"]), ("torchtitan", "megatron_stock")
        )
        self.assertEqual(titan["extra_flags"], "")
        self.assertEqual(megatron["extra_flags"], "--moe-permute-fusion")
        self.assertEqual(megatron["config"]["precision"], "stock")

    def test_a_schema_18_cell_is_read(self) -> None:
        cell = self.root / "golden"
        cell.mkdir()
        for name in ("manifest.json", "results.json"):
            shutil.copy(GOLDEN_RUN / name, cell / name)
        rows = [
            row
            for row in collect_matrix.collect(self.root, None)
            if row["cell"] == "golden"
        ]
        self.assertEqual(
            [row["arm"] for row in rows],
            ["titan_compiled", "titan_eager", "megatron_stock"],
        )
        self.assertEqual(rows[2]["engine"], "megatron_stock")
        self.assertEqual(rows[0]["model_size"], "1b")

    def test_a_schema_19_cell_is_read(self) -> None:
        cell = self.root / "recorded-19"
        cell.mkdir()
        shutil.copy(
            REPO_ROOT / "tests/fixtures/manifest_v19/titan-compiled-fa3-dp4-ep4.json",
            cell / "manifest.json",
        )
        payload = results(tokens_per_second=1000.0)
        payload["arms"] = ["titan_compiled_fa3"]
        payload["results"] = {"titan_compiled_fa3": payload["results"]["titan_compiled"]}
        (cell / "results.json").write_text(json.dumps(payload))
        rows = [
            row
            for row in collect_matrix.collect(self.root, None)
            if row["cell"] == "recorded-19"
        ]
        self.assertEqual([row["arm"] for row in rows], ["titan_compiled_fa3"])
        self.assertIs(rows[0]["config"]["packed_offsets"], False)

    def test_contamination_marker_marks_the_rows(self) -> None:
        (self.root / "model-size-1b-ac-none.CONTAMINATED").write_text("foreign")
        rows = collect_matrix.collect(self.root, None)
        self.assertTrue(rows[0]["contaminated"])
        self.assertFalse(rows[2]["contaminated"])

    def test_cell_without_results_yields_no_row(self) -> None:
        write_cell(
            self.root,
            "model-size-huge-ac-none",
            model_size="huge",
            dp=1,
            zero=0,
            results_data=None,
        )
        cells = [row["cell"] for row in collect_matrix.collect(self.root, None)]
        self.assertNotIn("model-size-huge-ac-none", cells)

    def test_moved_aside_cell_is_skipped(self) -> None:
        write_cell(
            self.root,
            "model-size-1b-ac-none.contaminated-20260920T000000Z",
            model_size="1b",
            dp=1,
            zero=0,
            results_data=results(tokens_per_second=1.0),
        )
        names = {cell.name for cell in collect_matrix.cell_dirs(self.root)}
        self.assertEqual(names, {
            "model-size-1b-ac-none",
            "model-size-1b-ac-none-dp-2-zero-1",
        })

    def test_size_filter_selects_one_size(self) -> None:
        write_cell(
            self.root,
            "model-size-huge-ac-none",
            model_size="huge",
            dp=1,
            zero=0,
            results_data=results(tokens_per_second=900.0),
        )
        sizes = {row["model_size"] for row in collect_matrix.collect(self.root, "huge")}
        self.assertEqual(sizes, {"huge"})

    def test_another_results_schema_is_refused_by_name(self) -> None:
        payload = results(tokens_per_second=1.0)
        payload["schema_version"] = 5
        (self.root / "model-size-1b-ac-none" / "results.json").write_text(
            json.dumps(payload)
        )
        with self.assertRaises(ValueError) as caught:
            collect_matrix.collect(self.root, None)
        self.assertIn("schema 5", str(caught.exception))
        self.assertIn("schema 6", str(caught.exception))

    def test_table_prints_one_line_per_row(self) -> None:
        rows = collect_matrix.collect(self.root, None)
        lines = collect_matrix.render(rows).splitlines()
        self.assertEqual(len(lines), len(rows) + 2)
        self.assertIn("tokens/s", lines[0])
        self.assertIn("zero", lines[0])

    def test_json_output_holds_the_same_rows(self) -> None:
        destination = self.root / "matrix.json"
        with contextlib.redirect_stdout(io.StringIO()):
            collect_matrix.main([str(self.root), "--json", str(destination)])
        payload = json.loads(destination.read_text())
        self.assertEqual(payload["row_count"], 4)
        self.assertEqual(payload["rows"], collect_matrix.collect(self.root, None))


if __name__ == "__main__":
    unittest.main()
