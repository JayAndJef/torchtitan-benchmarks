"""Analyzer tests over synthetic profiler-trace fixtures (no GPU needed)."""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from benchmarks.profile_regions import pooled_region_samples, trace_region_samples
from benchmarks.scenarios import Region

import compare_arms


REGIONS = (
    Region(name="backward_block", phase="backward", invocations_per_window=4),
    Region(name="forward_block", phase="forward", invocations_per_window=4),
)


def graph_name(graph_hash: str) -> str:
    return f"## Call CompiledFxGraph {graph_hash} ##"


def write_trace(path: Path, graphs: dict, extra_events: list = ()) -> Path:
    """Write a synthetic chrome trace mirroring torch profiler structure.

    ``graphs`` maps a graph hash to ``(phase, [gpu durations])``. Each
    invocation emits a CPU ``user_annotation`` (nested in a
    ``CompiledFunctionBackward`` frame when phase is "backward") plus a GPU
    ``gpu_user_annotation`` carrying the measured duration. Phase
    "gpu_only" emits no CPU side at all.
    """
    events = []
    slot = 0
    for graph_hash, (phase, durations) in graphs.items():
        for duration in durations:
            t0 = slot * 1000
            slot += 1
            if phase == "backward":
                events.append(
                    {"ph": "X", "cat": "cpu_op", "name": "CompiledFunctionBackward",
                     "tid": 1, "ts": t0, "dur": 900}
                )
            if phase != "gpu_only":
                # CPU-side span durations are large on purpose: they must
                # never be what gets measured.
                events.append(
                    {"ph": "X", "cat": "user_annotation", "name": graph_name(graph_hash),
                     "tid": 1, "ts": t0 + 10, "dur": 800}
                )
            events.append(
                {"ph": "X", "cat": "gpu_user_annotation", "name": graph_name(graph_hash),
                 "tid": 100, "ts": t0 + 20, "dur": duration}
            )
    events.extend(extra_events)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as trace_file:
        json.dump({"traceEvents": events}, trace_file)
    return path


class ExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def test_measures_only_gpu_compiled_graph_annotations(self) -> None:
        # CPU annotations (dur 800) classify but must not be measured, and
        # kernel / unrelated-annotation decoys must be invisible.
        decoys = [
            {"ph": "X", "cat": "kernel", "name": "triton_poi_fused_0",
             "tid": 100, "ts": 0, "dur": 9e9},
            {"ph": "X", "cat": "gpu_user_annotation", "name": "ProfilerStep#20",
             "tid": 100, "ts": 0, "dur": 9e9},
        ]
        trace = write_trace(
            self.root / "trace.json.gz",
            {"bwd": ("backward", [100.0] * 4), "fwd": ("forward", [10.0] * 4)},
            decoys,
        )
        samples = trace_region_samples(trace, REGIONS)
        self.assertEqual(samples["backward_block"], [100.0] * 4)
        self.assertEqual(samples["forward_block"], [10.0] * 4)

    def test_phase_comes_from_autograd_context_not_size(self) -> None:
        # The forward graph is the LARGER one here; duration-rank matching
        # would mislabel it as backward. Context-based matching must not.
        trace = write_trace(
            self.root / "trace.json.gz",
            {"big_fwd": ("forward", [100.0] * 4), "small_bwd": ("backward", [10.0] * 4)},
        )
        samples = trace_region_samples(trace, REGIONS)
        self.assertEqual(samples["forward_block"], [100.0] * 4)
        self.assertEqual(samples["backward_block"], [10.0] * 4)

    def test_count_disambiguates_same_phase_partitions(self) -> None:
        # A second forward graph with a different invocation count (like the
        # embedding/loss partitions in real piper traces) must be ignored.
        trace = write_trace(
            self.root / "trace.json.gz",
            {
                "fwd": ("forward", [10.0] * 4),
                "bwd": ("backward", [100.0] * 4),
                "other_fwd": ("forward", [999.0] * 2),
                "other_bwd": ("backward", [999.0] * 2),
            },
        )
        samples = trace_region_samples(trace, REGIONS)
        self.assertEqual(samples["forward_block"], [10.0] * 4)
        self.assertEqual(samples["backward_block"], [100.0] * 4)

    def test_pools_across_two_windows(self) -> None:
        window_20 = write_trace(
            self.root / "iteration_20.json.gz",
            {"bwd": ("backward", [100.0] * 4), "fwd": ("forward", [10.0] * 4)},
        )
        window_40 = write_trace(
            self.root / "iteration_40.json.gz",
            {"bwd": ("backward", [102.0] * 4), "fwd": ("forward", [12.0] * 4)},
        )
        pooled = pooled_region_samples([window_20, window_40], REGIONS)
        self.assertEqual(pooled["backward_block"], [100.0] * 4 + [102.0] * 4)
        self.assertEqual(pooled["forward_block"], [10.0] * 4 + [12.0] * 4)

    def test_malformed_trace_fails(self) -> None:
        not_gzip = self.root / "corrupt.json.gz"
        not_gzip.write_bytes(b"this is not gzip data")
        with self.assertRaisesRegex(ValueError, "unreadable profiler trace"):
            trace_region_samples(not_gzip, REGIONS)

        not_json = self.root / "not_json.json.gz"
        with gzip.open(not_json, "wt") as trace_file:
            trace_file.write("{not json")
        with self.assertRaisesRegex(ValueError, "unreadable profiler trace"):
            trace_region_samples(not_json, REGIONS)

        no_events_key = self.root / "no_events.json.gz"
        with gzip.open(no_events_key, "wt") as trace_file:
            json.dump({"other": []}, trace_file)
        with self.assertRaisesRegex(ValueError, "unreadable profiler trace"):
            trace_region_samples(no_events_key, REGIONS)

    def test_missing_phase_fails(self) -> None:
        trace = write_trace(
            self.root / "trace.json.gz", {"fwd": ("forward", [10.0] * 4)}
        )
        with self.assertRaisesRegex(ValueError, "exactly one backward compiled graph"):
            trace_region_samples(trace, REGIONS)

    def test_unexpected_invocation_count_fails(self) -> None:
        # A partitioning change (5 invocations instead of 4) must fail loudly
        # instead of silently reporting the wrong graph.
        trace = write_trace(
            self.root / "trace.json.gz",
            {"bwd": ("backward", [100.0] * 5), "fwd": ("forward", [10.0] * 4)},
        )
        with self.assertRaisesRegex(ValueError, "graph-to-region mapping"):
            trace_region_samples(trace, REGIONS)

    def test_ambiguous_matching_graphs_fail(self) -> None:
        trace = write_trace(
            self.root / "trace.json.gz",
            {
                "fwd_a": ("forward", [10.0] * 4),
                "fwd_b": ("forward", [20.0] * 4),
                "bwd": ("backward", [100.0] * 4),
            },
        )
        with self.assertRaisesRegex(ValueError, "found 2"):
            trace_region_samples(trace, REGIONS)

    def test_gpu_span_without_cpu_context_fails(self) -> None:
        trace = write_trace(
            self.root / "trace.json.gz",
            {"bwd": ("backward", [100.0] * 4), "fwd": ("gpu_only", [10.0] * 4)},
        )
        with self.assertRaisesRegex(ValueError, "lacks the CPU-side autograd context"):
            trace_region_samples(trace, REGIONS)

    def test_mixed_context_fails(self) -> None:
        # Same graph hash appearing inside AND outside backward frames.
        mixed = [
            {"ph": "X", "cat": "user_annotation", "name": graph_name("bwd"),
             "tid": 1, "ts": 990000, "dur": 100},
        ]
        trace = write_trace(
            self.root / "trace.json.gz",
            {"bwd": ("backward", [100.0] * 4), "fwd": ("forward", [10.0] * 4)},
            mixed,
        )
        with self.assertRaisesRegex(ValueError, "cannot classify"):
            trace_region_samples(trace, REGIONS)

    def test_empty_window_list_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "no profiler trace windows"):
            pooled_region_samples([], REGIONS)


class ComparisonTests(unittest.TestCase):
    def test_statistical_comparison_of_pooled_regions(self) -> None:
        base = {
            "backward_block": [100.0, 101.0, 99.0, 100.0, 100.5, 99.5, 100.0, 100.0],
            "forward_block": [10.0, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.0],
        }
        arm = {
            "backward_block": [90.0, 91.0, 89.0, 90.0, 90.5, 89.5, 90.0, 90.0],
            "forward_block": [10.0, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0, 10.0],
        }
        rows = compare_arms.region_comparison(base, arm, REGIONS)
        self.assertEqual([row["region"] for row in rows], ["backward_block", "forward_block"])

        backward, forward = rows
        self.assertEqual(backward["n_base"], 8)
        self.assertEqual(backward["n_arm"], 8)
        self.assertAlmostEqual(backward["delta_us"], -10.0)
        self.assertAlmostEqual(backward["ratio"], 0.9)
        self.assertAlmostEqual(backward["base_median_us"], 100.0)
        self.assertLess(backward["welch_p"], 1e-6)
        self.assertLess(backward["mwu_p"], 1e-3)
        self.assertLess(backward["cohens_d"], -2.0)

        self.assertAlmostEqual(forward["delta_us"], 0.0)
        self.assertAlmostEqual(forward["ratio"], 1.0)
        self.assertGreater(forward["welch_p"], 0.9)

    def test_loss_parsing_flags_non_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text(
                "step:  1  loss:  7.44780\n"
                "step:  2  loss:  nan\n"
            )
            parsed = compare_arms.losses(log)
        self.assertEqual(parsed[0], (1, 7.4478))
        self.assertNotEqual(parsed[1][1], parsed[1][1])  # NaN


if __name__ == "__main__":
    unittest.main()
