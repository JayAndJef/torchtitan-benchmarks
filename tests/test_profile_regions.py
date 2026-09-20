"""Analyzer tests over synthetic profiler-trace fixtures (no GPU needed)."""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.results import (
    evaluate_run,
    losses,
    render_evaluation,
    write_results,
)
from benchmarks.traces.extraction import (
    pooled_window_metrics,
    trace_window_metrics,
)


def graph_name(graph_hash: str) -> str:
    return f"## Call CompiledFxGraph {graph_hash} ##"


def write_trace(path: Path, graphs: dict, extra_events: list = ()) -> Path:
    """Write a synthetic chrome trace mirroring torch profiler structure.

    ``graphs`` maps a graph hash to ``(phase, [gpu durations])``. Each
    invocation emits a CPU ``user_annotation`` (nested in a
    ``CompiledFunctionBackward`` frame when phase is "backward"), a GPU
    ``gpu_user_annotation`` carrying the measured duration, and one ``kernel``
    event covering exactly half the annotation span. Phase "gpu_only" emits no
    CPU side at all.
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
            events.append(
                {"ph": "X", "cat": "kernel", "name": "triton_generated",
                 "tid": 100, "ts": t0 + 20, "dur": duration / 2}
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

    def test_profiled_steps_deduplicate_and_launch_latency_averages(self) -> None:
        # All four launch APIs count (runtime and driver, plain and Ex
        # variants); non-launch runtime calls do not.
        extra = [
            {"ph": "X", "cat": "cpu_op", "name": "ProfilerStep#21",
             "tid": 1, "ts": 0, "dur": 100},
            {"ph": "X", "cat": "python_function", "name": "ProfilerStep#21",
             "tid": 2, "ts": 0, "dur": 100},
            {"ph": "X", "cat": "cpu_op", "name": "ProfilerStep#22",
             "tid": 1, "ts": 200, "dur": 100},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
             "tid": 1, "ts": 10, "dur": 4.0},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernelExC",
             "tid": 1, "ts": 20, "dur": 8.0},
            {"ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernel",
             "tid": 1, "ts": 30, "dur": 2.0},
            {"ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernelEx",
             "tid": 1, "ts": 40, "dur": 2.0},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaMemsetAsync",
             "tid": 1, "ts": 50, "dur": 100.0},
            {"ph": "X", "cat": "python_function", "name": "cuLaunchKernel",
             "tid": 1, "ts": 60, "dur": 100.0},
        ]
        trace = write_trace(
            self.root / "trace.json.gz",
            {"bwd": ("backward", [100.0] * 4), "fwd": ("forward", [10.0] * 4)},
            extra,
        )
        window = trace_window_metrics(trace)
        self.assertEqual(window.profiled_steps, 2)
        self.assertEqual(window.launch_count, 4)
        self.assertEqual(window.launch_total_us, 16.0)

    def test_mixed_stepped_and_stepless_windows_fail(self) -> None:
        stepped = write_trace(
            self.root / "iteration_20.json.gz",
            {"bwd": ("backward", [100.0] * 4), "fwd": ("forward", [10.0] * 4)},
            [{"ph": "X", "cat": "cpu_op", "name": "ProfilerStep#20",
              "tid": 1, "ts": 0, "dur": 100}],
        )
        stepless = write_trace(
            self.root / "iteration_40.json.gz",
            {"bwd": ("backward", [100.0] * 4), "fwd": ("forward", [10.0] * 4)},
        )
        with self.assertRaisesRegex(ValueError, "per-step totals would be wrong"):
            pooled_window_metrics([stepped, stepless])

    def test_pools_measurements_across_two_windows(self) -> None:
        steps = [
            {"ph": "X", "cat": "cpu_op", "name": "ProfilerStep#20",
             "tid": 1, "ts": 0, "dur": 100},
        ]
        window_20 = write_trace(
            self.root / "iteration_20.json.gz",
            {"bwd": ("backward", [100.0] * 4), "fwd": ("forward", [10.0] * 4)},
            steps,
        )
        window_40 = write_trace(
            self.root / "iteration_40.json.gz",
            {"bwd": ("backward", [102.0] * 4), "fwd": ("forward", [12.0] * 4)},
            steps,
        )
        pooled = pooled_window_metrics([window_20, window_40])
        self.assertEqual(pooled.windows, 2)
        self.assertEqual(pooled.profiled_steps, 2)
        self.assertAlmostEqual(
            pooled.kernel_total_us, (4 * 50.0 + 4 * 5.0) + (4 * 51.0 + 4 * 6.0)
        )
        self.assertAlmostEqual(
            pooled.kernel_ms_per_step, pooled.kernel_total_us / 2 / 1000.0
        )
        self.assertIsNone(pooled.launch_latency_us)

    def test_malformed_trace_fails(self) -> None:
        not_gzip = self.root / "corrupt.json.gz"
        not_gzip.write_bytes(b"this is not gzip data")
        with self.assertRaisesRegex(ValueError, "unreadable profiler trace"):
            trace_window_metrics(not_gzip)

        not_json = self.root / "not_json.json.gz"
        with gzip.open(not_json, "wt") as trace_file:
            trace_file.write("{not json")
        with self.assertRaisesRegex(ValueError, "unreadable profiler trace"):
            trace_window_metrics(not_json)

        no_events_key = self.root / "no_events.json.gz"
        with gzip.open(no_events_key, "wt") as trace_file:
            json.dump({"other": []}, trace_file)
        with self.assertRaisesRegex(ValueError, "unreadable profiler trace"):
            trace_window_metrics(no_events_key)

    def test_empty_window_list_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "no profiler trace windows"):
            pooled_window_metrics([])


class EvaluationTests(unittest.TestCase):
    def test_loss_parsing_flags_non_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text(
                "step:  1  loss:  7.44780\n"
                "step:  2  loss:  nan\n"
            )
            parsed = losses(log)
        self.assertEqual(parsed[0], (1, 7.4478))
        self.assertNotEqual(parsed[1][1], parsed[1][1])  # NaN

    def test_uneven_host_latency_across_arms_warns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            manifest = {
                "schema_version": 5,
                "scenario": "synthetic",
                "hardware": "test-gpu",
                "workload": {},
                "selected_arms": ["baseline", "optimized"],
            }
            (out_dir / "manifest.json").write_text(json.dumps(manifest))
            for arm, launch_us in (("baseline", 4.0), ("optimized", 6.0)):
                write_trace(
                    out_dir
                    / arm
                    / "profiling/traces/iteration_20/rank0_trace.json.gz",
                    {
                        "backward": ("backward", [100.0] * 4),
                        "forward": ("forward", [10.0] * 4),
                    },
                    [
                        {"ph": "X", "cat": "cuda_runtime",
                         "name": "cudaLaunchKernel",
                         "tid": 1, "ts": 10, "dur": launch_us},
                    ],
                )
                (out_dir / f"{arm}.log").write_text(
                    "step: 2 loss: 1.0 memory: 3.00GiB tps: 1000\n"
                )
            result = evaluate_run(out_dir)
        self.assertTrue(
            any("host launch latency varies" in warning
                for warning in result.warnings),
            result.warnings,
        )

    def test_complete_evaluation_is_human_and_machine_readable(self) -> None:
        step_events = [
            {"ph": "X", "cat": "cpu_op", "name": "ProfilerStep#20",
             "tid": 1, "ts": 0, "dur": 100},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
             "tid": 1, "ts": 10, "dur": 6.0},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            manifest = {
                "schema_version": 5,
                "scenario": "synthetic",
                "hardware": "test-gpu",
                "workload": {
                    "profile_freq": 20,
                    "profiler_warmup": 5,
                    "profiler_active": 5,
                },
                "selected_arms": ["baseline", "optimized"],
            }
            (out_dir / "manifest.json").write_text(json.dumps(manifest))
            for arm, backward, forward, tps, loss in (
                (
                    "baseline",
                    [99.0, 100.0, 101.0, 100.0],
                    [9.0, 10.0, 11.0, 10.0],
                    1000,
                    "1.0",
                ),
                (
                    "optimized",
                    [89.0, 90.0, 91.0, 90.0],
                    [7.0, 8.0, 9.0, 8.0],
                    1200,
                    # Finite: evaluate_run refuses a non-finite trajectory
                    # before it publishes anything (test_throughput pins it).
                    "0.9",
                ),
            ):
                write_trace(
                    out_dir
                    / arm
                    / "profiling/traces/iteration_20/rank0_trace.json.gz",
                    {
                        "backward": ("backward", backward),
                        "forward": ("forward", forward),
                    },
                    step_events,
                )
                (out_dir / f"{arm}.log").write_text(
                    f"step: 2 loss: {loss} grad_norm: 2.0 "
                    f"memory: 3.00GiB tps: {tps}\n"
                )

            result = evaluate_run(out_dir)
            results_path = write_results(result)
            machine = json.loads(results_path.read_text())
            report = render_evaluation(result)

        self.assertEqual(machine["schema_version"], 5)
        self.assertEqual(
            machine["training"]["optimized"]["stable_tokens_per_second"], 1200
        )
        optimized_gpu = machine["gpu_time"]["optimized"]
        self.assertAlmostEqual(
            optimized_gpu["kernel_ms_per_step"], (4 * 45.0 + 4 * 4.0) / 1000.0
        )
        self.assertAlmostEqual(optimized_gpu["launch_latency_us"], 6.0)
        self.assertAlmostEqual(
            optimized_gpu["baseline_kernel_ratio"],
            (4 * 45.0 + 4 * 4.0) / (4 * 50.0 + 4 * 5.0),
        )
        methodology = machine["significance_methodology"]
        self.assertEqual(
            methodology["interpretation"],
            "invocation_distribution_diagnostic",
        )
        self.assertFalse(methodology["independence_assumption_met"])
        self.assertEqual(methodology["sample_unit"], "compiled_region_invocation")
        self.assertEqual(machine["losses"]["optimized"][0]["value"], 0.9)
        self.assertIn("stable tokens/s", report)
        self.assertIn("gpu kernel time", report)
        self.assertIn("kernel ms/step", report)


if __name__ == "__main__":
    unittest.main()
