"""Rank-aware trace reading, and the proof that it changed no old number.

Every run recorded before this existed holds exactly one rank, so the whole
point of this module is two-sided:

* **Nothing moved.** ``LegacyInertnessTests`` evaluates a real single-rank arm
  through the real publication path and compares the result to the
  ``results.json`` that run actually wrote. Exact equality, field by field.
* **Two ranks cannot become a mean.** The rest of the module drives two ranks
  through the same path and pins the maximum, the sum and the per-rank vector,
  and pins that the one-rank pooler refuses a two-rank call outright.

The mean is the failure this module exists to prevent. It is neither one
rank's cost nor the step's total, it passes every other check silently, and
widening a glob is all it takes to introduce it.
"""

import gzip
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.layout import trace_files, trace_files_by_rank
from benchmarks.artifacts.summaries import summarize
from benchmarks.e2e.results import (
    busiest_rank,
    evaluate_run,
    render_evaluation,
)
from benchmarks.e2e.registry import scenario_by_name
from benchmarks.e2e.validation import validate_arm
from benchmarks.traces.extraction import (
    busy_union,
    per_rank_pooled_metrics,
    pooled_window_metrics,
    trace_window_metrics,
)
from benchmarks.traces.schema import TRACE_FILE_GLOB, Region, rank_of_trace
from tests.test_runner import _SAC_LINE, _SIZE_LINE, _compiled_line


REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_RESULTS_3 = (
    Path(__file__).resolve().parent / "fixtures" / "legacy" / "e2e_results3"
)

REGIONS = (
    Region(name="backward_block", phase="backward", invocations_per_window=4),
    Region(name="forward_block", phase="forward", invocations_per_window=4),
)


def graph_name(graph_hash: str) -> str:
    return f"## Call CompiledFxGraph {graph_hash} ##"


def write_trace(path: Path, graphs: dict, extra_events: list = ()) -> Path:
    """Synthetic chrome trace, in the shape torch's profiler writes.

    Deliberately the same construction ``tests/test_profile_regions.py`` uses:
    one CPU ``user_annotation`` per invocation (nested in a
    ``CompiledFunctionBackward`` frame when the phase is backward), one GPU
    ``gpu_user_annotation`` carrying the measured span, and one kernel event
    covering exactly half of it.
    """
    events = []
    slot = 0
    for graph_hash, (phase, durations) in graphs.items():
        for duration in durations:
            t0 = slot * 1000
            slot += 1
            if phase == "backward":
                events.append(
                    {"ph": "X", "cat": "cpu_op",
                     "name": "CompiledFunctionBackward",
                     "tid": 1, "ts": t0, "dur": 900}
                )
            events.append(
                {"ph": "X", "cat": "user_annotation",
                 "name": graph_name(graph_hash),
                 "tid": 1, "ts": t0 + 10, "dur": 800}
            )
            events.append(
                {"ph": "X", "cat": "gpu_user_annotation",
                 "name": graph_name(graph_hash),
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


def step_event(duration: float) -> dict:
    return {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#20",
            "tid": 1, "ts": 0, "dur": duration}


def collective_event(duration: float, start: float = 500_000.0) -> dict:
    """One NCCL kernel, on a stream of its own, outside every region span."""
    return {"ph": "X", "cat": "kernel",
            "name": "ncclDevKernel_AllReduce_Sum_bf16_RING_LL",
            "pid": 0, "tid": 200, "ts": start, "dur": duration}


def two_rank_arm(arm_dir: Path) -> None:
    """One arm, two ranks, one profiler window each.

    Rank 0 is the cheap rank and rank 1 the busy one, and rank 1 is the only
    one that runs a collective. The numbers are chosen so the maximum, the
    mean and the sum are three different values:

    ======  ===========  ==========  ============
    rank    kernel ms    nccl ms     compute ms
    ======  ===========  ==========  ============
    0       0.220        0.000       0.220
    1       0.500        0.060       0.440
    ======  ===========  ==========  ============

    max 0.500, mean 0.360, sum 0.720.
    """
    write_trace(
        arm_dir / "profiling/traces/iteration_20/rank0_trace.json.gz",
        {"bwd": ("backward", [100.0] * 4), "fwd": ("forward", [10.0] * 4)},
        [step_event(1000.0)],
    )
    write_trace(
        arm_dir / "profiling/traces/iteration_20/rank1_trace.json.gz",
        {"bwd": ("backward", [200.0] * 4), "fwd": ("forward", [20.0] * 4)},
        [step_event(1500.0), collective_event(60.0)],
    )


def two_rank_run(out_dir: Path) -> None:
    manifest = {
        "schema_version": 9,
        "scenario": "synthetic_parallel",
        "hardware": "test-gpu",
        "workload": {},
        "regions": [
            {"name": region.name, "phase": region.phase,
             "invocations_per_window": region.invocations_per_window}
            for region in REGIONS
        ],
        "selected_arms": ["baseline"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    two_rank_arm(out_dir / "baseline")


class RankGroupingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def test_the_glob_finds_every_rank_not_only_rank_zero(self) -> None:
        two_rank_arm(self.root)
        found = [path.name for path in trace_files(self.root)]
        self.assertEqual(
            found, ["rank0_trace.json.gz", "rank1_trace.json.gz"]
        )

    def test_files_group_by_the_rank_their_name_carries(self) -> None:
        two_rank_arm(self.root)
        by_rank = trace_files_by_rank(self.root)
        self.assertEqual(sorted(by_rank), [0, 1])
        self.assertEqual(
            [path.name for path in by_rank[0]], ["rank0_trace.json.gz"]
        )
        self.assertEqual(
            [path.name for path in by_rank[1]], ["rank1_trace.json.gz"]
        )

    def test_every_window_of_one_rank_lands_in_that_rank(self) -> None:
        for iteration in (20, 40):
            for rank in (0, 1):
                write_trace(
                    self.root
                    / f"profiling/traces/iteration_{iteration}"
                    / f"rank{rank}_trace.json.gz",
                    {"fwd": ("forward", [10.0] * 4)},
                )
        by_rank = trace_files_by_rank(self.root)
        self.assertEqual({rank: len(paths) for rank, paths in by_rank.items()},
                         {0: 2, 1: 2})

    def test_a_two_digit_rank_parses_as_a_number(self) -> None:
        self.assertEqual(rank_of_trace(Path("rank12_trace.json.gz")), 12)
        self.assertEqual(rank_of_trace(Path("a/b/rank7_trace.json.gz")), 7)

    def test_a_name_that_does_not_declare_a_rank_returns_none(self) -> None:
        # Nothing under out/ is named this way; the tests' own synthetic
        # traces are, and they must stay usable.
        self.assertIsNone(rank_of_trace(Path("iteration_20.json.gz")))

    def test_a_globbed_name_that_does_not_parse_is_refused(self) -> None:
        path = (
            self.root
            / "profiling/traces/iteration_20/rankX_trace.json.gz"
        )
        write_trace(path, {"fwd": ("forward", [10.0] * 4)})
        self.assertEqual(len(trace_files(self.root)), 1)
        with self.assertRaisesRegex(ValueError, "does not name a rank"):
            trace_files_by_rank(self.root)


class PoolingStaysPerRankTests(unittest.TestCase):
    """The single-rank pooler must refuse two ranks rather than average them."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def test_two_ranks_in_one_pooling_call_are_refused(self) -> None:
        two_rank_arm(self.root)
        paths = trace_files(self.root)
        with self.assertRaisesRegex(ValueError, "average across ranks"):
            pooled_window_metrics(paths, REGIONS)

    def test_one_rank_many_windows_is_not_refused(self) -> None:
        for iteration in (20, 40):
            write_trace(
                self.root
                / f"profiling/traces/iteration_{iteration}/rank0_trace.json.gz",
                {"bwd": ("backward", [100.0] * 4),
                 "fwd": ("forward", [10.0] * 4)},
                [step_event(1000.0)],
            )
        pooled = pooled_window_metrics(trace_files(self.root), REGIONS)
        self.assertEqual(pooled.windows, 2)
        self.assertEqual(pooled.profiled_steps, 2)

    def test_per_rank_pooling_keeps_each_rank_whole(self) -> None:
        two_rank_arm(self.root)
        per_rank = per_rank_pooled_metrics(
            trace_files_by_rank(self.root), REGIONS
        )
        self.assertEqual(sorted(per_rank), [0, 1])
        self.assertAlmostEqual(per_rank[0].kernel_ms_per_step, 0.220)
        self.assertAlmostEqual(per_rank[1].kernel_ms_per_step, 0.500)
        # The mean of the two is 0.360 and appears nowhere.
        self.assertNotIn(
            0.360,
            [per_rank[0].kernel_ms_per_step, per_rank[1].kernel_ms_per_step],
        )

    def test_no_ranks_at_all_is_an_error_not_an_empty_answer(self) -> None:
        with self.assertRaisesRegex(ValueError, "no profiler trace windows"):
            per_rank_pooled_metrics({}, REGIONS)

    def test_a_failure_inside_one_rank_names_that_rank(self) -> None:
        two_rank_arm(self.root)
        wrong = (Region(name="forward_block", phase="forward",
                        invocations_per_window=99),)
        with self.assertRaisesRegex(ValueError, "rank 0:"):
            per_rank_pooled_metrics(trace_files_by_rank(self.root), wrong)

    def test_only_the_extraction_module_calls_the_single_rank_pooler(
        self,
    ) -> None:
        """No shipped module may pool a set of paths it did not split by rank.

        ``per_rank_pooled_metrics`` is the entry point; the one-rank pooler is
        its implementation. A call site that reaches past it is exactly how a
        published figure would silently become a mean.
        """
        sources = [
            path
            for directory in ("benchmarks", "tools")
            for path in sorted((REPO_ROOT / directory).rglob("*.py"))
        ]
        call = re.compile(r"(?<!_)pooled_window_metrics\s*\(")
        callers = sorted(
            str(path.relative_to(REPO_ROOT))
            for path in sources
            if call.search(path.read_text())
        )
        self.assertEqual(callers, ["benchmarks/traces/extraction.py"])


class CollectiveAndBasisTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def test_a_collective_leaves_the_total_and_leaves_compute(self) -> None:
        two_rank_arm(self.root)
        per_rank = per_rank_pooled_metrics(
            trace_files_by_rank(self.root), REGIONS
        )
        busy = per_rank[1]
        self.assertAlmostEqual(busy.collective_ms_per_step, 0.060)
        self.assertAlmostEqual(busy.compute_ms_per_step, 0.440)
        self.assertAlmostEqual(
            busy.kernel_ms_per_step,
            busy.compute_ms_per_step + busy.collective_ms_per_step,
        )
        quiet = per_rank[0]
        self.assertEqual(quiet.collective_ms_per_step, 0.0)
        self.assertEqual(
            quiet.compute_ms_per_step, quiet.kernel_ms_per_step
        )

    def test_the_step_wall_comes_from_the_profiler_annotation(self) -> None:
        two_rank_arm(self.root)
        per_rank = per_rank_pooled_metrics(
            trace_files_by_rank(self.root), REGIONS
        )
        self.assertAlmostEqual(per_rank[0].wall_ms_per_step, 1.0)
        self.assertAlmostEqual(per_rank[1].wall_ms_per_step, 1.5)
        # The reason it is recorded: bubble = wall - busy, per rank. No
        # kernel-derived total can see idle time, because a bubble runs no
        # kernel.
        for rank, busy in ((0, 0.220), (1, 0.500)):
            self.assertAlmostEqual(
                per_rank[rank].busy_kernel_ms_per_step, busy
            )
            self.assertGreater(
                per_rank[rank].wall_ms_per_step
                - per_rank[rank].busy_kernel_ms_per_step,
                0.0,
            )

    def test_busy_union_is_below_the_sum_when_streams_overlap(self) -> None:
        overlapping = [
            {"ph": "X", "cat": "kernel", "name": "a",
             "pid": 0, "tid": 100, "ts": 0.0, "dur": 100.0},
            {"ph": "X", "cat": "kernel", "name": "b",
             "pid": 0, "tid": 101, "ts": 50.0, "dur": 100.0},
            step_event(1000.0),
        ]
        write_trace(
            self.root / "profiling/traces/iteration_20/rank0_trace.json.gz",
            {},
            overlapping,
        )
        pooled = per_rank_pooled_metrics(
            trace_files_by_rank(self.root), ()
        )[0]
        self.assertAlmostEqual(pooled.kernel_total_us, 200.0)
        self.assertAlmostEqual(pooled.busy_kernel_us, 150.0)

    def test_busy_union_merges_touching_and_nested_intervals(self) -> None:
        self.assertEqual(busy_union([]), 0.0)
        self.assertEqual(busy_union([(0.0, 10.0), (10.0, 20.0)]), 20.0)
        self.assertEqual(busy_union([(0.0, 10.0), (2.0, 4.0)]), 10.0)
        self.assertEqual(busy_union([(0.0, 1.0), (5.0, 6.0)]), 2.0)


class SteplessRankTests(unittest.TestCase):
    """A rank the profiler never measured must not be silently dropped.

    ``kernel_ms_per_step`` is ``None`` when a rank's windows carry no
    ``ProfilerStep`` annotation. Ranking ``None`` as zero would exclude that
    rank from a maximum, and the excluded rank may be the busiest one -- the
    same wrongness the reduction exists to prevent, arriving by another door.
    """

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.out_dir = Path(self._temporary.name)

    def _mixed_run(self) -> Path:
        """Rank 1 does ten times rank 0's work and declares no step."""
        manifest = {
            "schema_version": 9,
            "scenario": "synthetic_parallel",
            "hardware": "test-gpu",
            "workload": {},
            "regions": [],
            "selected_arms": ["baseline"],
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest))
        arm = self.out_dir / "baseline"
        write_trace(
            arm / "profiling/traces/iteration_20/rank0_trace.json.gz",
            {},
            [{"ph": "X", "cat": "kernel", "name": "a",
              "pid": 0, "tid": 100, "ts": 0.0, "dur": 220.0},
             step_event(1000.0)],
        )
        write_trace(
            arm / "profiling/traces/iteration_20/rank1_trace.json.gz",
            {},
            [{"ph": "X", "cat": "kernel", "name": "a",
              "pid": 0, "tid": 100, "ts": 0.0, "dur": 2200.0}],
        )
        return arm

    def test_a_mixture_of_stepped_and_stepless_ranks_is_refused(self) -> None:
        arm = self._mixed_run()
        per_rank = per_rank_pooled_metrics(trace_files_by_rank(arm), ())
        self.assertEqual(per_rank[0].kernel_ms_per_step, 0.220)
        self.assertIsNone(per_rank[1].kernel_ms_per_step)
        with self.assertRaisesRegex(ValueError, r"ranks \[1\] carry no"):
            busiest_rank(per_rank)

    def test_evaluate_run_names_the_arm_and_does_not_publish(self) -> None:
        self._mixed_run()
        with self.assertRaisesRegex(ValueError, "baseline: ranks"):
            evaluate_run(self.out_dir)

    def test_a_run_nothing_profiled_keeps_its_old_answer(self) -> None:
        """All-``None`` is not a mixture; it is a run with no steps at all."""
        arm = self.out_dir / "baseline"
        for rank in (0, 1):
            write_trace(
                arm / f"profiling/traces/iteration_20/rank{rank}_trace.json.gz",
                {},
                [{"ph": "X", "cat": "kernel", "name": "a",
                  "pid": 0, "tid": 100, "ts": 0.0, "dur": 220.0}],
            )
        per_rank = per_rank_pooled_metrics(trace_files_by_rank(arm), ())
        self.assertEqual(busiest_rank(per_rank), 0)


class CollectivesAreNotRegionComputeTests(unittest.TestCase):
    """A collective that shares a region's stream is not that region's work."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def test_a_collective_inside_a_region_span_is_not_region_kernel(
        self,
    ) -> None:
        write_trace(
            self.root / "profiling/traces/iteration_20/rank0_trace.json.gz",
            {"bwd": ("backward", [100.0] * 4),
             "fwd": ("forward", [10.0] * 4)},
            [step_event(1000.0),
             # Same stream as the region's own annotation (tid 100), sitting
             # inside the first backward span.
             {"ph": "X", "cat": "kernel",
              "name": "ncclDevKernel_AllReduce_Sum_bf16_RING_LL",
              "tid": 100, "ts": 30.0, "dur": 40.0}],
        )
        pooled = per_rank_pooled_metrics(
            trace_files_by_rank(self.root), REGIONS
        )[0]
        self.assertAlmostEqual(pooled.kernel_total_us, 260.0)
        self.assertAlmostEqual(pooled.collective_us, 40.0)
        # Every backward invocation reads its own 50 us of compute. The one
        # that hosted the collective does not read 90.
        self.assertEqual(pooled.region_kernel["backward_block"], [50.0] * 4)
        self.assertAlmostEqual(pooled.region_kernel_ms_per_step, 0.220)
        self.assertAlmostEqual(pooled.compute_ms_per_step, 0.220)
        self.assertLessEqual(
            pooled.region_kernel_ms_per_step, pooled.compute_ms_per_step
        )


class StepWallReadsTheHostAnnotationTests(unittest.TestCase):
    """The step's wall clock is the host annotation, not the longest one."""

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def _window(self, extra: list):
        write_trace(
            self.root / "profiling/traces/iteration_20/rank0_trace.json.gz",
            {},
            extra,
        )
        return per_rank_pooled_metrics(trace_files_by_rank(self.root), ())[0]

    def test_a_device_annotation_cannot_set_the_wall(self) -> None:
        pooled = self._window([
            step_event(1000.0),
            # The decoy tests/test_profile_regions.py already writes: a device
            # annotation carrying the step's name and an absurd duration.
            {"ph": "X", "cat": "gpu_user_annotation", "name": "ProfilerStep#20",
             "tid": 100, "ts": 0.0, "dur": 9e9},
        ])
        self.assertEqual(pooled.profiled_steps, 1)
        self.assertAlmostEqual(pooled.wall_ms_per_step, 1.0)

    def test_a_step_named_only_on_the_device_reports_no_wall(self) -> None:
        pooled = self._window([
            {"ph": "X", "cat": "gpu_user_annotation", "name": "ProfilerStep#20",
             "tid": 100, "ts": 0.0, "dur": 500.0},
        ])
        self.assertEqual(pooled.profiled_steps, 1)
        self.assertIsNone(pooled.wall_ms_per_step)


class PublishedNumberIsTheMaximumTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.out_dir = Path(self._temporary.name)
        two_rank_run(self.out_dir)
        self.result = evaluate_run(self.out_dir)
        self.gpu = self.result.gpu_time["baseline"]

    def test_the_step_cost_is_the_busiest_rank_not_the_mean(self) -> None:
        self.assertAlmostEqual(self.gpu.kernel_ms_per_step, 0.500)
        self.assertNotAlmostEqual(self.gpu.kernel_ms_per_step, 0.360)
        self.assertEqual(self.gpu.rank_reduction, "max_over_ranks")
        self.assertEqual(self.gpu.published_rank, 1)

    def test_the_sum_over_ranks_is_recorded_beside_it(self) -> None:
        self.assertAlmostEqual(
            self.gpu.kernel_ms_per_step_summed_over_ranks, 0.720
        )

    def test_the_per_rank_vector_is_recorded(self) -> None:
        self.assertEqual(self.gpu.ranks, (0, 1))
        self.assertEqual([row.rank for row in self.gpu.per_rank], [0, 1])
        self.assertAlmostEqual(self.gpu.per_rank[0].kernel_ms_per_step, 0.220)
        self.assertAlmostEqual(self.gpu.per_rank[1].kernel_ms_per_step, 0.500)

    def test_the_split_columns_come_from_the_published_rank(self) -> None:
        self.assertAlmostEqual(self.gpu.collective_ms_per_step, 0.060)
        self.assertAlmostEqual(self.gpu.compute_ms_per_step, 0.440)
        self.assertAlmostEqual(self.gpu.wall_ms_per_step, 1.5)

    def test_every_published_scalar_belongs_to_one_rank(self) -> None:
        """Coherence, not a per-field maximum.

        ``other = kernel - regions``. Taking the two sides from different
        ranks can make it negative, which is why the reduction picks a rank
        and then reads that rank's whole row.
        """
        published = next(
            row for row in self.gpu.per_rank
            if row.rank == self.gpu.published_rank
        )
        self.assertEqual(
            self.gpu.region_kernel_ms_per_step,
            published.region_kernel_ms_per_step,
        )
        self.assertEqual(self.gpu.launch_latency_us, published.launch_latency_us)
        self.assertGreaterEqual(self.gpu.other_kernel_ms_per_step, 0.0)

    def test_the_busiest_rank_helper_breaks_ties_toward_the_lowest(self) -> None:
        per_rank = per_rank_pooled_metrics(
            trace_files_by_rank(self.out_dir / "baseline"), REGIONS
        )
        self.assertEqual(busiest_rank(per_rank), 1)
        self.assertEqual(busiest_rank({3: per_rank[0], 7: per_rank[0]}), 3)

    def test_the_report_names_the_reduction_and_shows_each_rank(self) -> None:
        report = render_evaluation(self.result)
        self.assertIn("MAX over ranks, never the mean", report)
        self.assertIn("per-rank gpu time", report)

    def test_the_written_file_declares_schema_five(self) -> None:
        machine = self.result.to_dict()
        self.assertEqual(machine["schema_version"], 5)
        gpu = machine["gpu_time"]["baseline"]
        self.assertEqual(gpu["ranks"], [0, 1])
        self.assertEqual([row["rank"] for row in gpu["per_rank"]], [0, 1])


class ValidationRulesGotStricterTests(unittest.TestCase):
    """Rules 5 and 7 now run per rank, and neither reading is weaker.

    Rule 5 asked for a window count over the whole arm and now asks it of
    every rank, which is the same question at one rank and a stronger one
    above it.

    Rule 7 changed less than an earlier draft of this docstring claimed. Its
    structural predicate already ran per window, because
    ``trace_window_metrics`` reads one file, so a rank whose graphs did not
    match failed under the old code as well. What changed is the attribution
    -- the error now names the rank -- and the fact that the per-window
    step-consistency check no longer spans two ranks. The test below
    discriminates on the message for that reason, and claims nothing more.

    Rules 6 and 9 read every rank's traces as one set, which is what they did
    when one rank was all there was. They are deliberately not per rank; the
    comment at their call site records why.
    """

    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.arm_dir = self.root / "baseline"
        self.log_path = self.root / "baseline.log"
        self.log_path.write_text(
            _compiled_line("default") + _SAC_LINE + _SIZE_LINE
            + "Training completed\n"
        )
        scenario = scenario_by_name("piper1b_rope")
        self.arm = next(arm for arm in scenario.arms if arm.name == "baseline")
        self.workload = scenario.workload

    def _window(self, rank: int, iteration: int, invocations: int = 80) -> None:
        write_trace(
            self.arm_dir
            / f"profiling/traces/iteration_{iteration}/rank{rank}_trace.json.gz",
            {"bwd": ("backward", [100.0] * invocations),
             "fwd": ("forward", [10.0] * invocations)},
            [step_event(1000.0)],
        )

    def _validate(self, regions=()) -> None:
        validate_arm(
            self.arm,
            self.arm_dir,
            self.log_path,
            self.workload,
            regions=regions,
        )

    def test_rule_five_fires_on_the_rank_that_is_short(self) -> None:
        for iteration in (20, 40):
            self._window(0, iteration)
        self._window(1, 20)
        with self.assertRaisesRegex(RuntimeError, r"found 1 for rank 1 under"):
            self._validate()

    def test_rule_five_keeps_its_one_rank_message(self) -> None:
        self._window(0, 20)
        with self.assertRaisesRegex(
            RuntimeError, r"expected at least 2 profiler windows, found 1 under"
        ):
            self._validate()

    def test_rule_five_still_fires_when_no_rank_wrote_anything(self) -> None:
        self.arm_dir.mkdir(parents=True)
        with self.assertRaisesRegex(
            RuntimeError, r"expected at least 2 profiler windows, found 0 under"
        ):
            self._validate()

    def test_rule_seven_asks_each_rank_its_own_structural_question(
        self,
    ) -> None:
        """Rank 1's graphs are repartitioned; rank 0's are fine."""
        regions = (
            Region(name="forward_block", phase="forward",
                   invocations_per_window=80),
        )
        for iteration in (20, 40):
            self._window(0, iteration)
            self._window(1, iteration, invocations=40)
        with self.assertRaisesRegex(RuntimeError, "rank 1:"):
            self._validate(regions=regions)

    def test_a_well_formed_two_rank_arm_passes_both_rules(self) -> None:
        regions = (
            Region(name="forward_block", phase="forward",
                   invocations_per_window=80),
        )
        for rank in (0, 1):
            for iteration in (20, 40):
                self._window(rank, iteration)
        self._validate(regions=regions)


class LegacyInertnessTests(unittest.TestCase):
    """The change moved no number that ``out/`` already holds.

    The fixture is the ``baseline`` arm of
    ``out/20260807T175156Z/piper1b_qkv/nvidia-h200`` -- a real 2-window,
    single-rank run at manifest schema 8, whose ``results.json`` (schema 3,
    the version this commit replaces) is checked in **verbatim**. That file is
    the ground truth: it was written by the code as it stood before ranks
    existed, and nothing in this test recomputes it.

    The two traces are the one derived part, and the derivation is stated so a
    reader can check it: every event that reaches a state-mutating branch of
    ``trace_window_metrics`` is kept, carrying only the seven fields those
    branches read, and everything else -- the ``args`` payloads, the
    un-annotated ``cpu_op`` frames, the non-launch runtime calls -- is
    dropped. That takes each window from 35,451 and 35,450 events to 11,775,
    and from 1.0 MB to 175 KB. The tests below are what prove the derivation
    lossless: if any dropped event had mattered, the recomputed figures would
    not equal the recorded ones.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()
        cls.out_dir = Path(cls._temporary.name) / "run"
        # Copy so no assertion here can touch the fixture.
        shutil.copytree(LEGACY_RESULTS_3, cls.out_dir)
        cls.recorded = json.loads((cls.out_dir / "results.json").read_text())
        cls.result = evaluate_run(cls.out_dir, arms_override=("baseline",))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_the_fixture_predates_this_change(self) -> None:
        manifest = json.loads((self.out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["schema_version"], 8)
        self.assertEqual(manifest["scenario"], "piper1b_qkv")
        self.assertEqual(self.recorded["schema_version"], 3)
        self.assertEqual(self.result.to_dict()["schema_version"], 5)

    def test_the_widened_glob_finds_exactly_what_the_old_one_found(self) -> None:
        arm_dir = self.out_dir / "baseline"
        old = sorted(
            arm_dir.glob("profiling/traces*/iteration_*/rank0_trace.json.gz")
        )
        self.assertEqual(trace_files(arm_dir), old)
        self.assertEqual(len(old), 2)

    def test_a_recorded_run_holds_exactly_one_rank(self) -> None:
        by_rank = trace_files_by_rank(self.out_dir / "baseline")
        self.assertEqual(list(by_rank), [0])
        per_rank = per_rank_pooled_metrics(by_rank, self._regions())
        self.assertEqual(list(per_rank), [0])
        self.assertEqual(busiest_rank(per_rank), 0)

    def _regions(self) -> tuple[Region, ...]:
        manifest = json.loads((self.out_dir / "manifest.json").read_text())
        return tuple(Region(**region) for region in manifest["regions"])

    def test_every_published_gpu_time_field_is_bit_identical(self) -> None:
        recorded = self.recorded["gpu_time"]["baseline"]
        published = self.result.gpu_time["baseline"]
        for field, value in recorded.items():
            with self.subTest(field=field):
                self.assertEqual(getattr(published, field), value)

    def test_every_published_region_summary_is_bit_identical(self) -> None:
        recorded = self.recorded["regions"]["baseline"]
        published = self.result.regions["baseline"]
        self.assertEqual(set(recorded), set(published))
        for name, summaries in recorded.items():
            for measure in ("span", "kernel"):
                with self.subTest(region=name, measure=measure):
                    self.assertEqual(
                        vars(getattr(published[name], measure)),
                        summaries[measure],
                    )

    def test_the_window_count_is_unchanged(self) -> None:
        self.assertEqual(
            self.result.trace_windows["baseline"],
            self.recorded["trace_windows"]["baseline"],
        )

    def test_the_new_fields_say_one_rank_and_no_collective(self) -> None:
        published = self.result.gpu_time["baseline"]
        self.assertEqual(published.ranks, (0,))
        self.assertEqual(published.published_rank, 0)
        self.assertEqual(
            published.kernel_ms_per_step_summed_over_ranks,
            published.kernel_ms_per_step,
        )
        self.assertEqual(published.collective_ms_per_step, 0.0)
        self.assertEqual(
            published.compute_ms_per_step, published.kernel_ms_per_step
        )

    def test_the_two_bases_agree_on_this_single_stream_trace(self) -> None:
        """A single-stream arm has nothing to overlap, so the bases agree.

        They are not bit-identical -- the union walks the intervals in time
        order and the sum walks them in file order, so they accumulate the
        same values differently -- and the residual here is 5e-4 us in
        402,484. Reading that residual as evidence of overlap would be wrong.
        The overlap case is
        ``CollectiveAndBasisTests.test_busy_union_is_below_the_sum_when_streams_overlap``.
        """
        published = self.result.gpu_time["baseline"]
        self.assertGreater(published.busy_kernel_ms_per_step, 0.0)
        self.assertAlmostEqual(
            published.busy_kernel_ms_per_step,
            published.kernel_ms_per_step,
            places=6,
        )

    def test_the_fixture_traces_reproduce_the_summaries_they_are_derived_from(
        self,
    ) -> None:
        """The proof is the fixture's, not the harness's.

        Recompute the region summaries straight from the traces and compare
        them to the recorded file. A fixture that had lost a needed event
        would fail here rather than quietly agreeing with itself.
        """
        pooled = per_rank_pooled_metrics(
            trace_files_by_rank(self.out_dir / "baseline"), self._regions()
        )[0]
        recorded = self.recorded["regions"]["baseline"]
        for name, summaries in recorded.items():
            self.assertEqual(
                vars(summarize(pooled.region_spans[name])), summaries["span"]
            )
            self.assertEqual(
                vars(summarize(pooled.region_kernel[name])),
                summaries["kernel"],
            )

    def test_the_trace_glob_constant_is_the_name_on_disk(self) -> None:
        self.assertTrue(
            re.fullmatch(
                TRACE_FILE_GLOB.replace("*", r".*"), "rank0_trace.json.gz"
            )
        )


def launch_events(count: int, start: float = 700_000.0) -> list:
    """``count`` kernel launches, in the two categories the extraction reads."""
    return [
        {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
         "pid": 0, "tid": 1, "ts": start + index, "dur": 1.0}
        for index in range(count)
    ]


def host_step(name: str, duration: float, start: float = 0.0) -> dict:
    """One profiler step, named on the host. This is the normal case."""
    return {"ph": "X", "cat": "user_annotation", "name": name,
            "tid": 1, "ts": start, "dur": duration}


def device_step(name: str, duration: float, start: float = 0.0) -> dict:
    """One profiler step, named on the device side only."""
    return {"ph": "X", "cat": "gpu_user_annotation", "name": name,
            "tid": 100, "ts": start, "dur": duration}


class LaunchCountsSkipSteplessRanksTests(unittest.TestCase):
    """``tools/collect_matrix.py`` must not divide a stepless rank by one.

    A rank whose windows carry no ``ProfilerStep`` has no per-step figure. If
    the reduction gives it a denominator of 1, it reports the whole window's
    launches as one step's worth. That is the largest value in the set, so it
    wins the maximum and becomes the arm's published figure -- a rank the
    profiler never measured, publishing a number no rank paid.

    The numbers here separate the two answers. Rank 0 is stepless and runs
    500 launches. Rank 1 runs the same 500 launches over 5 steps, so it pays
    100 per step. The correct answer is 100.0 and the defect answers 500.0.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        arm = self.tmp / "baseline"
        write_trace(
            arm / "profiling/traces/iteration_20/rank0_trace.json.gz",
            {"fwd": ("forward", [10.0])},
            launch_events(500),
        )
        write_trace(
            arm / "profiling/traces/iteration_20/rank1_trace.json.gz",
            {"fwd": ("forward", [10.0])},
            [host_step(f"ProfilerStep#{20 + index}", 1000.0, start=index * 2000.0)
             for index in range(5)] + launch_events(500),
        )

    def test_the_stepless_rank_is_left_out_of_the_maximum(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from collect_matrix import launch_counts

        counts = launch_counts(self.tmp, ["baseline"])
        self.assertAlmostEqual(counts["baseline"], 100.0)

    def test_the_stepless_rank_would_win_if_it_were_divided_by_one(self) -> None:
        """The defect this pins, stated as arithmetic rather than as a claim."""
        by_rank = trace_files_by_rank(self.tmp / "baseline")
        pooled = per_rank_pooled_metrics(by_rank, ())
        self.assertEqual(pooled[0].profiled_steps, 0)
        self.assertEqual(pooled[0].launch_count, 500)
        self.assertEqual(pooled[1].profiled_steps, 5)
        divided_by_one = max(
            rank.launch_count / max(rank.profiled_steps, 1)
            for rank in pooled.values()
        )
        self.assertAlmostEqual(divided_by_one, 500.0)


class PartialStepAnnotationIsRefusedTests(unittest.TestCase):
    """Some steps named on the host and others not is refused, not averaged.

    ``wall_ms_per_step`` divides by the steps that carried a host annotation.
    Every other per-step figure divides by ``profiled_steps``. Where the two
    counts disagree, ``wall - busy`` subtracts one average from another taken
    over a different denominator, and publishes the difference as a bubble.

    No trace under ``out/`` is in this state, on either engine. The refusal
    keeps it that way rather than returning a number nobody can read.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_one_window_with_a_host_step_and_a_device_step_is_refused(self) -> None:
        path = write_trace(
            self.tmp / "rank0_trace.json.gz",
            {"fwd": ("forward", [10.0])},
            [host_step("ProfilerStep#20", 1000.0),
             device_step("ProfilerStep#21", 1000.0, start=2000.0)],
        )
        with self.assertRaises(ValueError) as raised:
            trace_window_metrics(path, ())
        self.assertIn("host annotation", str(raised.exception))

    def test_an_all_host_window_pooled_with_an_all_device_window_is_refused(
        self,
    ) -> None:
        """The window check alone cannot see this one. Each window is pure."""
        host_only = write_trace(
            self.tmp / "iteration_20/rank0_trace.json.gz",
            {"fwd": ("forward", [10.0])},
            [host_step("ProfilerStep#20", 1000.0)],
        )
        device_only = write_trace(
            self.tmp / "iteration_40/rank0_trace.json.gz",
            {"fwd": ("forward", [10.0])},
            [device_step("ProfilerStep#40", 1000.0)],
        )
        trace_window_metrics(host_only, ())
        trace_window_metrics(device_only, ())
        with self.assertRaises(ValueError) as raised:
            pooled_window_metrics((host_only, device_only), ())
        self.assertIn("on the device", str(raised.exception))

    def test_every_step_on_the_device_still_reports_no_wall(self) -> None:
        """The documented carve-out. No mixture, so no refusal."""
        path = write_trace(
            self.tmp / "rank0_trace.json.gz",
            {"fwd": ("forward", [10.0])},
            [device_step("ProfilerStep#20", 1000.0)],
        )
        pooled = pooled_window_metrics((path,), ())
        self.assertEqual(pooled.profiled_steps, 1)
        self.assertIsNone(pooled.wall_ms_per_step)

    def test_every_step_on_the_host_is_the_normal_case(self) -> None:
        path = write_trace(
            self.tmp / "rank0_trace.json.gz",
            {"fwd": ("forward", [10.0])},
            [host_step("ProfilerStep#20", 1000.0)],
        )
        pooled = pooled_window_metrics((path,), ())
        self.assertEqual(pooled.profiled_steps, 1)
        self.assertAlmostEqual(pooled.wall_ms_per_step, 1.0)


if __name__ == "__main__":
    unittest.main()
