"""What a tokens/s figure counts, and what a mesh does with several of them.

Three things are under test here, and they are one concern read at three
places. The megatron driver divides a rank's own token count by the pipeline
degree, which is what TorchTitan already did. The manifest records that
definition. Evaluation publishes the MINIMUM over the ranks, records every
rank's own figure beside it, and warns when they spread.

**The single-rank case is the one that must not move.** Every number this
repo has published was taken at one rank, so each section below pins the
one-rank answer against the value the old code produced.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.artifacts.manifests import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    THROUGHPUT_DEFINITION,
    ArmRecord,
    manifest_data,
    write_manifest,
)
from benchmarks.e2e.engines.megatron_stock.driver.step_log import (  # noqa: E402
    tokens_per_second,
)
from benchmarks.e2e.engines.megatron_stock.steps import step_record  # noqa: E402
from benchmarks.e2e.parallelism import TRIVIAL_SPEC, ParallelismSpec  # noqa: E402
from benchmarks.e2e.engines.api import (  # noqa: E402
    Arm,
    CompileMode,
    DroppedLine,
    RunSpec,
    StepRead,
    StepSample,
)
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig  # noqa: E402
from benchmarks.e2e.registry import ENGINES, SCENARIOS, SEED  # noqa: E402
from benchmarks.models.piper_qwen3.shape import PIPER_1B  # noqa: E402
from benchmarks.e2e.results import (  # noqa: E402
    arm_steps,
    dropped_line_warnings,
    evaluate_run,
    pinning_warnings,
    RESULTS_SCHEMA_VERSION,
    extras_statistics,
    loss_visible_rank,
    lost_step_refusals,
    rank_result,
    rate_mean,
    refuse_non_finite_trajectories,
    render_evaluation,
    sampled_steps,
    step_ms,
    trajectory,
    write_results,
)


from tests.engine_helpers import (  # noqa: E402
    TEST_METADATA,
    run_spec,
    titan_step_line,
    write_run_manifest,
)


FIXTURE_RUN = run_spec(seq_len=1024, local_batch_size=4, steps=10)
"""A profiled run of 10 steps at sequence length 1024 and local batch 4; its rule takes steps 3 to 10."""


TITAN_ARM = Arm(
    name="baseline",
    description="baseline",
    config=TorchTitanConfig(compile=CompileMode.NONE),
)
"""A TorchTitan arm, whose engine reads the fork's step line."""


def _samples(arm: Arm, log: Path) -> dict:
    """The step samples of each rank in the log."""
    return {rank: read.samples for rank, read in arm_steps(arm, log).items()}


def _step_lines(*, tps: int) -> str:
    """The step lines that a rank prints for every step of ``FIXTURE_RUN``."""
    return "".join(
        titan_step_line(step, tps=tps)
        for step in range(1, FIXTURE_RUN.data.steps + 1)
    )


def _prefixed(text: str, rank: int) -> str:
    return "".join(f"[rank{rank}]:{line}" for line in text.splitlines(True))


class DriverArithmeticTests(unittest.TestCase):
    """``tokens_per_second`` is the driver's whole throughput rule."""

    def test_one_rank_keeps_the_value_the_driver_always_printed(self) -> None:
        # The old expression was round(batch * seq_len / elapsed).
        self.assertEqual(tokens_per_second(4096, 0.5, 1), round(4096 / 0.5))

    def test_a_pipeline_degree_divides_the_rate(self) -> None:
        self.assertEqual(tokens_per_second(4096, 0.5, 2), round(4096 / 1.0))
        self.assertEqual(tokens_per_second(4096, 0.5, 4), round(4096 / 2.0))

    def test_the_global_rate_is_the_per_device_rate_times_the_degree(self) -> None:
        # The ranks of one pipeline share a batch, so multiplying the
        # published per-device figure by the world size recovers the job's
        # own rate.
        per_device = tokens_per_second(4096, 0.5, 2)
        self.assertEqual(per_device * 2, round(4096 / 0.5))

    def test_a_degree_below_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            tokens_per_second(4096, 0.5, 0)


class ManifestRecordsTheDefinitionTests(unittest.TestCase):
    def _manifest(self) -> dict:
        scenario = SCENARIOS["engines"]
        run = RunSpec(
            shape=PIPER_1B,
            data=scenario.data,
            parallelism=TRIVIAL_SPEC,
            ac_mode="none",
            profile=False,
            window=scenario.window,
            warmup_steps=10,
            seed=SEED,
        )
        return manifest_data(
            scenario=scenario,
            hardware="test-gpu",
            metadata={"requested_gpu": "0"},
            run=run,
            arms=(
                ArmRecord(
                    arm=scenario.arms[0],
                    command=("python", "-m", "x"),
                    env_delta={},
                    cpu_pinning="none: test",
                    execution_model="test",
                ),
            ),
        )

    def test_the_manifest_names_what_a_tokens_per_second_figure_counts(self) -> None:
        self.assertEqual(
            self._manifest()["throughput_definition"], THROUGHPUT_DEFINITION
        )
        self.assertEqual(THROUGHPUT_DEFINITION, "tokens_per_second_per_device")

    def test_the_schema_moved_with_the_new_key(self) -> None:
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 20)
        self.assertEqual(self._manifest()["schema_version"], 20)


class PerRankLogParsingTests(unittest.TestCase):
    """One file, two ranks: the samples belong to whoever printed them."""

    def test_an_unprefixed_log_reads_as_one_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text(_step_lines(tps=1000))
            by_rank = arm_steps(TITAN_ARM, log)
        self.assertEqual(list(by_rank), [0])
        self.assertEqual(
            [sample.tokens_per_second for sample in by_rank[0].samples], [1000] * 10
        )

    def test_two_ranks_do_not_pool_into_one_series(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text(
                _prefixed(_step_lines(tps=1000), 0)
                + _prefixed(_step_lines(tps=800), 1)
            )
            by_rank = arm_steps(TITAN_ARM, log)
        self.assertEqual(sorted(by_rank), [0, 1])
        self.assertEqual(
            [sample.tokens_per_second for sample in by_rank[0].samples], [1000] * 10
        )
        self.assertEqual(
            [sample.tokens_per_second for sample in by_rank[1].samples], [800] * 10
        )
        self.assertEqual({sample.rank for sample in by_rank[1].samples}, {1})

    def test_a_missing_log_reads_as_no_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(arm_steps(TITAN_ARM, Path(temporary) / "absent.log"), {})


class LossVisibleRankTests(unittest.TestCase):
    """TorchTitan's own ``_get_metrics_rank`` arithmetic, restated."""

    def test_a_single_gpu_run_reads_rank_zero(self) -> None:
        self.assertEqual(loss_visible_rank(world_size=1, pp=1), 0)

    def test_a_two_stage_pipeline_reads_the_last_stage(self) -> None:
        self.assertEqual(loss_visible_rank(world_size=2, pp=2), 1)

    def test_a_pure_data_parallel_mesh_reads_rank_zero(self) -> None:
        self.assertEqual(loss_visible_rank(world_size=4, pp=1), 0)

    def test_the_last_stage_is_a_whole_data_parallel_group_away(self) -> None:
        self.assertEqual(loss_visible_rank(world_size=4, pp=2), 2)


class _RunFixture:
    """A minimal output directory: a manifest and one log per arm."""

    @staticmethod
    def build(
        out_dir: Path,
        logs: dict[str, str],
        parallelism: dict | None,
    ) -> None:
        spec = ParallelismSpec(
            **{
                key: value
                for key, value in (parallelism or {}).items()
                if key != "world_size"
            }
        )
        if spec.pp > 1:
            spec = ParallelismSpec(
                dp=spec.dp, pp=spec.pp, ep=spec.ep, zero=spec.zero,
                pp_schedule="1F1B",
            )
        arms = tuple(
            ArmRecord(
                arm=Arm(name=name, description=name, config=TorchTitanConfig(compile=CompileMode.NONE)),
                command=("python",),
                env_delta={},
                cpu_pinning="none: test",
                execution_model="test",
            )
            for name in sorted(logs)
        )
        write_manifest(
            out_dir,
            scenario=SCENARIOS["engines"],
            hardware="test-gpu",
            metadata=TEST_METADATA,
            run=replace(FIXTURE_RUN, parallelism=spec),
            arms=arms,
        )
        for arm, text in logs.items():
            (out_dir / f"{arm}.log").write_text(text)


class SingleRankIsUnchangedTests(unittest.TestCase):
    """The published figure at one rank is the median it has always been."""

    def _evaluate(self, parallelism: dict | None):
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    "baseline": _step_lines(tps=1000),
                    "optimized": _step_lines(tps=1200),
                },
                parallelism,
            )
            result = evaluate_run(out_dir)
            return result, render_evaluation(result)

    def test_one_rank_publishes_the_median_it_always_published(self) -> None:
        result, _ = self._evaluate({"world_size": 1, "pp": 1})
        summary = result.results["baseline"]
        self.assertEqual(summary.tokens_per_second.median, 1000)
        self.assertEqual(summary.tokens_per_second.median_rank, 0)
        self.assertEqual([row.rank for row in summary.per_rank], [0])

    def test_one_rank_prints_one_row_per_arm_and_raises_no_warning(self) -> None:
        result, report = self._evaluate({"world_size": 1, "pp": 1})
        self.assertNotIn("per-rank tokens/s", report)
        self.assertNotIn("WARNING", report)
        self.assertEqual(result.warnings, ())


class ZeroWarningsReachTheArtifactTests(unittest.TestCase):
    """The runner says them when the run starts. A reader of results.json
    was not there, and the file is what a report quotes.
    """

    def _warnings(self, parallelism: dict | None) -> list[str]:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir, {"baseline": _step_lines(tps=1000)}, parallelism
            )
            return list(evaluate_run(out_dir).warnings)

    def test_a_zero1_cell_at_dp_one_carries_both_warnings(self) -> None:
        warnings = self._warnings(
            {
                "world_size": 1,
                "dp": 1,
                "pp": 1,
                "ep": 1,
                "zero": 1,
            }
        )
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("at dp 1" in warning for warning in warnings))
        self.assertTrue(any("ZeRO-2" in warning for warning in warnings))

    def test_the_replicated_parity_carries_none(self) -> None:
        self.assertEqual(
            self._warnings(
                {
                    "world_size": 1,
                    "dp": 1,
                    "pp": 1,
                    "ep": 1,
                    "zero": 0,
                }
            ),
            [],
        )


class PinningWarningTests(unittest.TestCase):
    """A run whose arms mix pinned and unpinned processes says so in results.json."""

    PINNED = "numactl --cpunodebind=1 --membind=1"

    def _warnings(self, pinnings: dict[str, str]) -> list[str]:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            arms = tuple(
                ArmRecord(
                    arm=Arm(
                        name=name,
                        description=name,
                        config=TorchTitanConfig(compile=CompileMode.NONE),
                    ),
                    command=("python",),
                    env_delta={},
                    cpu_pinning=pinning,
                    execution_model="test",
                )
                for name, pinning in pinnings.items()
            )
            write_manifest(
                out_dir,
                scenario=SCENARIOS["engines"],
                hardware="test-gpu",
                metadata=TEST_METADATA,
                run=FIXTURE_RUN,
                arms=arms,
            )
            for name in pinnings:
                (out_dir / f"{name}.log").write_text(_step_lines(tps=1000))
            return json.loads(
                write_results(evaluate_run(out_dir)).read_text()
            )["warnings"]

    def test_a_pinned_arm_beside_a_declined_arm_warns(self) -> None:
        (warning,) = self._warnings(
            {"pinned": self.PINNED, "declined": "declined by engine"}
        )
        self.assertIn("the arms mix CPU pinning", warning)
        self.assertIn("declined: declined by engine", warning)

    def test_arms_with_one_pinning_do_not_warn(self) -> None:
        for pinning in (self.PINNED, "none: numactl not available"):
            with self.subTest(pinning=pinning):
                self.assertEqual(
                    self._warnings({"first": pinning, "second": pinning}), []
                )

    def test_two_unpinned_reasons_are_one_pinning(self) -> None:
        self.assertEqual(
            pinning_warnings(
                [
                    ArmRecord(
                        arm=Arm(
                            name=name,
                            description=name,
                            config=TorchTitanConfig(compile=CompileMode.NONE),
                        ),
                        command=(),
                        env_delta=None,
                        cpu_pinning=pinning,
                        execution_model=None,
                    )
                    for name, pinning in (
                        ("a", "declined by engine"),
                        ("b", "none: numactl not available"),
                    )
                ]
            ),
            [],
        )


class NoArmIsSpecialTests(unittest.TestCase):
    """Every arm publishes absolutes, and no arm anchors the others."""

    def test_a_run_without_a_baseline_arm_evaluates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    "titan_compiled": _step_lines(tps=1100),
                    "megatron_stock": _step_lines(tps=1200),
                },
                {"world_size": 1, "pp": 1},
            )
            result = evaluate_run(out_dir)
            report = render_evaluation(result)

        self.assertEqual(
            result.results["titan_compiled"].tokens_per_second.median, 1100
        )
        self.assertEqual(
            result.results["megatron_stock"].tokens_per_second.median, 1200
        )
        self.assertNotIn("ratio", report)
        self.assertNotIn("vs base", report)


class TwoRanksPublishTheSlowestTests(unittest.TestCase):
    """A schedule holds the ranks in step, so the mesh runs at the slowest."""

    def _evaluate(self, fast: int, slow: int):
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    "baseline": _prefixed(_step_lines(tps=fast), 0)
                    + _prefixed(_step_lines(tps=slow), 1)
                },
                {"world_size": 2, "pp": 2},
            )
            result = evaluate_run(out_dir)
            return result, render_evaluation(result)

    def test_the_published_figure_is_the_minimum_and_never_the_mean_over_ranks(
        self,
    ) -> None:
        result, _ = self._evaluate(1000, 900)
        summary = result.results["baseline"]
        self.assertEqual(summary.tokens_per_second.median, 900)
        self.assertEqual(summary.tokens_per_second.median_rank, 1)
        self.assertEqual(summary.tokens_per_second.mean, 900)
        self.assertEqual(summary.tokens_per_second.mean_rank, 1)
        self.assertEqual(summary.rank_reduction, "slowest_rank_per_statistic")
        # The mean over the ranks would be 950, which no device achieved.
        self.assertNotEqual(summary.tokens_per_second.median, 950)

    def test_every_rank_reaches_the_file_beside_the_published_one(self) -> None:
        result, _ = self._evaluate(1000, 900)
        summary = result.results["baseline"]
        self.assertEqual(
            [(row.rank, row.tokens_per_second.median) for row in summary.per_rank],
            [(0, 1000), (1, 900)],
        )
        self.assertEqual([len(row.steps) for row in summary.per_rank], [8, 8])
        self.assertEqual(summary.sample_count, 8)

    def test_the_published_step_cost_is_the_slowest_rank_own(self) -> None:
        result, _ = self._evaluate(1000, 900)
        summary = result.results["baseline"]
        # pp 2 halves the per-rank step cost of 4 x 1024 tokens.
        self.assertAlmostEqual(
            summary.step_ms.median, 1000.0 * 4 * 1024 / (900 * 2)
        )
        self.assertEqual(summary.step_ms.median_rank, 1)
        self.assertAlmostEqual(
            summary.per_rank[0].step_ms.median, 1000.0 * 4 * 1024 / (1000 * 2)
        )

    def test_a_narrow_spread_raises_nothing(self) -> None:
        result, _ = self._evaluate(1000, 900)
        self.assertEqual(
            [warning for warning in result.warnings if "varies" in warning], []
        )

    def test_a_wide_spread_warns_and_names_both_ranks(self) -> None:
        result, _ = self._evaluate(1000, 500)
        warning = next(
            warning for warning in result.warnings if "across ranks" in warning
        )
        self.assertIn("2.00x", warning)
        self.assertIn("rank 0 1,000", warning)
        self.assertIn("rank 1 500", warning)

    def test_the_written_file_carries_the_per_rank_rows(self) -> None:
        result, _ = self._evaluate(1000, 900)
        machine = result.to_dict()["results"]["baseline"]
        self.assertEqual(machine["rank_reduction"], "slowest_rank_per_statistic")
        self.assertEqual([row["rank"] for row in machine["per_rank"]], [0, 1])


class ARankWithNoSampleRefusesTheArmTests(unittest.TestCase):
    """"No sample" is a measurement that did not happen, so the arm publishes nothing."""

    def _refusal(self, logs: str) -> str:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir, {"baseline": logs}, {"world_size": 2, "pp": 2}
            )
            with self.assertRaises(ValueError) as caught:
                evaluate_run(out_dir)
            self.assertFalse((out_dir / "results.json").exists())
        return str(caught.exception)

    def test_a_silent_rank_refuses_the_arm(self) -> None:
        refusal = self._refusal(
            _prefixed(_step_lines(tps=1000), 0) + "[rank1]:starting up\n"
        )
        self.assertIn("baseline: rank 1 lacks sampled step 3", refusal)

    def test_a_rank_absent_from_the_log_refuses_the_arm(self) -> None:
        refusal = self._refusal(_prefixed(_step_lines(tps=1000), 0))
        self.assertIn("baseline: rank 1 lacks sampled step 3", refusal)


class TornStepLineTests(unittest.TestCase):
    """A step line that another rank's prefix cut is dropped, and results.json names it."""

    def _lines(self, cut_step: int) -> str:
        """Both ranks' step lines of every step, with rank 1's line of ``cut_step`` cut by a rank prefix."""
        cut = titan_step_line(cut_step)[:-60] + "[rank0]:USDT: profiler_stop\n"
        return "".join(
            f"[rank0]:{titan_step_line(step)}"
            + f"[rank1]:{cut if step == cut_step else titan_step_line(step)}"
            for step in range(1, FIXTURE_RUN.data.steps + 1)
        )

    def test_the_warning_names_the_arm_the_rank_the_step_and_the_log_line(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir, {"baseline": self._lines(1)}, {"world_size": 2, "pp": 2}
            )
            result = evaluate_run(out_dir)
        self.assertIn(
            "baseline: rank 1 step 1: a rank prefix cut the step line "
            "at line 2 of baseline.log, so the evaluation drops that step",
            result.warnings,
        )
        self.assertEqual(len(result.results["baseline"].per_rank[1].steps), 8)

    def test_a_cut_sampled_step_refuses_the_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir, {"baseline": self._lines(4)}, {"world_size": 2, "pp": 2}
            )
            with self.assertRaisesRegex(
                ValueError, r"^baseline: rank 1 lacks sampled step 4; "
            ):
                evaluate_run(out_dir)

    def test_a_step_that_the_cut_removed_reads_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "megatron_stock.log"
            log.write_text("a\nb\n")
            read = StepRead(samples=(), dropped=(DroppedLine(rank=0, line=2, step=None),))
            (warning,) = dropped_line_warnings("megatron_stock", log, {0: read})
        self.assertEqual(
            warning,
            "megatron_stock: rank 0 step unknown: a rank prefix cut the step "
            "line at line 2 of megatron_stock.log, so the evaluation drops that step",
        )

    def test_the_log_line_of_a_one_rank_log_is_its_own(self) -> None:
        cut = titan_step_line(2)[:-60] + "[rank0]:USDT: profiler_stop\n"
        rest = "".join(
            titan_step_line(step) for step in range(3, FIXTURE_RUN.data.steps + 1)
        )
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {"baseline": "header\n" + titan_step_line(1) + cut + rest},
                {"world_size": 1, "pp": 1},
            )
            warnings = evaluate_run(out_dir).warnings
        self.assertTrue(
            any("rank 0 step 2" in w and "at line 3 of" in w for w in warnings),
            warnings,
        )


def _megatron_record(step: int, *, tps: int = 1000) -> str:
    """One step record of the stock Megatron driver."""
    return (
        step_record(
            step=step,
            tokens_per_second=tps,
            peak_memory_gib=3.0,
            loss=1.0,
            grad_norm=2.0,
            tflops=12.5,
            mfu=1.26,
        )
        + "\n"
    )


STEP_LINES = {"titan_eager": titan_step_line, "megatron_stock": _megatron_record}
"""The step line of each engine, by arm."""


class LostStepTests(unittest.TestCase):
    """A rank that lacks a sampled step refuses the arm, whichever engine reads the log."""

    RUN = run_spec(
        ac_mode="none",
        seq_len=1024,
        local_batch_size=4,
        steps=10,
        parallelism=ParallelismSpec(dp=2),
    )

    def _evaluate(self, arm: str, lost: int | None):
        """Evaluate a two-rank run of ``arm`` in which rank 1 does not log step ``lost``."""
        line = STEP_LINES[arm]
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            write_run_manifest(out_dir, self.RUN, (ENGINES.arm(arm),))
            (out_dir / f"{arm}.log").write_text(
                "".join(
                    f"[rank{rank}]:{line(step)}"
                    for step in range(1, self.RUN.data.steps + 1)
                    for rank in (0, 1)
                    if not (rank == 1 and step == lost)
                )
            )
            return evaluate_run(out_dir)

    def test_each_engine_refuses_a_rank_that_lacks_a_sampled_step(self) -> None:
        for arm in STEP_LINES:
            with self.subTest(arm=arm), self.assertRaisesRegex(
                ValueError, rf"^{arm}: rank 1 lacks sampled step 7; "
            ):
                self._evaluate(arm, 7)

    def test_a_lost_step_outside_the_rule_refuses_nothing(self) -> None:
        for arm in STEP_LINES:
            with self.subTest(arm=arm):
                result = self._evaluate(arm, 2)
                self.assertEqual(
                    [len(row.steps) for row in result.results[arm].per_rank], [8, 8]
                )

    def test_the_profiled_rule_takes_35_steps_of_80(self) -> None:
        steps = sampled_steps(run_spec(steps=80))
        self.assertEqual(
            steps,
            (*range(3, 11), *range(22, 31), *range(42, 51), *range(62, 71)),
        )
        self.assertEqual(len(steps), 35)

    def test_the_unprofiled_rule_takes_every_step_after_the_warmup(self) -> None:
        self.assertEqual(
            sampled_steps(run_spec(profile=False, warmup_steps=10, steps=40)),
            tuple(range(11, 41)),
        )

    def test_a_sampled_step_past_the_run_is_refused(self) -> None:
        run = run_spec(steps=10)
        samples = [
            StepSample(
                rank=0,
                step=step,
                tokens_per_second=1000,
                peak_memory_gib=3.0,
                loss=1.0,
                grad_norm=2.0,
            )
            for step in (*range(3, 11), 22)
        ]
        self.assertEqual(
            lost_step_refusals(run, {0: samples}),
            ["rank 0 logs sampled step 22, and the run has 10 steps"],
        )


class NonFiniteTrajectoryTests(unittest.TestCase):
    """A ``nan`` or an ``inf`` on a step line fails the arm before publication.

    The step-line patterns read both words on purpose. This is the guard
    the 2026-08-30 report asked for before anyone turns stock Megatron's
    own NaN check off: it runs on every arm, whatever the engine's guard
    did, and it names the rank and the step.
    """

    def _lines(self, *, loss: float = 1.0, grad_norm: float = 2.0, at: int = 3):
        return "".join(
            titan_step_line(
                step,
                loss=loss if step == at else 1.0,
                grad_norm=grad_norm if step == at else 2.0,
            )
            for step in range(1, FIXTURE_RUN.data.steps + 1)
        )

    def _evaluate(self, logs: dict[str, str], parallelism=None):
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(out_dir, logs, parallelism)
            evaluate_run(out_dir)
            self.assertFalse((out_dir / "results.json").exists())

    def test_a_nan_loss_fails_the_arm_and_names_the_step(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"baseline: rank 0 logged a non-finite loss at step 3"
        ):
            self._evaluate({"baseline": self._lines(loss=float("nan"))})

    def test_an_inf_grad_norm_fails_the_arm_and_names_the_step(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"optimized: rank 0 logged a non-finite grad_norm at step 4",
        ):
            self._evaluate(
                {
                    "baseline": self._lines(),
                    "optimized": self._lines(grad_norm=float("inf"), at=4),
                }
            )

    def test_a_negative_inf_is_not_finite_either(self) -> None:
        with self.assertRaisesRegex(ValueError, r"step 5 \(-inf\)"):
            self._evaluate({"baseline": self._lines(loss=float("-inf"), at=5)})

    def test_every_rank_is_read_and_the_failure_names_the_rank(self) -> None:
        """The published trajectory is one rank's; the guard reads all of
        them, because no rank prints a non-finite value on purpose."""
        with self.assertRaisesRegex(
            ValueError, r"baseline: rank 1 logged a non-finite loss at step 3"
        ):
            self._evaluate(
                {
                    "baseline": _prefixed(self._lines(), 0)
                    + _prefixed(self._lines(loss=float("nan")), 1)
                },
                {"world_size": 2, "pp": 2},
            )

    def test_the_titan_sentinel_reads_as_no_loss(self) -> None:
        """A TorchTitan rank without the loss prints ``-1.0``, which is not a loss."""
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    "baseline": _prefixed(self._lines(loss=-1.0, at=3), 0)
                    + _prefixed(self._lines(), 1)
                },
                {"world_size": 2, "pp": 2},
            )
            result = evaluate_run(out_dir)
            steps = arm_steps(TITAN_ARM, out_dir / "baseline.log")
        self.assertEqual(result.losses["baseline"][2], (3, 1.0))
        self.assertIsNone(steps[0].samples[2].loss)

    def test_the_guard_reads_the_samples_directly(self) -> None:
        """Callable on its own, so a reader of an old directory can ask."""
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text(self._lines())
            refuse_non_finite_trajectories("baseline", _samples(TITAN_ARM, log), log)
            log.write_text(self._lines(grad_norm=float("nan"), at=2))
            with self.assertRaisesRegex(ValueError, r"grad_norm at step 2"):
                refuse_non_finite_trajectories(
                    "baseline", _samples(TITAN_ARM, log), log
                )


class WholeEvaluationTests(unittest.TestCase):
    """The published payload, read end to end on a synthetic run.

    The assertions pin the loss parser, the host-latency warning, and the
    machine-readable and human-readable halves of one evaluation.
    """

    def test_loss_parsing_flags_non_finite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text(
                titan_step_line(1, loss=7.4478)
                + titan_step_line(2, loss=float("nan"))
            )
            parsed = trajectory(arm_steps(TITAN_ARM, log)[0].samples, "loss")
        self.assertEqual(parsed[0], (1, 7.4478))
        self.assertNotEqual(parsed[1][1], parsed[1][1])  # NaN

    def test_complete_evaluation_is_human_and_machine_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    "titan_compiled": _step_lines(tps=1000),
                    "megatron_stock": _step_lines(tps=1200),
                },
                {"world_size": 1, "pp": 1},
            )
            result = evaluate_run(out_dir)
            results_path = write_results(result)
            machine = json.loads(results_path.read_text())
            report = render_evaluation(result)

        self.assertEqual(machine["schema_version"], RESULTS_SCHEMA_VERSION)
        self.assertEqual(RESULTS_SCHEMA_VERSION, 7)
        self.assertEqual(
            machine["results"]["megatron_stock"]["tokens_per_second"]["median"],
            1200,
        )
        self.assertEqual(machine["losses"]["megatron_stock"][0]["value"], 1.0)
        self.assertIn("tokens/s", report)
        self.assertIn("mean tok/s", report)
        self.assertIn("step ms", report)
        self.assertIn("mean ms", report)
        self.assertIn("total tokens over the total time", report)
        self.assertIn("p95 ms", report)
        self.assertIn("peak GiB", report)
        self.assertNotIn("gpu kernel time", report)

    def test_the_payload_carries_the_schema_seven_keys_and_no_others(
        self,
    ) -> None:
        """The exact key tree, top level, per arm and per rank."""
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {"titan_compiled": _step_lines(tps=1000)},
                {"world_size": 1, "pp": 1},
            )
            machine = evaluate_run(out_dir).to_dict()

        self.assertEqual(
            list(machine),
            [
                "schema_version",
                "scenario",
                "hardware",
                "output_dir",
                "arms",
                "results",
                "losses",
                "gradient_norms",
                "warnings",
            ],
        )
        arm = machine["results"]["titan_compiled"]
        self.assertEqual(
            list(arm),
            [
                "sample_count",
                "tokens_per_second",
                "step_ms",
                "peak_memory_gib",
                "extras",
                "rank_reduction",
                "per_rank",
            ],
        )
        self.assertEqual(
            list(arm["tokens_per_second"]),
            ["median", "median_rank", "mean", "mean_rank"],
        )
        self.assertEqual(
            list(arm["step_ms"]),
            ["median", "median_rank", "mean", "mean_rank", "p95", "p95_rank"],
        )
        self.assertEqual(
            list(arm["peak_memory_gib"]),
            ["max", "max_rank", "median", "median_rank", "mean", "mean_rank"],
        )
        self.assertEqual(list(arm["extras"]), ["torchtitan"])
        self.assertEqual(
            arm["extras"]["torchtitan"]["tflops"],
            {"median": 12.5, "median_rank": 0, "mean": 12.5, "mean_rank": 0},
        )
        self.assertEqual(sorted(arm["extras"]["torchtitan"]), ["mfu", "tflops"])
        (rank,) = arm["per_rank"]
        self.assertEqual(
            list(rank),
            [
                "rank",
                "tokens_per_second",
                "step_ms",
                "peak_memory_gib",
                "extras",
                "steps",
            ],
        )
        self.assertEqual(list(rank["tokens_per_second"]), ["median", "mean"])
        self.assertEqual(list(rank["step_ms"]), ["median", "mean", "p95"])
        self.assertEqual(list(rank["peak_memory_gib"]), ["max", "median", "mean"])
        self.assertEqual(rank["extras"]["mfu"], {"median": 1.26, "mean": 1.26})
        self.assertEqual([step["step"] for step in rank["steps"]], list(range(3, 11)))
        self.assertEqual(
            rank["steps"][0],
            {
                "step": 3,
                "tokens_per_second": 1000,
                "step_ms": 1000.0 * 4 * 1024 / 1000,
                "peak_memory_gib": 3.0,
                "extras": {"tflops": 12.5, "mfu": 1.26},
            },
        )


class MeanAndMedianTests(unittest.TestCase):
    """Each figure publishes a median and a mean, and each statistic names its own slowest rank."""

    def _evaluate(self, logs: dict[int, str]):
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {"baseline": "".join(_prefixed(text, rank) for rank, text in logs.items())},
                {"world_size": len(logs), "pp": len(logs)},
            )
            return evaluate_run(out_dir).results["baseline"]

    @staticmethod
    def _lines(tps: dict[int, int], memory: dict[int, float] | None = None) -> str:
        """Every step line of ``FIXTURE_RUN`` at 1000 tokens/s and 3 GiB, except the steps that ``tps`` and ``memory`` name."""
        memory = memory or {}
        return "".join(
            titan_step_line(step, tps=tps.get(step, 1000), memory=memory.get(step, 3.0))
            for step in range(1, FIXTURE_RUN.data.steps + 1)
        )

    def test_the_rate_mean_is_the_total_tokens_over_the_total_time(self) -> None:
        # Seven sampled steps at 1000 tokens/s and one at 250: 8 steps of
        # one token count take 7/1000 + 1/250 = 0.011 s per token.
        summary = self._evaluate({0: self._lines({6: 250})})
        self.assertEqual(summary.tokens_per_second.median, 1000)
        self.assertAlmostEqual(summary.tokens_per_second.mean, 8 / 0.011)
        # pp 1: one step holds 4 x 1024 tokens.
        tokens = 4 * 1024
        self.assertAlmostEqual(
            summary.step_ms.mean, (7 * tokens + tokens * 4) / 8
        )
        self.assertAlmostEqual(
            summary.step_ms.mean, 1000.0 * tokens / summary.tokens_per_second.mean
        )
        self.assertAlmostEqual(summary.step_ms.median, 1000.0 * tokens / 1000)
        self.assertAlmostEqual(summary.step_ms.p95, 1000.0 * tokens / 250)

    def test_the_rate_mean_is_the_harmonic_mean(self) -> None:
        self.assertAlmostEqual(rate_mean([1000, 1000, 1000, 250]), 4 / 0.007)

    def test_an_extra_takes_the_rate_mean(self) -> None:
        samples = [
            StepSample(
                rank=0,
                step=step,
                tokens_per_second=1000,
                peak_memory_gib=3.0,
                loss=1.0,
                grad_norm=2.0,
                extras={"tflops": tflops},
            )
            for step, tflops in ((3, 100.0), (4, 100.0), (5, 25.0))
        ]
        (tflops,) = extras_statistics(samples).values()
        self.assertEqual(tflops.median, 100.0)
        self.assertAlmostEqual(tflops.mean, 3 / (2 / 100 + 1 / 25))

    def test_the_median_and_the_mean_can_name_different_ranks(self) -> None:
        # Rank 0 runs every step at 900. Rank 1 runs at 1000 with one stall
        # at 200, so its median is higher and its mean is lower than rank 0's.
        summary = self._evaluate(
            {
                0: self._lines({step: 900 for step in range(1, 11)}),
                1: self._lines({5: 200}),
            }
        )
        tps = summary.tokens_per_second
        self.assertEqual((tps.median, tps.median_rank), (900, 0))
        self.assertAlmostEqual(tps.mean, 8 / (7 / 1000 + 1 / 200))
        self.assertEqual(tps.mean_rank, 1)
        cost = summary.step_ms
        self.assertEqual((cost.median_rank, cost.mean_rank, cost.p95_rank), (0, 1, 1))
        self.assertEqual(
            [row.tokens_per_second.median for row in summary.per_rank], [900, 1000]
        )

    def test_a_tie_goes_to_the_lower_rank(self) -> None:
        summary = self._evaluate({0: self._lines({}), 1: self._lines({})})
        self.assertEqual(summary.tokens_per_second.median_rank, 0)
        self.assertEqual(summary.step_ms.mean_rank, 0)
        self.assertEqual(summary.peak_memory_gib.max_rank, 0)

    def test_the_memory_maximum_reads_every_step_and_names_its_rank(self) -> None:
        # Step 1 is outside the sample rule, and it still holds the peak.
        summary = self._evaluate(
            {
                0: self._lines({}, {step: 5.0 for step in range(3, 11)}),
                1: self._lines({}, {1: 50.0, 7: 9.0}),
            }
        )
        memory = summary.peak_memory_gib
        self.assertEqual((memory.max, memory.max_rank), (50.0, 1))
        self.assertEqual((memory.median, memory.median_rank), (5.0, 0))
        self.assertEqual(memory.mean_rank, 0)
        self.assertAlmostEqual(summary.per_rank[1].peak_memory_gib.mean, (7 * 3.0 + 9.0) / 8)

    def test_the_spread_warning_reads_the_medians(self) -> None:
        # Rank 1's mean falls below 1/1.15 of rank 0's, and its median does not.
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    "baseline": _prefixed(self._lines({}), 0)
                    + _prefixed(self._lines({5: 100}), 1)
                },
                {"world_size": 2, "pp": 2},
            )
            warnings = evaluate_run(out_dir).warnings
        self.assertEqual(
            [warning for warning in warnings if "across ranks" in warning], []
        )

    def test_a_sampled_step_without_a_rate_is_refused(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"baseline: rank 0 logs tokens/s 0 at sampled step 4; "
        ):
            self._evaluate({0: self._lines({4: 0})})

    def _rank_result(self, tokens_per_second: float, extras: dict[str, float]):
        sample = StepSample(
            rank=0,
            step=4,
            tokens_per_second=tokens_per_second,
            peak_memory_gib=3.0,
            loss=1.0,
            grad_norm=2.0,
            extras=extras,
        )
        return rank_result(
            "baseline", 0, [sample], [sample], tokens_per_step=4096, pp=1
        )

    def test_a_non_finite_tokens_per_second_is_refused(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                rf"^baseline: rank 0 logs tokens/s {value} at sampled step 4; ",
            ):
                self._rank_result(value, {"tflops": 12.5})

    def test_an_extra_that_is_not_a_positive_finite_rate_is_refused(self) -> None:
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                rf"^baseline: rank 0 logs mfu {value} at sampled step 4; ",
            ):
                self._rank_result(1000, {"tflops": 12.5, "mfu": value})

    def test_a_zero_extra_on_an_unsampled_step_refuses_nothing(self) -> None:
        # Step 1 is outside the sample rule.
        first = titan_step_line(1).replace("mfu: 1.26%", "mfu: 0.00%")
        self.assertNotEqual(first, titan_step_line(1))
        lines = self._lines({}).replace(titan_step_line(1), first)
        summary = self._evaluate({0: lines})
        self.assertEqual(summary.extras["torchtitan"]["mfu"].median, 1.26)


class StepMsArithmeticTests(unittest.TestCase):
    """``step_ms`` is the tokens of one step divided by the measured rate."""

    def test_three_samples_give_three_step_costs(self) -> None:
        summary = step_ms(
            [1000, 2000, 4000], tokens_per_step=4 * 1024, pp=1
        )
        self.assertAlmostEqual(summary.mean, (4096.0 + 2048.0 + 1024.0) / 3)
        self.assertAlmostEqual(summary.median, 2048.0)
        self.assertAlmostEqual(summary.p95, 4096.0)

    def test_the_pipeline_degree_divides_the_cost(self) -> None:
        summary = step_ms([1000], tokens_per_step=4 * 1024, pp=4)
        self.assertAlmostEqual(summary.median, 1024.0)

    def test_p95_takes_the_nearest_rank_and_never_interpolates(self) -> None:
        # Twenty samples: ceil(0.95 * 20) = 19, so the 19th smallest.
        summary = step_ms(
            [1000] * 19 + [500], tokens_per_step=1000, pp=1
        )
        self.assertAlmostEqual(summary.p95, 1000.0)

    def test_an_empty_series_raises(self) -> None:
        with self.assertRaises(ValueError):
            step_ms([], tokens_per_step=1000, pp=1)


if __name__ == "__main__":
    unittest.main()
