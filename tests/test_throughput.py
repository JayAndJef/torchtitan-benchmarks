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
from benchmarks.e2e.parallelism import TRIVIAL_SPEC, ParallelismSpec  # noqa: E402
from benchmarks.e2e.engines.api import (  # noqa: E402
    Arm,
    CompileMode,
    DroppedLine,
    RunSpec,
    StepRead,
)
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig  # noqa: E402
from benchmarks.e2e.registry import SCENARIOS, SEED  # noqa: E402
from benchmarks.models.piper_qwen3.shape import PIPER_1B  # noqa: E402
from benchmarks.e2e.results import (  # noqa: E402
    arm_steps,
    dropped_line_warnings,
    evaluate_run,
    pinning_warnings,
    loss_visible_rank,
    refuse_non_finite_trajectories,
    render_evaluation,
    step_ms,
    trajectory,
    write_results,
)


from tests.engine_helpers import (  # noqa: E402
    TEST_METADATA,
    run_spec,
    titan_step_line,
)


FIXTURE_RUN = run_spec(seq_len=1024, local_batch_size=4)
"""A profiled run at sequence length 1024 and local batch 4."""


TITAN_ARM = Arm(
    name="baseline",
    description="baseline",
    config=TorchTitanConfig(compile=CompileMode.NONE),
)
"""A TorchTitan arm, whose engine reads the fork's step line."""


def _samples(arm: Arm, log: Path) -> dict:
    """The step samples of each rank in the log."""
    return {rank: read.samples for rank, read in arm_steps(arm, log).items()}


def _step_lines(*, tps: int, first_step: int = 2, count: int = 4) -> str:
    """Step lines a rank prints, inside the stable window rule."""
    return "".join(
        titan_step_line(step, tps=tps)
        for step in range(first_step, first_step + count)
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
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 19)
        self.assertEqual(self._manifest()["schema_version"], 19)


class PerRankLogParsingTests(unittest.TestCase):
    """One file, two ranks: the samples belong to whoever printed them."""

    def test_an_unprefixed_log_reads_as_one_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text(_step_lines(tps=1000))
            by_rank = arm_steps(TITAN_ARM, log)
        self.assertEqual(list(by_rank), [0])
        self.assertEqual(
            [sample.tokens_per_second for sample in by_rank[0].samples], [1000] * 4
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
            [sample.tokens_per_second for sample in by_rank[0].samples], [1000] * 4
        )
        self.assertEqual(
            [sample.tokens_per_second for sample in by_rank[1].samples], [800] * 4
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
        self.assertEqual(summary.stable_tokens_per_second, 1000)
        self.assertEqual(summary.published_rank, 0)
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
            result.results["titan_compiled"].stable_tokens_per_second, 1100
        )
        self.assertEqual(
            result.results["megatron_stock"].stable_tokens_per_second, 1200
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

    def test_the_published_figure_is_the_minimum_and_never_the_mean(self) -> None:
        result, _ = self._evaluate(1000, 900)
        summary = result.results["baseline"]
        self.assertEqual(summary.stable_tokens_per_second, 900)
        self.assertEqual(summary.published_rank, 1)
        self.assertEqual(summary.rank_reduction, "min_over_ranks")
        # The mean would be 950, which no device achieved.
        self.assertNotEqual(summary.stable_tokens_per_second, 950)

    def test_every_rank_reaches_the_file_beside_the_published_one(self) -> None:
        result, _ = self._evaluate(1000, 900)
        summary = result.results["baseline"]
        self.assertEqual(
            [(row.rank, row.stable_tokens_per_second) for row in summary.per_rank],
            [(0, 1000), (1, 900)],
        )
        self.assertEqual([row.stable_sample_count for row in summary.per_rank], [4, 4])

    def test_the_published_step_cost_is_the_published_rank_own(self) -> None:
        result, _ = self._evaluate(1000, 900)
        summary = result.results["baseline"]
        # pp 2 halves the per-rank step cost of 4 x 1024 tokens.
        self.assertAlmostEqual(
            summary.step_ms.median, 1000.0 * 4 * 1024 / (900 * 2)
        )
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
        self.assertEqual(machine["rank_reduction"], "min_over_ranks")
        self.assertEqual([row["rank"] for row in machine["per_rank"]], [0, 1])


class ARankWithNoSampleDoesNotWinTests(unittest.TestCase):
    """"No sample" is a measurement that did not happen, not a slow rank."""

    def test_a_silent_rank_does_not_become_the_published_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    "baseline": _prefixed(_step_lines(tps=1000), 0)
                    + "[rank1]:starting up\n"
                },
                {"world_size": 2, "pp": 2},
            )
            summary = evaluate_run(out_dir).results["baseline"]
        self.assertEqual(summary.published_rank, 0)
        self.assertEqual(summary.stable_tokens_per_second, 1000)
        self.assertEqual([row.rank for row in summary.per_rank], [0, 1])
        self.assertIsNone(summary.per_rank[1].stable_tokens_per_second)
        self.assertIsNone(summary.per_rank[1].step_ms.median)
        self.assertEqual(summary.per_rank[1].step_ms.series, ())


class TornStepLineTests(unittest.TestCase):
    """A step line that another rank's prefix cut is dropped, and results.json names it."""

    def test_the_warning_names_the_arm_the_rank_the_step_and_the_log_line(
        self,
    ) -> None:
        cut = titan_step_line(3)[:-60] + "[rank0]:USDT: profiler_stop\n"
        lines = [
            "[rank0]:" + titan_step_line(2),
            "[rank1]:" + titan_step_line(2),
            "[rank0]:" + titan_step_line(3),
            "[rank1]:" + cut,
            "[rank0]:" + titan_step_line(4),
            "[rank1]:" + titan_step_line(4),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir, {"baseline": "".join(lines)}, {"world_size": 2, "pp": 2}
            )
            result = evaluate_run(out_dir)
        self.assertIn(
            "baseline: rank 1 step 3: a rank prefix cut the step line "
            "at line 4 of baseline.log, so the evaluation drops that step",
            result.warnings,
        )
        self.assertEqual(result.results["baseline"].per_rank[1].stable_sample_count, 2)

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
        cut = titan_step_line(3)[:-60] + "[rank0]:USDT: profiler_stop\n"
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {"baseline": "header\n" + titan_step_line(2) + cut},
                {"world_size": 1, "pp": 1},
            )
            warnings = evaluate_run(out_dir).warnings
        self.assertTrue(
            any("rank 0 step 3" in w and "at line 3 of" in w for w in warnings),
            warnings,
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
            for step in range(2, 6)
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
        self.assertEqual(result.losses["baseline"][1], (3, 1.0))
        self.assertIsNone(steps[0].samples[1].loss)

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

        self.assertEqual(machine["schema_version"], 6)
        self.assertEqual(
            machine["results"]["megatron_stock"]["stable_tokens_per_second"],
            1200,
        )
        self.assertEqual(machine["losses"]["megatron_stock"][0]["value"], 1.0)
        self.assertIn("tokens/s", report)
        self.assertIn("step ms", report)
        self.assertIn("p95 ms", report)
        self.assertIn("peak GiB", report)
        self.assertNotIn("gpu kernel time", report)

    def test_the_payload_carries_the_schema_six_keys_and_no_others(
        self,
    ) -> None:
        """The exact key tree, top level and per arm."""
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {"titan_compiled": _step_lines(tps=1000)},
                {"world_size": 1, "pp": 1},
            )
            machine = evaluate_run(out_dir).to_dict()

        self.assertEqual(
            sorted(machine),
            sorted(
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
                ]
            ),
        )
        arm = machine["results"]["titan_compiled"]
        self.assertEqual(
            sorted(arm),
            sorted(
                [
                    "stable_tokens_per_second",
                    "stable_sample_count",
                    "peak_memory_gib",
                    "step_ms",
                    "rank_reduction",
                    "published_rank",
                    "per_rank",
                    "extras",
                ]
            ),
        )
        self.assertEqual(
            arm["extras"], {"torchtitan": {"mfu": 1.26, "tflops": 12.5}}
        )
        self.assertEqual(
            sorted(arm["step_ms"]), ["mean", "median", "p95", "series"]
        )
        self.assertEqual(
            sorted(arm["per_rank"][0]),
            sorted(
                [
                    "rank",
                    "stable_tokens_per_second",
                    "stable_sample_count",
                    "step_ms",
                ]
            ),
        )


class StepMsArithmeticTests(unittest.TestCase):
    """``step_ms`` is the tokens of one step divided by the measured rate."""

    def test_three_samples_give_three_step_costs(self) -> None:
        summary = step_ms(
            [1000, 2000, 4000], tokens_per_step=4 * 1024, pp=1
        )
        self.assertEqual(
            summary.series, (4096.0, 2048.0, 1024.0)
        )
        self.assertAlmostEqual(summary.mean, (4096.0 + 2048.0 + 1024.0) / 3)
        self.assertAlmostEqual(summary.median, 2048.0)

    def test_the_pipeline_degree_divides_the_cost(self) -> None:
        summary = step_ms([1000], tokens_per_step=4 * 1024, pp=4)
        self.assertAlmostEqual(summary.series[0], 1024.0)

    def test_the_series_keeps_step_order(self) -> None:
        summary = step_ms([4000, 1000, 2000], tokens_per_step=1000, pp=1)
        self.assertEqual(summary.series, (250.0, 1000.0, 500.0))

    def test_p95_takes_the_nearest_rank_and_never_interpolates(self) -> None:
        # Twenty samples: ceil(0.95 * 20) = 19, so the 19th smallest.
        summary = step_ms(
            [1000] * 19 + [500], tokens_per_step=1000, pp=1
        )
        self.assertAlmostEqual(summary.p95, 1000.0)
        self.assertIn(summary.p95, summary.series)

    def test_an_empty_series_publishes_no_statistic(self) -> None:
        summary = step_ms([], tokens_per_step=1000, pp=1)
        self.assertEqual(summary.series, ())
        self.assertIsNone(summary.mean)
        self.assertIsNone(summary.median)
        self.assertIsNone(summary.p95)


if __name__ == "__main__":
    unittest.main()
