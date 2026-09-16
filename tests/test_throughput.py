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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.artifacts.manifests import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    THROUGHPUT_DEFINITION,
    manifest_data,
)
from benchmarks.e2e.megatron.train import tokens_per_second  # noqa: E402
from benchmarks.e2e.parallelism import TRIVIAL_SPEC  # noqa: E402
from benchmarks.e2e.registry import SCENARIOS  # noqa: E402
from benchmarks.e2e.results import (  # noqa: E402
    evaluate_run,
    loss_visible_rank,
    per_rank_training_metrics,
    refuse_non_finite_trajectories,
    render_evaluation,
    training_metrics,
)
from tests.test_parallel_traces import step_event, write_trace  # noqa: E402


WORKLOAD = {"profile_freq": 20, "profiler_warmup": 5, "profiler_active": 5}


def _step_lines(*, tps: int, first_step: int = 2, count: int = 4) -> str:
    """Step lines a rank prints, inside the stable window rule."""
    return "".join(
        f"step: {step} loss: 1.0 grad_norm: 2.0 memory: 3.00GiB tps: {tps}\n"
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
        scenario = SCENARIOS["piper1b_megatron"]
        return manifest_data(
            scenario,
            [scenario.arms[0]],
            {scenario.arms[0].name: ["python", "-m", "x"]},
            "test-gpu",
            {"requested_gpu": "0"},
            (),
            "default",
            "none",
            "1b",
            parallelism=TRIVIAL_SPEC,
            megatron_p2p_sync="on",
            megatron_nan_guard="on",
            megatron_precision="stock",
        )

    def test_the_manifest_names_what_a_tokens_per_second_figure_counts(self) -> None:
        self.assertEqual(
            self._manifest()["throughput_definition"], THROUGHPUT_DEFINITION
        )
        self.assertEqual(THROUGHPUT_DEFINITION, "tokens_per_second_per_device")

    def test_the_schema_moved_with_the_new_key(self) -> None:
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 16)
        self.assertEqual(self._manifest()["schema_version"], 16)


class PerRankLogParsingTests(unittest.TestCase):
    """One file, two ranks: the rows belong to whoever printed them."""

    def test_an_unprefixed_log_reads_as_one_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text(_step_lines(tps=1000))
            by_rank = per_rank_training_metrics(log)
        self.assertEqual(list(by_rank), [0])
        self.assertEqual([row[2] for row in by_rank[0]], [1000] * 4)

    def test_two_ranks_do_not_pool_into_one_series(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text(
                _prefixed(_step_lines(tps=1000), 0)
                + _prefixed(_step_lines(tps=800), 1)
            )
            by_rank = per_rank_training_metrics(log)
            pooled = training_metrics(log)
        self.assertEqual(sorted(by_rank), [0, 1])
        self.assertEqual([row[2] for row in by_rank[0]], [1000] * 4)
        self.assertEqual([row[2] for row in by_rank[1]], [800] * 4)
        # The pooled reading is still available for peak memory, and it holds
        # both ranks' rows rather than one rank's.
        self.assertEqual(len(pooled), 8)

    def test_a_missing_log_reads_as_no_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(
                per_rank_training_metrics(Path(temporary) / "absent.log"), {}
            )


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
    """A minimal output directory: a manifest, a log and traces per arm.

    The traces exist because evaluation refuses an arm that has none. They
    declare no region, so every assertion here reads the throughput half.
    """

    @staticmethod
    def build(
        out_dir: Path,
        logs: dict[str, str],
        parallelism: dict | None,
        ranks: tuple[int, ...] = (0,),
    ) -> None:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "scenario": "synthetic",
            "hardware": "test-gpu",
            "workload": WORKLOAD,
            "regions": [],
            "selected_arms": sorted(logs),
        }
        if parallelism is not None:
            manifest["parallelism"] = parallelism
        (out_dir / "manifest.json").write_text(json.dumps(manifest))
        for arm, text in logs.items():
            (out_dir / f"{arm}.log").write_text(text)
            for window in (20, 40):
                for rank in ranks:
                    write_trace(
                        out_dir
                        / arm
                        / f"profiling/traces/iteration_{window}"
                        / f"rank{rank}_trace.json.gz",
                        {},
                        [step_event(1000.0)],
                    )


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

    def test_a_schema_nine_directory_carries_no_parallelism_and_still_reads(
        self,
    ) -> None:
        result, _ = self._evaluate(None)
        training = result.training["baseline"]
        self.assertEqual(training.stable_tokens_per_second, 1000)
        self.assertEqual(training.tokens_per_second_global, 1000)
        self.assertEqual(training.published_rank, 0)
        self.assertEqual(training.ranks, (0,))

    def test_the_ratio_is_still_taken_against_the_baseline(self) -> None:
        result, _ = self._evaluate({"world_size": 1, "pp": 1})
        self.assertEqual(result.training["optimized"].baseline_ratio, 1.2)

    def test_one_rank_prints_no_per_rank_table_and_raises_no_warning(self) -> None:
        result, report = self._evaluate({"world_size": 1, "pp": 1})
        self.assertNotIn("per-rank tokens/s", report)
        self.assertEqual(result.warnings, ())


class DenseShardingWarningsReachTheArtifactTests(unittest.TestCase):
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
                "dense_sharding": "zero1",
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
                    "dense_sharding": "replicate",
                }
            ),
            [],
        )

    def test_a_directory_written_before_the_rename_still_evaluates(self) -> None:
        """Three published cells record ``shard``, which this axis no longer
        declares. They predate the question these warnings ask, so they earn
        none -- and evaluating them must not fail.
        """
        self.assertEqual(
            self._warnings(
                {
                    "world_size": 1,
                    "dp": 1,
                    "pp": 1,
                    "ep": 1,
                    "dense_sharding": "shard",
                }
            ),
            [],
        )


class BaselineFreeSingletonTests(unittest.TestCase):
    """An eager TorchTitan-only run publishes absolutes, not fake ratios."""

    def test_singleton_evaluation_omits_every_baseline_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {"titan_stock": _step_lines(tps=1100)},
                {"world_size": 1, "pp": 1},
            )
            result = evaluate_run(out_dir)
            report = render_evaluation(result)

        self.assertEqual(result.arms, ("titan_stock",))
        self.assertEqual(
            result.training["titan_stock"].stable_tokens_per_second, 1100
        )
        self.assertIsNone(result.training["titan_stock"].baseline_ratio)
        self.assertIsNone(result.gpu_time["titan_stock"].baseline_kernel_ratio)
        self.assertEqual(result.comparisons, {})
        self.assertNotIn("compiled-region distributions", report)
        self.assertNotIn("Significance limitation", report)
        machine = result.to_dict()
        self.assertIsNone(machine["training"]["titan_stock"]["baseline_ratio"])
        self.assertIsNone(
            machine["gpu_time"]["titan_stock"]["baseline_kernel_ratio"]
        )

    def test_several_baseline_free_arms_still_need_a_comparison_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    "titan_stock": _step_lines(tps=1100),
                    "titan_other": _step_lines(tps=1200),
                },
                {"world_size": 1, "pp": 1},
            )
            with self.assertRaisesRegex(ValueError, "only a one-arm run"):
                evaluate_run(out_dir)


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
                ranks=(0, 1),
            )
            result = evaluate_run(out_dir)
            return result, render_evaluation(result)

    def test_the_published_figure_is_the_minimum_and_never_the_mean(self) -> None:
        result, _ = self._evaluate(1000, 900)
        training = result.training["baseline"]
        self.assertEqual(training.stable_tokens_per_second, 900)
        self.assertEqual(training.published_rank, 1)
        self.assertEqual(training.rank_reduction, "min_over_ranks")
        # The mean would be 950, which no device achieved.
        self.assertNotEqual(training.stable_tokens_per_second, 950)

    def test_every_rank_reaches_the_file_beside_the_published_one(self) -> None:
        result, _ = self._evaluate(1000, 900)
        training = result.training["baseline"]
        self.assertEqual(training.ranks, (0, 1))
        self.assertEqual(
            [(row.rank, row.stable_tokens_per_second) for row in training.per_rank],
            [(0, 1000), (1, 900)],
        )
        self.assertEqual([row.stable_sample_count for row in training.per_rank], [4, 4])

    def test_the_global_figure_is_the_published_one_times_the_world_size(
        self,
    ) -> None:
        result, _ = self._evaluate(1000, 900)
        self.assertEqual(result.training["baseline"].tokens_per_second_global, 1800)

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

    def test_the_report_names_the_reduction_and_shows_each_rank(self) -> None:
        _, report = self._evaluate(1000, 900)
        self.assertIn("MIN over ranks, never the mean", report)
        self.assertIn("per-rank tokens/s", report)
        self.assertIn("throughput_definition", report)

    def test_the_written_file_carries_the_per_rank_rows(self) -> None:
        result, _ = self._evaluate(1000, 900)
        machine = result.to_dict()["training"]["baseline"]
        self.assertEqual(machine["rank_reduction"], "min_over_ranks")
        self.assertEqual(machine["ranks"], [0, 1])
        self.assertEqual([row["rank"] for row in machine["per_rank"]], [0, 1])


class TheThroughputRatioSaysWhichRanksItDividedTests(unittest.TestCase):
    """The twin of the caption on ``baseline_kernel_ratio``.

    Each side of the ratio names its own slowest rank. Under a pipeline
    split those two rank indices hold different partitions of the model, so
    the file says so rather than leaving a reader to derive it.
    """

    def _evaluate(self, baseline: tuple[int, int], optimized: tuple[int, int]):
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    arm: _prefixed(_step_lines(tps=rows[0]), 0)
                    + _prefixed(_step_lines(tps=rows[1]), 1)
                    for arm, rows in (
                        ("baseline", baseline),
                        ("optimized", optimized),
                    )
                },
                {"world_size": 2, "pp": 2},
                ranks=(0, 1),
            )
            return evaluate_run(out_dir)

    def _ratio_warnings(self, result) -> list[str]:
        return [
            warning for warning in result.warnings if "tokens/s 'ratio'" in warning
        ]

    def test_one_rank_each_raises_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    "baseline": _step_lines(tps=1000),
                    "optimized": _step_lines(tps=1200),
                },
                {"world_size": 1, "pp": 1},
            )
            result = evaluate_run(out_dir)
        self.assertEqual(self._ratio_warnings(result), [])

    def test_two_arms_that_agree_on_the_slowest_rank_raise_nothing(self) -> None:
        result = self._evaluate((1000, 900), (1200, 1100))
        self.assertEqual(result.training["baseline"].published_rank, 1)
        self.assertEqual(result.training["optimized"].published_rank, 1)
        self.assertEqual(self._ratio_warnings(result), [])

    def test_two_arms_that_disagree_name_both_ranks(self) -> None:
        result = self._evaluate((1000, 900), (1100, 1200))
        self.assertEqual(result.training["baseline"].published_rank, 1)
        self.assertEqual(result.training["optimized"].published_rank, 0)
        warning = self._ratio_warnings(result)[0]
        self.assertIn("rank 0", warning)
        self.assertIn("baseline rank 1", warning)

    def test_the_ratio_is_still_published(self) -> None:
        """The caption qualifies the number. It does not withhold it."""
        result = self._evaluate((1000, 900), (1100, 1200))
        self.assertAlmostEqual(
            result.training["optimized"].baseline_ratio, 1100 / 900
        )


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
                ranks=(0, 1),
            )
            training = evaluate_run(out_dir).training["baseline"]
        self.assertEqual(training.published_rank, 0)
        self.assertEqual(training.stable_tokens_per_second, 1000)
        self.assertEqual(training.ranks, (0, 1))
        self.assertIsNone(training.per_rank[1].stable_tokens_per_second)


class NonFiniteTrajectoryTests(unittest.TestCase):
    """A ``nan`` or an ``inf`` on a step line fails the arm before publication.

    The step-line patterns read both words on purpose. This is the guard
    the 2026-08-30 report asked for before anyone turns stock Megatron's
    own NaN check off: it runs on every arm, whatever the engine's guard
    did, and it names the rank and the step.
    """

    def _lines(self, *, loss: str = "1.0", grad_norm: str = "2.0", at: int = 3):
        lines = []
        for step in range(2, 6):
            value_loss = loss if step == at else "1.0"
            value_norm = grad_norm if step == at else "2.0"
            lines.append(
                f"step: {step} loss: {value_loss} grad_norm: {value_norm} "
                "memory: 3.00GiB tps: 1000\n"
            )
        return "".join(lines)

    def _evaluate(self, logs: dict[str, str], parallelism=None, ranks=(0,)):
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(out_dir, logs, parallelism, ranks=ranks)
            evaluate_run(out_dir)
            self.assertFalse((out_dir / "results.json").exists())

    def test_a_nan_loss_fails_the_arm_and_names_the_step(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"baseline: rank 0 logged a non-finite loss at step 3"
        ):
            self._evaluate({"baseline": self._lines(loss="nan")})

    def test_an_inf_grad_norm_fails_the_arm_and_names_the_step(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            r"optimized: rank 0 logged a non-finite grad_norm at step 4",
        ):
            self._evaluate(
                {
                    "baseline": self._lines(),
                    "optimized": self._lines(grad_norm="inf", at=4),
                }
            )

    def test_a_negative_inf_is_not_finite_either(self) -> None:
        with self.assertRaisesRegex(ValueError, r"step 5 \(-inf\)"):
            self._evaluate({"baseline": self._lines(loss="-inf", at=5)})

    def test_every_rank_is_read_and_the_failure_names_the_rank(self) -> None:
        """The published trajectory is one rank's; the guard reads all of
        them, because no rank prints a non-finite value on purpose."""
        with self.assertRaisesRegex(
            ValueError, r"baseline: rank 1 logged a non-finite loss at step 3"
        ):
            self._evaluate(
                {
                    "baseline": _prefixed(self._lines(), 0)
                    + _prefixed(self._lines(loss="nan"), 1)
                },
                {"world_size": 2, "pp": 2},
                ranks=(0, 1),
            )

    def test_the_titan_sentinel_is_finite_and_passes(self) -> None:
        """A rank without the loss prints ``-1.0``, which is a number."""
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            _RunFixture.build(
                out_dir,
                {
                    "baseline": _prefixed(self._lines(loss="-1.00000"), 0)
                    + _prefixed(self._lines(), 1)
                },
                {"world_size": 2, "pp": 2},
                ranks=(0, 1),
            )
            result = evaluate_run(out_dir)
        self.assertEqual(result.losses["baseline"][1], (3, 1.0))

    def test_the_guard_reads_a_bare_log_directly(self) -> None:
        """Callable on its own, so a reader of an old directory can ask."""
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text(self._lines())
            refuse_non_finite_trajectories("baseline", log)
            log.write_text(self._lines(grad_norm="nan", at=2))
            with self.assertRaisesRegex(ValueError, r"grad_norm at step 2"):
                refuse_non_finite_trajectories("baseline", log)
            log.unlink()
            # A missing log is a validation failure, not this guard's.
            refuse_non_finite_trajectories("baseline", log)


if __name__ == "__main__":
    unittest.main()
