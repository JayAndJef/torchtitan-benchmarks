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
from benchmarks.e2e.megatron_stock.step_log import (  # noqa: E402
    tokens_per_second,
)
from benchmarks.e2e.parallelism import TRIVIAL_SPEC  # noqa: E402
from benchmarks.e2e.axes import RunAxes  # noqa: E402
from benchmarks.e2e.registry import SCENARIOS  # noqa: E402
from benchmarks.e2e.results import (  # noqa: E402
    evaluate_run,
    loss_visible_rank,
    losses,
    per_rank_training_metrics,
    refuse_non_finite_trajectories,
    render_evaluation,
    step_ms,
    training_metrics,
    write_results,
)


WORKLOAD = {
    "profile_freq": 20,
    "profiler_warmup": 5,
    "profiler_active": 5,
    "local_batch_size": 4,
    "seq_len": 1024,
}


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
        scenario = SCENARIOS["engines"]
        return manifest_data(
            scenario,
            [scenario.arms[0]],
            {scenario.arms[0].name: ["python", "-m", "x"]},
            "test-gpu",
            {"requested_gpu": "0"},
            torchtitan_args=(),
            megatron_args=(),
            axes=RunAxes(
                ac_mode="none",
                model_size="1b",
                parallelism=TRIVIAL_SPEC,
                megatron_p2p_sync="on",
                megatron_nan_guard="on",
                megatron_precision="stock",
                profile=False,
                warmup_steps=10,
            ),
        )

    def test_the_manifest_names_what_a_tokens_per_second_figure_counts(self) -> None:
        self.assertEqual(
            self._manifest()["throughput_definition"], THROUGHPUT_DEFINITION
        )
        self.assertEqual(THROUGHPUT_DEFINITION, "tokens_per_second_per_device")

    def test_the_schema_moved_with_the_new_key(self) -> None:
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 18)
        self.assertEqual(self._manifest()["schema_version"], 18)


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
    """A minimal output directory: a manifest and one log per arm.

    Evaluation reads the logs alone, so a fixture needs no trace.
    """

    @staticmethod
    def build(
        out_dir: Path,
        logs: dict[str, str],
        parallelism: dict | None,
    ) -> None:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "profile": True,
            "scenario": "synthetic",
            "hardware": "test-gpu",
            "workload": WORKLOAD,
            "selected_arms": sorted(logs),
            "arms": [
                {"name": arm, "engine": "torchtitan"}
                for arm in sorted(logs)
            ],
        }
        manifest["parallelism"] = (
            {"world_size": 1, "dp": 1, "pp": 1, "ep": 1}
            if parallelism is None
            else parallelism
        )
        (out_dir / "manifest.json").write_text(json.dumps(manifest))
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


class WholeEvaluationTests(unittest.TestCase):
    """The published payload, read end to end on a synthetic run.

    The assertions pin the loss parser, the host-latency warning, and the
    machine-readable and human-readable halves of one evaluation.
    """

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
                ]
            ),
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
