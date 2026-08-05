"""CPU-only tests for the Click benchmark interface."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import click
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cli import cli
from benchmarks.scenarios import PIPER_1B_ROPE, SCENARIOS


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_root_help_and_scenario_listing(self) -> None:
        help_result = self.runner.invoke(cli, ["--help"])
        self.assertEqual(help_result.exit_code, 0)
        self.assertIn("run-all", help_result.output)
        result = self.runner.invoke(cli, ["scenarios"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("piper1b_lm_head", result.output)
        self.assertIn("piper_optimized_te_ce", result.output)

    def test_execution_help_shows_environment_variables(self) -> None:
        result = self.runner.invoke(cli, ["run-all", "--help"])
        self.assertEqual(result.exit_code, 0)
        for envvar in (
            "OUT",
            "SEQ",
            "STEPS",
            "BATCH",
            "BENCHMARK_CACHE_ROOT",
            "BENCH_COMPILER_ENV",
        ):
            self.assertIn(envvar, result.output)

    def test_run_preserves_torchtitan_passthrough_arguments(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(PIPER_1B_ROPE.arm("baseline"),),
        )
        with mock.patch(
            "benchmarks.cli.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    "2",
                    "--scenario",
                    "piper1b_rope",
                    "--arm",
                    "baseline",
                    "--",
                    "--debug.seed",
                    "42",
                    "--debug.deterministic",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        request = execute.call_args.args[0]
        self.assertEqual(request.gpu, "2")
        self.assertEqual(request.arm_name, "baseline")
        self.assertEqual(
            request.extra_args,
            ("--debug.seed", "42", "--debug.deterministic"),
        )

    def test_run_all_executes_then_evaluates_same_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            completed = SimpleNamespace(out_dir=out_dir)
            with mock.patch(
                "benchmarks.cli.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli._evaluate") as evaluate:
                result = self.runner.invoke(
                    cli,
                    ["run-all", "6", "--scenario", "piper1b_rope"],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        request = execute.call_args.args[0]
        self.assertIsNone(request.arm_name)
        evaluate.assert_called_once_with(out_dir, (), None)

    def test_all_scenarios_runs_each_scenario_under_one_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli._evaluate"):
                result = self.runner.invoke(cli, ["run-all", "0", "--all-scenarios"])
        self.assertEqual(result.exit_code, 0, result.output)
        requests = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(
            [request.scenario_name for request in requests], list(SCENARIOS)
        )
        self.assertEqual(len({request.timestamp for request in requests}), 1)
        self.assertIsNotNone(requests[0].timestamp)

    def test_all_scenarios_stops_at_the_first_failing_scenario(self) -> None:
        with mock.patch(
            "benchmarks.cli.execute_run", side_effect=RuntimeError("arm failed")
        ) as execute, mock.patch("benchmarks.cli._evaluate") as evaluate:
            result = self.runner.invoke(cli, ["run-all", "0", "--all-scenarios"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(execute.call_count, 1)
        evaluate.assert_not_called()

    def test_all_scenarios_rejects_conflicting_options(self) -> None:
        for conflicting in (
            ["--scenario", "piper1b_rope"],
            ["--out", "/tmp/output"],
            ["--results", "/tmp/results.json"],
        ):
            with self.subTest(option=conflicting[0]):
                result = self.runner.invoke(
                    cli, ["run-all", "0", "--all-scenarios", *conflicting]
                )
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("--all-scenarios cannot be combined", result.output)

    def test_run_all_does_not_evaluate_a_failed_execution(self) -> None:
        with mock.patch(
            "benchmarks.cli.execute_run", side_effect=RuntimeError("arm failed")
        ), mock.patch("benchmarks.cli._evaluate") as evaluate:
            result = self.runner.invoke(cli, ["run-all", "0"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("arm failed", result.output)
        evaluate.assert_not_called()

    def test_run_all_records_evaluation_failure_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            (out_dir / "run_state.json").write_text(
                json.dumps({"schema_version": 1, "status": "arms_completed"})
            )
            completed = SimpleNamespace(out_dir=out_dir)
            with mock.patch(
                "benchmarks.cli.execute_run", return_value=completed
            ), mock.patch(
                "benchmarks.cli._evaluate",
                side_effect=click.ClickException("bad trace"),
            ):
                result = self.runner.invoke(cli, ["run-all", "0"])
            state = json.loads((out_dir / "run_state.json").read_text())

        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(state["status"], "evaluation_failed")
        self.assertIn("bad trace", state["evaluation"]["error"])


if __name__ == "__main__":
    unittest.main()
