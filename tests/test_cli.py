"""CPU-only tests for the Click benchmark interface."""

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import click
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cli.e2e import run_all_command, run_command
from benchmarks.cli.main import cli
from benchmarks.e2e.parallelism import MEGATRON_LAUNCHERS, NAN_GUARD_LAUNCHERS
from benchmarks.e2e.registry import PIPER_1B_ROPE, SCENARIOS
from benchmarks.e2e.runner import execute_run
from benchmarks.execution.affinity import CpuPinning
from benchmarks.models.piper_qwen3.shape import HUGE


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_importing_main_alone_registers_every_command(self) -> None:
        """The group is fully populated by importing ``cli.main`` and nothing else.

        ``run``, ``run-all`` and ``evaluate`` are defined in
        ``benchmarks/cli/e2e.py`` and ``kernel-bench`` in
        ``benchmarks/cli/kernel.py``, with plain ``@click.command``;
        ``main.py`` attaches them with ``cli.add_command``. Binding them with
        ``@cli.command`` in their own modules instead would invert that edge,
        and ``from benchmarks.cli.main import cli`` -- what ``__main__.py``
        and this file do -- would then yield a group holding only whichever
        commands some earlier import had loaded. Nothing else would notice: a
        CLI missing ``run-all`` starts fine and prints a usage message.

        A subprocess, because the rest of the suite imports the command
        modules for its patch targets; in this process the group would be
        populated either way.
        """
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from benchmarks.cli.main import cli\n"
                "print(' '.join(sorted(cli.commands)))",
            ],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.split(),
            ["evaluate", "kernel-bench", "run", "run-all", "scenarios"],
        )

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
            "COMPILE_MODE",
        ):
            self.assertIn(envvar, result.output)

    def test_compile_mode_reaches_the_request(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(PIPER_1B_ROPE.arm("baseline"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(
                cli, ["run", "2", "--compile-mode", "cuda-graph", "--ac", "none"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].compile_mode, "cuda-graph")
        self.assertEqual(execute.call_args.args[0].ac_mode, "none")

    def test_compile_mode_defaults_to_unrequested(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(PIPER_1B_ROPE.arm("baseline"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(cli, ["run", "2"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(execute.call_args.args[0].compile_mode)
        self.assertIsNone(execute.call_args.args[0].ac_mode)

    def test_compile_mode_applies_to_every_swept_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli,
                    [
                        "run-all",
                        "0",
                        "--all-scenarios",
                        "--compile-mode",
                        "cuda-graph",
                        "--ac",
                        "none",
                        "--model-size",
                        "huge",
                    ],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        modes = {call.args[0].compile_mode for call in execute.call_args_list}
        self.assertEqual(modes, {"cuda-graph"})
        ac_modes = {call.args[0].ac_mode for call in execute.call_args_list}
        self.assertEqual(ac_modes, {"none"})
        # The third global axis must reach every swept scenario too.
        sizes = {call.args[0].model_size for call in execute.call_args_list}
        self.assertEqual(sizes, {"huge"})

    def test_model_size_defaults_to_unrequested_and_rejects_unknowns(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(PIPER_1B_ROPE.arm("baseline"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(cli, ["run", "2"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(execute.call_args.args[0].model_size)

        rejected = self.runner.invoke(cli, ["run", "2", "--model-size", "enormous"])
        self.assertNotEqual(rejected.exit_code, 0)

    def test_model_size_reaches_the_recorded_training_command(self) -> None:
        """--model-size arrives as --config-arg, not as a mangled config name.

        Runs the real runner behind a fake training process, so the assertion
        is on the command the manifest actually records.
        """

        def fake_process(command, **kwargs):
            kwargs["stdout"].write(
                "[titan] - root - INFO - Compiling each TransformerBlock with "
                "torch.compile (mode=default)\n"
                "[titan] - root - INFO - Model qwen3 piper_1B "
                f"size: {HUGE.param_count:,} total parameters\n"
                "Training completed\n"
            )
            arm_dir = Path(command[command.index("--dump-folder") + 1])
            for iteration in (20, 40):
                trace = (
                    arm_dir
                    / f"profiling/traces/iteration_{iteration}/rank0_trace.json.gz"
                )
                trace.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(trace, "wt") as trace_file:
                    json.dump({"traceEvents": []}, trace_file)
            return SimpleNamespace(returncode=0)

        def run_with_fake_process(request, **kwargs):
            return execute_run(request, process_runner=fake_process, **kwargs)

        metadata = {
            "requested_gpu": "0",
            "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
            "torch_version": "test",
            "torchtitan_git_rev": "titan-rev",
            "benchmarks_git_rev": "bench-rev",
        }
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.cli.e2e.execute_run", side_effect=run_with_fake_process
        ), mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            out_dir = Path(temporary) / "run"
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    "0",
                    "--scenario",
                    "piper1b_attention",
                    "--arm",
                    "baseline",
                    "--ac",
                    "none",
                    "--model-size",
                    "huge",
                    "--out",
                    str(out_dir),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            manifest = json.loads((out_dir / "manifest.json").read_text())

        self.assertEqual(manifest["model_size"], "huge")
        command = manifest["commands"]["baseline"]
        self.assertEqual(command[command.index("--config") + 1], "qwen3_piper_1b")
        self.assertEqual(command[command.index("--config-arg") + 1], "size=huge")
        self.assertFalse([token for token in command if token.endswith("_huge")])

    def test_unknown_compile_mode_is_rejected(self) -> None:
        result = self.runner.invoke(cli, ["run", "2", "--compile-mode", "turbo"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value", result.output)

    def test_run_preserves_torchtitan_passthrough_arguments(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(PIPER_1B_ROPE.arm("baseline"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
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
        self.assertEqual(request.arm_names, ("baseline",))
        self.assertEqual(
            request.extra_args,
            ("--debug.seed", "42", "--debug.deterministic"),
        )
        self.assertIn("Evaluate with:", result.output)

    def test_run_collects_repeated_arms_in_command_line_order(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(
                PIPER_1B_ROPE.arm("helion"),
                PIPER_1B_ROPE.arm("baseline"),
            ),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    "2",
                    "--scenario",
                    "piper1b_rope",
                    "--arm",
                    "helion",
                    "--arm",
                    "baseline",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            execute.call_args.args[0].arm_names, ("helion", "baseline")
        )

    def test_run_all_executes_then_evaluates_same_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            completed = SimpleNamespace(out_dir=out_dir)
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate") as evaluate:
                result = self.runner.invoke(
                    cli,
                    ["run-all", "6", "--scenario", "piper1b_rope"],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        request = execute.call_args.args[0]
        self.assertEqual(request.arm_names, ())
        evaluate.assert_called_once_with(out_dir, (), None)

    def test_all_scenarios_runs_each_scenario_under_one_timestamp(self) -> None:
        # At the default ac mode (sac), scenarios that only support ac=none
        # (piper1b_megatron) are skipped with a message rather than run.
        supported_at_sac = [
            name
            for name, scenario in SCENARIOS.items()
            if "sac" in scenario.supported_ac_modes
        ]
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(cli, ["run-all", "0", "--all-scenarios"])
        self.assertEqual(result.exit_code, 0, result.output)
        requests = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(
            [request.scenario_name for request in requests], supported_at_sac
        )
        self.assertNotIn("piper1b_megatron", supported_at_sac)
        self.assertIn("skipped: does not support ac mode 'sac'", result.output)
        self.assertEqual(len({request.timestamp for request in requests}), 1)
        self.assertIsNotNone(requests[0].timestamp)

    def test_all_scenarios_at_compile_none_skips_the_megatron_scenario(self) -> None:
        # The uncompiled mode names a titan treatment Megatron never has, so
        # the sweep skips that scenario with a message instead of recording a
        # mode one of its arms could not receive.
        supported = [
            name
            for name, scenario in SCENARIOS.items()
            if "none" in scenario.supported_compile_modes
        ]
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli,
                    ["run-all", "0", "--all-scenarios", "--compile-mode", "none"],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        names = [call.args[0].scenario_name for call in execute.call_args_list]
        self.assertEqual(names, supported)
        self.assertNotIn("piper1b_megatron", supported)
        self.assertIn("skipped: does not support compile mode 'none'", result.output)

    def test_all_scenarios_at_ac_none_includes_the_megatron_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli, ["run-all", "0", "--all-scenarios", "--ac", "none"]
                )
        self.assertEqual(result.exit_code, 0, result.output)
        names = [call.args[0].scenario_name for call in execute.call_args_list]
        self.assertEqual(names, list(SCENARIOS))

    def test_all_scenarios_stops_at_the_first_failing_scenario(self) -> None:
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", side_effect=RuntimeError("arm failed")
        ) as execute, mock.patch("benchmarks.cli.e2e._evaluate") as evaluate:
            result = self.runner.invoke(cli, ["run-all", "0", "--all-scenarios"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(execute.call_count, 1)
        evaluate.assert_not_called()

    def test_an_omitted_scenario_fails_on_both_execution_commands(self) -> None:
        """``--scenario`` is required, and an omitted one runs nothing.

        ``piper1b_rope`` was the default until this change. A default could
        only be reached by an omission, and would then measure that scenario
        under whatever label the operator assumed. ``benchmarks/e2e/runner.py``
        holds the rule in one place, so both commands that take the flag
        inherit it and the CLI adds no default of its own.

        Click cannot state the rule as ``required=True``: the option block is
        shared with ``run-all``, whose ``--all-scenarios`` and ``--resume``
        each supply the scenario themselves. Two tests above cover that half.

        The patch on ``hardware_metadata`` proves the refusal lands before the
        runner probes the host, so no scenario runs.
        """
        for command in (["run", "2"], ["run-all", "0"]):
            with self.subTest(command=command[0]):
                with mock.patch(
                    "benchmarks.e2e.runner.hardware_metadata",
                    side_effect=AssertionError("the host was probed"),
                ):
                    result = self.runner.invoke(cli, command)
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("no scenario requested", result.output)
                self.assertIn("--scenario", result.output)

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
            "benchmarks.cli.e2e.execute_run", side_effect=RuntimeError("arm failed")
        ), mock.patch("benchmarks.cli.e2e._evaluate") as evaluate:
            result = self.runner.invoke(cli, ["run-all", "0"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("arm failed", result.output)
        evaluate.assert_not_called()

    def test_megatron_p2p_sync_reaches_the_request(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(PIPER_1B_ROPE.arm("baseline"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(
                cli, ["run", "2", "--megatron-p2p-sync", "off"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].megatron_p2p_sync, "off")

    def test_megatron_p2p_sync_defaults_to_unrequested(self) -> None:
        """``None`` is what lets a resume inherit the recorded value."""
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(PIPER_1B_ROPE.arm("baseline"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(cli, ["run", "2"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(execute.call_args.args[0].megatron_p2p_sync)

    def test_an_unknown_megatron_p2p_sync_value_is_rejected(self) -> None:
        result = self.runner.invoke(
            cli, ["run", "2", "--megatron-p2p-sync", "false"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value", result.output)

    def test_megatron_p2p_sync_takes_no_environment_variable(self) -> None:
        """Its ``off`` value has to agree with ``<gpu>`` and with ``--arm``.

        An exported value would make a plain ``run 0 --scenario X`` fail a
        refusal naming a flag the operator never passed.
        """
        for command in (run_command, run_all_command):
            parameters = {
                option: parameter
                for parameter in command.params
                for option in parameter.opts
            }
            with self.subTest(command=command.name):
                self.assertIsNone(parameters["--megatron-p2p-sync"].envvar)

    def test_all_scenarios_at_p2p_sync_off_skips_titan_only_scenarios(
        self,
    ) -> None:
        """The value reaches megatron arms alone.

        ``_resolve_run`` refuses ``off`` for a run with no megatron arm, so
        a sweep that reached such a scenario would abort at its first
        titan-only entry. The sweep skips it with a message instead, as it
        skips a scenario that declines a mode.
        """
        holds_megatron = [
            name
            for name, scenario in SCENARIOS.items()
            if any(arm.launcher in MEGATRON_LAUNCHERS for arm in scenario.arms)
        ]
        self.assertTrue(holds_megatron)
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli,
                    [
                        "run-all",
                        "0,1",
                        "--all-scenarios",
                        "--ac",
                        "none",
                        "--pp",
                        "2",
                        "--pp-schedule",
                        "1F1B",
                        "--megatron-p2p-sync",
                        "off",
                    ],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        requests = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(
            [request.scenario_name for request in requests], holds_megatron
        )
        self.assertEqual(
            {request.megatron_p2p_sync for request in requests}, {"off"}
        )
        self.assertIn(
            "skipped: --megatron-p2p-sync 'off' reaches no arm", result.output
        )

    def test_megatron_nan_guard_reaches_the_request(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(PIPER_1B_ROPE.arm("baseline"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(
                cli, ["run", "2", "--megatron-nan-guard", "off"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].megatron_nan_guard, "off")

    def test_megatron_nan_guard_defaults_to_unrequested(self) -> None:
        """``None`` is what lets a resume inherit the recorded value."""
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(PIPER_1B_ROPE.arm("baseline"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(cli, ["run", "2"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(execute.call_args.args[0].megatron_nan_guard)

    def test_an_unknown_megatron_nan_guard_value_is_rejected(self) -> None:
        result = self.runner.invoke(
            cli, ["run", "2", "--megatron-nan-guard", "false"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value", result.output)

    def test_megatron_nan_guard_takes_no_environment_variable(self) -> None:
        """Its ``off`` value has to agree with ``--scenario`` and ``--arm``.

        An exported value would make a plain titan run fail a refusal
        naming a flag the operator never passed.
        """
        for command in (run_command, run_all_command):
            parameters = {
                option: parameter
                for parameter in command.params
                for option in parameter.opts
            }
            with self.subTest(command=command.name):
                self.assertIsNone(parameters["--megatron-nan-guard"].envvar)

    def test_all_scenarios_at_nan_guard_off_runs_the_stock_scenario_alone(
        self,
    ) -> None:
        """The value reaches the stock megatron arm alone.

        ``_resolve_run`` refuses ``off`` for a run with no such arm, and for
        a run holding the tuned arm, so the sweep would abort at its first
        titan-only entry. It skips both kinds with the refusal's own
        reason instead, as it skips a scenario that declines a mode.
        """
        holds_stock = [
            name
            for name, scenario in SCENARIOS.items()
            if any(arm.launcher in NAN_GUARD_LAUNCHERS for arm in scenario.arms)
        ]
        self.assertEqual(holds_stock, ["piper_megatron_stock"])
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli,
                    [
                        "run-all",
                        "0",
                        "--all-scenarios",
                        "--ac",
                        "none",
                        "--megatron-nan-guard",
                        "off",
                    ],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        requests = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(
            [request.scenario_name for request in requests], holds_stock
        )
        self.assertEqual(
            {request.megatron_nan_guard for request in requests}, {"off"}
        )
        # Both refusals appear, each on the scenario it names.
        self.assertIn(
            "skipped: --megatron-nan-guard 'off' reaches no arm", result.output
        )
        self.assertIn(
            "skipped: --megatron-nan-guard 'off' was requested with baseline, "
            "whose driver benchmarks/e2e/megatron/train.py has no NaN guard",
            result.output,
        )

    def test_run_all_records_evaluation_failure_for_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            (out_dir / "run_state.json").write_text(
                json.dumps({"schema_version": 1, "status": "arms_completed"})
            )
            completed = SimpleNamespace(out_dir=out_dir)
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ), mock.patch(
                "benchmarks.cli.e2e._evaluate",
                side_effect=click.ClickException("bad trace"),
            ):
                result = self.runner.invoke(cli, ["run-all", "0"])
            state = json.loads((out_dir / "run_state.json").read_text())

        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(state["status"], "evaluation_failed")
        self.assertIn("bad trace", state["evaluation"]["error"])


if __name__ == "__main__":
    unittest.main()
