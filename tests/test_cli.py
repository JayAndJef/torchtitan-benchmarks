"""CPU-only tests for the Click benchmark interface."""

import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import click
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.layout import _default_output_dir
from benchmarks.cli.e2e import run_command
from benchmarks.cli.main import cli
from benchmarks.e2e.parallelism import (
    MEGATRON_ENGINES,
)
from benchmarks.e2e.registry import (
    DEFAULT_AC_MODE,
    DEFAULT_MEGATRON_NAN_GUARD,
    DEFAULT_MEGATRON_P2P_SYNC,
    ENGINES,
    SCENARIOS,
)
from benchmarks.e2e.runner import execute_run
from benchmarks.execution.affinity import CpuPinning
from benchmarks.models.piper_qwen3.shape import HUGE


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        # ``run`` always evaluates, and every test here patches the runner,
        # so no output directory holds a manifest. A test that asserts on
        # the evaluation patches the name again itself.
        evaluate = mock.patch("benchmarks.cli.e2e._evaluate")
        self.addCleanup(evaluate.stop)
        evaluate.start()

    def _two_scenarios(self) -> dict:
        """The registry with a second scenario, for the multi-scenario rules.

        One scenario is declared today, so the rules that need two are
        exercised against a copy of it under another name.
        """
        engines = SCENARIOS["engines"]
        return {
            "engines": engines,
            "engines_copy": replace(engines, name="engines_copy"),
        }

    def test_importing_main_alone_registers_every_command(self) -> None:
        """The group is fully populated by importing ``cli.main`` and nothing else.

        ``run`` and ``evaluate`` are defined in
        ``benchmarks/cli/e2e.py`` and ``kernel-bench`` in
        ``benchmarks/cli/kernel.py``, with plain ``@click.command``;
        ``main.py`` attaches them with ``cli.add_command``. Binding them with
        ``@cli.command`` in their own modules instead would invert that edge,
        and ``from benchmarks.cli.main import cli`` -- what ``__main__.py``
        and this file do -- would then yield a group holding only whichever
        commands some earlier import had loaded. Nothing else would notice: a
        CLI missing ``evaluate`` starts fine and prints a usage message.

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
            ["evaluate", "kernel-bench", "run", "scenarios"],
        )

    def test_root_help_and_scenario_listing(self) -> None:
        help_result = self.runner.invoke(cli, ["--help"])
        self.assertEqual(help_result.exit_code, 0)
        self.assertIn("run", help_result.output)
        result = self.runner.invoke(cli, ["scenarios"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("engines", result.output)
        self.assertIn("titan_compiled", result.output)

    def test_scenarios_listing_truncates_to_single_sentence(self) -> None:
        result = self.runner.invoke(cli, ["scenarios"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "engines: Stock TorchTitan, compiled and eager, against stock Megatron-LM on one pre-tokenized c4_test stream.",
            result.output,
        )
        self.assertNotIn(
            "This is a systems-throughput claim about configured engines",
            result.output,
        )

    def test_scenarios_e2e_lists_only_e2e_scenarios(self) -> None:
        result = self.runner.invoke(cli, ["scenarios", "e2e"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("end-to-end scenarios (run --scenario):", result.output)
        self.assertIn("engines", result.output)
        self.assertIn("titan_compiled", result.output)
        self.assertNotIn("kernel scenarios", result.output)
        self.assertNotIn("kernel spans", result.output)
        self.assertNotIn("rope", result.output)

    def test_scenarios_detail_e2e(self) -> None:
        result = self.runner.invoke(cli, ["scenarios", "detail", "engines"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("End-to-end scenario: engines", result.output)
        self.assertIn(
            "This is a systems-throughput claim about configured engines",
            result.output,
        )
        self.assertIn("titan_compiled:", result.output)
        self.assertIn("megatron_stock:", result.output)

    def test_scenarios_shorthand_detail_e2e(self) -> None:
        result = self.runner.invoke(cli, ["scenarios", "engines"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("End-to-end scenario: engines", result.output)
        self.assertIn(
            "This is a systems-throughput claim about configured engines",
            result.output,
        )

    def test_scenarios_detail_arm(self) -> None:
        result = self.runner.invoke(
            cli, ["scenarios", "detail", "engines/megatron_stock"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("End-to-end arm: engines/megatron_stock", result.output)
        self.assertIn("NOT PLAIN BF16", result.output)

    def test_scenarios_detail_unknown_target(self) -> None:
        result = self.runner.invoke(cli, ["scenarios", "detail", "nonexistent"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(
            "Unknown scenario, span, or arm 'nonexistent'.", result.output
        )
        self.assertIn("Available e2e scenarios: engines", result.output)


    def test_execution_help_shows_environment_variables(self) -> None:
        result = self.runner.invoke(cli, ["run", "--help"])
        self.assertEqual(result.exit_code, 0)
        for envvar in (
            "OUT",
            "SEQ",
            "STEPS",
            "BATCH",
            "BENCHMARK_CACHE_ROOT",
            "BENCH_COMPILER_ENV",
            "AC_MODE",
        ):
            self.assertIn(envvar, result.output)

    def test_ac_mode_reaches_the_request(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_compiled"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(cli, ["run", "2", "--ac", "none"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].axes.ac_mode, "none")

    def test_ac_mode_defaults_to_unrequested(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_compiled"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(cli, ["run", "2"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(execute.call_args.args[0].axes.ac_mode)

    def test_ac_mode_applies_to_every_swept_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli,
                    ["run", "0", "--ac", "none", "--model-size", "huge"],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        ac_modes = {call.args[0].axes.ac_mode for call in execute.call_args_list}
        self.assertEqual(ac_modes, {"none"})
        # The third global axis must reach every swept scenario too.
        sizes = {call.args[0].axes.model_size for call in execute.call_args_list}
        self.assertEqual(sizes, {"huge"})

    def test_model_size_defaults_to_unrequested_and_rejects_unknowns(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_compiled"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(cli, ["run", "2"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(execute.call_args.args[0].axes.model_size)

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
                    "engines",
                    "--arm",
                    "titan_compiled",
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
        command = manifest["commands"]["titan_compiled"]
        self.assertEqual(
            command[command.index("--config") + 1], "qwen3_piper_1b_pretokenized"
        )
        self.assertEqual(command[command.index("--config-arg") + 1], "size=huge")
        self.assertFalse([token for token in command if token.endswith("_huge")])

    def test_unknown_ac_mode_is_rejected(self) -> None:
        result = self.runner.invoke(cli, ["run", "2", "--ac", "turbo"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value", result.output)

    def test_run_preserves_torchtitan_passthrough_arguments(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_compiled"),),
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
                    "engines",
                    "--arm",
                    "titan_compiled",
                    "--",
                    "--debug.seed",
                    "42",
                    "--debug.deterministic",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        request = execute.call_args.args[0]
        self.assertEqual(request.gpu, "2")
        self.assertEqual(request.arm_names, ("titan_compiled",))
        self.assertEqual(
            request.extra_args,
            ("--debug.seed", "42", "--debug.deterministic"),
        )

    def test_run_collects_repeated_arms_in_command_line_order(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(
                ENGINES.arm("titan_eager"),
                ENGINES.arm("titan_compiled"),
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
                    "engines",
                    "--arm",
                    "titan_eager",
                    "--arm",
                    "titan_compiled",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            execute.call_args.args[0].arm_names, ("titan_eager", "titan_compiled")
        )

    def test_run_executes_then_evaluates_same_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            completed = SimpleNamespace(out_dir=out_dir)
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate") as evaluate:
                result = self.runner.invoke(
                    cli,
                    ["run", "6", "--scenario", "engines", "--ac", "none"],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        request = execute.call_args.args[0]
        self.assertEqual(request.arm_names, ())
        evaluate.assert_called_once_with(out_dir, (), None)

    def test_all_scenarios_runs_each_scenario_under_one_timestamp(self) -> None:
        # Every scenario declines the default ac mode (sac), so the sweep at
        # ac none is the one that runs them all, under one timestamp.
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli, ["run", "0", "--ac", "none"]
                )
        self.assertEqual(result.exit_code, 0, result.output)
        requests = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(
            [request.scenario_name for request in requests], list(SCENARIOS)
        )

    def test_all_scenarios_skips_a_scenario_that_declines_the_ac_mode(self) -> None:
        """The skip says why, and a sweep that ran nothing exits nonzero."""
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli, ["run", "0", "--ac", "sac"]
                )
        self.assertNotEqual(result.exit_code, 0)
        execute.assert_not_called()
        self.assertIn("skipped: does not support ac mode 'sac'", result.output)

    def test_a_sweep_that_skips_every_scenario_is_refused(self) -> None:
        """Nothing ran, so the command exits nonzero.

        The message names each skipped scenario and the axis that excluded
        it. An exit code of 0 would report a measurement nobody took.
        """
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(cli, ["run", "0", "--ac", "sac"])
        self.assertEqual(result.exit_code, 1, result.output)
        execute.assert_not_called()
        self.assertIn("nothing ran", result.output)
        for name in SCENARIOS:
            self.assertIn(
                f"{name}: does not support ac mode 'sac'", result.output
            )

    def test_a_sweep_that_runs_one_scenario_of_two_is_not_refused(self) -> None:
        """The refusal covers an empty sweep alone, not a partial skip."""
        registry = self._two_scenarios()
        registry["engines_copy"] = replace(
            registry["engines_copy"], supported_ac_modes=("sac",)
        )
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.SCENARIOS", registry
            ), mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(cli, ["run", "0", "--ac", "none"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            [call.args[0].scenario_name for call in execute.call_args_list],
            ["engines"],
        )

    def test_the_cli_ac_default_is_the_registry_default(self) -> None:
        """An omitted --ac resolves to the registry default, and the sweep
        reads the same constant."""
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(cli, ["run", "0"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(DEFAULT_AC_MODE, "none")
        self.assertEqual(
            [call.args[0].scenario_name for call in execute.call_args_list],
            list(SCENARIOS),
        )

    def test_all_scenarios_at_ac_none_includes_the_megatron_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli, ["run", "0", "--ac", "none"]
                )
        self.assertEqual(result.exit_code, 0, result.output)
        names = [call.args[0].scenario_name for call in execute.call_args_list]
        self.assertEqual(names, list(SCENARIOS))

    def test_all_scenarios_stops_at_the_first_failing_scenario(self) -> None:
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", side_effect=RuntimeError("arm failed")
        ) as execute, mock.patch("benchmarks.cli.e2e._evaluate") as evaluate:
            result = self.runner.invoke(
                cli, ["run", "0", "--ac", "none"]
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(execute.call_count, 1)
        evaluate.assert_not_called()

    def test_an_omitted_scenario_runs_every_scenario(self) -> None:
        """There is no default scenario of one name, and no refusal either.

        An omitted ``--scenario`` selects the whole roster, so the run
        measures every scenario rather than one under whatever label the
        operator assumed. ``benchmarks/e2e/runner.py`` keeps the refusal
        for a programmatic request that names none, and this command
        passes a name for every run but a resume.
        """
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute:
                result = self.runner.invoke(cli, ["run", "0", "--ac", "none"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            [call.args[0].scenario_name for call in execute.call_args_list],
            list(SCENARIOS),
        )

    def test_a_single_directory_option_needs_exactly_one_scenario(self) -> None:
        """``--out``, ``--resume`` and ``--results`` each name one path.

        Two scenarios sharing one would overwrite each other's manifest and
        results, so each option needs the selection narrowed to one.
        """
        with tempfile.TemporaryDirectory() as temporary:
            for conflicting in (
                ["--out", f"{temporary}/output"],
                ["--resume", temporary],
                ["--results", f"{temporary}/results.json"],
            ):
                with self.subTest(option=conflicting[0]):
                    with mock.patch.dict(
                        "benchmarks.cli.e2e.SCENARIOS",
                        self._two_scenarios(),
                        clear=True,
                    ):
                        result = self.runner.invoke(
                            cli, ["run", "0", *conflicting]
                        )
                    self.assertNotEqual(result.exit_code, 0)
                    self.assertIn("names one directory", result.output)
                    self.assertIn("2 scenarios", result.output)

    def test_one_scenario_accepts_the_single_directory_options(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute:
                result = self.runner.invoke(
                    cli,
                    [
                        "run",
                        "0",
                        "--scenario",
                        "engines",
                        "--ac",
                        "none",
                        "--out",
                        f"{temporary}/output",
                    ],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_count, 1)

    def test_an_unknown_scenario_is_refused_and_named(self) -> None:
        with mock.patch(
            "benchmarks.cli.e2e.execute_run",
            side_effect=AssertionError("a run started"),
        ):
            result = self.runner.invoke(
                cli, ["run", "0", "--scenario", "kernels"]
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("no such scenario: 'kernels'", result.output)

    def test_an_arm_absent_from_a_selected_scenario_is_refused(self) -> None:
        """``--arm`` applies to every selected scenario.

        A name one of them does not declare would run a smaller matrix
        than the operator asked for, so it is refused before a GPU is
        claimed.
        """
        with mock.patch(
            "benchmarks.cli.e2e.execute_run",
            side_effect=AssertionError("a run started"),
        ):
            result = self.runner.invoke(
                cli, ["run", "0", "--arm", "no_such_arm"]
            )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("has no arm(s) 'no_such_arm'", result.output)

    def test_several_scenarios_share_one_timestamp(self) -> None:
        """One stamp groups the whole sweep under one out/<stamp>/ root."""
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch.dict(
                "benchmarks.cli.e2e.SCENARIOS",
                self._two_scenarios(),
                clear=True,
            ), mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute:
                result = self.runner.invoke(cli, ["run", "0", "--ac", "none"])
        self.assertEqual(result.exit_code, 0, result.output)
        requests = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(
            [request.scenario_name for request in requests],
            ["engines", "engines_copy"],
        )
        stamps = {request.timestamp for request in requests}
        self.assertEqual(len(stamps), 1)
        self.assertIsNotNone(requests[0].timestamp)
        self.assertIn("===== scenario: engines_copy =====", result.output)

    def test_a_repeated_scenario_runs_again_under_its_own_directory(self) -> None:
        """A name given twice is two runs, and the second takes a suffix."""
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute:
                result = self.runner.invoke(
                    cli,
                    ["run", "0", "--scenario", "engines", "--scenario", "engines"],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        requests = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(
            [request.scenario_name for request in requests], ["engines", "engines"]
        )
        self.assertEqual([request.occurrence for request in requests], [1, 2])
        directories = {
            _default_output_dir(
                ENGINES, "h200", None, {}, request.timestamp, request.occurrence
            )
            for request in requests
        }
        self.assertEqual(len(directories), 2)

    def test_two_different_scenarios_each_keep_their_plain_name(self) -> None:
        """The suffix counts one name, so two names never take one."""
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch.dict(
                "benchmarks.cli.e2e.SCENARIOS",
                self._two_scenarios(),
                clear=True,
            ), mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute:
                result = self.runner.invoke(cli, ["run", "0", "--ac", "none"])
        self.assertEqual(result.exit_code, 0, result.output)
        requests = [call.args[0] for call in execute.call_args_list]
        self.assertEqual([request.occurrence for request in requests], [1, 1])
        self.assertEqual(
            _default_output_dir(ENGINES, "h200", None, {}, "STAMP", 1).name, "h200"
        )
        self.assertEqual(
            _default_output_dir(ENGINES, "h200", None, {}, "STAMP", 1).parent.name,
            "engines",
        )
        self.assertEqual(
            _default_output_dir(ENGINES, "h200", None, {}, "STAMP", 2).parent.name,
            "engines-run2",
        )

    def test_run_does_not_evaluate_a_failed_execution(self) -> None:
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", side_effect=RuntimeError("arm failed")
        ), mock.patch("benchmarks.cli.e2e._evaluate") as evaluate:
            result = self.runner.invoke(cli, ["run", "0"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("arm failed", result.output)
        evaluate.assert_not_called()

    def test_megatron_p2p_sync_reaches_the_request(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_compiled"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(
                cli, ["run", "2", "--megatron-p2p-sync", "off"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].axes.megatron_p2p_sync, "off")

    def test_megatron_p2p_sync_defaults_to_unrequested(self) -> None:
        """``None`` is what lets a resume inherit the recorded value."""
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_compiled"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(cli, ["run", "2"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(execute.call_args.args[0].axes.megatron_p2p_sync)

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
        for command in (run_command,):
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
            if any(arm.engine in MEGATRON_ENGINES for arm in scenario.arms)
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
                        "run",
                        "0,1",
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
            {request.axes.megatron_p2p_sync for request in requests}, {"off"}
        )

    def test_megatron_nan_guard_reaches_the_request(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_compiled"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(
                cli, ["run", "2", "--megatron-nan-guard", "off"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].axes.megatron_nan_guard, "off")

    def test_megatron_nan_guard_defaults_to_unrequested(self) -> None:
        """``None`` is what lets a resume inherit the recorded value."""
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_compiled"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(cli, ["run", "2"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(execute.call_args.args[0].axes.megatron_nan_guard)

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
        for command in (run_command,):
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
            if any(arm.engine in MEGATRON_ENGINES for arm in scenario.arms)
        ]
        self.assertEqual(holds_stock, ["engines"])
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli,
                    [
                        "run",
                        "0",
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
            {request.axes.megatron_nan_guard for request in requests}, {"off"}
        )

    def test_megatron_precision_reaches_the_request(self) -> None:
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_compiled"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(
                cli,
                [
                    "run",
                    "2",
                    # Named, so the sweep's own lean-needs-zero-1 skip does
                    # not apply and the value reaches the request.
                    "--scenario",
                    "engines",
                    "--zero",
                    "1",
                    "--megatron-precision",
                    "lean",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].axes.megatron_precision, "lean")

    def test_megatron_precision_defaults_to_unrequested(self) -> None:
        """``None`` is what lets a later resume inherit the recorded value."""
        completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_compiled"),),
        )
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=completed
        ) as execute:
            result = self.runner.invoke(cli, ["run", "2"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(execute.call_args.args[0].axes.megatron_precision)

    def test_an_unknown_megatron_precision_value_is_rejected(self) -> None:
        result = self.runner.invoke(
            cli, ["run", "2", "--megatron-precision", "bf16"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value", result.output)

    def test_megatron_precision_takes_no_environment_variable(self) -> None:
        """``lean`` has to agree with ``--scenario``, ``--arm`` AND
        ``--zero``. An exported value would fail every replicated
        run on a flag the operator never passed.
        """
        for command in (run_command,):
            parameters = {
                option: parameter
                for parameter in command.params
                for option in parameter.opts
            }
            with self.subTest(command=command.name):
                self.assertIsNone(parameters["--megatron-precision"].envvar)

    def test_all_scenarios_at_lean_runs_the_stock_scenario_alone(self) -> None:
        """The value reaches the stock megatron arm alone, and it needs a
        sharded dense value.

        The sweep skips what ``_resolve_run`` would refuse, with the
        refusal's own reason, rather than aborting at its first titan-only
        entry.
        """
        holds_stock = [
            name
            for name, scenario in SCENARIOS.items()
            if any(arm.engine in MEGATRON_ENGINES for arm in scenario.arms)
        ]
        self.assertEqual(holds_stock, ["engines"])
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli,
                    [
                        "run",
                        "0",
                        "--ac",
                        "none",
                        "--zero",
                        1,
                        "--megatron-precision",
                        "lean",
                    ],
                )
        self.assertEqual(result.exit_code, 0, result.output)
        requests = [call.args[0] for call in execute.call_args_list]
        self.assertEqual(
            [request.scenario_name for request in requests], holds_stock
        )
        self.assertEqual(
            {request.axes.megatron_precision for request in requests}, {"lean"}
        )

    def test_all_scenarios_at_lean_under_zero_zero_skips_everything(
        self,
    ) -> None:
        """Megatron asserts the distributed optimizer under the
        precision-aware optimizer, and ``--zero`` is the one
        owner of that flag. No scenario can honour lean without it.
        """
        with tempfile.TemporaryDirectory() as temporary:
            completed = SimpleNamespace(out_dir=Path(temporary))
            with mock.patch(
                "benchmarks.cli.e2e.execute_run", return_value=completed
            ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"):
                result = self.runner.invoke(
                    cli,
                    ["run", "0", "--ac", "none", "--megatron-precision", "lean"],
                )
        # Nothing ran, so the sweep refuses rather than exiting 0.
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(execute.call_args_list, [])
        self.assertIn(
            "skipped: --megatron-precision 'lean' needs --zero 1",
            result.output,
        )

    def test_run_records_evaluation_failure_for_resume(self) -> None:
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
                result = self.runner.invoke(cli, ["run", "0"])
            state = json.loads((out_dir / "run_state.json").read_text())

        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(state["status"], "evaluation_failed")
        self.assertIn("bad trace", state["evaluation"]["error"])


if __name__ == "__main__":
    unittest.main()
