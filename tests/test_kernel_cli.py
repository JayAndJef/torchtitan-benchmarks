"""CPU-only tests for kernel-benchmark CLI wiring and orchestration."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cli import cli
from benchmarks.kernel_runner import KernelRunRequest, execute_kernel_run
from benchmarks.kernels import KERNEL_SCENARIOS
from benchmarks.runtime import CpuPinning
from tests.test_kernel_results import sample_result


METADATA = {
    "requested_gpu": "7",
    "nvidia_smi": "7, Test GPU, GPU-uuid, driver",
    "torch_version": "test",
    "torchtitan_git_rev": "titan-rev",
    "benchmarks_git_rev": "bench-rev",
}
PINNING = CpuPinning(("numactl", "--cpunodebind=1", "--membind=1"), "numactl test")


def patched_environment():
    return (
        mock.patch(
            "benchmarks.kernel_runner.hardware_metadata",
            return_value=("test-gpu", dict(METADATA)),
        ),
        mock.patch(
            "benchmarks.kernel_runner.resolve_cpu_pinning", return_value=PINNING
        ),
    )


class KernelCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_scenarios_lists_kernel_scenarios(self) -> None:
        result = self.runner.invoke(cli, ["scenarios"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("kernel scenarios", result.output)
        for name in KERNEL_SCENARIOS:
            self.assertIn(name, result.output)

    def test_flags_map_onto_the_request(self) -> None:
        with mock.patch(
            "benchmarks.cli.execute_kernel_run", return_value=()
        ) as execute:
            result = self.runner.invoke(
                cli,
                [
                    "kernel-bench",
                    "7",
                    "--scenario",
                    "swiglu",
                    "--n",
                    "20",
                    "--warmup",
                    "5",
                    "--burst",
                    "--batch",
                    "1",
                    "--seq-len",
                    "512",
                    "--seed",
                    "3",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        request = execute.call_args.args[0]
        self.assertEqual(request.gpu, "7")
        self.assertEqual(request.scenario_names, ("swiglu",))
        self.assertEqual((request.n, request.warmup, request.seed), (20, 5, 3))
        self.assertTrue(request.burst)
        self.assertEqual((request.batch, request.seq_len), (1, 512))

    def test_defaults_to_every_scenario(self) -> None:
        with mock.patch(
            "benchmarks.cli.execute_kernel_run", return_value=()
        ) as execute:
            result = self.runner.invoke(cli, ["kernel-bench", "7"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            execute.call_args.args[0].scenario_names, tuple(KERNEL_SCENARIOS)
        )

    def test_out_requires_a_single_scenario(self) -> None:
        result = self.runner.invoke(
            cli, ["kernel-bench", "7", "--out", "/tmp/kernels"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--out requires exactly one --scenario", result.output)

    def test_failed_scenario_exits_nonzero_after_rendering(self) -> None:
        outcome = SimpleNamespace(
            scenario="rope",
            out_dir=Path("/tmp/out"),
            result=sample_result(),
            correctness_failed=True,
            error=None,
            failed=True,
        )
        with mock.patch(
            "benchmarks.cli.execute_kernel_run", return_value=(outcome,)
        ):
            result = self.runner.invoke(cli, ["kernel-bench", "7"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("kernel scenarios failed: rope (correctness)", result.output)
        self.assertIn("correctness", result.output)


class KernelRunnerTests(unittest.TestCase):
    def test_failures_are_reported_before_later_scenarios_run(self) -> None:
        """A first-scenario failure must not wait for the whole sweep."""
        events = []

        def fake_process(command, **kwargs):
            name = command[command.index("--scenario") + 1]
            if name == "rope":
                kwargs["stdout"].write("boom\n")
                return SimpleNamespace(returncode=1)
            out_dir = Path(command[command.index("--out-dir") + 1])
            (out_dir / "results.json").write_text(
                json.dumps(sample_result().to_dict())
            )
            return SimpleNamespace(returncode=0)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel_runner.BENCH_DIR", Path(temporary)
        ):
            compiler_env = Path(temporary) / "enable.sh"
            compiler_env.write_text("# no-op compiler environment\n")
            execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("rope", "swiglu"),
                    timestamp="stamp",
                    compiler_env=compiler_env,
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
                event_handler=events.append,
            )
        kinds = [(event.kind, event.message) for event in events]
        error_index = next(
            index for index, (kind, _) in enumerate(kinds) if kind == "error"
        )
        swiglu_index = next(
            index
            for index, (kind, message) in enumerate(kinds)
            if kind == "arm" and "swiglu" in message
        )
        self.assertLess(error_index, swiglu_index)
        self.assertIn("boom", kinds[error_index][1])

    def test_broken_compiler_env_fails_only_its_scenario(self) -> None:
        def fake_process(command, **kwargs):
            out_dir = Path(command[command.index("--out-dir") + 1])
            (out_dir / "results.json").write_text(
                json.dumps(sample_result().to_dict())
            )
            return SimpleNamespace(returncode=0)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel_runner.BENCH_DIR", Path(temporary)
        ):
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("rope", "swiglu"),
                    timestamp="stamp",
                    compiler_env=Path(temporary) / "missing.sh",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        by_name = {outcome.scenario: outcome for outcome in outcomes}
        self.assertTrue(by_name["rope"].failed)
        self.assertIn("compiler environment", by_name["rope"].error)
        self.assertFalse(by_name["swiglu"].failed)

    def test_worker_command_carries_pinning_env_and_manifest(self) -> None:
        captured = {}

        def fake_process(command, **kwargs):
            captured["command"] = command
            captured["env"] = kwargs["env"]
            out_dir = Path(command[command.index("--out-dir") + 1])
            payload = sample_result().to_dict()
            payload["scenario"] = "swiglu"
            (out_dir / "results.json").write_text(json.dumps(payload))
            return SimpleNamespace(returncode=0)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            out_dir = Path(temporary) / "kernels"
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu",),
                    n=5,
                    warmup=2,
                    out_dir=out_dir,
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        command = captured["command"]
        self.assertEqual(command[:3], list(PINNING.prefix))
        self.assertIn("-m", command)
        self.assertEqual(command[command.index("-m") + 1], "benchmarks.kernel_worker")
        self.assertEqual(command[command.index("--n") + 1], "5")
        self.assertNotIn("--burst", command)
        self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(captured["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

        self.assertEqual(manifest["kind"], "kernel")
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["scenario"], "swiglu")
        self.assertEqual(manifest["hardware_metadata"]["cpu_pinning"], "numactl test")
        self.assertEqual(manifest["spec"]["dim"], 1024)
        self.assertEqual(manifest["shapes"]["x"], [8192, 1024])
        self.assertEqual(manifest["n"], 5)

        self.assertEqual(len(outcomes), 1)
        self.assertFalse(outcomes[0].failed)
        self.assertIsNotNone(outcomes[0].result)

    def test_compiler_environment_only_for_scenarios_that_need_it(self) -> None:
        def fake_process(command, **kwargs):
            out_dir = Path(command[command.index("--out-dir") + 1])
            (out_dir / "results.json").write_text(
                json.dumps(sample_result().to_dict())
            )
            return SimpleNamespace(returncode=0)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel_runner.BENCH_DIR", Path(temporary)
        ), mock.patch(
            "benchmarks.kernel_runner.add_compiler_environment",
            side_effect=lambda env, script: {**env, "SOURCED": "1"},
        ) as compiler:
            execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu", "rope"),
                    timestamp="stamp",
                    compiler_env=Path(temporary) / "enable.sh",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        self.assertEqual(compiler.call_count, 1)

    def test_missing_compiler_env_fails_only_the_te_scenario(self) -> None:
        def fake_process(command, **kwargs):
            out_dir = Path(command[command.index("--out-dir") + 1])
            (out_dir / "results.json").write_text(
                json.dumps(sample_result().to_dict())
            )
            return SimpleNamespace(returncode=0)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel_runner.BENCH_DIR", Path(temporary)
        ), mock.patch(
            "benchmarks.kernel_runner.RuntimePaths.resolve"
        ) as resolve:
            resolve.return_value = SimpleNamespace(
                bench_dir=Path(temporary),
                titan_dir=Path(temporary),
                cache_root=Path(temporary) / "cache",
                compiler_env=None,
            )
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu", "rope"),
                    timestamp="stamp",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        by_name = {outcome.scenario: outcome for outcome in outcomes}
        self.assertFalse(by_name["swiglu"].failed)
        self.assertTrue(by_name["rope"].failed)
        self.assertIn("C++20 host compiler", by_name["rope"].error)

    def test_worker_crash_surfaces_the_log_tail(self) -> None:
        def crashing_process(command, **kwargs):
            kwargs["stdout"].write("boom: build failed\n")
            return SimpleNamespace(returncode=1)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu",),
                    out_dir=Path(temporary) / "kernels",
                ),
                process_runner=crashing_process,
                environment={"PATH": "/usr/bin"},
            )
        self.assertTrue(outcomes[0].failed)
        self.assertIn("boom: build failed", outcomes[0].error)

    def test_correctness_exit_code_keeps_results(self) -> None:
        def failing_process(command, **kwargs):
            out_dir = Path(command[command.index("--out-dir") + 1])
            payload = sample_result().to_dict()
            payload["all_correctness_passed"] = False
            (out_dir / "results.json").write_text(json.dumps(payload))
            return SimpleNamespace(returncode=3)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu",),
                    out_dir=Path(temporary) / "kernels",
                ),
                process_runner=failing_process,
                environment={"PATH": "/usr/bin"},
            )
        self.assertTrue(outcomes[0].correctness_failed)
        self.assertIsNone(outcomes[0].error)
        self.assertIsNotNone(outcomes[0].result)


if __name__ == "__main__":
    unittest.main()
