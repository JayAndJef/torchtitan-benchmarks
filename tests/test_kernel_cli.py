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

from benchmarks.cli.main import cli
from benchmarks.execution.affinity import CpuPinning
from benchmarks.kernel.registry import KERNEL_SCENARIOS, kernel_scenario_by_name
from benchmarks.kernel.results.merge import (
    CORRECTNESS_FRAGMENT_KIND,
    TIMING_FRAGMENT_KIND,
)
from benchmarks.kernel.runner import (
    KernelRunRequest,
    execute_kernel_run,
    resolve_arm_skips,
)
from benchmarks.kernel.schema import (
    CorrectnessCheck,
    KernelArm,
    KernelScenario,
)
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
            "benchmarks.kernel.runner.hardware_metadata",
            return_value=("test-gpu", dict(METADATA)),
        ),
        mock.patch(
            "benchmarks.kernel.runner.resolve_cpu_pinning", return_value=PINNING
        ),
    )


def correctness_fragment(scenario: str, passed: bool = True) -> dict:
    return {
        "kind": CORRECTNESS_FRAGMENT_KIND,
        "scenario": scenario,
        "rows": [
            {
                "arm": kernel_scenario_by_name(scenario).arms[-1].name,
                "reference": kernel_scenario_by_name(scenario).baseline_arm,
                "kind": "tolerance",
                "output": "out",
                "metric": "rel_l2",
                "value": 1e-3,
                "threshold": 2e-2,
                "passed": passed,
                "informational": False,
            }
        ],
        "all_passed": passed,
        "environment": {"device": "Test GPU", "torch_version": "test"},
    }


def timing_fragment(scenario: str, arm: str, replicate: int) -> dict:
    """A worker's output, shaped exactly as ``run_timing_pass`` writes it."""
    declaration = kernel_scenario_by_name(scenario).arm(arm)
    return {
        "kind": TIMING_FRAGMENT_KIND,
        "scenario": scenario,
        "arm": arm,
        "replicate": replicate,
        # Distinct per replicate, so a merge that scrambled the boundaries
        # would produce different statistics rather than the same ones.
        "modes": {
            mode: [10.0 + replicate + index for index in range(3)]
            for mode in declaration.modes
        },
        "bytes_moved": None,
        "peak_memory_gib": 1.5 if replicate == 0 else None,
        "burst_us_per_call": None,
    }


def fragment_writer(
    *,
    correctness_passes: bool = True,
    correctness_code: dict[str, int] | None = None,
    skip_timing: tuple[tuple[str, str, int], ...] = (),
):
    """A ``process_runner`` that plays the worker protocol.

    ``correctness_code`` maps a scenario to a return code whose worker also
    writes no fragment, which is how a build failure looks to the parent.
    ``skip_timing`` names (scenario, arm, replicate) triples whose timing
    worker crashes the same way.
    """
    codes = correctness_code or {}

    def fake_process(command, **kwargs):
        scenario = command[command.index("--scenario") + 1]
        mode = command[command.index("--mode") + 1]
        fragment = Path(command[command.index("--fragment") + 1])
        if mode == "correctness":
            if scenario in codes:
                kwargs["stdout"].write("boom: build failed\n")
                return SimpleNamespace(returncode=codes[scenario])
            fragment.write_text(
                json.dumps(correctness_fragment(scenario, correctness_passes))
            )
            return SimpleNamespace(returncode=0 if correctness_passes else 3)
        arm = command[command.index("--arm") + 1]
        replicate = int(command[command.index("--replicate") + 1])
        if (scenario, arm, replicate) in skip_timing:
            kwargs["stdout"].write(f"boom: {arm} r{replicate}\n")
            return SimpleNamespace(returncode=1)
        fragment.write_text(
            json.dumps(timing_fragment(scenario, arm, replicate))
        )
        return SimpleNamespace(returncode=0)

    return fake_process


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
            "benchmarks.cli.kernel.execute_kernel_run", return_value=()
        ) as execute:
            result = self.runner.invoke(
                cli,
                [
                    "kernel-bench",
                    "7",
                    "--scenario",
                    "swiglu",
                    "--replicates",
                    "3",
                    "--samples-per-replicate",
                    "20",
                    "--burst-k",
                    "8",
                    "--warmup-calls",
                    "5",
                    "--burst",
                    "--model-size",
                    "huge",
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
        self.assertEqual(
            (
                request.replicates,
                request.samples_per_replicate,
                request.burst_k,
                request.warmup_calls,
                request.seed,
            ),
            (3, 20, 8, 5, 3),
        )
        self.assertTrue(request.burst)
        self.assertEqual(request.model_size, "huge")
        self.assertEqual((request.batch, request.seq_len), (1, 512))

    def test_model_size_defaults_to_normal_and_rejects_unknown(self) -> None:
        with mock.patch(
            "benchmarks.cli.kernel.execute_kernel_run", return_value=()
        ) as execute:
            result = self.runner.invoke(cli, ["kernel-bench", "7"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].model_size, "normal")

        result = self.runner.invoke(
            cli, ["kernel-bench", "7", "--model-size", "enormous"]
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_defaults_to_every_scenario(self) -> None:
        with mock.patch(
            "benchmarks.cli.kernel.execute_kernel_run", return_value=()
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
            "benchmarks.cli.kernel.execute_kernel_run", return_value=(outcome,)
        ):
            result = self.runner.invoke(cli, ["kernel-bench", "7"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("kernel scenarios failed: rope (correctness)", result.output)
        self.assertIn("correctness", result.output)


class KernelRunnerTests(unittest.TestCase):
    def test_failures_are_reported_before_later_scenarios_run(self) -> None:
        """A first-scenario failure must not wait for the whole sweep."""
        events = []
        fake_process = fragment_writer(correctness_code={"rope": 1})

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel.runner.BENCH_DIR", Path(temporary)
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

    def test_a_skip_closes_over_correctness_references(self) -> None:
        """An arm whose reference is skipped is skipped too. Timing an arm
        that nothing checked is the silent wrongness the gates exist for.

        No scenario reaches this today -- rope's ``te`` is a referrer, never a
        reference -- so the closure is guarding the roster's growth, and the
        synthetic scenario is how it gets exercised at all.
        """

        def arm(name, requires_gcc=False, reference=None):
            return KernelArm(
                name=name,
                description=name,
                builder=f"benchmarks.kernel.operations.rope:build_{name}",
                modes=("forward",),
                requires_gcc_toolset=requires_gcc,
                correctness=(
                    ()
                    if reference is None
                    else (
                        CorrectnessCheck(
                            kind="tolerance",
                            reference=reference,
                            outputs=("out",),
                            max_rel_l2=1e-2,
                        ),
                    )
                ),
            )

        scenario = KernelScenario(
            name="rope",
            description="synthetic",
            inputs_builder="benchmarks.kernel.operations.rope:rope_inputs",
            reference_builder=None,
            baseline_arm="anchor",
            arms=(
                arm("anchor"),
                arm("needs_compiler", requires_gcc=True),
                arm("gated_on_it", reference="needs_compiler"),
                arm("gated_on_that", reference="gated_on_it"),
                arm("independent", reference="anchor"),
            ),
        )
        self.assertEqual(
            resolve_arm_skips(scenario, compiler_unavailable=None), {}
        )
        skipped = resolve_arm_skips(scenario, compiler_unavailable="no gcc")
        self.assertEqual(
            sorted(skipped), ["gated_on_it", "gated_on_that", "needs_compiler"]
        )
        self.assertEqual(skipped["needs_compiler"], "no gcc")
        self.assertIn("'needs_compiler' is skipped", skipped["gated_on_it"])
        self.assertIn("'gated_on_it' is skipped", skipped["gated_on_that"])

    def test_a_broken_compiler_env_costs_the_te_arm_only(self) -> None:
        """The requirement belongs to the arm. Rope still measures its other
        three arms, which the former scenario-level check threw away."""
        fake_process = fragment_writer()

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel.runner.BENCH_DIR", Path(temporary)
        ):
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("rope", "swiglu"),
                    replicates=2,
                    timestamp="stamp",
                    compiler_env=Path(temporary) / "missing.sh",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        by_name = {outcome.scenario: outcome for outcome in outcomes}
        self.assertFalse(by_name["rope"].failed)
        self.assertFalse(by_name["swiglu"].failed)
        arms = by_name["rope"].result.arms
        self.assertEqual(arms["te"].status, "skipped")
        self.assertIn("compiler environment", arms["te"].status_reason)
        self.assertEqual(arms["te"].modes, {})
        for name in ("baseline", "helion", "copy_floor"):
            self.assertEqual(arms[name].status, "ok", name)

    def test_a_skipped_arm_is_spawned_in_neither_pass(self) -> None:
        commands = []

        writer = fragment_writer()

        def fake_process(command, **kwargs):
            commands.append(command)
            return writer(command, **kwargs)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel.runner.BENCH_DIR", Path(temporary)
        ):
            execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("rope",),
                    replicates=2,
                    timestamp="stamp",
                    compiler_env=Path(temporary) / "missing.sh",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        self.assertEqual(commands[0][commands[0].index("--mode") + 1], "correctness")
        self.assertEqual(
            commands[0][commands[0].index("--skip-arm") + 1], "te"
        )
        timed = {
            command[command.index("--arm") + 1] for command in commands[1:]
        }
        self.assertEqual(timed, {"copy_floor", "baseline", "helion"})

    def test_worker_command_carries_pinning_env_and_manifest(self) -> None:
        captured = []
        writer = fragment_writer()

        def fake_process(command, **kwargs):
            captured.append((command, kwargs["env"]))
            return writer(command, **kwargs)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            out_dir = Path(temporary) / "kernels"
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu",),
                    replicates=2,
                    samples_per_replicate=2,
                    burst_k=8,
                    warmup_calls=2,
                    out_dir=out_dir,
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        command, environment = captured[0]
        self.assertEqual(command[:3], list(PINNING.prefix))
        self.assertIn("-m", command)
        self.assertEqual(command[command.index("-m") + 1], "benchmarks.kernel.worker")
        self.assertEqual(command[command.index("--mode") + 1], "correctness")
        self.assertEqual(command[command.index("--replicates") + 1], "2")
        self.assertEqual(command[command.index("--burst-k") + 1], "8")
        self.assertEqual(command[command.index("--warmup-calls") + 1], "2")
        self.assertNotIn("--burst", command)
        # The correctness pass covers every arm at once, so it names none.
        self.assertNotIn("--arm", command)
        # Forwarded unconditionally, unlike the optional overrides.
        self.assertEqual(command[command.index("--model-size") + 1], "normal")
        self.assertNotIn("--batch", command)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

        # One correctness pass, then replicates x arms timing passes, in
        # replicate-major order.
        arms = [arm.name for arm in kernel_scenario_by_name("swiglu").arms]
        timing = [
            (
                argv[argv.index("--arm") + 1],
                int(argv[argv.index("--replicate") + 1]),
            )
            for argv, _ in captured[1:]
        ]
        self.assertEqual(len(captured), 1 + 2 * len(arms))
        self.assertEqual(
            timing, [(arm, r) for r in range(2) for arm in arms]
        )

        self.assertEqual(manifest["kind"], "kernel")
        self.assertEqual(manifest["schema_version"], 4)
        self.assertEqual(manifest["scenario"], "swiglu")
        self.assertEqual(manifest["hardware_metadata"]["cpu_pinning"], "numactl test")
        self.assertNotIn("spec", manifest)
        self.assertEqual(manifest["model_size"], "normal")
        self.assertEqual(manifest["model_shape"]["dim"], 1024)
        self.assertEqual(manifest["model_shape"]["n_layers"], 16)
        self.assertEqual(manifest["workload"], {"batch": 4, "seq_len": 1024})
        self.assertEqual(manifest["shapes"]["x"], [8192, 1024])
        self.assertEqual(manifest["replicates"], 2)
        self.assertEqual(manifest["burst_k"], 8)
        self.assertNotIn("n", manifest)
        # Every argv the run will issue, not just the first.
        self.assertNotIn("command", manifest)
        self.assertEqual(
            [list(argv) for argv, _ in captured], manifest["commands"]
        )

        self.assertEqual(len(outcomes), 1)
        self.assertFalse(outcomes[0].failed)
        self.assertIsNotNone(outcomes[0].result)

    def test_the_compiler_environment_is_resolved_once_per_run(self) -> None:
        """It shells out to bash, and the answer cannot change between two
        scenarios of one run. A run needing it at all pays exactly once; a run
        needing it nowhere pays nothing."""

        def run(scenario_names, temporary, compiler):
            execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=scenario_names,
                    replicates=2,
                    timestamp="-".join(scenario_names),
                    compiler_env=Path(temporary) / "enable.sh",
                ),
                process_runner=fragment_writer(),
                environment={"PATH": "/usr/bin"},
            )
            return compiler.call_count

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel.runner.BENCH_DIR", Path(temporary)
        ), mock.patch(
            "benchmarks.kernel.runner.add_compiler_environment",
            side_effect=lambda env, script: {**env, "SOURCED": "1"},
        ) as compiler:
            self.assertEqual(run(("swiglu",), temporary, compiler), 0)
            self.assertEqual(run(("swiglu", "rope"), temporary, compiler), 1)

    def test_missing_compiler_env_skips_only_the_te_arm(self) -> None:
        fake_process = fragment_writer()

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel.runner.BENCH_DIR", Path(temporary)
        ), mock.patch(
            "benchmarks.kernel.runner.RuntimePaths.resolve"
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
                    replicates=2,
                    timestamp="stamp",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        by_name = {outcome.scenario: outcome for outcome in outcomes}
        self.assertFalse(by_name["swiglu"].failed)
        self.assertFalse(by_name["rope"].failed)
        arms = by_name["rope"].result.arms
        self.assertEqual(arms["te"].status, "skipped")
        self.assertIn("C++20 host compiler", arms["te"].status_reason)
        self.assertEqual(arms["baseline"].status, "ok")

    def test_unbalanced_routing_skips_only_swiglu(self) -> None:
        """An odd batch x seq_len breaks swiglu's balanced split (rows =
        batch * seq_len * top_k against 4 experts); the other scenarios run."""
        events = []
        fake_process = fragment_writer()

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel.runner.BENCH_DIR", Path(temporary)
        ), mock.patch(
            "benchmarks.kernel.runner.add_compiler_environment",
            side_effect=lambda env, script: env,
        ):
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu", "rope"),
                    batch=3,
                    seq_len=1025,
                    timestamp="stamp",
                    compiler_env=Path(temporary) / "enable.sh",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
                event_handler=events.append,
            )
        by_name = {outcome.scenario: outcome for outcome in outcomes}
        self.assertTrue(by_name["swiglu"].failed)
        self.assertIn(
            "6150 routed rows (batch 3 x seq 1025 x top_k 2) do not divide "
            "evenly among 4 experts",
            by_name["swiglu"].error,
        )
        self.assertIsNone(by_name["swiglu"].result)
        self.assertFalse(by_name["rope"].failed)
        self.assertIsNotNone(by_name["rope"].result)
        # Loud, not silent: the skip is streamed as it happens.
        self.assertTrue(
            any(
                event.kind == "error" and "6150 routed rows" in event.message
                for event in events
            )
        )
        # No output directory is created for a scenario that never ran.
        self.assertFalse(by_name["swiglu"].out_dir.exists())

    def test_worker_crash_surfaces_the_log_tail(self) -> None:
        fake_process = fragment_writer(correctness_code={"swiglu": 1})

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu",),
                    out_dir=Path(temporary) / "kernels",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        self.assertTrue(outcomes[0].failed)
        self.assertIn("boom: build failed", outcomes[0].error)

    def test_failed_gates_keep_results_and_time_nothing(self) -> None:
        """A gate failure is the result; measuring past it wastes the GPU."""
        seen = []
        writer = fragment_writer(correctness_passes=False)

        def fake_process(command, **kwargs):
            seen.append(command[command.index("--mode") + 1])
            return writer(command, **kwargs)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            out_dir = Path(temporary) / "kernels"
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu",),
                    replicates=2,
                    out_dir=out_dir,
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
            self.assertTrue((out_dir / "results.json").exists())

        self.assertEqual(seen, ["correctness"])
        self.assertTrue(outcomes[0].correctness_failed)
        self.assertIsNone(outcomes[0].error)
        result = outcomes[0].result
        self.assertIsNotNone(result)
        self.assertFalse(result.all_correctness_passed)
        # Every declared arm is still in the file, so a reader can tell a
        # scenario that measured nothing from one that declared nothing.
        self.assertEqual(
            {name: arm.status for name, arm in result.arms.items()},
            {arm.name: "skipped" for arm in kernel_scenario_by_name("swiglu").arms},
        )
        # The request is recorded, not the zero replicates that ran.
        self.assertEqual(result.replicates, 2)
        self.assertTrue(
            any("no arm was timed" in warning for warning in result.warnings)
        )

    def test_anchor_loss_writes_no_results(self) -> None:
        """Every comparison is a ratio against the anchor; without it there is
        no coherent table to write."""
        anchor = kernel_scenario_by_name("swiglu").baseline_arm
        fake_process = fragment_writer(
            skip_timing=(("swiglu", anchor, 1),)
        )

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            out_dir = Path(temporary) / "kernels"
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu",),
                    replicates=2,
                    out_dir=out_dir,
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
            self.assertFalse((out_dir / "results.json").exists())

        self.assertTrue(outcomes[0].failed)
        self.assertIsNone(outcomes[0].result)
        self.assertIn("anchor arm", outcomes[0].error)
        self.assertEqual(outcomes[0].failed_passes, (f"{anchor} r1",))

    def test_one_lost_arm_keeps_the_others_and_still_fails(self) -> None:
        scenario = kernel_scenario_by_name("swiglu")
        lost = scenario.arms[-1].name
        fake_process = fragment_writer(skip_timing=(("swiglu", lost, 0),))

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("swiglu",),
                    replicates=2,
                    out_dir=Path(temporary) / "kernels",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        result = outcomes[0].result
        self.assertIsNotNone(result)
        self.assertEqual(result.arms[lost].status, "failed")
        self.assertEqual(result.arms[lost].modes, {})
        self.assertIn("1 of 2 replicates", result.arms[lost].status_reason)
        self.assertEqual(result.arms[scenario.baseline_arm].status, "ok")
        self.assertTrue(
            any(lost in warning for warning in result.warnings), result.warnings
        )
        # Reported, not silently absorbed: a partial roster exits nonzero.
        self.assertTrue(outcomes[0].failed)
        self.assertEqual(outcomes[0].failed_passes, (f"{lost} r0",))

    def test_replicate_boundaries_survive_the_merge(self) -> None:
        """The merge orders fragments by replicate index, not by arrival."""
        fake_process = fragment_writer()

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("qkv",),
                    replicates=3,
                    out_dir=Path(temporary) / "kernels",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        result = outcomes[0].result
        anchor = result.arms[kernel_scenario_by_name("qkv").baseline_arm]
        forward = anchor.modes["forward"]
        self.assertEqual(len(forward.replicates_us), 3)
        # timing_fragment offsets each replicate's samples by its index.
        self.assertEqual(
            [replicate[0] for replicate in forward.replicates_us],
            [10.0, 11.0, 12.0],
        )
        self.assertEqual(len(forward.samples_us), 9)
        # Measured once per arm, by replicate 0's worker.
        self.assertEqual(anchor.peak_memory_gib, 1.5)


if __name__ == "__main__":
    unittest.main()
