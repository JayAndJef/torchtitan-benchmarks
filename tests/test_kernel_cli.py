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
from benchmarks.kernel.runner import (
    KernelRunRequest,
    execute_kernel_run,
    resolve_arm_skips,
)
from benchmarks.kernel.schema import (
    CORRECTNESS_FRAGMENT_KIND,
    CorrectnessCheck,
    KernelArm,
    KernelScenario,
    TIMING_FRAGMENT_KIND,
    resolve_shape_and_workload,
    timing_fragment_path,
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


def timing_fragment(
    scenario: str,
    arm: str,
    replicate: int,
    measured: bool = True,
    bytes_moved: int | None = None,
    scale: float = 1.0,
) -> dict:
    """A worker's output, shaped exactly as ``run_timing_pass`` writes it.

    ``measured=False`` is the arm that ran and timed nothing: the worker
    completed and wrote its fragment, and every mode in it is empty.

    ``bytes_moved`` is what the parent turns into the GB/s column. Only a
    bandwidth floor and its neighbours report it, so it defaults to None as
    most arms write it.
    ``scale`` multiplies the samples, which is how one arm is given a
    different median from its neighbours.
    """
    declaration = kernel_scenario_by_name(scenario).arm(arm)
    return {
        "kind": TIMING_FRAGMENT_KIND,
        "scenario": scenario,
        "arm": arm,
        "replicate": replicate,
        # Distinct per replicate, so a merge that scrambled the boundaries
        # would produce different statistics rather than the same ones.
        "modes": {
            mode: [(10.0 + replicate + index) * scale for index in range(3)]
            for mode in declaration.modes
        }
        if measured
        else {},
        "bytes_moved": bytes_moved,
        "peak_memory_gib": 1.5 if replicate == 0 else None,
        "burst_us_per_call": None,
    }


def fragment_writer(
    *,
    correctness_passes: bool = True,
    correctness_code: dict[str, int] | None = None,
    correctness_exit: int | None = None,
    timing_exit: int = 0,
    skip_timing: tuple[tuple[str, str, int], ...] = (),
    empty_arms: tuple[str, ...] = (),
    bytes_moved: int | None = None,
    arm_scale: dict[str, float] | None = None,
):
    """A ``process_runner`` that plays the worker protocol.

    ``correctness_code`` maps a scenario to a return code whose worker also
    writes no fragment, which is how a build failure looks to the parent.
    ``skip_timing`` names (scenario, arm, replicate) triples whose timing
    worker crashes the same way.

    ``correctness_exit`` and ``timing_exit`` decouple the exit code from the
    fragment, which is the real worker's window: it writes the fragment first
    and computes the code afterwards, so a death in between reports one
    verdict in the code and the opposite one on disk.

    ``empty_arms`` names arms whose workers all succeed and time nothing.
    ``bytes_moved`` is reported by every arm, which is what the GB/s column
    is derived from. ``arm_scale`` multiplies one named arm's samples, so a
    scenario can carry arms with different medians.
    """
    scales = arm_scale or {}
    codes = correctness_code or {}

    def fake_process(command, **kwargs):
        scenario = command[command.index("--scenario") + 1]
        mode = command[command.index("--mode") + 1]
        if mode == "correctness":
            fragment = Path(command[command.index("--fragment") + 1])
            if scenario in codes:
                kwargs["stdout"].write("boom: build failed\n")
                return SimpleNamespace(returncode=codes[scenario])
            fragment.write_text(
                json.dumps(correctness_fragment(scenario, correctness_passes))
            )
            if correctness_exit is not None:
                return SimpleNamespace(returncode=correctness_exit)
            return SimpleNamespace(returncode=0 if correctness_passes else 3)
        fragments_dir = Path(command[command.index("--fragments-dir") + 1])
        arm = command[command.index("--arm") + 1]
        first = int(command[command.index("--replicate") + 1])
        count = (
            int(command[command.index("--replicate-count") + 1])
            if "--replicate-count" in command
            else 1
        )
        # A batched worker writes one fragment per replicate it covered, and
        # a crashed one writes none of them -- so a skip_timing entry
        # anywhere in the block costs the whole block, exactly as a real
        # worker's traceback does.
        block = range(first, first + count)
        if any((scenario, arm, replicate) in skip_timing for replicate in block):
            kwargs["stdout"].write(f"boom: {arm} r{first}+{count}\n")
            return SimpleNamespace(returncode=1)
        for replicate in block:
            timing_fragment_path(fragments_dir, arm, replicate).write_text(
                json.dumps(
                    timing_fragment(
                        scenario,
                        arm,
                        replicate,
                        measured=arm not in empty_arms,
                        bytes_moved=bytes_moved,
                        scale=scales.get(arm, 1.0),
                    )
                )
            )
        return SimpleNamespace(returncode=timing_exit)

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
                    "expert_mlp",
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
        self.assertEqual(request.scenario_names, ("expert_mlp",))
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

    def test_model_size_defaults_to_1b_and_rejects_unknown(self) -> None:
        with mock.patch(
            "benchmarks.cli.kernel.execute_kernel_run", return_value=()
        ) as execute:
            result = self.runner.invoke(cli, ["kernel-bench", "7"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].model_size, "1b")

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

    def test_the_three_counts_refuse_a_zero(self) -> None:
        """Each names a repetition the run cannot have none of, and each used
        to reach the GPU and fail there, or not fail at all."""
        for flag in ("--replicates", "--samples-per-replicate", "--burst-k"):
            with self.subTest(flag=flag):
                result = self.runner.invoke(
                    cli, ["kernel-bench", "7", flag, "0"]
                )
                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("is not in the range", result.output)

    def test_out_requires_a_single_scenario(self) -> None:
        result = self.runner.invoke(
            cli, ["kernel-bench", "7", "--out", "/tmp/kernels"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--out requires exactly one --scenario", result.output)

    def test_arm_selection_reaches_the_request(self) -> None:
        with mock.patch(
            "benchmarks.cli.kernel.execute_kernel_run", return_value=()
        ) as execute:
            result = self.runner.invoke(
                cli,
                [
                    "kernel-bench",
                    "7",
                    "--scenario",
                    "attention_core",
                    "--arm",
                    "mcore/base",
                    "--arm",
                    "titan",
                ],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            execute.call_args.args[0].arm_names, ("mcore/base", "titan")
        )

        # Omitted means every arm, exactly as before the flag existed.
        with mock.patch(
            "benchmarks.cli.kernel.execute_kernel_run", return_value=()
        ) as execute:
            result = self.runner.invoke(cli, ["kernel-bench", "7"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].arm_names, ())

    def test_arm_requires_a_single_scenario(self) -> None:
        """An arm name belongs to one roster, so a selection across scenarios
        would mean a different thing in each of them."""
        result = self.runner.invoke(
            cli, ["kernel-bench", "7", "--arm", "mcore/base"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--arm requires exactly one --scenario", result.output)

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
                    scenario_names=("rope", "expert_mlp"),
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
        expert_mlp_index = next(
            index
            for index, (kind, message) in enumerate(kinds)
            if kind == "arm" and "expert_mlp" in message
        )
        self.assertLess(error_index, expert_mlp_index)
        self.assertIn("boom", kinds[error_index][1])

    def test_a_skip_closes_over_correctness_references(self) -> None:
        """An arm whose reference is skipped is skipped too. Timing an arm
        that nothing checked is the silent wrongness the gates exist for.

        No scenario reaches this today. The only arm with a requirement is
        rope's ``titan/te``, and it is a referrer -- it points at
        ``mcore/base``, which needs no compiler -- never a reference. So the
        closure is guarding the roster's growth, and the synthetic scenario
        is how it gets exercised at all.
        """

        def arm(name, requires_gcc=False, reference=None):
            return KernelArm(
                name=name,
                description=name,
                builder=f"benchmarks.kernel.operations.rope:build_{name}",
                modes=("forward",),
                requires_gcc_toolset=requires_gcc,
                eager_reason="a stub, not an implementation",
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
        shape, workload = resolve_shape_and_workload()
        self.assertEqual(
            resolve_arm_skips(
                scenario,
                compiler_unavailable=None,
                shape=shape,
                workload=workload,
            ),
            {},
        )
        skipped = resolve_arm_skips(
            scenario,
            compiler_unavailable="no gcc",
            shape=shape,
            workload=workload,
        )
        self.assertEqual(
            sorted(skipped), ["gated_on_it", "gated_on_that", "needs_compiler"]
        )
        self.assertEqual(skipped["needs_compiler"], "no gcc")
        self.assertIn("'needs_compiler' is skipped", skipped["gated_on_it"])
        self.assertIn("'gated_on_it' is skipped", skipped["gated_on_that"])

    def test_a_broken_compiler_env_costs_the_te_arm_only(self) -> None:
        """The requirement belongs to the arm. Rope still measures its other
        four arms, which the former scenario-level check threw away."""
        fake_process = fragment_writer()

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch, mock.patch(
            "benchmarks.kernel.runner.BENCH_DIR", Path(temporary)
        ):
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("rope", "expert_mlp"),
                    replicates=2,
                    timestamp="stamp",
                    compiler_env=Path(temporary) / "missing.sh",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        by_name = {outcome.scenario: outcome for outcome in outcomes}
        self.assertFalse(by_name["rope"].failed)
        self.assertFalse(by_name["expert_mlp"].failed)
        arms = by_name["rope"].result.arms
        self.assertEqual(arms["titan/te"].status, "skipped")
        self.assertIn("compiler environment", arms["titan/te"].status_reason)
        self.assertEqual(arms["titan/te"].modes, {})
        for name in ("mcore/base", "mcore/no_rope_fusion", "titan", "titan/helion"):
            self.assertEqual(arms[name].status, "ok", name)

    def test_a_skipped_arm_is_spawned_in_neither_pass(self) -> None:
        commands = []

        writer = fragment_writer()

        def fake_process(command, **kwargs):
            commands.append(command)
            return writer(command, **kwargs)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            out_dir = Path(temporary) / "kernels"
            execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("rope",),
                    replicates=2,
                    timestamp="stamp",
                    compiler_env=Path(temporary) / "missing.sh",
                    out_dir=out_dir,
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertEqual(commands[0][commands[0].index("--mode") + 1], "correctness")
        self.assertEqual(
            commands[0][commands[0].index("--skip-arm") + 1], "titan/te"
        )
        timed = {
            command[command.index("--arm") + 1] for command in commands[1:]
        }
        self.assertEqual(
            timed,
            {"mcore/base", "mcore/no_rope_fusion", "titan", "titan/helion"},
        )

        # The manifest says so on its own. "arms" is the registry roster and
        # still lists titan/te, so without this a reader must diff it against
        # "commands" to learn that the arm never ran.
        self.assertEqual(manifest["schema_version"], 7)
        self.assertEqual(manifest["unit_kind"], "scenario")
        self.assertEqual(list(manifest["skipped_arms"]), ["titan/te"])
        self.assertIn(
            "compiler environment", manifest["skipped_arms"]["titan/te"]
        )
        self.assertIn("titan/te", [arm["name"] for arm in manifest["arms"]])

    def test_an_unselected_arm_is_spawned_in_neither_pass(self) -> None:
        """``--arm`` rides the same skip machinery a missing requirement
        does, so one selection reaches both passes: the correctness worker is
        told which arms it must not gate, and no timing worker is spawned for
        them in any replicate."""
        commands = []
        writer = fragment_writer()

        def fake_process(command, **kwargs):
            commands.append(command)
            return writer(command, **kwargs)

        unselected = [
            "mcore/attn_flash3",
            "mcore/attn_unfused",
            "titan/flex_flash",
            "titan/flash_attention_3",
        ]
        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            out_dir = Path(temporary) / "kernels"
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("attention_core",),
                    arm_names=("mcore/base", "titan"),
                    replicates=2,
                    timestamp="stamp",
                    out_dir=out_dir,
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        correctness = commands[0]
        self.assertEqual(
            correctness[correctness.index("--mode") + 1], "correctness"
        )
        self.assertEqual(
            [
                correctness[index + 1]
                for index, token in enumerate(correctness)
                if token == "--skip-arm"
            ],
            unselected,
        )
        # Two replicates of the two selected arms, and no other worker.
        self.assertEqual(
            [
                (
                    command[command.index("--arm") + 1],
                    command[command.index("--replicate") + 1],
                )
                for command in commands[1:]
            ],
            [
                ("mcore/base", "0"),
                ("titan", "0"),
                ("mcore/base", "1"),
                ("titan", "1"),
            ],
        )

        # The reason names the flag. An arm nobody asked for and an arm this
        # host cannot run are different facts, and both land in the same
        # field.
        self.assertFalse(outcomes[0].failed)
        arms = outcomes[0].result.arms
        for name in unselected:
            self.assertEqual(arms[name].status, "skipped", name)
            self.assertIn("--arm did not select it", arms[name].status_reason)
        for name in ("mcore/base", "titan"):
            self.assertEqual(arms[name].status, "ok", name)
        self.assertEqual(list(manifest["skipped_arms"]), unselected)

    def test_a_selection_this_scenario_cannot_honour_is_refused(self) -> None:
        """Three refusals, each naming what is missing. Repairing any of them
        would measure a roster the operator did not ask for, and dropping the
        arm instead would publish a table with no ratios or an ungated
        number."""
        cases = (
            (
                "attention_core",
                ("mcore/base", "titn"),
                ("no such arm: titn", "Available arms: mcore/base"),
            ),
            (
                "attention_core",
                ("titan",),
                ("must include the anchor arm 'mcore/base'",),
            ),
            (
                "expert_mlp",
                ("titan", "titan/piper_optimized_triton"),
                (
                    "correctness reference 'titan/fused_grouped_experts'",
                    "add --arm titan/fused_grouped_experts",
                ),
            ),
        )
        shape, workload = resolve_shape_and_workload()
        for scenario_name, selection, expected in cases:
            with self.subTest(scenario=scenario_name, selection=selection):
                with self.assertRaises(ValueError) as raised:
                    resolve_arm_skips(
                        kernel_scenario_by_name(scenario_name),
                        compiler_unavailable=None,
                        shape=shape,
                        workload=workload,
                        selected=selection,
                    )
                for phrase in expected:
                    self.assertIn(phrase, str(raised.exception))

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
                    scenario_names=("expert_mlp",),
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
        self.assertEqual(command[command.index("--model-size") + 1], "1b")
        self.assertNotIn("--batch", command)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "7")
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

        # One correctness pass, then replicates x arms timing passes, in
        # replicate-major order.
        arms = [arm.name for arm in kernel_scenario_by_name("expert_mlp").arms]
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

        # Schema 7 renames the value: two kinds of unit write a manifest,
        # and "kernel" named the family and one member of it at once.
        self.assertEqual(manifest["kind"], "kernel_scenario")
        self.assertEqual(manifest["schema_version"], 7)
        self.assertIsNone(manifest["span_scenarios"])
        self.assertEqual(manifest["skipped_arms"], {})
        self.assertEqual(manifest["scenario"], "expert_mlp")
        # And no span name in it. The two names have separate fields, so a
        # reader of either never has to ask which roster it belongs to.
        self.assertIsNone(manifest["span"])
        self.assertEqual(manifest["hardware_metadata"]["cpu_pinning"], "numactl test")
        self.assertNotIn("spec", manifest)
        self.assertEqual(manifest["model_size"], "1b")
        self.assertEqual(manifest["model_shape"]["dim"], 1024)
        self.assertEqual(manifest["model_shape"]["n_layers"], 16)
        self.assertEqual(manifest["workload"], {"batch": 4, "seq_len": 1024})
        self.assertEqual(manifest["shapes"]["x"], [8192, 1024])
        self.assertEqual(manifest["replicates"], 2)
        self.assertEqual(manifest["replicates_per_process"], 1)
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

    def test_replicates_per_process_batches_without_mixing_arms(self) -> None:
        """A batched sweep spawns one worker per (arm, block) and still reads
        every replicate back.

        The two properties that must survive the batch: no worker names two
        arms -- the isolation that keeps one arm's dependencies out of
        another's interpreter -- and the outer loop is still the block, so
        every arm is measured once before any arm is measured again.
        """
        captured = []

        def recording(command, **kwargs):
            captured.append(list(command))
            return fragment_writer()(command, **kwargs)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            out_dir = Path(temporary) / "kernels"
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("lm_head_projection",),
                    replicates=5,
                    replicates_per_process=2,
                    out_dir=out_dir,
                ),
                process_runner=recording,
                environment={"PATH": "/usr/bin"},
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        arms = [
            arm.name
            for arm in kernel_scenario_by_name("lm_head_projection").arms
        ]
        timing = [
            (
                argv[argv.index("--arm") + 1],
                int(argv[argv.index("--replicate") + 1]),
                int(argv[argv.index("--replicate-count") + 1])
                if "--replicate-count" in argv
                else 1,
            )
            for argv in captured[1:]
        ]
        # Blocks of two, the last one short, and the arms alternate inside
        # each block rather than one arm running its whole ladder first.
        self.assertEqual(
            timing,
            [
                (arm, first, count)
                for first, count in ((0, 2), (2, 2), (4, 1))
                for arm in arms
            ],
        )
        for argv in captured[1:]:
            self.assertEqual(argv.count("--arm"), 1)
        self.assertEqual(manifest["replicates_per_process"], 2)

        result = outcomes[0].result
        self.assertFalse(outcomes[0].failed)
        for arm in arms:
            self.assertEqual(result.arms[arm].status, "ok")
            forward = result.arms[arm].modes["forward"]
            self.assertEqual(len(forward.replicates_us), 5)
        self.assertEqual(result.methodology["replicates_per_process"], 2)
        self.assertEqual(
            result.methodology["arm_isolation"],
            "one_process_per_arm_replicate_block",
        )

    def test_a_batched_worker_that_dies_costs_every_replicate_it_owed(
        self,
    ) -> None:
        """The merge counts replicates, not workers.

        One crashed batched worker takes several replicates with it, and each
        missing one is named: an arm short of a replicate is incomplete
        however few processes lost it.
        """
        fake_process = fragment_writer(
            skip_timing=(("lm_head_projection", "titan", 3),)
        )
        events = []
        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("lm_head_projection",),
                    replicates=4,
                    replicates_per_process=2,
                    out_dir=Path(temporary) / "kernels",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
                event_handler=events.append,
            )
        self.assertTrue(outcomes[0].failed)
        self.assertEqual(
            outcomes[0].failed_passes, ("titan r2", "titan r3")
        )
        self.assertEqual(outcomes[0].result.arms["titan"].status, "failed")
        self.assertEqual(
            outcomes[0].result.arms["mcore/base"].status, "ok"
        )
        # Each one is also named as it happens. The outcome above is what the
        # merge reads; this is what the operator watching a long run sees, and
        # the docstring's claim is about the second.
        lost = [
            event.message
            for event in events
            if "wrote no fragment" in event.message
        ]
        self.assertEqual(len(lost), 2)
        self.assertTrue(any("titan r2" in message for message in lost))
        self.assertTrue(any("titan r3" in message for message in lost))

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
            self.assertEqual(run(("expert_mlp",), temporary, compiler), 0)
            self.assertEqual(run(("expert_mlp", "rope"), temporary, compiler), 1)

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
                    scenario_names=("expert_mlp", "rope"),
                    replicates=2,
                    timestamp="stamp",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        by_name = {outcome.scenario: outcome for outcome in outcomes}
        self.assertFalse(by_name["expert_mlp"].failed)
        self.assertFalse(by_name["rope"].failed)
        arms = by_name["rope"].result.arms
        self.assertEqual(arms["titan/te"].status, "skipped")
        self.assertIn("C++20 host compiler", arms["titan/te"].status_reason)
        self.assertEqual(arms["titan"].status, "ok")

    def test_unbalanced_routing_skips_only_expert_mlp(self) -> None:
        """An odd batch x seq_len breaks expert_mlp's balanced split (rows =
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
                    scenario_names=("expert_mlp", "rope"),
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
        self.assertTrue(by_name["expert_mlp"].failed)
        self.assertIn(
            "6150 routed rows (batch 3 x seq 1025 x top_k 2) do not divide "
            "evenly among 4 experts",
            by_name["expert_mlp"].error,
        )
        self.assertIsNone(by_name["expert_mlp"].result)
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
        self.assertFalse(by_name["expert_mlp"].out_dir.exists())

    def test_worker_crash_surfaces_the_log_tail(self) -> None:
        fake_process = fragment_writer(correctness_code={"expert_mlp": 1})

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("expert_mlp",),
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
                    scenario_names=("expert_mlp",),
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
            {arm.name: "skipped" for arm in kernel_scenario_by_name("expert_mlp").arms},
        )
        # The request is recorded, not the zero replicates that ran.
        self.assertEqual(result.replicates, 2)
        self.assertTrue(
            any("no arm was timed" in warning for warning in result.warnings)
        )

    def test_a_failed_gate_fragment_outranks_a_clean_exit_code(self) -> None:
        """The verdict is the fragment as well as the code.

        The worker writes the fragment before it computes the code, so a
        signal or an OOM kill in that window leaves a recorded failure beside
        an exit code that is not 3. Reading the code alone timed every arm and
        published the numbers next to ``all_correctness_passed: false``.
        """
        seen = []
        writer = fragment_writer(correctness_passes=False, correctness_exit=0)

        def fake_process(command, **kwargs):
            seen.append(command[command.index("--mode") + 1])
            return writer(command, **kwargs)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("expert_mlp",),
                    replicates=2,
                    out_dir=Path(temporary) / "kernels",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        self.assertEqual(seen, ["correctness"])
        self.assertTrue(outcomes[0].correctness_failed)
        self.assertTrue(outcomes[0].failed)
        self.assertFalse(outcomes[0].result.all_correctness_passed)

    def test_a_timing_worker_that_dies_after_writing_is_reported(self) -> None:
        """Its samples stand, and its exit code is still said out loud."""
        events = []
        fake_process = fragment_writer(timing_exit=9)

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("lm_head_projection",),
                    replicates=2,
                    out_dir=Path(temporary) / "kernels",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
                event_handler=events.append,
            )
        self.assertFalse(outcomes[0].failed)
        self.assertEqual(
            {name: arm.status for name, arm in outcomes[0].result.arms.items()},
            {"mcore/base": "ok", "titan": "ok"},
        )
        self.assertTrue(
            any(
                "wrote its fragments and then exited with 9" in event.message
                for event in events
            ),
            [event.message for event in events],
        )

    def test_anchor_loss_writes_no_results(self) -> None:
        """Every comparison is a ratio against the anchor; without it there is
        no coherent table to write."""
        anchor = kernel_scenario_by_name("expert_mlp").baseline_arm
        fake_process = fragment_writer(
            skip_timing=(("expert_mlp", anchor, 1),)
        )

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            out_dir = Path(temporary) / "kernels"
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("expert_mlp",),
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
        scenario = kernel_scenario_by_name("expert_mlp")
        lost = scenario.arms[-1].name
        fake_process = fragment_writer(skip_timing=(("expert_mlp", lost, 0),))

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("expert_mlp",),
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

    def test_an_arm_that_measured_nothing_is_not_published_as_ok(self) -> None:
        """Every replicate wrote a fragment, and every fragment was empty.

        ``status`` is a dataclass default, so this arm used to reach the file
        as ``ok`` with no modes under it -- and the reporter tabulates mode by
        mode, so it appeared in no table and in no unmeasured list. It
        vanished, and the scenario exited 0.
        """
        scenario = kernel_scenario_by_name("expert_mlp")
        empty = scenario.arms[-1].name
        fake_process = fragment_writer(empty_arms=(empty,))

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("expert_mlp",),
                    replicates=2,
                    out_dir=Path(temporary) / "kernels",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        result = outcomes[0].result
        self.assertEqual(result.arms[empty].status, "failed")
        self.assertEqual(result.arms[empty].modes, {})
        self.assertIn("no samples", result.arms[empty].status_reason)
        self.assertEqual(result.arms[scenario.baseline_arm].status, "ok")
        self.assertTrue(
            any(empty in warning for warning in result.warnings), result.warnings
        )
        self.assertTrue(outcomes[0].failed)

    def test_an_empty_anchor_writes_no_results(self) -> None:
        """The same failure one level up. Every comparison is a ratio against
        the anchor, so an anchor that measured nothing has no table."""
        anchor = kernel_scenario_by_name("expert_mlp").baseline_arm
        fake_process = fragment_writer(empty_arms=(anchor,))

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            out_dir = Path(temporary) / "kernels"
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("expert_mlp",),
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
        self.assertIn("no samples", outcomes[0].error)

    def test_the_derived_columns_are_computed_from_the_registry(self) -> None:
        """GB/s and x-floor, the two numbers no worker can produce.

        Only the parent holds a second arm, so the floor ratio is its work;
        and since ``is_floor`` moved from ``BuiltArm`` to the registry, the
        parent reads it from the declaration rather than from the fragment.
        Deleting the whole derived block left the suite green: the fixture
        reported no ``bytes_moved``, so GB/s was never computed at all.

        Driven against ``qk_norm``. This test ran on ``rope`` until scenario 4
        made that scenario cross-engine and retired its ``copy_floor``: a
        bandwidth floor answers no cross-engine question. The x_floor column
        is derived from a floor, so the test has to sit on a scenario that
        still declares one. ``qk_norm`` does, and it has the same shape --
        one floor and two arms above it.
        """
        fake_process = fragment_writer(
            bytes_moved=4_000_000, arm_scale={"copy_floor": 0.5}
        )

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("qk_norm",),
                    replicates=2,
                    out_dir=Path(temporary) / "kernels",
                    compiler_env=Path(temporary) / "missing.sh",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        arms = outcomes[0].result.arms
        # Samples are 10..13 over two replicates, so the pooled median is
        # 11.5 us; copy_floor's are halved, so its median is 5.75 us.
        anchor = arms["mcore/base"].modes["forward"].derived
        self.assertAlmostEqual(anchor["gbps"], 4e6 / 11.5e-6 / 1e9)
        self.assertAlmostEqual(anchor["x_floor"], 2.0)
        self.assertAlmostEqual(arms["titan"].modes["forward"].derived["x_floor"], 2.0)
        # The floor is the denominator, so it carries no ratio to itself.
        floor = arms["copy_floor"].modes["forward"].derived
        self.assertNotIn("x_floor", floor)
        self.assertAlmostEqual(floor["gbps"], 4e6 / 5.75e-6 / 1e9)

    def test_replicate_boundaries_survive_the_merge(self) -> None:
        """The merge orders fragments by replicate index, not by arrival."""
        fake_process = fragment_writer()

        metadata_patch, pinning_patch = patched_environment()
        with tempfile.TemporaryDirectory() as temporary, metadata_patch, pinning_patch:
            outcomes = execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=("lm_head_projection",),
                    replicates=3,
                    out_dir=Path(temporary) / "kernels",
                ),
                process_runner=fake_process,
                environment={"PATH": "/usr/bin"},
            )
        result = outcomes[0].result
        anchor = result.arms[
            kernel_scenario_by_name("lm_head_projection").baseline_arm
        ]
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
