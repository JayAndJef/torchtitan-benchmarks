"""CPU-only tests for benchmark scenario construction and validation helpers."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.runner import (
    command_for_arm,
    trace_files,
    validate_arm,
    write_manifest,
)
from benchmarks.scenarios import (
    PIPER_1B_ROPE,
    PIPER_1B_SWIGLU,
    PIPER_1B_WORKLOAD,
    scenario_by_name,
)


class ScenarioTests(unittest.TestCase):
    def test_piper_swiglu_uses_only_grouped_experts_override(self) -> None:
        scenario = scenario_by_name("piper1b_swiglu")
        self.assertEqual(
            [arm.name for arm in scenario.arms], ["baseline", "fused_grouped_experts"]
        )
        fused = scenario.arm("fused_grouped_experts")
        self.assertEqual(
            fused.override_imports,
            ("torchtitan.overrides.fused_swiglu.fused_grouped_experts",),
        )
        self.assertEqual(fused.expected_override_count, 16)

    def test_both_scenarios_use_the_fixed_piper_workload(self) -> None:
        for scenario in (PIPER_1B_ROPE, PIPER_1B_SWIGLU):
            self.assertIs(scenario.workload, PIPER_1B_WORKLOAD)
        self.assertEqual(PIPER_1B_WORKLOAD.module, "piper1b")
        self.assertEqual(PIPER_1B_WORKLOAD.config, "qwen3_piper_1b")
        self.assertEqual(PIPER_1B_WORKLOAD.local_batch_size, 4)
        self.assertEqual(PIPER_1B_WORKLOAD.seq_len, 1024)
        self.assertEqual(PIPER_1B_WORKLOAD.steps, 40)

    def test_both_scenarios_declare_the_piper_regions(self) -> None:
        for scenario in (PIPER_1B_ROPE, PIPER_1B_SWIGLU):
            self.assertEqual(
                [(r.name, r.phase, r.invocations_per_window) for r in scenario.regions],
                [("backward_block", "backward", 80), ("forward_block", "forward", 80)],
            )


class CommandTests(unittest.TestCase):
    def test_command_adds_only_the_arm_override_and_dump_folder(self) -> None:
        arm = PIPER_1B_SWIGLU.arm("fused_grouped_experts")
        command = command_for_arm(
            PIPER_1B_SWIGLU.workload, arm, Path("/out/fused"), ["--debug.seed", "42"]
        )
        override_index = command.index("--override.imports")
        self.assertEqual(
            command[override_index + 1],
            "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
        )
        self.assertNotIn("torchtitan.overrides.fused_swiglu.fused_swiglu", command)
        self.assertEqual(command[-2:], ["--dump-folder", "/out/fused"])
        self.assertIn("--debug.seed", command)

    def test_baseline_command_has_no_override(self) -> None:
        command = command_for_arm(
            PIPER_1B_ROPE.workload, PIPER_1B_ROPE.arm("baseline"), Path("/out/baseline"), []
        )
        self.assertNotIn("--override.imports", command)
        self.assertEqual(command[:5], ["./run_train.sh", "--module", "piper1b",
                                       "--config", "qwen3_piper_1b"])
        self.assertIn("--compile.enable", command)
        self.assertIn("--profiler.enable_profiling", command)

    def test_each_arm_gets_its_own_dump_folder(self) -> None:
        for scenario in (PIPER_1B_ROPE, PIPER_1B_SWIGLU):
            for arm in scenario.arms:
                command = command_for_arm(
                    scenario.workload, arm, Path("/out") / arm.name, []
                )
                self.assertEqual(command[-1], f"/out/{arm.name}")


class ManifestTests(unittest.TestCase):
    def test_manifest_records_run_configuration(self) -> None:
        scenario = scenario_by_name("piper1b_swiglu")
        selected = (scenario.arm("fused_grouped_experts"),)
        extra_args = ["--debug.seed", "42"]
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            commands = {
                arm.name: command_for_arm(
                    scenario.workload, arm, out_dir / arm.name, extra_args
                )
                for arm in selected
            }
            metadata = {"requested_gpu": "3", "nvidia_smi": "3, RTX A6000, uuid, 550"}
            write_manifest(
                out_dir, scenario, selected, commands, "rtx-a6000", metadata, extra_args
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        self.assertEqual(manifest["scenario"], "piper1b_swiglu")
        self.assertEqual(manifest["hardware"], "rtx-a6000")
        self.assertEqual(manifest["hardware_metadata"], metadata)
        self.assertEqual(manifest["workload"]["local_batch_size"], 4)
        self.assertEqual(manifest["workload"]["seq_len"], 1024)
        self.assertEqual(manifest["selected_arms"], ["fused_grouped_experts"])
        self.assertEqual(
            [region["name"] for region in manifest["regions"]],
            ["backward_block", "forward_block"],
        )
        self.assertEqual(manifest["extra_torchtitan_args"], extra_args)
        fused_command = manifest["commands"]["fused_grouped_experts"]
        self.assertIn(
            "torchtitan.overrides.fused_swiglu.fused_grouped_experts", fused_command
        )
        self.assertIn("--debug.seed", fused_command)
        self.assertEqual(fused_command[-2], "--dump-folder")


class ValidationTests(unittest.TestCase):
    def test_validation_requires_completion_overrides_and_trace_windows(self) -> None:
        arm = PIPER_1B_SWIGLU.arm("fused_grouped_experts")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for iteration in ("iteration_20", "iteration_40"):
                trace = root / "profiling" / "traces" / iteration / "rank0_trace.json.gz"
                trace.parent.mkdir(parents=True, exist_ok=True)
                trace.touch()
            # Mirrors torchtitan's log format: "[Override] <import path>: <fqn> ..."
            applied = (
                "[Override] torchtitan.overrides.fused_swiglu.fused_grouped_experts: "
                "model_spec.model.layers.0.moe ...\n"
            )
            log = root / "fused_grouped_experts.log"
            log.write_text("Training completed\n" + applied * 16)
            self.assertEqual(len(trace_files(root)), 2)
            validate_arm(arm, root, log, PIPER_1B_SWIGLU.workload)

            log.write_text("Training completed\n" + applied * 15)
            with self.assertRaisesRegex(RuntimeError, "expected 16 override"):
                validate_arm(arm, root, log, PIPER_1B_SWIGLU.workload)

            log.write_text(
                "Training completed\n"
                + "[Override] torchtitan.overrides.other.thing: fqn ...\n" * 16
            )
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(arm, root, log, PIPER_1B_SWIGLU.workload)


if __name__ == "__main__":
    unittest.main()
