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
from analysis.compare_arms import stable_tps, training_metrics
from benchmarks.scenarios import (
    PIPER_1B_LM_HEAD,
    PIPER_1B_QKV,
    PIPER_1B_ROPE,
    PIPER_1B_SWIGLU,
    PIPER_1B_WORKLOAD,
    scenario_by_name,
)
from piper1b.config_registry import (
    qwen3_piper_1b,
    qwen3_piper_1b_full_logits,
    qwen3_piper_1b_fused_linear_ce,
    qwen3_piper_1b_fused_linear_ce_full,
    qwen3_piper_1b_te_fused_ce,
    qwen3_piper_1b_te_fused_ce_full,
    qwen3_piper_1b_unfused_qkv,
)
from piper1b.fused_losses import FusedLinearCrossEntropyLoss, TECrossEntropyLoss
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.models.common import FusedQKVLinear, QKVLinear


class ScenarioTests(unittest.TestCase):
    def test_piper_lm_head_has_six_explicit_configs(self) -> None:
        scenario = scenario_by_name("piper1b_lm_head")
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            [
                "baseline",
                "chunked",
                "fused_linear_ce",
                "fused_linear_ce_full",
                "te_fused_ce",
                "te_fused_ce_full",
            ],
        )
        self.assertEqual(scenario.workload.config, "qwen3_piper_1b_full_logits")
        self.assertEqual(scenario.workload.seed, 42)
        self.assertEqual(
            [arm.config for arm in scenario.arms],
            [
                None,
                "qwen3_piper_1b",
                "qwen3_piper_1b_fused_linear_ce",
                "qwen3_piper_1b_fused_linear_ce_full",
                "qwen3_piper_1b_te_fused_ce",
                "qwen3_piper_1b_te_fused_ce_full",
            ],
        )
        self.assertEqual(
            scenario.arm("te_fused_ce").trace_kernel_markers,
            ("online_softmax_kernel", "cross_entropy_kernel"),
        )
        self.assertEqual(
            scenario.arm("te_fused_ce_full").trace_kernel_markers,
            ("online_softmax_kernel", "cross_entropy_kernel"),
        )

    def test_piper_lm_head_configs_use_expected_losses(self) -> None:
        full = qwen3_piper_1b_full_logits().loss
        chunked = qwen3_piper_1b().loss
        fused = qwen3_piper_1b_fused_linear_ce().loss
        fused_full = qwen3_piper_1b_fused_linear_ce_full().loss
        te = qwen3_piper_1b_te_fused_ce().loss
        te_full = qwen3_piper_1b_te_fused_ce_full().loss

        self.assertIsInstance(full, CrossEntropyLoss.Config)
        self.assertIsInstance(chunked, ChunkedLossWrapper.Config)
        self.assertEqual(chunked.num_chunks, 8)
        self.assertIsInstance(fused, FusedLinearCrossEntropyLoss.Config)
        self.assertEqual(fused.batch_chunk_size, 1024)
        self.assertIsNone(fused.chunking_method)
        self.assertIsInstance(fused_full, FusedLinearCrossEntropyLoss.Config)
        self.assertEqual(fused_full.num_chunks, 1)
        self.assertIsNone(fused_full.batch_chunk_size)
        self.assertIsNone(fused_full.chunking_method)
        self.assertIsInstance(te, ChunkedLossWrapper.Config)
        self.assertEqual(te.num_chunks, 8)
        self.assertIsInstance(te.loss_fn, TECrossEntropyLoss.Config)
        self.assertIsInstance(te_full, ChunkedLossWrapper.Config)
        self.assertEqual(te_full.num_chunks, 1)
        self.assertIsInstance(te_full.loss_fn, TECrossEntropyLoss.Config)

    def test_piper_qkv_compares_unfused_baseline_to_fused_config(self) -> None:
        scenario = scenario_by_name("piper1b_qkv")
        self.assertEqual([arm.name for arm in scenario.arms], ["baseline", "fused_qkv"])
        self.assertEqual(scenario.workload.config, "qwen3_piper_1b_unfused_qkv")
        self.assertEqual(scenario.workload.seed, 42)
        self.assertIsNone(scenario.arm("baseline").config)
        self.assertEqual(scenario.arm("fused_qkv").config, "qwen3_piper_1b")

    def test_piper_qkv_configs_use_expected_projection_types(self) -> None:
        fused = qwen3_piper_1b().model_spec.model.layers[0].attention.qkv_linear
        unfused = (
            qwen3_piper_1b_unfused_qkv()
            .model_spec.model.layers[0]
            .attention.qkv_linear
        )
        self.assertIsInstance(fused, FusedQKVLinear.Config)
        self.assertIsInstance(unfused, QKVLinear.Config)

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

    def test_existing_scenarios_use_the_fixed_piper_workload(self) -> None:
        for scenario in (PIPER_1B_ROPE, PIPER_1B_SWIGLU):
            self.assertIs(scenario.workload, PIPER_1B_WORKLOAD)
        self.assertEqual(PIPER_1B_WORKLOAD.module, "piper1b")
        self.assertEqual(PIPER_1B_WORKLOAD.config, "qwen3_piper_1b")
        self.assertEqual(PIPER_1B_WORKLOAD.local_batch_size, 4)
        self.assertEqual(PIPER_1B_WORKLOAD.seq_len, 1024)
        self.assertEqual(PIPER_1B_WORKLOAD.steps, 40)

    def test_all_scenarios_declare_the_piper_regions(self) -> None:
        for scenario in (
            PIPER_1B_ROPE,
            PIPER_1B_SWIGLU,
            PIPER_1B_QKV,
            PIPER_1B_LM_HEAD,
        ):
            self.assertEqual(
                [(r.name, r.phase, r.invocations_per_window) for r in scenario.regions],
                [("backward_block", "backward", 80), ("forward_block", "forward", 80)],
            )


class CommandTests(unittest.TestCase):
    def test_qkv_arm_selects_fused_config(self) -> None:
        command = command_for_arm(
            PIPER_1B_QKV.workload,
            PIPER_1B_QKV.arm("fused_qkv"),
            Path("/out/fused_qkv"),
            [],
        )
        self.assertEqual(
            command[command.index("--config") + 1],
            "qwen3_piper_1b",
        )
        self.assertEqual(command[command.index("--debug.seed") + 1], "42")

        baseline = command_for_arm(
            PIPER_1B_QKV.workload,
            PIPER_1B_QKV.arm("baseline"),
            Path("/out/baseline"),
            [],
        )
        self.assertEqual(
            baseline[baseline.index("--config") + 1],
            "qwen3_piper_1b_unfused_qkv",
        )
        self.assertEqual(baseline[baseline.index("--debug.seed") + 1], "42")

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
        self.assertNotIn("--debug.seed", command)
        self.assertEqual(command[:5], ["./run_train.sh", "--module", "piper1b",
                                       "--config", "qwen3_piper_1b"])
        self.assertIn("--compile.enable", command)
        self.assertIn("--profiler.enable_profiling", command)

    def test_each_arm_gets_its_own_dump_folder(self) -> None:
        for scenario in (
            PIPER_1B_ROPE,
            PIPER_1B_SWIGLU,
            PIPER_1B_QKV,
            PIPER_1B_LM_HEAD,
        ):
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


class TrainingMetricsTests(unittest.TestCase):
    def test_stable_tps_excludes_compile_and_profiler_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "arm.log"
            log.write_text(
                "step:  1  loss: 1.0  memory: 10.00GiB(20%)  tps: 100\n"
                "step:  2  loss: 1.0  memory: 11.00GiB(22%)  tps: 9,900\n"
                "step: 10  loss: 1.0  memory: 12.00GiB(24%)  tps: 10,100\n"
                "step: 11  loss: 1.0  memory: 12.00GiB(24%)  tps: 8,000\n"
                "step: 21  loss: 1.0  memory: 12.00GiB(24%)  tps: 200\n"
                "step: 22  loss: 1.0  memory: 12.00GiB(24%)  tps: 10,000\n"
            )
            rows = training_metrics(log)

        self.assertEqual(
            stable_tps(
                rows,
                {
                    "profile_freq": 20,
                    "profiler_warmup": 5,
                    "profiler_active": 5,
                },
            ),
            [9900, 10100, 10000],
        )


if __name__ == "__main__":
    unittest.main()
