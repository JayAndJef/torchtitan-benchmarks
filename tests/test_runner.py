"""CPU-only tests for benchmark scenario construction and validation helpers."""

import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts import trace_files
from benchmarks.runner import (
    RunRequest,
    command_for_arm,
    execute_run,
    validate_arm,
    write_manifest,
)
from benchmarks.metrics import stable_tps, training_metrics
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
    qwen3_piper_1b_piper_optimized_te_ce,
    qwen3_piper_1b_te_fused_ce,
    qwen3_piper_1b_unfused_qkv,
)
from piper1b.lm_head.losses import (
    FusedLinearCrossEntropyLoss,
    PiperOptimizedCrossEntropyLoss,
    TECrossEntropyLoss,
)
from torchtitan.components.loss import (
    ChunkedLossWrapper,
    CrossEntropyLoss,
    LossWithLMHead,
)
from torchtitan.config import CompileConfig
from torchtitan.models.common import FusedQKVLinear, QKVLinear


PIPER_OPTIMIZED_SWIGLU_OVERRIDE = (
    "piper1b.swiglu.combined_swiglu.piper_optimized_fused_grouped_experts"
)


class ScenarioTests(unittest.TestCase):
    def test_piper_lm_head_has_four_full_token_configs(self) -> None:
        scenario = scenario_by_name("piper1b_lm_head")
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            [
                "baseline",
                "fused_linear_ce",
                "te_fused_ce",
                "piper_optimized_te_ce",
            ],
        )
        self.assertEqual(scenario.workload.config, "qwen3_piper_1b_full_logits")
        self.assertEqual(scenario.workload.seed, 42)
        self.assertEqual(
            [arm.config for arm in scenario.arms],
            [
                None,
                "qwen3_piper_1b_fused_linear_ce",
                "qwen3_piper_1b_te_fused_ce",
                "qwen3_piper_1b_piper_optimized_te_ce",
            ],
        )
        self.assertEqual(
            scenario.arm("te_fused_ce").trace_kernel_markers,
            ("online_softmax_kernel", "cross_entropy_kernel"),
        )
        self.assertEqual(
            scenario.arm("piper_optimized_te_ce").trace_kernel_markers,
            ("piper_optimized_cross_entropy_kernel",),
        )

    def test_piper_lm_head_configs_use_expected_losses(self) -> None:
        full = qwen3_piper_1b_full_logits().loss
        default = qwen3_piper_1b().loss
        fused = qwen3_piper_1b_fused_linear_ce().loss
        te = qwen3_piper_1b_te_fused_ce().loss
        optimized = qwen3_piper_1b_piper_optimized_te_ce().loss

        self.assertIsInstance(full, CrossEntropyLoss.Config)
        self.assertIsInstance(default, CrossEntropyLoss.Config)
        self.assertIsInstance(fused, FusedLinearCrossEntropyLoss.Config)
        self.assertIsNone(fused.batch_chunk_size)
        self.assertIsNone(fused.chunking_method)
        self.assertIsInstance(te, TECrossEntropyLoss.Config)
        self.assertIsInstance(optimized, PiperOptimizedCrossEntropyLoss.Config)

        fused_loss = fused.build(compile_config=None)
        te_loss = te.build(compile_config=None)
        optimized_loss = optimized.build(compile_config=None)
        self.assertIsInstance(fused_loss, LossWithLMHead)
        self.assertNotIsInstance(fused_loss, ChunkedLossWrapper)
        self.assertNotIsInstance(te_loss, LossWithLMHead)
        self.assertNotIsInstance(te_loss, ChunkedLossWrapper)
        self.assertNotIsInstance(optimized_loss, LossWithLMHead)
        self.assertNotIsInstance(optimized_loss, ChunkedLossWrapper)

    def test_custom_lm_head_losses_honor_loss_compilation(self) -> None:
        compile_config = CompileConfig(enable=True, components=["loss"])

        def passthrough(fn, **kwargs):
            return fn

        with mock.patch("torch.compile", side_effect=passthrough) as compile_fn:
            FusedLinearCrossEntropyLoss.Config().build(
                compile_config=compile_config
            )
            PiperOptimizedCrossEntropyLoss.Config().build(
                compile_config=compile_config
            )
        self.assertEqual(compile_fn.call_count, 2)

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
            (PIPER_OPTIMIZED_SWIGLU_OVERRIDE,),
        )
        self.assertEqual(fused.expected_override_count, 16)
        self.assertEqual(
            fused.trace_kernel_markers,
            (
                "_combined_silu_and_mul_forward_kernel",
                "_combined_silu_and_mul_backward_kernel",
            ),
        )

    def test_existing_scenarios_use_the_fixed_piper_workload(self) -> None:
        for scenario in (PIPER_1B_ROPE, PIPER_1B_SWIGLU):
            self.assertIs(scenario.workload, PIPER_1B_WORKLOAD)
        self.assertEqual(PIPER_1B_WORKLOAD.module, "piper1b")
        self.assertEqual(PIPER_1B_WORKLOAD.config, "qwen3_piper_1b")
        self.assertEqual(PIPER_1B_WORKLOAD.local_batch_size, 4)
        self.assertEqual(PIPER_1B_WORKLOAD.seq_len, 1024)
        self.assertEqual(PIPER_1B_WORKLOAD.steps, 40)

    def test_te_rope_trace_uses_position_aware_kernel(self) -> None:
        self.assertEqual(
            PIPER_1B_ROPE.arm("te").trace_kernel_markers,
            ("fused_rope_forward_positions_kernel",),
        )

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
            PIPER_OPTIMIZED_SWIGLU_OVERRIDE,
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
        self.assertIn(PIPER_OPTIMIZED_SWIGLU_OVERRIDE, fused_command)
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
                with gzip.open(trace, "wt") as trace_file:
                    trace_file.write("_combined_silu_and_mul_forward_kernel\n")
                    trace_file.write("_combined_silu_and_mul_backward_kernel\n")
            # Mirrors torchtitan's log format: "[Override] <import path>: <fqn> ..."
            applied = (
                f"[Override] {PIPER_OPTIMIZED_SWIGLU_OVERRIDE}: "
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


class ResumeTests(unittest.TestCase):
    def test_resume_skips_valid_arm_and_retries_incomplete_arm(self) -> None:
        metadata = {
            "requested_gpu": "0",
            "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
            "torch_version": "test",
            "torchtitan_git_rev": "titan-rev",
            "benchmarks_git_rev": "bench-rev",
        }
        events = []

        def fake_process(command, **kwargs):
            import gzip as gzip_module

            kwargs["stdout"].write("Training completed\n")
            arm_dir = Path(command[-1])
            for iteration in (20, 40):
                trace = (
                    arm_dir
                    / f"profiling/traces/iteration_{iteration}/rank0_trace.json.gz"
                )
                trace.parent.mkdir(parents=True, exist_ok=True)
                trace_events = []
                slot = 0
                for graph, phase in (
                    ("backward", "backward"),
                    ("forward", "forward"),
                ):
                    graph_name = f"## Call CompiledFxGraph {graph} ##"
                    for _ in range(80):
                        start = slot * 1000
                        slot += 1
                        if phase == "backward":
                            trace_events.append(
                                {
                                    "ph": "X",
                                    "cat": "cpu_op",
                                    "name": "CompiledFunctionBackward",
                                    "tid": 1,
                                    "ts": start,
                                    "dur": 900,
                                }
                            )
                        trace_events.extend(
                            (
                                {
                                    "ph": "X",
                                    "cat": "user_annotation",
                                    "name": graph_name,
                                    "tid": 1,
                                    "ts": start + 10,
                                    "dur": 800,
                                },
                                {
                                    "ph": "X",
                                    "cat": "gpu_user_annotation",
                                    "name": graph_name,
                                    "tid": 100,
                                    "ts": start + 20,
                                    "dur": 100,
                                },
                            )
                        )
                with gzip_module.open(trace, "wt") as trace_file:
                    json.dump({"traceEvents": trace_events}, trace_file)
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.runner.hardware_metadata",
            return_value=("test-gpu", metadata),
        ):
            out_dir = Path(temporary) / "run"
            environment = {"PATH": os.environ["PATH"]}
            request = RunRequest(
                gpu="0",
                scenario_name="piper1b_rope",
                arm_name="baseline",
                out_dir=out_dir,
                seq_len=512,
                steps=60,
                batch=2,
                extra_args=("--debug.deterministic",),
            )
            execute_run(
                request,
                process_runner=fake_process,
                environment=environment,
            )

            resumed = RunRequest(
                gpu="0",
                scenario_name=None,
                arm_name="baseline",
                resume_dir=out_dir,
            )
            process = mock.Mock(side_effect=fake_process)
            execute_run(
                resumed,
                process_runner=process,
                environment=environment,
                event_handler=events.append,
            )
            process.assert_not_called()
            self.assertTrue(any(event.kind == "skip" for event in events))

            (out_dir / "baseline.log").write_text("interrupted\n")
            retry_process = mock.Mock(side_effect=fake_process)
            execute_run(
                resumed,
                process_runner=retry_process,
                environment=environment,
            )
            retry_command = retry_process.call_args.args[0]
            self.assertEqual(
                retry_command[retry_command.index("--training.seq-len") + 1], "512"
            )
            self.assertEqual(
                retry_command[retry_command.index("--training.steps") + 1], "60"
            )
            self.assertEqual(
                retry_command[
                    retry_command.index("--training.local-batch-size") + 1
                ],
                "2",
            )
            self.assertIn("--debug.deterministic", retry_command)
            archived_logs = list(
                (out_dir / "attempts").glob("*/baseline/baseline.log")
            )
            self.assertEqual(len(archived_logs), 1)
            self.assertIn("interrupted", archived_logs[0].read_text())
            self.assertIn("Training completed", (out_dir / "baseline.log").read_text())

            incompatible = RunRequest(
                gpu="0",
                scenario_name=None,
                arm_name="baseline",
                resume_dir=out_dir,
                steps=80,
            )
            with self.assertRaisesRegex(ValueError, "conflicts with the recorded"):
                execute_run(
                    incompatible,
                    process_runner=fake_process,
                    environment=environment,
                )

            conflicting_args = RunRequest(
                gpu="0",
                scenario_name=None,
                arm_name="baseline",
                resume_dir=out_dir,
                extra_args=("--debug.seed", "7"),
            )
            with self.assertRaisesRegex(ValueError, "extra_torchtitan_args"):
                execute_run(
                    conflicting_args,
                    process_runner=fake_process,
                    environment=environment,
                )


if __name__ == "__main__":
    unittest.main()
