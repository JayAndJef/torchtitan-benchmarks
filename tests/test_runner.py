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

from benchmarks.artifacts.layout import trace_files
from benchmarks.artifacts.manifests import write_manifest
from benchmarks.e2e.launch import command_for_arm
from benchmarks.e2e.parallelism import ParallelismSpec, TRIVIAL_SPEC
from benchmarks.e2e.registry import (
    Arm,
    COMPILE_MODES,
    SCENARIOS,
    TORCH_COMPILE_MODE,
    PIPER_1B_LM_HEAD,
    PIPER_1B_QKV,
    PIPER_1B_ROPE,
    PIPER_1B_SWIGLU,
    PIPER_1B_WORKLOAD,
    scenario_by_name,
)
from benchmarks.e2e.results import stable_tps, training_metrics
from benchmarks.e2e.runner import RunRequest, _resolve_run, execute_run, select_arms
from benchmarks.e2e.validation import validate_arm
from benchmarks.execution.affinity import CpuPinning, resolve_cpu_pinning
from dataclasses import replace
from benchmarks.models.piper_qwen3.components.lm_head.losses import (
    FusedLinearCrossEntropyLoss,
    PiperOptimizedCrossEntropyLoss,
    TECrossEntropyLoss,
)
from benchmarks.models.piper_qwen3.config_registry import (
    qwen3_piper_1b,
    qwen3_piper_1b_full_logits,
    qwen3_piper_1b_fused_linear_ce,
    qwen3_piper_1b_piper_optimized_te_ce,
    qwen3_piper_1b_te_fused_ce,
    qwen3_piper_1b_unfused_qkv,
)
from torchtitan.components.loss import (
    ChunkedLossWrapper,
    CrossEntropyLoss,
    LossWithLMHead,
)
from benchmarks.models.piper_qwen3.parallelize import (
    DATA_PARALLEL_LINE,
    parallelize_piper1b,
    skip_data_parallel,
)
from benchmarks.models.piper_qwen3.shape import HUGE, PIPER_1B, PIPER_SHAPES
from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.models.common import FusedQKVLinear, QKVLinear


PIPER_OPTIMIZED_SWIGLU_OVERRIDE = (
    "benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu."
    "piper_optimized_triton_fused_grouped_experts"
)


class ScenarioTests(unittest.TestCase):
    def test_a_request_without_a_scenario_is_refused_and_starts_nothing(
        self,
    ) -> None:
        """There is no default scenario, and an omission fails the run.

        A default could only be reached by an omission, and it would then
        measure ``piper1b_rope`` under whatever label the operator assumed --
        a wrong result rather than a missing one. ``_resolve_run`` holds the
        rule, so the CLI and every programmatic caller inherit it, and no
        scenario name is written anywhere as a default.

        The two patches prove the refusal lands before any work starts.
        ``hardware_metadata`` is the first host probe ``_resolve_run`` makes
        after it resolves the scenario, and ``process_runner`` launches the
        training subprocess; neither may run.
        """

        def never(*args, **kwargs):
            raise AssertionError("a run started without a scenario")

        self.assertIsNone(RunRequest(gpu="0").scenario_name)
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata", side_effect=never
        ):
            with self.assertRaises(ValueError) as caught:
                execute_run(
                    RunRequest(gpu="0"),
                    process_runner=never,
                    environment={"PATH": os.environ["PATH"]},
                )
        self.assertIn("no scenario requested", str(caught.exception))
        self.assertIn("--scenario", str(caught.exception))

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
            [arm.name for arm in scenario.arms],
            ["baseline", "piper_optimized_triton", "piper_optimized_inductor"],
        )
        fused = scenario.arm("piper_optimized_triton")
        self.assertEqual(
            fused.override_imports,
            (PIPER_OPTIMIZED_SWIGLU_OVERRIDE,),
        )
        self.assertEqual(fused.overrides_per_block, 1)
        self.assertEqual(
            fused.trace_kernel_markers,
            (
                "_combined_silu_and_mul_forward_kernel",
                "_combined_silu_and_mul_backward_kernel",
            ),
        )
        inductor = scenario.arm("piper_optimized_inductor")
        self.assertEqual(
            inductor.override_imports,
            (
                "benchmarks.models.piper_qwen3.components.swiglu."
                "combined_swiglu.piper_optimized_inductor_fused_grouped_experts",
            ),
        )
        self.assertEqual(inductor.overrides_per_block, 1)
        # Plain-ops activation has no distinctive kernel name by design;
        # the override count is the application check.
        self.assertEqual(inductor.trace_kernel_markers, ())

    def test_existing_scenarios_use_the_fixed_piper_workload(self) -> None:
        for scenario in (PIPER_1B_ROPE, PIPER_1B_SWIGLU):
            self.assertIs(scenario.workload, PIPER_1B_WORKLOAD)
        self.assertEqual(
            PIPER_1B_WORKLOAD.module, "benchmarks.models.piper_qwen3"
        )
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


class SelectedArmTests(unittest.TestCase):
    """Repeated ``--arm`` is an ordered subset, never a second scenario."""

    def setUp(self) -> None:
        self.scenario = scenario_by_name("piper1b_megatron")
        self.metadata = {
            "requested_gpu": "0",
            "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
            "torch_version": "test",
            "torchtitan_git_rev": "titan-rev",
            "benchmarks_git_rev": "bench-rev",
            "megatron_git_rev": "megatron-rev",
        }

    def test_no_names_selects_the_complete_roster(self) -> None:
        self.assertEqual(select_arms(self.scenario, ()), self.scenario.arms)

    def test_one_name_selects_one_arm(self) -> None:
        selected = select_arms(self.scenario, ("titan_stock",))
        self.assertEqual([arm.name for arm in selected], ["titan_stock"])

    def test_several_names_preserve_request_order(self) -> None:
        selected = select_arms(
            self.scenario, ("titan_stock", "baseline")
        )
        self.assertEqual(
            [arm.name for arm in selected], ["titan_stock", "baseline"]
        )

    def test_a_duplicate_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, r"--arm repeats 'baseline'"):
            select_arms(self.scenario, ("baseline", "baseline"))

    def test_unknown_names_are_refused_together(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"has no arm\(s\) 'missing', 'also_missing'"
        ):
            select_arms(self.scenario, ("missing", "also_missing"))

    def _resolve(self, names: tuple[str, ...], compile_mode: str):
        request = RunRequest(
            gpu="0",
            scenario_name=self.scenario.name,
            arm_names=names,
            out_dir=Path("/tmp/selected-arm-test"),
            compile_mode=compile_mode,
            ac_mode="none",
        )
        return _resolve_run(request, {"PATH": os.environ["PATH"]})

    def test_compile_mode_acceptance_depends_on_the_selected_engines(self) -> None:
        cases = (
            (("baseline", "titan_stock"), "default", True),
            (("titan_stock",), "none", True),
            (("baseline",), "none", False),
            (("baseline", "titan_stock"), "none", False),
            ((), "none", False),
        )
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", self.metadata),
        ) as hardware, mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            for names, compile_mode, allowed in cases:
                with self.subTest(names=names, compile_mode=compile_mode):
                    hardware.reset_mock()
                    if allowed:
                        resolved = self._resolve(names, compile_mode)
                        self.assertEqual(
                            [arm.name for arm in resolved[2]], list(names)
                        )
                        hardware.assert_called_once()
                    else:
                        with self.assertRaisesRegex(
                            ValueError, "does not support compile mode 'none'"
                        ):
                            self._resolve(names, compile_mode)
                        hardware.assert_not_called()

    def test_order_reaches_execution_manifest_state_and_resume_gate(self) -> None:
        launched: list[str] = []

        def fake_process(command, **kwargs):
            launched.append(Path(kwargs["stdout"].name).stem)
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", self.metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ), mock.patch("benchmarks.e2e.runner.validate_arm"):
            out_dir = Path(temporary) / "run"
            execute_run(
                RunRequest(
                    gpu="0",
                    scenario_name="piper1b_qkv",
                    arm_names=("fused_qkv", "baseline"),
                    out_dir=out_dir,
                ),
                process_runner=fake_process,
                environment={"PATH": os.environ["PATH"]},
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())
            state = json.loads((out_dir / "run_state.json").read_text())
            self.assertEqual(launched, ["fused_qkv", "baseline"])
            self.assertEqual(
                manifest["selected_arms"], ["fused_qkv", "baseline"]
            )
            self.assertEqual(
                list(manifest["commands"]), ["fused_qkv", "baseline"]
            )
            self.assertEqual(
                list(state["arms"]), ["fused_qkv", "baseline"]
            )

            with self.assertRaisesRegex(ValueError, "selected_arms"):
                execute_run(
                    RunRequest(
                        gpu="0",
                        arm_names=("baseline", "fused_qkv"),
                        resume_dir=out_dir,
                    ),
                    process_runner=fake_process,
                    environment={"PATH": os.environ["PATH"]},
                )


def _p2p_flags(command: list[str]) -> list[str]:
    """The flag tokens of ``command`` that name the p2p sync.

    Flags only: a path in the argv can carry the substring too, and the
    interpreter's own path does on a checkout named after this option.
    """
    return [
        token for token in command if token.startswith("--") and "p2p" in token
    ]


class MegatronP2pSyncResolutionTests(unittest.TestCase):
    """What ``_resolve_run`` does with ``--megatron-p2p-sync``.

    Both refusals are parent-side and land before any host probe, so a
    refused request claims no GPU. The value reaches the megatron commands
    alone; a TorchTitan argv is untouched under either value.
    """

    PP2 = ParallelismSpec(pp=2, pp_schedule="1F1B")

    def setUp(self) -> None:
        self.metadata = {
            "requested_gpu": "0",
            "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
            "torch_version": "test",
            "torchtitan_git_rev": "titan-rev",
            "benchmarks_git_rev": "bench-rev",
            "megatron_git_rev": "mcore-rev",
        }

    def _resolve(
        self,
        names: tuple[str, ...],
        *,
        gpu: str = "0,1",
        parallelism: ParallelismSpec | None = None,
        megatron_p2p_sync: str | None = "off",
        scenario_name: str = "piper1b_megatron",
    ):
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", self.metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return _resolve_run(
                RunRequest(
                    gpu=gpu,
                    scenario_name=scenario_name,
                    arm_names=names,
                    out_dir=Path("/tmp/p2p-sync-test"),
                    ac_mode="none",
                    parallelism=self.PP2 if parallelism is None else parallelism,
                    megatron_p2p_sync=megatron_p2p_sync,
                ),
                {"PATH": os.environ["PATH"]},
            )

    def _refused_before_any_probe(self, pattern: str, **keywords) -> None:
        def never(*args, **kwargs):
            raise AssertionError("a host probe ran for a refused request")

        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata", side_effect=never
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning", side_effect=never
        ):
            with self.assertRaisesRegex(ValueError, pattern):
                _resolve_run(
                    RunRequest(
                        scenario_name="piper1b_megatron",
                        out_dir=Path("/tmp/p2p-sync-test"),
                        ac_mode="none",
                        **keywords,
                    ),
                    {"PATH": os.environ["PATH"]},
                )

    def test_off_at_pp_one_is_refused_before_any_host_probe(self) -> None:
        """No pipeline message exists, so the manifest would record a
        treatment the run did not have."""
        self._refused_before_any_probe(
            "no pipeline message",
            gpu="0",
            arm_names=("baseline",),
            megatron_p2p_sync="off",
        )

    def test_off_without_a_megatron_arm_is_refused_before_any_host_probe(
        self,
    ) -> None:
        """The ``--compile-mode none`` exception, the other way round.

        That mode needs every selected arm to be TorchTitan. This value
        needs at least one not to be, because TorchTitan sends no pipeline
        message through Megatron and the value would reach nothing.
        """
        self._refused_before_any_probe(
            "reaches no arm",
            gpu="0,1",
            arm_names=("titan_stock",),
            parallelism=self.PP2,
            megatron_p2p_sync="off",
        )

    def test_an_unknown_value_is_refused(self) -> None:
        self._refused_before_any_probe(
            "unknown megatron p2p sync",
            gpu="0,1",
            arm_names=("baseline",),
            parallelism=self.PP2,
            megatron_p2p_sync="false",
        )

    def test_off_reaches_the_megatron_command_and_not_the_titan_one(
        self,
    ) -> None:
        """``run --arm`` narrows the engine set, and a mixed selection is
        legal: the megatron arm gets the flag and the titan arm gets
        nothing."""
        resolved = self._resolve(("baseline", "titan_stock"))
        self.assertEqual(resolved[12], "off")
        commands = resolved[6]
        megatron = commands["baseline"]
        self.assertEqual(megatron[-3:-1], ["--batch-p2p-sync", "off"])
        self.assertEqual(_p2p_flags(commands["titan_stock"]), [])

    def test_a_megatron_only_subset_passes(self) -> None:
        resolved = self._resolve(("baseline",))
        self.assertEqual([arm.name for arm in resolved[2]], ["baseline"])
        self.assertEqual(resolved[12], "off")

    def test_the_default_resolves_to_on_and_adds_no_token(self) -> None:
        for requested in (None, "on"):
            with self.subTest(requested=requested):
                resolved = self._resolve(
                    ("baseline", "titan_stock"), megatron_p2p_sync=requested
                )
                self.assertEqual(resolved[12], "on")
                for name, command in resolved[6].items():
                    self.assertEqual(_p2p_flags(command), [], name)

    def test_the_stock_scenario_takes_the_value_too(self) -> None:
        resolved = self._resolve(
            ("baseline",), scenario_name="piper_megatron_stock"
        )
        command = resolved[6]["baseline"]
        self.assertEqual(command[-2:], ["--bench-batch-p2p-sync", "off"])

    def _write_manifest(self, out_dir: Path, megatron_p2p_sync: str) -> None:
        scenario = scenario_by_name("piper1b_megatron")
        write_manifest(
            out_dir,
            scenario,
            (scenario.arm("baseline"),),
            {"baseline": ["cmd"]},
            "test-gpu",
            {**self.metadata, "cpu_pinning": "none: test"},
            (),
            "default",
            "none",
            "1b",
            parallelism=self.PP2,
            megatron_p2p_sync=megatron_p2p_sync,
        )

    def _resume(self, out_dir: Path, megatron_p2p_sync: str | None):
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", self.metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return _resolve_run(
                RunRequest(
                    gpu="0,1",
                    scenario_name=None,
                    arm_names=("baseline",),
                    resume_dir=out_dir,
                    parallelism=self.PP2,
                    megatron_p2p_sync=megatron_p2p_sync,
                ),
                {"PATH": os.environ["PATH"]},
            )

    def test_a_resume_inherits_the_recorded_value_and_refuses_another(
        self,
    ) -> None:
        """The gate reads like --compile-mode's: an omitted value inherits
        the recorded one and rebuilds the same argv, a different value is
        refused, and the refusal names the field."""
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            out_dir.mkdir()
            self._write_manifest(out_dir, "off")
            resolved = self._resume(out_dir, megatron_p2p_sync=None)
            self.assertEqual(resolved[12], "off")
            self.assertTrue(resolved[11])
            self.assertEqual(
                resolved[6]["baseline"][-3:-1], ["--batch-p2p-sync", "off"]
            )
            with self.assertRaisesRegex(ValueError, "megatron_p2p_sync"):
                self._resume(out_dir, megatron_p2p_sync="on")

    def test_a_resume_of_a_schema_twelve_directory_reads_as_on(self) -> None:
        """A directory written before the field exists carries no key, and
        no such run could have turned the sync off. So it resumes as ``on``
        with the argv it always had, and a request for ``off`` is refused
        rather than changing the treatment under the recorded label."""
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            out_dir.mkdir()
            self._write_manifest(out_dir, "on")
            manifest_path = out_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            del manifest["megatron_p2p_sync"]
            manifest["schema_version"] = 12
            manifest_path.write_text(json.dumps(manifest))
            resolved = self._resume(out_dir, megatron_p2p_sync=None)
            self.assertEqual(resolved[12], "on")
            self.assertEqual(_p2p_flags(resolved[6]["baseline"]), [])
            with self.assertRaisesRegex(ValueError, "megatron_p2p_sync"):
                self._resume(out_dir, megatron_p2p_sync="off")

    def test_execute_run_hands_the_value_to_validate_arm(self) -> None:
        """The value the run resolved is the value the gate reads."""

        def fake_process(command, **kwargs):
            kwargs["stdout"].write("Training completed\n")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", self.metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ), mock.patch("benchmarks.e2e.runner.validate_arm") as validate:
            execute_run(
                RunRequest(
                    gpu="0,1",
                    scenario_name="piper1b_megatron",
                    arm_names=("baseline",),
                    out_dir=Path(temporary) / "run",
                    ac_mode="none",
                    parallelism=self.PP2,
                    megatron_p2p_sync="off",
                ),
                process_runner=fake_process,
                environment={"PATH": os.environ["PATH"]},
            )
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(validate.call_args.kwargs["megatron_p2p_sync"], "off")
        self.assertEqual(validate.call_args.kwargs["parallelism"], self.PP2)

    def test_the_banner_names_the_value(self) -> None:
        """The banner names every comparability boundary the manifest
        gates, and this value is one."""
        events: list[str] = []

        def failing_process(command, **kwargs):
            kwargs["stdout"].write("nothing trained\n")
            return SimpleNamespace(returncode=1)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", self.metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            with self.assertRaises(RuntimeError):
                execute_run(
                    RunRequest(
                        gpu="0,1",
                        scenario_name="piper1b_megatron",
                        arm_names=("baseline",),
                        out_dir=Path(temporary) / "run",
                        ac_mode="none",
                        parallelism=self.PP2,
                        megatron_p2p_sync="off",
                    ),
                    event_handler=lambda event: events.append(
                        event.message if event.kind == "summary" else ""
                    ),
                    process_runner=failing_process,
                    environment={"PATH": os.environ["PATH"]},
                )
        self.assertIn("megatron p2p sync: off", events)


class ParallelizeTests(unittest.TestCase):
    def test_all_piper_configs_run_single_gpu_plain_bf16(self) -> None:
        for factory in (
            qwen3_piper_1b,
            qwen3_piper_1b_unfused_qkv,
            qwen3_piper_1b_full_logits,
            qwen3_piper_1b_fused_linear_ce,
            qwen3_piper_1b_te_fused_ce,
            qwen3_piper_1b_piper_optimized_te_ce,
        ):
            for size in PIPER_SHAPES:
                with self.subTest(config=factory.__name__, size=size):
                    config = factory(size=size)
                    self.assertIs(
                        config.model_spec.parallelize_fn, parallelize_piper1b
                    )
                    self.assertEqual(config.training.dtype, "bfloat16")

    # Every guard fires before the model is touched, so dummies suffice.
    # ``parallelism`` is real: the shard-degree guard reads the RAW
    # configured value, because the resolved mesh cannot show a dropped
    # flag once the harness asks for a sharded mesh too.
    _PARALLELIZE_COMMON = dict(
        model=object(),
        parallelism=ParallelismConfig(data_parallel_shard_degree=1),
        compile_config=None,
        ac_config=None,
        dump_folder="",
    )

    def test_parallelize_refuses_each_axis_for_its_own_reason(self) -> None:
        """One message per axis, because the axes fail for different reasons.

        One ``world_size != 1`` check stood here before, and it refused a
        pipeline rank -- which needs no gradient reduction and keeps exactly
        this plain-bf16 model -- for the data-parallel axis's reason.
        """
        for name, dims, expected in (
            (
                "tp",
                ParallelDims(
                    dp_replicate=1, dp_shard=1, cp=1, tp=2, pp=1, ep=1, world_size=2
                ),
                "tensor parallelism",
            ),
            (
                "cp",
                ParallelDims(
                    dp_replicate=1, dp_shard=1, cp=2, tp=1, pp=1, ep=1, world_size=2
                ),
                "context parallelism",
            ),
            (
                "hsdp",
                ParallelDims(
                    dp_replicate=2, dp_shard=2, cp=1, tp=1, pp=1, ep=1, world_size=4
                ),
                "one data-parallel treatment at a time",
            ),
        ):
            with self.subTest(axis=name):
                with self.assertRaisesRegex(RuntimeError, expected):
                    parallelize_piper1b(
                        parallel_dims=dims,
                        training=TrainingConfig(dtype="bfloat16"),
                        **self._PARALLELIZE_COMMON,
                    )

    def test_parallelize_rejects_non_bf16(self) -> None:
        single_gpu = ParallelDims(
            dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
        )
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            parallelize_piper1b(
                parallel_dims=single_gpu,
                training=TrainingConfig(dtype="float32"),
                **self._PARALLELIZE_COMMON,
            )

    def test_parallelize_lets_a_pipeline_rank_through(self) -> None:
        """A pipeline stage keeps the plain-bf16 model, and still skips DP.

        PP splits the layers and synchronizes no gradient, so ``dp`` stays 1
        and ``skip_dp`` stays right. That is what makes PP the cheapest
        honest parallel measurement this repo can take.
        """
        pipelined = ParallelDims(
            dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=2, ep=1, world_size=2
        )
        sentinel = object()
        with mock.patch(
            "benchmarks.models.piper_qwen3.parallelize.parallelize_qwen3",
            return_value=sentinel,
        ) as delegate:
            result = parallelize_piper1b(
                parallel_dims=pipelined,
                training=TrainingConfig(dtype="bfloat16"),
                **self._PARALLELIZE_COMMON,
            )
        self.assertIs(result, sentinel)
        self.assertIs(delegate.call_args.kwargs["skip_dp"], True)

    _DP2 = ParallelDims(
        dp_replicate=2, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=2
    )

    def test_skip_data_parallel_reads_the_delivered_mesh(self) -> None:
        """The subprocess-side twin of ``parallelism.skip_dp``.

        Both say "no data-parallel machinery is needed" exactly when the
        degree is 1, so a single-GPU run and a pipeline-only run keep the
        plain-bf16 model this repo has always measured.
        """
        for name, dims, expected in (
            (
                "single gpu",
                ParallelDims(
                    dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
                ),
                True,
            ),
            (
                "pipeline only",
                ParallelDims(
                    dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=2, ep=1, world_size=2
                ),
                True,
            ),
            ("dp 2", self._DP2, False),
        ):
            with self.subTest(mesh=name):
                self.assertIs(skip_data_parallel(dims), expected)

    def test_parallelize_runs_the_data_parallel_path_at_replicate_2(self) -> None:
        """dp 2 asks the delegate for FSDP and states what came back.

        ``skip_dp`` False is what makes ``parallelize_qwen3`` reach
        ``apply_fsdp_to_decoder``; without it the two ranks read different
        data, reduce no gradient, and report roughly twice the true speed.

        ``FSDPModule.__new__`` returns an instance of the class it was
        injected into, so a subclass of it is not one and a real wrapped
        module cannot be built without a process group. The name this module
        binds is patched instead: the branch under test is the count, and
        production keeps torch's own class.
        """
        import torch.nn as nn

        class _Wrapped(nn.Module):
            pass

        model = nn.Sequential(_Wrapped(), _Wrapped(), nn.Linear(2, 2))
        with mock.patch(
            "benchmarks.models.piper_qwen3.parallelize.FSDPModule", _Wrapped
        ):
            with mock.patch(
                "benchmarks.models.piper_qwen3.parallelize.parallelize_qwen3",
                return_value=model,
            ) as delegate:
                with self.assertLogs(level="INFO") as logs:
                    result = parallelize_piper1b(
                        parallel_dims=self._DP2,
                        training=TrainingConfig(dtype="bfloat16"),
                        **self._PARALLELIZE_COMMON,
                    )
        self.assertIs(result, model)
        self.assertIs(delegate.call_args.kwargs["skip_dp"], False)
        printed = "\n".join(logs.output)
        self.assertIn(
            DATA_PARALLEL_LINE.format(replicate=2, shard=1), printed
        )
        self.assertIn("2 FSDP units", printed)

    def test_parallelize_refuses_a_dp_run_the_delegate_left_unwrapped(
        self,
    ) -> None:
        """The count is a guard, not a log line.

        A future edit that lost ``skip_dp`` would leave every module bare and
        every other check would pass. Nothing wrapped means nothing reduces.
        """
        import torch.nn as nn

        class _Wrapped(nn.Module):
            pass

        with mock.patch(
            "benchmarks.models.piper_qwen3.parallelize.FSDPModule", _Wrapped
        ):
            with mock.patch(
                "benchmarks.models.piper_qwen3.parallelize.parallelize_qwen3",
                return_value=nn.Linear(2, 2),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "never reduce their gradients"
                ):
                    parallelize_piper1b(
                        parallel_dims=self._DP2,
                        training=TrainingConfig(dtype="bfloat16"),
                        **self._PARALLELIZE_COMMON,
                    )

    def test_a_single_gpu_run_logs_no_data_parallel_line(self) -> None:
        """The trivial spec's log text is a recorded fact; it must not move.

        ``--resume`` and every published directory read these lines, and a
        line that appeared at one rank would be a new recorded fact for a run
        whose treatment did not change.
        """
        import logging
        import torch.nn as nn

        single_gpu = ParallelDims(
            dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
        )
        with mock.patch(
            "benchmarks.models.piper_qwen3.parallelize.parallelize_qwen3",
            return_value=nn.Linear(2, 2),
        ):
            with self.assertLogs(level="INFO") as logs:
                logging.getLogger().info("probe")
                parallelize_piper1b(
                    parallel_dims=single_gpu,
                    training=TrainingConfig(dtype="bfloat16"),
                    **self._PARALLELIZE_COMMON,
                )
        self.assertEqual(
            [line for line in logs.output if "data parallel" in line], []
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

    def test_the_model_size_rides_as_a_config_argument(self) -> None:
        # The config name never carries the size: it is delivered as a keyword
        # argument to the config function via the fork's --config-arg.
        for size in PIPER_SHAPES:
            with self.subTest(size=size):
                command = command_for_arm(
                    PIPER_1B_QKV.workload,
                    PIPER_1B_QKV.arm("fused_qkv"),
                    Path("/out/fused_qkv"),
                    [],
                    model_size=size,
                )
                self.assertEqual(
                    command[command.index("--config") + 1], "qwen3_piper_1b"
                )
                self.assertEqual(
                    command[command.index("--config-arg") + 1], f"size={size}"
                )
                # The retired scheme spelled the size into the config
                # name. Checked as the mangled name itself rather than as
                # "no token ends in the size": the config family is called
                # qwen3_piper_1b, so at size "1b" the plain name ends in it.
                self.assertNotIn(f"qwen3_piper_1b_{size}", command)

    def test_command_adds_only_the_arm_override_and_dump_folder(self) -> None:
        arm = PIPER_1B_SWIGLU.arm("piper_optimized_triton")
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
        self.assertEqual(
            command[:5],
            [
                "./run_train.sh",
                "--module",
                "benchmarks.models.piper_qwen3",
                "--config",
                "qwen3_piper_1b",
            ],
        )
        self.assertIn("--compile.enable", command)
        self.assertIn("--profiler.enable_profiling", command)

    def test_compile_mode_reaches_torchtitan_as_the_torch_level_name(self) -> None:
        command = command_for_arm(
            PIPER_1B_ROPE.workload,
            PIPER_1B_ROPE.arm("baseline"),
            Path("/out/baseline"),
            [],
            "cuda-graph",
        )
        self.assertEqual(
            command[command.index("--compile.mode") + 1], "reduce-overhead"
        )
        self.assertEqual(command[-2:], ["--dump-folder", "/out/baseline"])

    def test_default_compile_mode_leaves_the_command_untouched(self) -> None:
        command = command_for_arm(
            PIPER_1B_ROPE.workload,
            PIPER_1B_ROPE.arm("baseline"),
            Path("/out/baseline"),
            [],
        )
        self.assertNotIn("--compile.mode", command)
        self.assertIn("--compile.enable", command)

    def test_uncompiled_mode_drops_only_the_compile_flag(self) -> None:
        # CompileConfig.enable is False in the fork, so the uncompiled command
        # omits the flag rather than negating it. Everything else must be the
        # command the default mode builds, token for token: this is the
        # assertion that keeps the new mode from moving an existing default.
        compiled = command_for_arm(
            PIPER_1B_ROPE.workload,
            PIPER_1B_ROPE.arm("baseline"),
            Path("/out/baseline"),
            [],
        )
        eager = command_for_arm(
            PIPER_1B_ROPE.workload,
            PIPER_1B_ROPE.arm("baseline"),
            Path("/out/baseline"),
            [],
            "none",
        )
        self.assertNotIn("--compile.enable", eager)
        self.assertNotIn("--compile.mode", eager)
        self.assertEqual(
            eager, [token for token in compiled if token != "--compile.enable"]
        )

    def test_a_megatron_command_refuses_an_uncompiled_mode(self) -> None:
        scenario = scenario_by_name("piper1b_megatron")
        with self.assertRaisesRegex(ValueError, "cannot apply to this arm"):
            command_for_arm(
                scenario.workload,
                scenario.arm("baseline"),
                Path("/out/baseline"),
                [],
                "none",
                "none",
            )

    def test_ac_none_adds_the_subcommand_token_last(self) -> None:
        command = command_for_arm(
            PIPER_1B_ROPE.workload,
            PIPER_1B_ROPE.arm("baseline"),
            Path("/out/baseline"),
            [],
            "default",
            "none",
        )
        # tyro attributes flags after a subcommand token to that subcommand,
        # so the token must trail everything, including --dump-folder.
        self.assertEqual(command[-1], "activation-checkpoint:none")
        self.assertEqual(command[-3:-1], ["--dump-folder", "/out/baseline"])

    def test_ac_sac_leaves_the_command_untouched(self) -> None:
        command = command_for_arm(
            PIPER_1B_ROPE.workload,
            PIPER_1B_ROPE.arm("baseline"),
            Path("/out/baseline"),
            [],
        )
        self.assertNotIn("activation-checkpoint:none", command)

    def test_megatron_launcher_builds_the_driver_command(self) -> None:
        workload = replace(PIPER_1B_ROPE.workload, seed=42)
        arm = Arm(name="baseline", description="megatron", launcher="megatron")
        command = command_for_arm(
            workload, arm, Path("/out/baseline"), [], "cuda-graph", "none"
        )
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1:3], ["-m", "benchmarks.e2e.megatron.train"])
        self.assertEqual(command[command.index("--mode") + 1], "cuda-graph")
        self.assertEqual(command[command.index("--seed") + 1], "42")
        self.assertEqual(command[command.index("--seq-len") + 1], "1024")
        self.assertEqual(command[-1], "/out/baseline")

    def test_megatron_launcher_refuses_unsupported_requests(self) -> None:
        workload = replace(PIPER_1B_ROPE.workload, seed=42)
        arm = Arm(name="baseline", description="megatron", launcher="megatron")
        with self.assertRaisesRegex(ValueError, "passthrough"):
            command_for_arm(
                workload, arm, Path("/out"), ["--debug.seed", "7"], "default", "none"
            )
        with self.assertRaisesRegex(ValueError, "without recompute"):
            command_for_arm(workload, arm, Path("/out"), [], "default", "sac")
        unseeded = Arm(name="baseline", description="megatron", launcher="megatron")
        with self.assertRaisesRegex(ValueError, "seeded"):
            command_for_arm(
                PIPER_1B_ROPE.workload, unseeded, Path("/out"), [], "default", "none"
            )
        with self.assertRaisesRegex(ValueError, "unknown launcher"):
            command_for_arm(
                workload,
                Arm(name="x", description="x", launcher="colossalai"),
                Path("/out"),
                [],
            )

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


class AttentionScenarioTests(unittest.TestCase):
    def test_scenario_registration(self) -> None:
        scenario = scenario_by_name("piper1b_attention")
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            ["baseline", "flash_attention_3", "flex_flash"],
        )
        # Swapping the inner attention does not change parameter structure,
        # so unlike qkv/lm_head this scenario needs no fixed seed.
        self.assertIsNone(scenario.workload.seed)

    def test_both_flash_arms_pin_their_own_kernel_names(self) -> None:
        """The markers are the only proof an arm ran the backend it claims."""
        scenario = scenario_by_name("piper1b_attention")
        fa3 = scenario.arm("flash_attention_3").trace_kernel_markers
        fa4 = scenario.arm("flex_flash").trace_kernel_markers
        self.assertTrue(fa3 and fa4)
        # FA4 spells "FlashAttention" out where FA3 abbreviates, so neither
        # arm's markers can be satisfied by the other's kernels.
        for marker in fa4:
            self.assertFalse(any(marker in other for other in fa3), marker)
        for marker in fa3:
            self.assertFalse(any(marker in other for other in fa4), marker)


class MegatronScenarioTests(unittest.TestCase):
    def test_scenario_registration(self) -> None:
        scenario = scenario_by_name("piper1b_megatron")
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            [
                "baseline",
                "titan_stock",
                "titan_swiglu",
                "titan_lm_head",
                "titan_swiglu_lm_head",
            ],
        )
        baseline = scenario.arm("baseline")
        self.assertEqual(baseline.launcher, "megatron")
        self.assertEqual(baseline.validation, "megatron")
        self.assertIn("--ac never affects this arm", baseline.description)
        self.assertIn("tuned BASE profile", baseline.description)
        self.assertIn("fastest-available TE fused CE", baseline.description)
        self.assertIn(
            "not accepted by stock pretrain_gpt.py", baseline.description
        )
        self.assertEqual(scenario.supported_ac_modes, ("none",))
        # The complete roster declines the titan-only uncompiled treatment;
        # an explicit all-titan subset is the narrow runner-level exception.
        self.assertEqual(
            scenario.supported_compile_modes, ("default", "cuda-graph")
        )
        self.assertEqual(scenario.regions, ())
        self.assertEqual(scenario.workload.seed, 42)
        for arm in scenario.arms[1:]:
            self.assertEqual(arm.launcher, "torchtitan")

    def test_pretokenized_config_twins_build(self) -> None:
        import benchmarks.models.piper_qwen3.config_registry as registry
        from benchmarks.e2e.data.piper_qwen3 import PretokenizedReplayDataLoader
        from benchmarks.models.piper_qwen3.components.lm_head.losses import (
            PiperOptimizedCrossEntropyLoss,
        )

        scenario = scenario_by_name("piper1b_megatron")
        config_names = {
            arm.config or scenario.workload.config for arm in scenario.arms[1:]
        }
        for name in sorted(config_names):
            config = getattr(registry, name)()
            self.assertIsInstance(
                config.dataloader, PretokenizedReplayDataLoader.Config, name
            )
            self.assertEqual(config.dataloader.replay_steps, 40, name)
        te_ce = registry.qwen3_piper_1b_piper_optimized_te_ce_pretokenized()
        self.assertIsInstance(te_ce.loss, PiperOptimizedCrossEntropyLoss.Config)

    def test_every_arm_command_builds_at_ac_none(self) -> None:
        scenario = scenario_by_name("piper1b_megatron")
        for arm in scenario.arms:
            command = command_for_arm(
                scenario.workload,
                arm,
                Path("/out") / arm.name,
                [],
                "cuda-graph",
                "none",
            )
            self.assertTrue(command, arm.name)

    def test_run_refuses_sac_for_the_megatron_scenario(self) -> None:
        request = RunRequest(
            gpu="0", scenario_name="piper1b_megatron", ac_mode="sac"
        )
        with self.assertRaisesRegex(ValueError, "does not support ac mode"):
            execute_run(request, environment={"PATH": os.environ["PATH"]})

    def test_run_refuses_the_uncompiled_mode_for_the_megatron_scenario(self) -> None:
        request = RunRequest(
            gpu="0",
            scenario_name="piper1b_megatron",
            compile_mode="none",
            ac_mode="none",
        )
        with self.assertRaisesRegex(ValueError, "does not support compile mode"):
            execute_run(request, environment={"PATH": os.environ["PATH"]})

    def test_every_titan_only_scenario_supports_every_compile_mode(self) -> None:
        """The exemption is derived from the cause, not from a name.

        A scenario restricts the compile axis because one of its arms cannot
        receive a treatment, and only a non-TorchTitan arm has that problem:
        the axis names whole-block ``torch.compile``, which every titan arm
        gets and no Megatron arm has. So a scenario whose arms are all
        TorchTitan must accept every mode. A name-based skip would let a
        future titan-only scenario restrict the axis for no stated reason.
        """
        for name, scenario in SCENARIOS.items():
            if any(arm.launcher != "torchtitan" for arm in scenario.arms):
                continue
            with self.subTest(scenario=name):
                self.assertEqual(scenario.supported_compile_modes, COMPILE_MODES)


class CpuPinningTests(unittest.TestCase):
    def _sysfs(self, root: Path, device: str, node: str) -> Path:
        node_dir = root / "bus/pci/devices" / device
        node_dir.mkdir(parents=True)
        (node_dir / "numa_node").write_text(node + "\n")
        return root

    def test_pins_to_the_gpu_numa_node(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.execution.affinity.run_text", return_value="00000000:E3:00.0\n"
        ), mock.patch("benchmarks.execution.affinity.shutil.which", return_value="/usr/bin/numactl"):
            sysfs = self._sysfs(Path(temporary), "0000:e3:00.0", "1")
            pinning = resolve_cpu_pinning("7", sysfs_root=sysfs)
        self.assertEqual(
            pinning.prefix, ("numactl", "--cpunodebind=1", "--membind=1")
        )
        self.assertEqual(pinning.description, "numactl --cpunodebind=1 --membind=1")

    def test_unpinned_when_prerequisites_are_missing(self) -> None:
        with mock.patch("benchmarks.execution.affinity.shutil.which", return_value=None):
            self.assertEqual(
                resolve_cpu_pinning("7").prefix, ()
            )

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.execution.affinity.run_text",
            return_value="unavailable: nvidia-smi not found",
        ), mock.patch(
            "benchmarks.execution.affinity.shutil.which", return_value="/usr/bin/numactl"
        ):
            pinning = resolve_cpu_pinning("7", sysfs_root=Path(temporary))
        self.assertEqual(pinning.prefix, ())
        self.assertIn("cannot resolve PCI bus id", pinning.description)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.execution.affinity.run_text", return_value="00000000:E3:00.0\n"
        ), mock.patch(
            "benchmarks.execution.affinity.shutil.which", return_value="/usr/bin/numactl"
        ):
            no_affinity = self._sysfs(Path(temporary), "0000:e3:00.0", "-1")
            pinning = resolve_cpu_pinning("7", sysfs_root=no_affinity)
        self.assertEqual(pinning.prefix, ())
        self.assertIn("no NUMA affinity", pinning.description)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.execution.affinity.run_text", return_value="00010001:03:00.0\n"
        ), mock.patch(
            "benchmarks.execution.affinity.shutil.which", return_value="/usr/bin/numactl"
        ):
            truncatable = self._sysfs(Path(temporary), "0001:03:00.0", "0")
            pinning = resolve_cpu_pinning("7", sysfs_root=truncatable)
        self.assertEqual(pinning.prefix, ())
        self.assertIn("unsupported PCI domain", pinning.description)


class ManifestTests(unittest.TestCase):
    def test_manifest_records_run_configuration(self) -> None:
        scenario = scenario_by_name("piper1b_swiglu")
        selected = (scenario.arm("piper_optimized_triton"),)
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
                out_dir,
                scenario,
                selected,
                commands,
                "rtx-a6000",
                metadata,
                extra_args,
                "cuda-graph",
                "none",
                "1b",
                parallelism=TRIVIAL_SPEC,
                megatron_p2p_sync="on",
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        self.assertEqual(manifest["schema_version"], 13)
        self.assertEqual(manifest["compile_mode"], "cuda-graph")
        self.assertEqual(manifest["ac_mode"], "none")
        self.assertEqual(manifest["model_size"], "1b")
        self.assertEqual(manifest["megatron_p2p_sync"], "on")
        self.assertEqual(manifest["model_shape"], PIPER_1B.describe(seq_len=1024))
        self.assertEqual(
            manifest["execution_model"], "single-gpu-plain-bf16-no-fsdp"
        )
        self.assertEqual(manifest["scenario"], "piper1b_swiglu")
        self.assertEqual(manifest["hardware"], "rtx-a6000")
        self.assertEqual(manifest["hardware_metadata"], metadata)
        self.assertEqual(manifest["workload"]["local_batch_size"], 4)
        self.assertEqual(manifest["workload"]["seq_len"], 1024)
        self.assertEqual(manifest["selected_arms"], ["piper_optimized_triton"])
        self.assertEqual(
            [region["name"] for region in manifest["regions"]],
            ["backward_block", "forward_block"],
        )
        self.assertEqual(manifest["extra_torchtitan_args"], extra_args)
        fused_command = manifest["commands"]["piper_optimized_triton"]
        self.assertIn(PIPER_OPTIMIZED_SWIGLU_OVERRIDE, fused_command)
        self.assertIn("--debug.seed", fused_command)
        self.assertEqual(fused_command[-2], "--dump-folder")


class UncompiledRunTests(unittest.TestCase):
    """What --compile-mode none records, and what it declines to claim."""

    def _run(self, out_dir: Path) -> dict:
        metadata = {
            "requested_gpu": "0",
            "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
            "torch_version": "test",
            "torchtitan_git_rev": "titan-rev",
            "benchmarks_git_rev": "bench-rev",
        }

        def fake_process(command, **kwargs):
            # No compile line: this is what an eager arm's log looks like.
            kwargs["stdout"].write(_SAC_LINE + _SIZE_LINE + "Training completed\n")
            arm_dir = Path(command[command.index("--dump-folder") + 1])
            for iteration in (20, 40):
                trace = (
                    arm_dir
                    / f"profiling/traces/iteration_{iteration}/rank0_trace.json.gz"
                )
                trace.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(trace, "wt") as trace_file:
                    trace_file.write("cudaLaunchKernel\n")
            return SimpleNamespace(returncode=0)

        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            execute_run(
                RunRequest(
                    gpu="0",
                    scenario_name="piper1b_rope",
                    arm_names=("baseline",),
                    out_dir=out_dir,
                    compile_mode="none",
                ),
                process_runner=fake_process,
                environment={"PATH": os.environ["PATH"]},
            )
        return json.loads((out_dir / "manifest.json").read_text())

    def test_an_uncompiled_run_records_the_mode_and_declares_no_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._run(Path(temporary) / "run")

        self.assertEqual(manifest["schema_version"], 13)
        self.assertEqual(manifest["compile_mode"], "none")
        # Region pooling reads Inductor's compiled-graph annotations, and an
        # eager run emits none. The run says so rather than declare a region
        # rule 7 would then fail to find.
        self.assertEqual(manifest["regions"], [])
        self.assertNotIn("--compile.enable", manifest["commands"]["baseline"])

    def test_a_resume_refuses_to_cross_the_uncompiled_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            self._run(out_dir)
            with self.assertRaisesRegex(Exception, "compile_mode"):
                execute_run(
                    RunRequest(
                        gpu="0",
                        arm_names=("baseline",),
                        resume_dir=out_dir,
                        compile_mode="default",
                    ),
                    environment={"PATH": os.environ["PATH"]},
                )


def _compiled_line(torch_mode: str) -> str:
    """The apply_compile log line validate_arm matches, as torchtitan emits it.

    Takes the torch-level mode name ("default"/"reduce-overhead"), which is
    what reaches the log; the harness-level "cuda-graph" maps onto
    "reduce-overhead".
    """
    return (
        "[titan] - root - INFO - Compiling each TransformerBlock with "
        f"torch.compile (mode={torch_mode})\n"
    )


# The SelectiveAC application line validate_arm requires under ac mode "sac"
# and rejects under "none".
_SAC_LINE = (
    "[titan] - root - INFO - Applied SelectiveAC activation checkpointing "
    "to the model\n"
)

# Validation rule 11's marker: both engines print the parameter count, and a
# run whose --model-size silently failed to apply would otherwise pass every
# other rule. Exported so tests/test_run_validation.py builds the same log.
_SIZE_LINE = (
    "[titan] - root - INFO - Model qwen3 piper_1B "
    f"size: {PIPER_SHAPES['1b'].param_count:,} total parameters\n"
)


class ValidationTests(unittest.TestCase):
    def test_validation_requires_completion_overrides_and_trace_windows(self) -> None:
        arm = PIPER_1B_SWIGLU.arm("piper_optimized_triton")
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
            log = root / "piper_optimized.log"
            completed = _compiled_line("default") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            log.write_text(completed + applied * 16)
            self.assertEqual(len(trace_files(root)), 2)
            validate_arm(arm, root, log, PIPER_1B_SWIGLU.workload)

            log.write_text(completed + applied * 15)
            with self.assertRaisesRegex(RuntimeError, "expected 16 override"):
                validate_arm(arm, root, log, PIPER_1B_SWIGLU.workload)

            log.write_text(
                completed
                + "[Override] torchtitan.overrides.other.thing: fqn ...\n" * 16
            )
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(arm, root, log, PIPER_1B_SWIGLU.workload)

    def _rope_baseline_fixture(self, root: Path, *, cudagraphs: bool) -> Path:
        for iteration in ("iteration_20", "iteration_40"):
            trace = root / "profiling" / "traces" / iteration / "rank0_trace.json.gz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(trace, "wt") as trace_file:
                trace_file.write("cudaLaunchKernel\n")
                if cudagraphs:
                    trace_file.write("cudaGraphLaunch\n")
        return root / "baseline.log"

    def test_the_applied_mode_must_match_the_requested_one(self) -> None:
        arm = PIPER_1B_ROPE.arm("baseline")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root, cudagraphs=True)

            # cuda-graph is delivered to torch.compile as reduce-overhead, so
            # that is the name the log must carry.
            log.write_text(
                _compiled_line("reduce-overhead") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            validate_arm(
                arm, root, log, PIPER_1B_ROPE.workload, compile_mode="cuda-graph"
            )

            log.write_text(_SAC_LINE + _SIZE_LINE + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(
                    arm, root, log, PIPER_1B_ROPE.workload, compile_mode="cuda-graph"
                )

            log.write_text(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(
                    arm, root, log, PIPER_1B_ROPE.workload, compile_mode="cuda-graph"
                )

    def test_default_run_requires_the_default_mode_line(self) -> None:
        arm = PIPER_1B_ROPE.arm("baseline")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root, cudagraphs=False)

            log.write_text(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            validate_arm(arm, root, log, PIPER_1B_ROPE.workload)

            log.write_text(
                _compiled_line("reduce-overhead") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(arm, root, log, PIPER_1B_ROPE.workload)

    def test_cudagraph_mode_requires_a_graph_launch_in_the_traces(self) -> None:
        arm = PIPER_1B_ROPE.arm("baseline")
        applied = _compiled_line("reduce-overhead") + _SAC_LINE
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root, cudagraphs=False)
            log.write_text(applied + _SIZE_LINE + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "cudaGraphLaunch"):
                validate_arm(
                    arm,
                    root,
                    log,
                    PIPER_1B_ROPE.workload,
                    compile_mode="cuda-graph",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root, cudagraphs=True)
            log.write_text(applied + _SIZE_LINE + "Training completed\n")
            validate_arm(
                arm,
                root,
                log,
                PIPER_1B_ROPE.workload,
                compile_mode="cuda-graph",
            )

    def test_megatron_validation_profile(self) -> None:
        arm = Arm(
            name="baseline",
            description="megatron",
            launcher="megatron",
            validation="megatron",
        )
        mode_line = (
            "Megatron-LM training loop (mode=cuda-graph, "
            "cuda_graph_impl=full_iteration)\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root, cudagraphs=True)

            # No torch.compile line, no SelectiveAC line: still valid, and
            # rule 9's cudaGraphLaunch requirement applies to megatron too.
            log.write_text(mode_line + _SIZE_LINE + "Training completed\n")
            validate_arm(
                arm,
                root,
                log,
                PIPER_1B_ROPE.workload,
                compile_mode="cuda-graph",
                ac_mode="none",
            )

            # The driver's mode line must name the requested mode.
            log.write_text(
                "Megatron-LM training loop (mode=default, cuda_graph_impl=none)\n"
                + _SIZE_LINE + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(
                    arm,
                    root,
                    log,
                    PIPER_1B_ROPE.workload,
                    compile_mode="cuda-graph",
                    ac_mode="none",
                )

        # And the graph launch requirement still bites.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root, cudagraphs=False)
            log.write_text(mode_line + _SIZE_LINE + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "cudaGraphLaunch"):
                validate_arm(
                    arm,
                    root,
                    log,
                    PIPER_1B_ROPE.workload,
                    compile_mode="cuda-graph",
                    ac_mode="none",
                )

    def test_an_uncompiled_mode_requires_the_compile_line_to_be_absent(self) -> None:
        arm = PIPER_1B_ROPE.arm("baseline")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root, cudagraphs=False)

            # No compile line at all: this is what an eager arm looks like.
            log.write_text(_SAC_LINE + _SIZE_LINE + "Training completed\n")
            validate_arm(
                arm, root, log, PIPER_1B_ROPE.workload, compile_mode="none"
            )

            # The block compile line proves the arm was compiled, so the run
            # measured the wrong treatment under an eager label.
            log.write_text(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE
                + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "engine compiled the model"):
                validate_arm(
                    arm, root, log, PIPER_1B_ROPE.workload, compile_mode="none"
                )

            # The loss compile line proves it too: one marker covers every
            # component --compile.enable switches on.
            log.write_text(
                "[titan] - root - INFO - Compiling the loss function with "
                "torch.compile\n" + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "engine compiled the model"):
                validate_arm(
                    arm, root, log, PIPER_1B_ROPE.workload, compile_mode="none"
                )

    def test_an_engine_that_cannot_prove_eager_execution_is_refused(self) -> None:
        # megatron-core compiles its jit_fuser regions at import, so its
        # profile declares no compiled_marker. An uncompiled mode that
        # reached it must fail rather than pass unchecked.
        arm = Arm(
            name="baseline",
            description="megatron",
            launcher="megatron",
            validation="megatron",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root, cudagraphs=False)
            log.write_text(_SIZE_LINE + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "cannot prove compile mode"):
                validate_arm(
                    arm,
                    root,
                    log,
                    PIPER_1B_ROPE.workload,
                    compile_mode="none",
                    ac_mode="none",
                )

    def test_the_compiled_marker_is_part_of_the_mode_line(self) -> None:
        # The two halves of rule 8 must name the same log line: mode_line is
        # required to be present under a compiled mode, compiled_marker to be
        # absent under an uncompiled one.
        from benchmarks.e2e.validation import VALIDATION_PROFILES

        profile = VALIDATION_PROFILES["torchtitan"]
        for mode in TORCH_COMPILE_MODE:
            self.assertIn(profile.compiled_marker, profile.mode_line(mode))
            self.assertIn(
                profile.compiled_marker,
                _compiled_line(TORCH_COMPILE_MODE[mode]),
            )

    def test_megatron_mode_line_matches_the_driver_constant(self) -> None:
        # The validation profile and the driver define the contract in two
        # places; this pins them together without importing megatron.
        from benchmarks.e2e.megatron.train import MODE_LINE
        from benchmarks.e2e.validation import VALIDATION_PROFILES

        for mode in ("default", "cuda-graph"):
            rendered = MODE_LINE.format(mode=mode, impl="anything")
            self.assertIn(VALIDATION_PROFILES["megatron"].mode_line(mode), rendered)

    def test_ac_mode_must_match_the_applied_treatment(self) -> None:
        arm = PIPER_1B_ROPE.arm("baseline")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root, cudagraphs=False)

            # sac requested, SelectiveAC absent: the run measured no-AC.
            log.write_text(_compiled_line("default") + _SIZE_LINE + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "ac mode 'sac'"):
                validate_arm(arm, root, log, PIPER_1B_ROPE.workload)

            # none requested, SelectiveAC applied: the run measured SAC.
            log.write_text(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "ac mode 'none'"):
                validate_arm(
                    arm, root, log, PIPER_1B_ROPE.workload, ac_mode="none"
                )

            # none requested, SelectiveAC absent: valid.
            log.write_text(_compiled_line("default") + _SIZE_LINE + "Training completed\n")
            validate_arm(
                arm, root, log, PIPER_1B_ROPE.workload, ac_mode="none"
            )


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


def _write_block_traces(arm_dir: Path, *, cudagraphs: bool = False) -> None:
    """Write two profiler windows with the region structure the runner expects."""
    for iteration in (20, 40):
        trace = (
            arm_dir / f"profiling/traces/iteration_{iteration}/rank0_trace.json.gz"
        )
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace_events = []
        slot = 0
        for graph, phase in (("backward", "backward"), ("forward", "forward")):
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
        if cudagraphs:
            trace_events.append(
                {
                    "ph": "X",
                    "cat": "cuda_runtime",
                    "name": "cudaGraphLaunch",
                    "tid": 1,
                    "ts": 0,
                    "dur": 5,
                }
            )
        with gzip.open(trace, "wt") as trace_file:
            json.dump({"traceEvents": trace_events}, trace_file)


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
            kwargs["stdout"].write(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            _write_block_traces(Path(command[-1]))
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            out_dir = Path(temporary) / "run"
            environment = {"PATH": os.environ["PATH"]}
            request = RunRequest(
                gpu="0",
                scenario_name="piper1b_rope",
                arm_names=("baseline",),
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
                arm_names=("baseline",),
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
                arm_names=("baseline",),
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
                arm_names=("baseline",),
                resume_dir=out_dir,
                extra_args=("--debug.seed", "7"),
            )
            with self.assertRaisesRegex(ValueError, "extra_torchtitan_args"):
                execute_run(
                    conflicting_args,
                    process_runner=fake_process,
                    environment=environment,
                )

            conflicting_mode = RunRequest(
                gpu="0",
                scenario_name=None,
                arm_names=("baseline",),
                resume_dir=out_dir,
                compile_mode="cuda-graph",
            )
            with self.assertRaisesRegex(ValueError, "compile_mode"):
                execute_run(
                    conflicting_mode,
                    process_runner=fake_process,
                    environment=environment,
                )

            conflicting_ac = RunRequest(
                gpu="0",
                scenario_name=None,
                arm_names=("baseline",),
                resume_dir=out_dir,
                ac_mode="none",
            )
            with self.assertRaisesRegex(ValueError, "ac_mode"):
                execute_run(
                    conflicting_ac,
                    process_runner=fake_process,
                    environment=environment,
                )

    def test_resume_rehydrates_the_recorded_compile_mode(self) -> None:
        metadata = {
            "requested_gpu": "0",
            "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
            "torch_version": "test",
            "torchtitan_git_rev": "titan-rev",
            "benchmarks_git_rev": "bench-rev",
        }
        mode = "cuda-graph"

        def fake_process(command, **kwargs):
            kwargs["stdout"].write(
                _compiled_line("reduce-overhead") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            _write_block_traces(Path(command[-1]), cudagraphs=True)
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            out_dir = Path(temporary) / "run"
            environment = {"PATH": os.environ["PATH"]}
            execute_run(
                RunRequest(
                    gpu="0",
                    scenario_name="piper1b_rope",
                    arm_names=("baseline",),
                    out_dir=out_dir,
                    compile_mode=mode,
                ),
                process_runner=fake_process,
                environment=environment,
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["compile_mode"], mode)

            (out_dir / "baseline.log").write_text("interrupted\n")
            retry_process = mock.Mock(side_effect=fake_process)
            execute_run(
                RunRequest(
                    gpu="0",
                    scenario_name=None,
                    arm_names=("baseline",),
                    resume_dir=out_dir,
                ),
                process_runner=retry_process,
                environment=environment,
            )
            retry_command = retry_process.call_args.args[0]
            self.assertEqual(
                retry_command[retry_command.index("--compile.mode") + 1],
                "reduce-overhead",
            )
            self.assertIn(
                "Training completed", (out_dir / "baseline.log").read_text()
            )


if __name__ == "__main__":
    unittest.main()
