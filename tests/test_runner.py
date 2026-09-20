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
    SCENARIOS,
    ENGINES,
    PIPER_1B_MEGATRON_WORKLOAD,
    scenario_by_name,
)
from benchmarks.e2e.results import stable_tps, training_metrics
from benchmarks.e2e.runner import RunRequest, _resolve_run, execute_run, select_arms
from benchmarks.e2e.validation import validate_arm
from benchmarks.execution.affinity import CpuPinning, resolve_cpu_pinning
from dataclasses import replace
from benchmarks.models.piper_qwen3.components.lm_head.losses import (
    PiperOptimizedCrossEntropyLoss,
)
from benchmarks.models.piper_qwen3.config_registry import (
    qwen3_piper_1b,
    qwen3_piper_1b_pretokenized,
)
from torchtitan.components.loss import CrossEntropyLoss
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


PIPER_OPTIMIZED_SWIGLU_OVERRIDE = (
    "benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu."
    "piper_optimized_inductor_fused_grouped_experts"
)

# No registered arm carries an override today. The plumbing stays, so the
# rules that guard it are exercised against a synthetic arm.
OVERRIDE_ARM = Arm(
    name="override_arm",
    description="a synthetic arm that swaps one config node per block",
    compile="torch",
    override_imports=(PIPER_OPTIMIZED_SWIGLU_OVERRIDE,),
    overrides_per_block=1,
)


class ScenarioTests(unittest.TestCase):
    def test_a_request_without_a_scenario_is_refused_and_starts_nothing(
        self,
    ) -> None:
        """There is no default scenario, and an omission fails the run.

        A default could only be reached by an omission, and it would then
        measure one scenario under whatever label the operator assumed --
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

    def test_the_stock_config_uses_plain_cross_entropy(self) -> None:
        self.assertIsInstance(qwen3_piper_1b().loss, CrossEntropyLoss.Config)

    def test_custom_lm_head_losses_honor_loss_compilation(self) -> None:
        compile_config = CompileConfig(enable=True, components=["loss"])

        def passthrough(fn, **kwargs):
            return fn

        with mock.patch("torch.compile", side_effect=passthrough) as compile_fn:
            PiperOptimizedCrossEntropyLoss.Config().build(
                compile_config=compile_config
            )
        self.assertEqual(compile_fn.call_count, 1)

    def test_every_scenario_uses_the_fixed_piper_workload(self) -> None:
        for scenario in SCENARIOS.values():
            self.assertIs(scenario.workload, PIPER_1B_MEGATRON_WORKLOAD)
        self.assertEqual(
            PIPER_1B_MEGATRON_WORKLOAD.module, "benchmarks.models.piper_qwen3"
        )
        self.assertEqual(
            PIPER_1B_MEGATRON_WORKLOAD.config, "qwen3_piper_1b_pretokenized"
        )
        self.assertEqual(PIPER_1B_MEGATRON_WORKLOAD.local_batch_size, 4)
        self.assertEqual(PIPER_1B_MEGATRON_WORKLOAD.seq_len, 1024)
        self.assertEqual(PIPER_1B_MEGATRON_WORKLOAD.steps, 40)


class SelectedArmTests(unittest.TestCase):
    """Repeated ``--arm`` is an ordered subset, never a second scenario."""

    def setUp(self) -> None:
        self.scenario = scenario_by_name("engines")
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
        selected = select_arms(self.scenario, ("titan_compiled",))
        self.assertEqual([arm.name for arm in selected], ["titan_compiled"])

    def test_several_names_preserve_request_order(self) -> None:
        selected = select_arms(
            self.scenario, ("titan_compiled", "megatron_stock")
        )
        self.assertEqual(
            [arm.name for arm in selected], ["titan_compiled", "megatron_stock"]
        )

    def test_a_duplicate_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, r"--arm repeats 'megatron_stock'"):
            select_arms(self.scenario, ("megatron_stock", "megatron_stock"))

    def test_unknown_names_are_refused_together(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"has no arm\(s\) 'missing', 'also_missing'"
        ):
            select_arms(self.scenario, ("missing", "also_missing"))

    def _resolve(self, names: tuple[str, ...]):
        request = RunRequest(
            gpu="0",
            scenario_name=self.scenario.name,
            arm_names=names,
            out_dir=Path("/tmp/selected-arm-test"),
            ac_mode="none",
        )
        return _resolve_run(request, {"PATH": os.environ["PATH"]})

    def test_every_subset_resolves_and_keeps_its_order(self) -> None:
        cases = (
            ("megatron_stock", "titan_compiled"),
            ("titan_eager",),
            ("megatron_stock",),
            (),
        )
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", self.metadata),
        ) as hardware, mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            for names in cases:
                with self.subTest(names=names):
                    hardware.reset_mock()
                    resolved = self._resolve(names)
                    expected = list(names) or [
                        arm.name for arm in self.scenario.arms
                    ]
                    self.assertEqual(
                        [arm.name for arm in resolved[2]], expected
                    )
                    hardware.assert_called_once()

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
                    scenario_name="engines",
                    arm_names=("titan_eager", "titan_compiled"),
                    out_dir=out_dir,
                    ac_mode="none",
                ),
                process_runner=fake_process,
                environment={"PATH": os.environ["PATH"]},
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())
            state = json.loads((out_dir / "run_state.json").read_text())
            self.assertEqual(launched, ["titan_eager", "titan_compiled"])
            self.assertEqual(
                manifest["selected_arms"], ["titan_eager", "titan_compiled"]
            )
            self.assertEqual(
                list(manifest["commands"]), ["titan_eager", "titan_compiled"]
            )
            self.assertEqual(
                list(state["arms"]), ["titan_eager", "titan_compiled"]
            )

            with self.assertRaisesRegex(ValueError, "selected_arms"):
                execute_run(
                    RunRequest(
                        gpu="0",
                        arm_names=("titan_compiled", "titan_eager"),
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
        scenario_name: str = "engines",
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
                        scenario_name="engines",
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
            arm_names=("megatron_stock",),
            megatron_p2p_sync="off",
        )

    def test_off_without_a_megatron_arm_is_refused_before_any_host_probe(
        self,
    ) -> None:
        """The value needs at least one arm that is not TorchTitan.

        TorchTitan sends no pipeline message through Megatron, so the
        value would reach nothing.
        """
        self._refused_before_any_probe(
            "reaches no arm",
            gpu="0,1",
            arm_names=("titan_compiled",),
            parallelism=self.PP2,
            megatron_p2p_sync="off",
        )

    def test_an_unknown_value_is_refused(self) -> None:
        self._refused_before_any_probe(
            "unknown megatron p2p sync",
            gpu="0,1",
            arm_names=("megatron_stock",),
            parallelism=self.PP2,
            megatron_p2p_sync="false",
        )

    def test_off_reaches_the_megatron_command_and_not_the_titan_one(
        self,
    ) -> None:
        """``run --arm`` narrows the engine set, and a mixed selection is
        legal: the megatron arm gets the flag and the titan arm gets
        nothing."""
        resolved = self._resolve(("megatron_stock", "titan_compiled"))
        self.assertEqual(resolved[11], "off")
        commands = resolved[6]
        megatron = commands["megatron_stock"]
        self.assertEqual(megatron[-2:], ["--bench-batch-p2p-sync", "off"])
        self.assertEqual(_p2p_flags(commands["titan_compiled"]), [])

    def test_a_megatron_only_subset_passes(self) -> None:
        resolved = self._resolve(("megatron_stock",))
        self.assertEqual([arm.name for arm in resolved[2]], ["megatron_stock"])
        self.assertEqual(resolved[11], "off")

    def test_the_default_resolves_to_on_and_adds_no_token(self) -> None:
        for requested in (None, "on"):
            with self.subTest(requested=requested):
                resolved = self._resolve(
                    ("megatron_stock", "titan_compiled"), megatron_p2p_sync=requested
                )
                self.assertEqual(resolved[11], "on")
                for name, command in resolved[6].items():
                    self.assertEqual(_p2p_flags(command), [], name)

    def test_the_stock_scenario_takes_the_value_too(self) -> None:
        resolved = self._resolve(
            ("megatron_stock",), scenario_name="engines"
        )
        command = resolved[6]["megatron_stock"]
        self.assertEqual(command[-2:], ["--bench-batch-p2p-sync", "off"])

    def _write_manifest(self, out_dir: Path, megatron_p2p_sync: str) -> None:
        scenario = scenario_by_name("engines")
        write_manifest(
            out_dir,
            scenario,
            (scenario.arm("megatron_stock"),),
            {"megatron_stock": ["cmd"]},
            "test-gpu",
            {**self.metadata, "cpu_pinning": "none: test"},
            (),
            "none",
            "1b",
            parallelism=self.PP2,
            megatron_p2p_sync=megatron_p2p_sync,
            megatron_nan_guard="on",
            megatron_precision="stock",
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
                    arm_names=("megatron_stock",),
                    resume_dir=out_dir,
                    parallelism=self.PP2,
                    megatron_p2p_sync=megatron_p2p_sync,
                ),
                {"PATH": os.environ["PATH"]},
            )

    def test_a_resume_inherits_the_recorded_value_and_refuses_another(
        self,
    ) -> None:
        """The gate reads like --ac's: an omitted value inherits
        the recorded one and rebuilds the same argv, a different value is
        refused, and the refusal names the field."""
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            out_dir.mkdir()
            self._write_manifest(out_dir, "off")
            resolved = self._resume(out_dir, megatron_p2p_sync=None)
            self.assertEqual(resolved[11], "off")
            self.assertTrue(resolved[10])
            self.assertEqual(
                resolved[6]["megatron_stock"][-2:],
                ["--bench-batch-p2p-sync", "off"],
            )
            with self.assertRaisesRegex(ValueError, "megatron_p2p_sync"):
                self._resume(out_dir, megatron_p2p_sync="on")

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
                    scenario_name="engines",
                    arm_names=("megatron_stock",),
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
                        scenario_name="engines",
                        arm_names=("megatron_stock",),
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


NO_NAN_CHECK = "--no-check-for-nan-in-loss-and-grad"


class MegatronNanGuardResolutionTests(unittest.TestCase):
    """What ``_resolve_run`` does with ``--megatron-nan-guard``.

    Both refusals are parent-side and land before any host probe. The value
    reaches the stock megatron command alone; the tuned driver has no guard
    and refuses ``off``; a TorchTitan argv is untouched under either value.
    Legal at every mesh, so every case here is the trivial spec.
    """

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
        megatron_nan_guard: str | None = "off",
        scenario_name: str = "engines",
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
                    gpu="0",
                    scenario_name=scenario_name,
                    arm_names=names,
                    out_dir=Path("/tmp/nan-guard-test"),
                    ac_mode="none",
                    megatron_nan_guard=megatron_nan_guard,
                ),
                {"PATH": os.environ["PATH"]},
            )

    def _refused_before_any_probe(
        self, pattern: str, scenario_name: str, **keywords
    ) -> None:
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
                        gpu="0",
                        scenario_name=scenario_name,
                        out_dir=Path("/tmp/nan-guard-test"),
                        ac_mode="none",
                        **keywords,
                    ),
                    {"PATH": os.environ["PATH"]},
                )

    def test_off_without_a_stock_megatron_arm_is_refused_before_any_probe(
        self,
    ) -> None:
        """A TorchTitan-only run gives the value nothing to reach."""
        for scenario_name, names in (
            ("engines", ("titan_compiled",)),
            ("engines", ("titan_compiled",)),
        ):
            with self.subTest(scenario=scenario_name):
                self._refused_before_any_probe(
                    "reaches no arm",
                    scenario_name,
                    arm_names=names,
                    megatron_nan_guard="off",
                )

    def test_an_unknown_value_is_refused(self) -> None:
        self._refused_before_any_probe(
            "unknown megatron nan guard",
            "engines",
            arm_names=("megatron_stock",),
            megatron_nan_guard="false",
        )

    def test_off_reaches_the_stock_command_and_not_the_titan_one(self) -> None:
        """A mixed selection is legal: the stock arm gets Megatron's own
        token, once, ahead of the harness group, and the titan arm gets
        nothing."""
        resolved = self._resolve(("megatron_stock", "titan_compiled"))
        self.assertEqual(resolved[12], "off")
        commands = resolved[6]
        stock = commands["megatron_stock"]
        self.assertEqual(stock.count(NO_NAN_CHECK), 1)
        self.assertLess(stock.index(NO_NAN_CHECK), stock.index("--bench-arm-dir"))
        self.assertNotIn(NO_NAN_CHECK, commands["titan_compiled"])

    def test_a_stock_only_subset_passes(self) -> None:
        resolved = self._resolve(("megatron_stock",))
        self.assertEqual([arm.name for arm in resolved[2]], ["megatron_stock"])
        self.assertEqual(resolved[12], "off")

    def test_the_default_resolves_to_on_and_adds_no_token(self) -> None:
        for requested in (None, "on"):
            with self.subTest(requested=requested):
                resolved = self._resolve(
                    ("megatron_stock", "titan_compiled"), megatron_nan_guard=requested
                )
                self.assertEqual(resolved[12], "on")
                for name, command in resolved[6].items():
                    self.assertNotIn(NO_NAN_CHECK, command, name)

    def test_the_banner_names_the_value(self) -> None:
        """The banner names every comparability boundary, and this value
        is one."""
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
                        gpu="0",
                        scenario_name="engines",
                        arm_names=("megatron_stock",),
                        out_dir=Path(temporary) / "run",
                        ac_mode="none",
                        megatron_nan_guard="off",
                    ),
                    event_handler=lambda event: events.append(
                        event.message if event.kind == "summary" else ""
                    ),
                    process_runner=failing_process,
                    environment={"PATH": os.environ["PATH"]},
                )
        self.assertIn("megatron nan guard: off", events)

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
                    gpu="0",
                    scenario_name="engines",
                    arm_names=("megatron_stock",),
                    out_dir=Path(temporary) / "run",
                    ac_mode="none",
                    megatron_nan_guard="off",
                ),
                process_runner=fake_process,
                environment={"PATH": os.environ["PATH"]},
            )
        self.assertEqual(validate.call_count, 1)
        self.assertEqual(validate.call_args.kwargs["megatron_nan_guard"], "off")
        self.assertEqual(validate.call_args.kwargs["megatron_p2p_sync"], "on")

    def _write_manifest(self, out_dir: Path, megatron_nan_guard: str) -> None:
        scenario = scenario_by_name("engines")
        write_manifest(
            out_dir,
            scenario,
            (scenario.arm("megatron_stock"),),
            {"megatron_stock": ["cmd"]},
            "test-gpu",
            {**self.metadata, "cpu_pinning": "none: test"},
            (),
            "none",
            "1b",
            parallelism=TRIVIAL_SPEC,
            megatron_p2p_sync="on",
            megatron_nan_guard=megatron_nan_guard,
            megatron_precision="stock",
        )

    def _resume(self, out_dir: Path, megatron_nan_guard: str | None):
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", self.metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return _resolve_run(
                RunRequest(
                    gpu="0",
                    scenario_name=None,
                    arm_names=("megatron_stock",),
                    resume_dir=out_dir,
                    megatron_nan_guard=megatron_nan_guard,
                ),
                {"PATH": os.environ["PATH"]},
            )

    def test_a_resume_inherits_the_recorded_value_and_refuses_another(
        self,
    ) -> None:
        """An omitted value inherits the recorded one and rebuilds the same
        argv; a different value is refused, and the refusal names the
        field."""
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            out_dir.mkdir()
            self._write_manifest(out_dir, "off")
            resolved = self._resume(out_dir, megatron_nan_guard=None)
            self.assertEqual(resolved[12], "off")
            self.assertTrue(resolved[10])
            self.assertIn(NO_NAN_CHECK, resolved[6]["megatron_stock"])
            with self.assertRaisesRegex(ValueError, "megatron_nan_guard"):
                self._resume(out_dir, megatron_nan_guard="on")


class MegatronPrecisionResolutionTests(unittest.TestCase):
    """What ``_resolve_run`` does with ``--megatron-precision``.

    All three refusals are parent-side and land before any host probe.
    The value reaches the stock megatron command alone. The tuned driver
    builds a plain torch AdamW, so it refuses ``lean``. And ``lean`` needs
    a sharded dense value, because Megatron asserts the distributed
    optimizer under the precision-aware optimizer.
    """

    LEAN_FLAGS = (
        "--use-precision-aware-optimizer",
        "--main-grads-dtype",
        "--exp-avg-dtype",
        "--exp-avg-sq-dtype",
    )

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
        megatron_precision: str | None = "lean",
        zero: str = 1,
        scenario_name: str = "engines",
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
                    gpu="0",
                    scenario_name=scenario_name,
                    arm_names=names,
                    out_dir=Path("/tmp/precision-test"),
                    ac_mode="none",
                    parallelism=ParallelismSpec(
                        dp=1, zero=zero
                    ),
                    megatron_precision=megatron_precision,
                ),
                {"PATH": os.environ["PATH"]},
            )

    def _refused_before_any_probe(
        self, pattern: str, scenario_name: str, **keywords
    ) -> None:
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
                        gpu="0",
                        scenario_name=scenario_name,
                        out_dir=Path("/tmp/precision-test"),
                        ac_mode="none",
                        **keywords,
                    ),
                    {"PATH": os.environ["PATH"]},
                )

    def test_lean_without_a_stock_arm_is_refused_before_any_probe(self) -> None:
        """TorchTitan holds its own bf16 optimizer states, so a titan-only
        run gives the value nothing to reach."""
        for scenario_name, names in (
            ("engines", ("titan_compiled",)),
            ("engines", ("titan_compiled",)),
        ):
            with self.subTest(scenario=scenario_name):
                self._refused_before_any_probe(
                    "reaches no arm",
                    scenario_name,
                    arm_names=names,
                    megatron_precision="lean",
                )

    def test_an_unknown_value_is_refused(self) -> None:
        self._refused_before_any_probe(
            "unknown megatron precision",
            "engines",
            arm_names=("megatron_stock",),
            megatron_precision="bf16",
        )

    def test_lean_reaches_the_stock_command_and_not_the_titan_one(self) -> None:
        """A mixed selection is legal: the stock arm gets the four flags
        and the titan arm gets none of them."""
        resolved = self._resolve(("megatron_stock", "titan_compiled"))
        self.assertEqual(resolved[13], "lean")
        commands = resolved[6]
        stock = commands["megatron_stock"]
        for flag in self.LEAN_FLAGS:
            with self.subTest(flag=flag):
                self.assertEqual(stock.count(flag), 1)
                self.assertNotIn(flag, commands["titan_compiled"])

    def test_the_default_resolves_to_stock_and_adds_no_flag(self) -> None:
        for requested in (None, "stock"):
            with self.subTest(requested=requested):
                resolved = self._resolve(
                    ("megatron_stock", "titan_compiled"), megatron_precision=requested
                )
                self.assertEqual(resolved[13], "stock")
                for name, command in resolved[6].items():
                    for flag in self.LEAN_FLAGS:
                        self.assertNotIn(flag, command, name)

    def _write_manifest(
        self,
        out_dir: Path,
        *,
        megatron_precision: str,
        parallelism: ParallelismSpec = TRIVIAL_SPEC,
    ) -> None:
        scenario = scenario_by_name("engines")
        write_manifest(
            out_dir,
            scenario,
            (scenario.arm("megatron_stock"),),
            {"megatron_stock": ["cmd"]},
            "test-gpu",
            {**self.metadata, "cpu_pinning": "none: test"},
            (),
            "none",
            "1b",
            parallelism=parallelism,
            megatron_p2p_sync="on",
            megatron_nan_guard="on",
            megatron_precision=megatron_precision,
        )

    def _resume(
        self,
        out_dir: Path,
        *,
        megatron_precision: str | None = None,
        parallelism: ParallelismSpec | None = None,
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
                    gpu="0",
                    scenario_name=None,
                    arm_names=("megatron_stock",),
                    resume_dir=out_dir,
                    parallelism=parallelism,
                    megatron_precision=megatron_precision,
                ),
                {"PATH": os.environ["PATH"]},
            )

    def test_a_resume_inherits_the_recorded_precision(self) -> None:
        """Schema 16 records the value, so an omitted one reads back and
        rebuilds the same argv; a different one is refused, and the
        refusal names the field.

        The recorded mesh is sharded, because an inherited ``lean`` under
        ``zero 0`` would meet the parent-side refusal before this gate.
        """
        spec = ParallelismSpec(dp=1, zero=1)
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            out_dir.mkdir()
            self._write_manifest(
                out_dir, megatron_precision="lean", parallelism=spec
            )
            resolved = self._resume(out_dir, parallelism=spec)
            self.assertEqual(resolved[13], "lean")
            self.assertIn(
                "--use-precision-aware-optimizer", resolved[6]["megatron_stock"]
            )
            with self.assertRaisesRegex(ValueError, "megatron_precision"):
                self._resume(
                    out_dir, megatron_precision="stock", parallelism=spec
                )

class ParallelizeTests(unittest.TestCase):
    def test_all_piper_configs_run_single_gpu_plain_bf16(self) -> None:
        for factory in (
            qwen3_piper_1b,
            qwen3_piper_1b_pretokenized,
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
    def test_the_workload_config_reaches_a_titan_arm(self) -> None:
        stock = command_for_arm(
            ENGINES.workload,
            ENGINES.arm("titan_compiled"),
            Path("/out/titan_compiled"),
            [],
        )
        self.assertEqual(
            stock[stock.index("--config") + 1],
            "qwen3_piper_1b_pretokenized",
        )
        self.assertEqual(stock[stock.index("--debug.seed") + 1], "42")

    def test_the_model_size_rides_as_a_config_argument(self) -> None:
        # The config name never carries the size: it is delivered as a keyword
        # argument to the config function via the fork's --config-arg.
        for size in PIPER_SHAPES:
            with self.subTest(size=size):
                command = command_for_arm(
                    ENGINES.workload,
                    ENGINES.arm("titan_compiled"),
                    Path("/out/titan_stock"),
                    [],
                    model_size=size,
                )
                self.assertEqual(
                    command[command.index("--config") + 1],
                    "qwen3_piper_1b_pretokenized",
                )
                self.assertEqual(
                    command[command.index("--config-arg") + 1], f"size={size}"
                )
                # The retired scheme spelled the size into the config
                # name. Checked as the mangled name itself rather than as
                # "no token ends in the size": the config family is called
                # qwen3_piper_1b, so at size "1b" the plain name ends in it.
                self.assertNotIn(f"qwen3_piper_1b_pretokenized_{size}", command)

    def test_command_adds_only_the_arm_override_and_dump_folder(self) -> None:
        command = command_for_arm(
            ENGINES.workload,
            OVERRIDE_ARM,
            Path("/out/fused"),
            ["--debug.seed", "42"],
        )
        override_index = command.index("--override.imports")
        self.assertEqual(
            command[override_index + 1],
            PIPER_OPTIMIZED_SWIGLU_OVERRIDE,
        )
        self.assertNotIn("torchtitan.overrides.fused_swiglu.fused_swiglu", command)
        self.assertEqual(command[-2:], ["--dump-folder", "/out/fused"])
        self.assertIn("--debug.seed", command)

    def test_a_stock_command_has_no_override(self) -> None:
        command = command_for_arm(
            ENGINES.workload,
            ENGINES.arm("titan_compiled"),
            Path("/out/titan_stock"),
            [],
        )
        self.assertNotIn("--override.imports", command)
        self.assertEqual(
            command[:5],
            [
                "./run_train.sh",
                "--module",
                "benchmarks.models.piper_qwen3",
                "--config",
                "qwen3_piper_1b_pretokenized",
            ],
        )
        self.assertIn("--compile.enable", command)
        self.assertIn("--profiler.enable_profiling", command)

    def test_an_eager_arm_drops_only_the_compile_flag(self) -> None:
        # CompileConfig.enable is False in the fork, so an eager arm omits
        # the flag rather than negating it. Everything else must be the
        # command the compiled arm builds, token for token: this is the
        # assertion that keeps the arm property from moving an existing
        # default.
        compiled = command_for_arm(
            ENGINES.workload,
            ENGINES.arm("titan_compiled"),
            Path("/out/baseline"),
            [],
        )
        eager = command_for_arm(
            ENGINES.workload,
            ENGINES.arm("titan_eager"),
            Path("/out/baseline"),
            [],
        )
        self.assertNotIn("--compile.enable", eager)
        self.assertEqual(
            eager, [token for token in compiled if token != "--compile.enable"]
        )

    def test_every_arm_declares_one_of_the_two_compile_values(self) -> None:
        """The field is required, and only two values exist."""
        for name, scenario in SCENARIOS.items():
            for arm in scenario.arms:
                with self.subTest(scenario=name, arm=arm.name):
                    self.assertIn(arm.compile, ("torch", "none"))

    def test_only_the_compiled_arm_asks_for_torch_compile(self) -> None:
        scenario = scenario_by_name("engines")
        self.assertEqual(scenario.arm("titan_compiled").compile, "torch")
        self.assertEqual(scenario.arm("titan_eager").compile, "none")
        self.assertEqual(scenario.arm("megatron_stock").compile, "none")

    def test_ac_none_adds_the_subcommand_token_last(self) -> None:
        command = command_for_arm(
            ENGINES.workload,
            ENGINES.arm("titan_compiled"),
            Path("/out/baseline"),
            [],
            "none",
        )
        # tyro attributes flags after a subcommand token to that subcommand,
        # so the token must trail everything, including --dump-folder.
        self.assertEqual(command[-1], "activation-checkpoint:none")
        self.assertEqual(command[-3:-1], ["--dump-folder", "/out/baseline"])

    def test_ac_sac_leaves_the_command_untouched(self) -> None:
        command = command_for_arm(
            ENGINES.workload,
            ENGINES.arm("titan_compiled"),
            Path("/out/baseline"),
            [],
        )
        self.assertNotIn("activation-checkpoint:none", command)

    def test_each_arm_gets_its_own_dump_folder(self) -> None:
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                command = command_for_arm(
                    scenario.workload, arm, Path("/out") / arm.name, [], "none"
                )
                self.assertIn(f"/out/{arm.name}", command)


class EnginesScenarioTests(unittest.TestCase):
    def test_scenario_registration(self) -> None:
        scenario = scenario_by_name("engines")
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            ["titan_compiled", "titan_eager", "megatron_stock"],
        )
        stock = scenario.arm("megatron_stock")
        self.assertEqual(stock.launcher, "megatron_stock")
        self.assertEqual(stock.validation, "megatron_stock")
        self.assertIn("NOT PLAIN BF16", stock.description)
        self.assertEqual(
            stock.trace_kernel_markers,
            ("cudnn_generated_fort_native_sdpa", "_mul_silu_split"),
        )
        self.assertEqual(scenario.supported_ac_modes, ("none",))
        self.assertEqual(scenario.workload.seed, 42)
        for arm in scenario.arms[:2]:
            self.assertEqual(arm.launcher, "torchtitan")

    def test_every_titan_arm_reads_the_replay_stream(self) -> None:
        import benchmarks.models.piper_qwen3.config_registry as registry
        from benchmarks.e2e.data.piper_qwen3 import PretokenizedReplayDataLoader

        scenario = scenario_by_name("engines")
        config_names = {
            arm.config or scenario.workload.config
            for arm in scenario.arms
            if arm.launcher == "torchtitan"
        }
        for name in sorted(config_names):
            config = getattr(registry, name)()
            self.assertIsInstance(
                config.dataloader, PretokenizedReplayDataLoader.Config, name
            )
            self.assertEqual(config.dataloader.replay_steps, 40, name)

    def test_every_arm_command_builds_at_ac_none(self) -> None:
        scenario = scenario_by_name("engines")
        for arm in scenario.arms:
            command = command_for_arm(
                scenario.workload,
                arm,
                Path("/out") / arm.name,
                [],
                "none",
            )
            self.assertTrue(command, arm.name)

    def test_run_refuses_sac_for_the_megatron_scenario(self) -> None:
        request = RunRequest(
            gpu="0", scenario_name="engines", ac_mode="sac"
        )
        with self.assertRaisesRegex(ValueError, "does not support ac mode"):
            execute_run(request, environment={"PATH": os.environ["PATH"]})


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
        scenario = scenario_by_name("engines")
        selected = (scenario.arm("titan_eager"),)
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
                "none",
                "1b",
                parallelism=TRIVIAL_SPEC,
                megatron_p2p_sync="on",
                megatron_nan_guard="on",
                megatron_precision="stock",
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        self.assertEqual(manifest["schema_version"], 17)
        self.assertEqual(manifest["ac_mode"], "none")
        self.assertEqual(manifest["model_size"], "1b")
        self.assertEqual(manifest["megatron_p2p_sync"], "on")
        self.assertEqual(manifest["megatron_nan_guard"], "on")
        self.assertEqual(manifest["model_shape"], PIPER_1B.describe(seq_len=1024))
        self.assertEqual(
            manifest["execution_model"], "single-gpu-plain-bf16-no-fsdp"
        )
        self.assertEqual(manifest["scenario"], "engines")
        self.assertEqual(manifest["hardware"], "rtx-a6000")
        self.assertEqual(manifest["hardware_metadata"], metadata)
        self.assertEqual(manifest["workload"]["local_batch_size"], 4)
        self.assertEqual(manifest["workload"]["seq_len"], 1024)
        self.assertEqual(manifest["selected_arms"], ["titan_eager"])
        self.assertEqual(manifest["extra_torchtitan_args"], extra_args)
        titan_command = manifest["commands"]["titan_eager"]
        self.assertIn("--debug.seed", titan_command)
        self.assertEqual(titan_command[-2], "--dump-folder")


class EagerArmTests(unittest.TestCase):
    """What the eager arm records, and what it declines to claim."""

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
            kwargs["stdout"].write(_SIZE_LINE + "Training completed\n")
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
                    scenario_name="engines",
                    arm_names=("titan_eager",),
                    out_dir=out_dir,
                    ac_mode="none",
                ),
                process_runner=fake_process,
                environment={"PATH": os.environ["PATH"]},
            )
        return json.loads((out_dir / "manifest.json").read_text())

    def test_an_eager_arm_builds_no_compile_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._run(Path(temporary) / "run")

        self.assertEqual(manifest["schema_version"], 17)
        self.assertNotIn("--compile.enable", manifest["commands"]["titan_eager"])

    def test_the_manifest_records_each_arm_compile_treatment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._run(Path(temporary) / "run")

        recorded = {arm["name"]: arm["compile"] for arm in manifest["arms"]}
        self.assertEqual(
            recorded,
            {
                "titan_compiled": "torch",
                "titan_eager": "none",
                "megatron_stock": "none",
            },
        )


def _compiled_line(torch_mode: str) -> str:
    """The apply_compile log line validate_arm matches, as torchtitan emits it.

    Takes the torch-level mode name, which is what reaches the log.
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
        arm = OVERRIDE_ARM
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
            validate_arm(arm, root, log, ENGINES.workload)

            log.write_text(completed + applied * 15)
            with self.assertRaisesRegex(RuntimeError, "expected 16 override"):
                validate_arm(arm, root, log, ENGINES.workload)

            log.write_text(
                completed
                + "[Override] torchtitan.overrides.other.thing: fqn ...\n" * 16
            )
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(arm, root, log, ENGINES.workload)

    def _rope_baseline_fixture(self, root: Path) -> Path:
        for iteration in ("iteration_20", "iteration_40"):
            trace = root / "profiling" / "traces" / iteration / "rank0_trace.json.gz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(trace, "wt") as trace_file:
                trace_file.write("cudaLaunchKernel\n")
        return root / "baseline.log"

    def test_a_compiled_arm_requires_the_compile_line(self) -> None:
        arm = ENGINES.arm("titan_compiled")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root)

            log.write_text(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            validate_arm(arm, root, log, ENGINES.workload)

            log.write_text(_SAC_LINE + _SIZE_LINE + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "did not apply it"):
                validate_arm(arm, root, log, ENGINES.workload)

    def test_an_eager_arm_refuses_the_compile_line(self) -> None:
        """Rule 8 inverts on an eager arm: a run that silently compiled
        cannot be published as eager."""
        arm = ENGINES.arm("titan_eager")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root)

            log.write_text(_SAC_LINE + _SIZE_LINE + "Training completed\n")
            validate_arm(arm, root, log, ENGINES.workload)

            log.write_text(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "compiled the model"):
                validate_arm(arm, root, log, ENGINES.workload)

    def test_ac_mode_must_match_the_applied_treatment(self) -> None:
        arm = ENGINES.arm("titan_compiled")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root)

            # sac requested, SelectiveAC absent: the run measured no-AC.
            log.write_text(_compiled_line("default") + _SIZE_LINE + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "ac mode 'sac'"):
                validate_arm(arm, root, log, ENGINES.workload)

            # none requested, SelectiveAC applied: the run measured SAC.
            log.write_text(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "ac mode 'none'"):
                validate_arm(
                    arm, root, log, ENGINES.workload, ac_mode="none"
                )

            # none requested, SelectiveAC absent: valid.
            log.write_text(_compiled_line("default") + _SIZE_LINE + "Training completed\n")
            validate_arm(
                arm, root, log, ENGINES.workload, ac_mode="none"
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


def _write_block_traces(arm_dir: Path) -> None:
    """Write the two profiler windows the runner expects."""
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
                _compiled_line("default") + _SIZE_LINE + "Training completed\n"
            )
            _write_block_traces(
                Path(command[command.index("--dump-folder") + 1])
            )
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
                scenario_name="engines",
                arm_names=("titan_compiled",),
                out_dir=out_dir,
                ac_mode="none",
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
                arm_names=("titan_compiled",),
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

            (out_dir / "titan_compiled.log").write_text("interrupted\n")
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
                (out_dir / "attempts").glob("*/titan_compiled/titan_compiled.log")
            )
            self.assertEqual(len(archived_logs), 1)
            self.assertIn("interrupted", archived_logs[0].read_text())
            self.assertIn("Training completed", (out_dir / "titan_compiled.log").read_text())

            incompatible = RunRequest(
                gpu="0",
                scenario_name=None,
                arm_names=("titan_compiled",),
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
                arm_names=("titan_compiled",),
                resume_dir=out_dir,
                extra_args=("--debug.seed", "7"),
            )
            with self.assertRaisesRegex(ValueError, "extra_torchtitan_args"):
                execute_run(
                    conflicting_args,
                    process_runner=fake_process,
                    environment=environment,
                )

            conflicting_size = RunRequest(
                gpu="0",
                scenario_name=None,
                arm_names=("titan_compiled",),
                resume_dir=out_dir,
                model_size="huge",
            )
            with self.assertRaisesRegex(ValueError, "model_size"):
                execute_run(
                    conflicting_size,
                    process_runner=fake_process,
                    environment=environment,
                )

    def test_resume_rehydrates_the_recorded_ac_mode(self) -> None:
        metadata = {
            "requested_gpu": "0",
            "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
            "torch_version": "test",
            "torchtitan_git_rev": "titan-rev",
            "benchmarks_git_rev": "bench-rev",
        }
        mode = "none"

        def fake_process(command, **kwargs):
            kwargs["stdout"].write(_SIZE_LINE + "Training completed\n")
            _write_block_traces(
                Path(command[command.index("--dump-folder") + 1])
            )
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
                    scenario_name="engines",
                    arm_names=("titan_eager",),
                    out_dir=out_dir,
                    ac_mode=mode,
                ),
                process_runner=fake_process,
                environment=environment,
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["ac_mode"], mode)

            (out_dir / "titan_eager.log").write_text("interrupted\n")
            retry_process = mock.Mock(side_effect=fake_process)
            execute_run(
                RunRequest(
                    gpu="0",
                    scenario_name=None,
                    arm_names=("titan_eager",),
                    resume_dir=out_dir,
                ),
                process_runner=retry_process,
                environment=environment,
            )
            retry_command = retry_process.call_args.args[0]
            self.assertNotIn("--compile.enable", retry_command)
            self.assertIn("activation-checkpoint:none", retry_command)
            self.assertIn(
                "Training completed", (out_dir / "titan_eager.log").read_text()
            )


if __name__ == "__main__":
    unittest.main()
