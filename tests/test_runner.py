"""CPU-only tests for benchmark scenario construction and validation helpers."""

import ast
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.layout import trace_files
from benchmarks.artifacts.manifest_v19 import upgrade_v19
from benchmarks.artifacts.manifests import load_manifest, run_record
from benchmarks.e2e.axes import RequestedAxes, RunRequest
from benchmarks.e2e.checks import check_run
from benchmarks.e2e.overrides import apply_overrides, parse_override
from benchmarks.e2e.engines.api import Arm, CompileMode
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig
from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.parallelism import TRIVIAL_SPEC
from benchmarks.e2e.registry import (
    SCENARIOS,
    C4_REPLAY_DATA,
    ENGINES,
    SEED,
    scenario_by_name,
)
from tests.engine_helpers import (
    command,
    configured,
    run_spec,
    titan_step_line,
    validate,
)
from benchmarks.e2e.engines.api import ProfileWindow, StepSample
from benchmarks.e2e.results import arm_steps, stable_samples
from benchmarks.e2e.runner import (
    _resolve_run,
    check_request,
    execute_run,
    select_arms,
)
from benchmarks.execution.affinity import CpuPinning, resolve_cpu_pinning
from dataclasses import asdict, fields, replace
from benchmarks.models.piper_qwen3.components.lm_head.losses import (
    PiperOptimizedCrossEntropyLoss,
)
from benchmarks.e2e.engines.torchtitan.plugins.config_registry import (
    qwen3_piper_1b_pretokenized,
)
from torchtitan.components.loss import CrossEntropyLoss
from benchmarks.e2e.engines.torchtitan.plugins.parallelize import (
    DATA_PARALLEL_LINE,
    parallelize_piper1b,
    skip_data_parallel,
)
from benchmarks.models.piper_qwen3.shape import HUGE, PIPER_1B, PIPER_SHAPES
from benchmarks.models.piper_qwen3.titan_model import apply_config_overrides
from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.override import parse_cli_imports
from torchtitan.distributed import ParallelDims


PIPER_OPTIMIZED_SWIGLU_OVERRIDE = (
    "benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu."
    "piper_optimized_inductor_fused_grouped_experts"
)

# No registered arm needs the host compiler today, so a synthetic arm
# exercises the override argv, the override rules and the
# compiler-environment branch of the runner.
OVERRIDE_ARM = Arm(
    name="override_arm",
    description="a synthetic arm that swaps one config node per block",
    config=TorchTitanConfig(
        compile=CompileMode.TORCH,
        override_imports=(PIPER_OPTIMIZED_SWIGLU_OVERRIDE,),
        overrides_per_block=1,
        requires_gcc_toolset=True,
    ),
)

# The eight names ``RequestedAxes`` owns. The refusal helpers below take one
# flat keyword mapping per case, so this list is what splits an axis from a
# plain request field.
_AXIS_KEYWORDS = tuple(field.name for field in fields(RequestedAxes))


def _request_with_axes(**keywords) -> RunRequest:
    """A ``RunRequest`` from flat keywords, axes and all."""
    axes = {
        name: keywords.pop(name)
        for name in list(keywords)
        if name in _AXIS_KEYWORDS
    }
    return RunRequest(axes=RequestedAxes(**axes), **keywords)


class ScenarioTests(unittest.TestCase):
    def test_a_request_without_a_scenario_is_refused_and_starts_nothing(
        self,
    ) -> None:
        """There is no default scenario, and an omission fails the run.

        A default could only be reached by an omission, and it would then
        measure one scenario under whatever label the operator assumed --
        a wrong result rather than a missing one. ``check_request`` holds the
        rule, so the CLI and every programmatic caller inherit it, and no
        scenario name is written anywhere as a default.

        The two patches prove the refusal lands before any work starts.
        ``hardware_metadata`` is the first host probe ``_resolve_run`` makes
        after the check resolves the scenario, and ``process_runner`` launches the
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
                    check_request(
                        RunRequest(gpu="0"),
                        environment={"PATH": os.environ["PATH"]},
                    ),
                    process_runner=never,
                )
        self.assertIn("no scenario requested", str(caught.exception))
        self.assertIn("--scenario", str(caught.exception))

    def test_the_check_reads_the_given_environment_even_when_it_is_empty(self) -> None:
        request = RunRequest(
            axes=RequestedAxes(ac_mode="none", model_size="1b"),
            gpu="0",
            scenario_name="engines",
            out_dir=Path("/tmp/check-environment-test"),
        )
        cache_root = Path("/tmp/check-environment-cache")
        with mock.patch.dict(
            os.environ, {"STEPS": "77", "BENCHMARK_CACHE_ROOT": str(cache_root)}
        ):
            checked = check_request(request, environment={})
            inherited = check_request(request)
        self.assertEqual(inherited.run.data.steps, 77)
        self.assertEqual(inherited.paths.cache_root, cache_root)
        self.assertEqual(checked.environment, {})
        self.assertIs(checked.request, request)
        self.assertEqual(checked.run.data.steps, ENGINES.data.steps)
        self.assertNotEqual(checked.paths.cache_root, cache_root)

    def test_the_stock_config_uses_plain_cross_entropy(self) -> None:
        self.assertIsInstance(
            qwen3_piper_1b_pretokenized().loss, CrossEntropyLoss.Config
        )

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
            self.assertIs(scenario.data, C4_REPLAY_DATA)
            for arm in scenario.arms:
                if engine_for(arm).name == "torchtitan":
                    self.assertEqual(
                        arm.config.module, "benchmarks.e2e.engines.torchtitan.plugins"
                    )
                    self.assertEqual(
                        arm.config.config, "qwen3_piper_1b_pretokenized"
                    )
        self.assertEqual(C4_REPLAY_DATA.local_batch_size, 4)
        self.assertEqual(C4_REPLAY_DATA.seq_len, 4096)
        self.assertEqual(C4_REPLAY_DATA.steps, 40)


class UncompiledScheduleRefusalTests(unittest.TestCase):
    """The refusal parallelism rule 6 used to hold.

    PyTorch's zero-bubble and DualPipeV classes call
    ``_check_torch_compile_compatibility``, which raises on a compiled
    stage module. Compile is a property of each arm, so a spec alone
    cannot answer this: ``check_request`` reads the selected arms and names
    the one that compiles.
    """

    ZBV = ParallelismSpec(pp=2, pp_schedule="ZBVZeroBubble")

    def setUp(self) -> None:
        self.metadata = {
            "requested_gpu": "0",
            "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
            "torch_version": "test",
            "torchtitan_git_rev": "titan-rev",
            "benchmarks_git_rev": "bench-rev",
            "megatron_git_rev": "megatron-rev",
        }

    def _resolve(self, names: tuple[str, ...]):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", self.metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return _resolve_run(
                check_request(
                    RunRequest(
                        axes=RequestedAxes(
                            ac_mode="none",
                            parallelism=self.ZBV,
                        ),
                        gpu="0,1",
                        scenario_name="engines",
                        arm_names=names,
                        out_dir=Path(temporary) / "run",
                        batch=8,
                    ),
                    environment={"PATH": os.environ["PATH"]},
                ),
            )

    def test_a_compiled_arm_is_refused_and_named(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"ZBVZeroBubble.*titan_compiled"
        ):
            self._resolve(("titan_compiled", "titan_eager"))

    def test_the_eager_arm_alone_resolves(self) -> None:
        """Rule 5 refuses the stock megatron arm at this schedule, so the
        eager titan arm is the whole legal selection here."""
        resolved = self._resolve(("titan_eager",))
        self.assertEqual([arm.name for arm in resolved.arms], ["titan_eager"])


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
            axes=RequestedAxes(
                ac_mode="none",
            ),
            gpu="0",
            scenario_name=self.scenario.name,
            arm_names=names,
            out_dir=Path("/tmp/selected-arm-test"),
        )
        return _resolve_run(
            check_request(
                request,
                environment={"PATH": os.environ["PATH"]},
            ),
        )

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
                        [arm.name for arm in resolved.arms], expected
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
                check_request(
                    RunRequest(
                        axes=RequestedAxes(
                            ac_mode="none",
                        ),
                        gpu="0",
                        scenario_name="engines",
                        arm_names=("titan_eager", "titan_compiled"),
                        out_dir=out_dir,
                    ),
                    environment={"PATH": os.environ["PATH"]},
                ),
                process_runner=fake_process,
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())
            state = json.loads((out_dir / "run_state.json").read_text())
            self.assertEqual(launched, ["titan_eager", "titan_compiled"])
            self.assertEqual(
                [arm["name"] for arm in manifest["arms"]],
                ["titan_eager", "titan_compiled"],
            )
            self.assertEqual(
                list(state["arms"]), ["titan_eager", "titan_compiled"]
            )

            with self.assertRaisesRegex(ValueError, "existing manifest: arms$"):
                execute_run(
                    check_request(
                        RunRequest(
                            gpu="0",
                            arm_names=("titan_compiled", "titan_eager"),
                            resume_dir=out_dir,
                        ),
                        environment={"PATH": os.environ["PATH"]},
                    ),
                    process_runner=fake_process,
                )


def _p2p_flags(command: list[str]) -> list[str]:
    """The flag tokens of ``command`` that name the p2p sync.

    Flags only: a path in the argv can carry the substring too.
    """
    return [
        token for token in command if token.startswith("--") and "p2p" in token
    ]


NO_NAN_CHECK = "--no-check-for-nan-in-loss-and-grad"

LEAN_FLAGS = (
    "--use-precision-aware-optimizer",
    "--main-grads-dtype",
    "--exp-avg-dtype",
    "--exp-avg-sq-dtype",
)

_METADATA = {
    "requested_gpu": "0",
    "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
    "torch_version": "test",
    "torchtitan_git_rev": "titan-rev",
    "benchmarks_git_rev": "bench-rev",
    "megatron_git_rev": "mcore-rev",
}

PP2 = ParallelismSpec(pp=2, pp_schedule="1F1B")


def _overrides(*texts: str) -> tuple:
    return tuple(parse_override(text) for text in texts)


def _resolve(
    names: tuple[str, ...],
    *overrides: str,
    gpu: str = "0",
    parallelism: ParallelismSpec | None = None,
    resume_dir: Path | None = None,
):
    """Resolve one ``engines`` request with stubbed host probes."""
    with mock.patch(
        "benchmarks.e2e.runner.hardware_metadata",
        return_value=("test-gpu", _METADATA),
    ), mock.patch(
        "benchmarks.e2e.runner.resolve_cpu_pinning",
        return_value=CpuPinning((), "none: test"),
    ):
        return _resolve_run(
            check_request(
                RunRequest(
                    axes=RequestedAxes(
                        ac_mode=None if resume_dir else "none",
                        parallelism=parallelism,
                    ),
                    gpu=gpu,
                    scenario_name=None if resume_dir else "engines",
                    arm_names=names,
                    out_dir=None if resume_dir else Path("/tmp/override-test"),
                    resume_dir=resume_dir,
                    overrides=_overrides(*overrides),
                ),
                environment={"PATH": os.environ["PATH"]},
            ),
        )


def _refused_before_any_probe(
    test: unittest.TestCase, pattern: str, names: tuple[str, ...], *overrides: str, **keywords
) -> None:
    def never(*args, **kwargs):
        raise AssertionError("a host probe ran for a refused request")

    with mock.patch(
        "benchmarks.e2e.runner.hardware_metadata", side_effect=never
    ), mock.patch(
        "benchmarks.e2e.runner.resolve_cpu_pinning", side_effect=never
    ):
        with test.assertRaisesRegex(ValueError, pattern):
            _resolve_run(
                check_request(
                    RunRequest(
                        axes=RequestedAxes(ac_mode="none", **keywords),
                        gpu="0,1" if keywords.get("parallelism") else "0",
                        scenario_name="engines",
                        arm_names=names,
                        out_dir=Path("/tmp/override-test"),
                        overrides=_overrides(*overrides),
                    ),
                    environment={"PATH": os.environ["PATH"]},
                ),
            )


class OverrideResolutionTests(unittest.TestCase):
    """What ``_resolve_run`` does with ``--set``.

    Each refusal lands before any host probe. A value reaches the arm that
    the override names, and no other arm.
    """

    def test_the_megatron_defaults_reach_the_stock_command(self) -> None:
        resolved = _resolve(("megatron_stock", "titan_compiled"), gpu="0,1", parallelism=PP2)
        stock = resolved.commands["megatron_stock"]
        self.assertEqual(stock[-2:], ["--bench-batch-p2p-sync", "off"])
        self.assertEqual(stock.count(NO_NAN_CHECK), 1)
        self.assertLess(stock.index(NO_NAN_CHECK), stock.index("--bench-arm-dir"))
        titan = resolved.commands["titan_compiled"]
        self.assertEqual(_p2p_flags(titan), [])
        self.assertNotIn(NO_NAN_CHECK, titan)

    def test_on_values_remove_the_stock_tokens(self) -> None:
        resolved = _resolve(
            ("megatron_stock",),
            "megatron_stock.p2p_sync=on",
            "megatron_stock.nan_guard=on",
            gpu="0,1",
            parallelism=PP2,
        )
        (arm,) = resolved.arms
        self.assertEqual((arm.config.p2p_sync, arm.config.nan_guard), ("on", "on"))
        command = resolved.commands["megatron_stock"]
        self.assertEqual(_p2p_flags(command), [])
        self.assertNotIn(NO_NAN_CHECK, command)

    def test_lean_reaches_the_stock_command_and_not_the_titan_one(self) -> None:
        resolved = _resolve(
            ("megatron_stock", "titan_compiled"),
            "megatron_stock.precision=lean",
            parallelism=ParallelismSpec(zero=1),
        )
        for flag in LEAN_FLAGS:
            with self.subTest(flag=flag):
                self.assertEqual(resolved.commands["megatron_stock"].count(flag), 1)
                self.assertNotIn(flag, resolved.commands["titan_compiled"])

    def test_extra_flags_reach_the_named_arm_alone(self) -> None:
        resolved = _resolve(
            ("titan_compiled", "titan_eager"),
            "titan_eager.extra_flags+=--training.gc-freq 7",
        )
        self.assertIn("--training.gc-freq", resolved.commands["titan_eager"])
        self.assertNotIn("--training.gc-freq", resolved.commands["titan_compiled"])

    def test_a_set_on_an_unselected_arm_is_refused_before_any_probe(self) -> None:
        _refused_before_any_probe(
            self,
            "'megatron_stock' is not a selected arm. Available: titan_compiled",
            ("titan_compiled",),
            "megatron_stock.nan_guard=on",
        )

    def test_an_unknown_value_is_refused_and_names_the_choices(self) -> None:
        _refused_before_any_probe(
            self,
            "'false' is not one of on, off",
            ("megatron_stock",),
            "megatron_stock.p2p_sync=false",
        )

    def test_the_engine_check_refuses_each_run_it_cannot_honour(self) -> None:
        _refused_before_any_probe(
            self,
            "no pipeline message.*'lean' needs --zero 1",
            ("megatron_stock",),
            "megatron_stock.p2p_sync=on",
            "megatron_stock.precision=lean",
        )

    def test_one_error_lists_the_shared_and_the_engine_refusals(self) -> None:
        with self.assertRaises(ValueError) as caught:
            check_run(
                run_spec(ac_mode="sac", profile=False, warmup_steps=40),
                ENGINES,
                (configured(ENGINES.arm("megatron_stock"), precision="lean"),),
                device_count=2,
                resumed=None,
            )
        message = str(caught.exception)
        for part in (
            "does not match",
            "more than the 40 warmup step(s)",
            "does not support ac mode 'sac'",
            "'lean' needs --zero 1",
        ):
            with self.subTest(part=part):
                self.assertIn(part, message)

    def test_execute_run_hands_the_config_to_validate_arm(self) -> None:
        def fake_process(command, **kwargs):
            kwargs["stdout"].write("Training completed\n")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", _METADATA),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ), mock.patch("benchmarks.e2e.runner.validate_arm") as validate:
            execute_run(
                check_request(
                    RunRequest(
                        axes=RequestedAxes(ac_mode="none"),
                        gpu="0",
                        scenario_name="engines",
                        arm_names=("megatron_stock",),
                        out_dir=Path(temporary) / "run",
                        overrides=_overrides("megatron_stock.nan_guard=on"),
                    ),
                    environment={"PATH": os.environ["PATH"]},
                ),
                process_runner=fake_process,
            )
        self.assertEqual(validate.call_count, 1)
        arm = validate.call_args.args[1]
        self.assertEqual(arm.config.nan_guard, "on")
        self.assertEqual(arm.config.p2p_sync, "off")

    def test_the_banner_names_each_arm_config(self) -> None:
        events: list[str] = []

        def failing_process(command, **kwargs):
            kwargs["stdout"].write("nothing trained\n")
            return SimpleNamespace(returncode=1)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", _METADATA),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            with self.assertRaises(RuntimeError):
                execute_run(
                    check_request(
                        RunRequest(
                            axes=RequestedAxes(ac_mode="none"),
                            gpu="0",
                            scenario_name="engines",
                            arm_names=("megatron_stock",),
                            out_dir=Path(temporary) / "run",
                        ),
                        environment={"PATH": os.environ["PATH"]},
                    ),
                    event_handler=lambda event: events.append(
                        event.message if event.kind == "summary" else ""
                    ),
                    process_runner=failing_process,
                )
        (line,) = [event for event in events if event.startswith("config ")]
        self.assertIn("config megatron_stock: megatron_stock {", line)
        self.assertIn('"nan_guard": "off"', line)


class OverrideResumeTests(unittest.TestCase):
    """A resume inherits each recorded config field and refuses a changed one."""

    def _record(self, out_dir: Path, *overrides: str, parallelism=None) -> None:
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", _METADATA),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ), mock.patch("benchmarks.e2e.runner.validate_arm"):
            execute_run(
                check_request(
                    RunRequest(
                        axes=RequestedAxes(ac_mode="none", parallelism=parallelism),
                        gpu="0",
                        scenario_name="engines",
                        arm_names=("megatron_stock",),
                        out_dir=out_dir,
                        overrides=_overrides(*overrides),
                    ),
                    environment={"PATH": os.environ["PATH"]},
                ),
                process_runner=lambda command, **kwargs: SimpleNamespace(returncode=0),
            )

    def test_an_omitted_field_inherits_the_recorded_value(self) -> None:
        spec = ParallelismSpec(zero=1)
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            self._record(
                out_dir,
                "megatron_stock.precision=lean",
                "megatron_stock.nan_guard=on",
                parallelism=spec,
            )
            resolved = _resolve(("megatron_stock",), resume_dir=out_dir, parallelism=spec)
            (arm,) = resolved.arms
            self.assertEqual((arm.config.precision, arm.config.nan_guard), ("lean", "on"))
            self.assertTrue(resolved.resumed)
            self.assertIn(
                "--use-precision-aware-optimizer", resolved.commands["megatron_stock"]
            )
            resolved = _resolve(
                ("megatron_stock",),
                "megatron_stock.nan_guard=on",
                resume_dir=out_dir,
                parallelism=spec,
            )
            self.assertEqual(resolved.arms[0].config.precision, "lean")

    def test_a_changed_field_is_refused_and_named(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            self._record(out_dir)
            with self.assertRaisesRegex(
                ValueError, "existing manifest: megatron_stock.config.nan_guard$"
            ):
                _resolve(
                    ("megatron_stock",),
                    "megatron_stock.nan_guard=on",
                    resume_dir=out_dir,
                )
            with self.assertRaisesRegex(
                ValueError, "megatron_stock.config.extra_flags"
            ):
                _resolve(
                    ("megatron_stock",),
                    "megatron_stock.extra_flags+=--moe-permute-fusion",
                    resume_dir=out_dir,
                )

    def test_a_schema_18_manifest_is_refused_by_name(self) -> None:
        golden = (
            Path(__file__).resolve().parent
            / "fixtures/golden/runs/dp1-1b"
        )
        with self.assertRaisesRegex(ValueError, "records manifest schema 18"):
            _resolve(("megatron_stock",), resume_dir=golden)


V19_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "manifest_v19"
"""Two manifests that the dp4 x ep4 attention matrix recorded under schema 19."""


class SchemaNineteenTests(unittest.TestCase):
    def test_a_recorded_schema_19_run_loads_with_packed_offsets_off(self) -> None:
        titan = run_record(
            json.loads((V19_FIXTURES / "titan-compiled-fa3-dp4-ep4.json").read_text()),
            "titan",
        )
        (arm,) = titan.arms
        self.assertEqual(arm.arm.name, "titan_compiled_fa3")
        self.assertIs(arm.arm.config.packed_offsets, False)
        recorded = json.loads(
            (V19_FIXTURES / "megatron-stock-dp4-ep4.json").read_text()
        )
        (megatron,) = run_record(recorded, "megatron").arms
        self.assertEqual(
            asdict(megatron.arm.config),
            {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in recorded["arms"][0]["config"].items()
            },
        )

    def test_a_resume_refuses_a_schema_19_manifest_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            shutil.copy(
                V19_FIXTURES / "titan-compiled-fa3-dp4-ep4.json",
                out_dir / "manifest.json",
            )
            with self.assertRaisesRegex(ValueError, "records manifest schema 19"):
                load_manifest(out_dir)

    def test_the_upgrade_refuses_a_field_that_schema_19_cannot_hold(self) -> None:
        manifest = json.loads(
            (V19_FIXTURES / "titan-compiled-fa3-dp4-ep4.json").read_text()
        )
        manifest["arms"][0]["config"]["packed_offsets"] = True
        with self.assertRaisesRegex(ValueError, "which schema 20 adds"):
            upgrade_v19(manifest)


class ParallelizeTests(unittest.TestCase):
    def test_all_piper_configs_run_single_gpu_plain_bf16(self) -> None:
        for factory in (qwen3_piper_1b_pretokenized,):
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
            "benchmarks.e2e.engines.torchtitan.plugins.parallelize.parallelize_qwen3",
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
            "benchmarks.e2e.engines.torchtitan.plugins.parallelize.FSDPModule", _Wrapped
        ):
            with mock.patch(
                "benchmarks.e2e.engines.torchtitan.plugins.parallelize.parallelize_qwen3",
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
            "benchmarks.e2e.engines.torchtitan.plugins.parallelize.FSDPModule", _Wrapped
        ):
            with mock.patch(
                "benchmarks.e2e.engines.torchtitan.plugins.parallelize.parallelize_qwen3",
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
            "benchmarks.e2e.engines.torchtitan.plugins.parallelize.parallelize_qwen3",
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
        stock = command(
            run_spec(), ENGINES.arm("titan_compiled"), "/out/titan_compiled"
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
                argv = command(
                    run_spec(size),
                    ENGINES.arm("titan_compiled"),
                    "/out/titan_stock",
                )
                self.assertEqual(
                    argv[argv.index("--config") + 1],
                    "qwen3_piper_1b_pretokenized",
                )
                self.assertEqual(
                    argv[argv.index("--config-arg") + 1], f"size={size}"
                )
                # The config family is qwen3_piper_1b, so check the whole name.
                self.assertNotIn(f"qwen3_piper_1b_pretokenized_{size}", argv)

    def test_command_adds_only_the_arm_override_and_dump_folder(self) -> None:
        argv = command(
            run_spec(),
            configured(OVERRIDE_ARM, extra_flags=("--training.gc-freq", "7")),
            "/out/fused",
        )
        override_index = argv.index("--override.imports")
        self.assertEqual(
            argv[override_index + 1],
            PIPER_OPTIMIZED_SWIGLU_OVERRIDE,
        )
        self.assertNotIn("torchtitan.overrides.fused_swiglu.fused_swiglu", argv)
        self.assertEqual(argv[-2:], ["--dump-folder", "/out/fused"])
        self.assertIn("--training.gc-freq", argv)

    def test_a_stock_command_has_no_override(self) -> None:
        argv = command(
            run_spec(), ENGINES.arm("titan_compiled"), "/out/titan_stock"
        )
        self.assertNotIn("--override.imports", argv)
        trainer = argv.index("torchtitan.train") - 1
        self.assertEqual(
            argv[trainer : trainer + 6],
            [
                "-m",
                "torchtitan.train",
                "--module",
                "benchmarks.e2e.engines.torchtitan.plugins",
                "--config",
                "qwen3_piper_1b_pretokenized",
            ],
        )
        self.assertIn("--compile.enable", argv)
        self.assertIn("--profiler.enable_profiling", argv)

    def test_an_eager_arm_drops_only_the_compile_flag(self) -> None:
        # CompileConfig.enable is False in the fork, so an eager arm omits
        # the flag rather than negating it. Everything else must be the
        # command the compiled arm builds, token for token: this is the
        # assertion that keeps the arm property from moving an existing
        # default.
        compiled = command(
            run_spec(), ENGINES.arm("titan_compiled"), "/out/baseline"
        )
        eager = command(run_spec(), ENGINES.arm("titan_eager"), "/out/baseline")
        self.assertNotIn("--compile.enable", eager)
        self.assertEqual(
            eager, [token for token in compiled if token != "--compile.enable"]
        )

    def test_every_titan_arm_declares_one_of_the_two_compile_values(self) -> None:
        """The field is required, and only two values exist."""
        for name, scenario in SCENARIOS.items():
            for arm in scenario.arms:
                if engine_for(arm).name != "torchtitan":
                    continue
                with self.subTest(scenario=name, arm=arm.name):
                    self.assertIsInstance(arm.config.compile, CompileMode)

    def test_only_the_compiled_arm_asks_for_torch_compile(self) -> None:
        scenario = scenario_by_name("engines")
        self.assertIs(scenario.arm("titan_compiled").config.compile, CompileMode.TORCH)
        self.assertIs(scenario.arm("titan_eager").config.compile, CompileMode.NONE)
        self.assertFalse(hasattr(scenario.arm("megatron_stock").config, "compile"))

    def test_ac_none_adds_the_subcommand_token_last(self) -> None:
        argv = command(
            run_spec(ac_mode="none"),
            ENGINES.arm("titan_compiled"),
            "/out/baseline",
        )
        # tyro attributes flags after a subcommand token to that subcommand,
        # so the token must trail everything, including --dump-folder.
        self.assertEqual(argv[-1], "activation-checkpoint:none")
        self.assertEqual(argv[-3:-1], ["--dump-folder", "/out/baseline"])

    def test_ac_sac_leaves_the_command_untouched(self) -> None:
        argv = command(run_spec(), ENGINES.arm("titan_compiled"), "/out/baseline")
        self.assertNotIn("activation-checkpoint:none", argv)

    def test_each_arm_gets_its_own_dump_folder(self) -> None:
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                argv = command(
                    run_spec(ac_mode="none"), arm, Path("/out") / arm.name
                )
                self.assertIn(f"/out/{arm.name}", argv)


class EnginesScenarioTests(unittest.TestCase):
    def test_scenario_registration(self) -> None:
        scenario = scenario_by_name("engines")
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            ["titan_compiled", "titan_eager", "megatron_stock"],
        )
        stock = scenario.arm("megatron_stock")
        self.assertEqual(engine_for(stock).name, "megatron_stock")
        self.assertIn("NOT PLAIN BF16", stock.description)
        self.assertEqual(
            stock.config.trace_kernel_markers,
            ("cudnn_generated_fort_native_sdpa", "_mul_silu_split"),
        )
        self.assertEqual(scenario.supported_ac_modes, ("none",))
        self.assertEqual(SEED, 42)
        for arm in scenario.arms[:2]:
            self.assertEqual(engine_for(arm).name, "torchtitan")

    def test_every_titan_arm_reads_the_replay_stream(self) -> None:
        import benchmarks.e2e.engines.torchtitan.plugins.config_registry as registry
        from benchmarks.e2e.engines.torchtitan.plugins.replay import PretokenizedReplayDataLoader

        scenario = scenario_by_name("engines")
        config_names = {
            arm.config.config
            for arm in scenario.arms
            if engine_for(arm).name == "torchtitan"
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
            argv = command(run_spec(ac_mode="none"), arm, Path("/out") / arm.name)
            self.assertTrue(argv, arm.name)

    def test_run_refuses_sac_for_the_megatron_scenario(self) -> None:
        request = RunRequest(
            axes=RequestedAxes(
                ac_mode="sac",
            ),
            gpu="0",
            scenario_name="engines",
        )
        with self.assertRaisesRegex(ValueError, "does not support ac mode"):
            execute_run(
                check_request(
                    request,
                    environment={"PATH": os.environ["PATH"]},
                ),
            )


FA3_TARGET = "benchmarks.models.piper_qwen3.components.attention.fa3_override.packed_fa3_attention"
"""The override that replaces each block's attention with FA3 varlen attention."""

FA3_MARKERS = ("FlashAttnFwdSm90",)
"""The Hopper forward kernel of FA3 varlen attention."""


class AttentionScenarioTests(unittest.TestCase):
    def test_scenario_registration(self) -> None:
        scenario = scenario_by_name("attention")
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            [
                "titan_compiled",
                "titan_compiled_fa3",
                "megatron_stock",
            ],
        )
        self.assertIs(scenario.data, C4_REPLAY_DATA)
        self.assertEqual(scenario.supported_ac_modes, ("none",))
        self.assertIs(scenario.arm("titan_compiled"), ENGINES.arm("titan_compiled"))
        self.assertIs(scenario.arm("megatron_stock"), ENGINES.arm("megatron_stock"))

    def test_each_kernel_arm_overrides_one_attention_per_block(self) -> None:
        arm = scenario_by_name("attention").arm("titan_compiled_fa3")
        config = arm.config
        self.assertEqual(engine_for(arm).name, "torchtitan")
        self.assertEqual(config.compile, CompileMode.TORCH)
        self.assertEqual(config.override_imports, (FA3_TARGET,))
        self.assertEqual(config.overrides_per_block, 1)
        self.assertEqual(config.trace_kernel_markers, FA3_MARKERS)

    def test_every_arm_command_builds_at_ac_none(self) -> None:
        for arm in scenario_by_name("attention").arms:
            argv = command(run_spec(ac_mode="none"), arm, Path("/out") / arm.name)
            self.assertTrue(argv, arm.name)

    def test_the_fa3_arm_sends_the_rows_of_one_pipeline_microbatch(self) -> None:
        scenario = scenario_by_name("attention")
        pp2 = ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2)
        for spec, rows in ((TRIVIAL_SPEC, "8"), (pp2, "2")):
            run = run_spec(ac_mode="none", parallelism=spec, local_batch_size=8)
            with self.subTest(pp=spec.pp):
                argv = command(run, scenario.arm("titan_compiled_fa3"), Path("/out"))
                self.assertEqual(argv[argv.index("--dataloader.offset-rows") + 1], rows)
                self.assertNotIn(
                    "--dataloader.offset-rows",
                    command(run, scenario.arm("titan_compiled"), Path("/out")),
                )

    def test_packed_offsets_without_a_packed_attention_override_is_refused(self) -> None:
        for imports in ((), ("benchmarks.models.piper_qwen3.components.rope.te_rope_override.te_rope",)):
            with self.subTest(imports=imports):
                arm = configured(
                    ENGINES.arm("titan_compiled"),
                    packed_offsets=True,
                    override_imports=imports,
                )
                (refusal,) = engine_for(arm).check(run_spec(ac_mode="none"), arm)
                self.assertIn("titan_compiled: packed_offsets is on", refusal)
                self.assertIn("benchmarks.models.piper_qwen3.components.attention.", refusal)

    def test_a_packed_attention_override_without_packed_offsets_is_refused(self) -> None:
        scenario = scenario_by_name("attention")
        (arm,) = apply_overrides(
            (scenario.arm("titan_compiled_fa3"),),
            _overrides("titan_compiled_fa3.packed_offsets=off"),
        )
        (refusal,) = engine_for(arm).check(run_spec(ac_mode="none"), arm)
        self.assertIn("titan_compiled_fa3: the override import", refusal)
        self.assertIn("fa3_override.packed_fa3_attention", refusal)
        self.assertIn("packed_offsets is off", refusal)
        self.assertEqual(
            engine_for(arm).check(run_spec(ac_mode="none"), scenario.arm("titan_compiled_fa3")),
            [],
        )


MOE_PACKAGE = "benchmarks.models.piper_qwen3.components.moe"

HOST_COUNT_TARGET = f"{MOE_PACKAGE}.host_count_dispatcher.host_count_dispatcher"

PER_EXPERT_TARGET = f"{MOE_PACKAGE}.te_per_expert_experts.te_per_expert_experts"

PER_EXPERT_MARKERS = ("torchtitan_benchmarks::te_per_expert_mm",)
"""The profiler range that each per-expert op opens, at every shape."""

EP2_SPEC = ParallelismSpec(dp=2, ep=2, zero=1)
"""A two-GPU mesh at ep 2, at which every arm of the experts and stacked scenarios can run."""

OVERRIDE_COUNT_SIZES = ("1b", "30b-a3b")
"""The model sizes at which each override arm is counted."""


def _override_counts(
    scenario_name: str, names: list[str]
) -> dict[str, dict[str, dict[str, int]]]:
    """Per arm of ``scenario_name``, size and override target: the number of nodes it replaced."""
    scenario = scenario_by_name(scenario_name)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for name in names:
        config = scenario.arm(name).config
        for size in OVERRIDE_COUNT_SIZES:
            lines = apply_config_overrides(
                qwen3_piper_1b_pretokenized(size=size),
                config.override_imports,
                expected=config.overrides_per_block * PIPER_SHAPES[size].n_layers,
            )
            counts.setdefault(name, {})[size] = {
                target: sum(line.startswith(f"[Override] {target}:") for line in lines)
                for target in config.override_imports
            }
    return counts


OVERRIDE_COUNT_SCRIPT = """
import json
import sys

sys.path.insert(0, sys.argv[1])
from tests.test_runner import _override_counts

print(json.dumps(_override_counts(sys.argv[2], sys.argv[3:])))
"""
"""Counts the arms ``argv[3:]`` of the scenario ``argv[2]`` in a fresh process and prints the counts as JSON."""


def _count_in_subprocess(test: unittest.TestCase, scenario: str, names: list[str]) -> dict:
    """The ``_override_counts`` of ``names`` from a fresh process, so TE does not patch this process."""
    repo = Path(__file__).resolve().parent.parent
    completed = subprocess.run(
        [sys.executable, "-c", OVERRIDE_COUNT_SCRIPT, str(repo), scenario, *names],
        cwd=repo,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=600,
    )
    test.assertEqual(completed.returncode, 0, completed.stderr)
    return json.loads(completed.stdout.splitlines()[-1])


class ExpertsScenarioTests(unittest.TestCase):
    def test_scenario_registration(self) -> None:
        scenario = scenario_by_name("experts")
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            [
                "titan_compiled",
                "titan_compiled_te_per_expert",
                "megatron_stock",
            ],
        )
        self.assertIs(scenario.data, C4_REPLAY_DATA)
        self.assertEqual(scenario.supported_ac_modes, ("none",))
        self.assertIs(scenario.arm("titan_compiled"), ENGINES.arm("titan_compiled"))
        self.assertIs(scenario.arm("megatron_stock"), ENGINES.arm("megatron_stock"))
        self.assertEqual(set(SCENARIOS), {"engines", "attention", "experts", "stacked"})

    def test_the_override_arm_declares_its_imports_count_and_markers(self) -> None:
        arm = scenario_by_name("experts").arm("titan_compiled_te_per_expert")
        config = arm.config
        self.assertEqual(engine_for(arm).name, "torchtitan")
        self.assertEqual(config.compile, CompileMode.TORCH)
        self.assertEqual(config.override_imports, (HOST_COUNT_TARGET, PER_EXPERT_TARGET))
        self.assertEqual(config.overrides_per_block, 2)
        self.assertEqual(config.trace_kernel_markers, PER_EXPERT_MARKERS)
        self.assertIs(config.packed_offsets, False)
        self.assertEqual(config.extra_flags, ())
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC)
        self.assertEqual(engine_for(arm).check(run, arm), [])

    def test_the_per_expert_marker_is_the_range_that_its_ops_open(self) -> None:
        """The source is parsed, so TE stays out of this process."""
        source = Path(__file__).resolve().parent.parent / (
            "benchmarks/models/piper_qwen3/components/moe/te_per_expert_experts.py"
        )
        (value,) = [
            node.value.value
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.Assign)
            and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["TRACE_MARKER"]
        ]
        self.assertEqual(PER_EXPERT_MARKERS, (value,))
        self.assertEqual(
            scenario_by_name("experts").arm("titan_compiled_te_per_expert").config.trace_kernel_markers,
            (value,),
        )

    def test_every_arm_command_builds_at_ac_none(self) -> None:
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC)
        for arm in scenario_by_name("experts").arms:
            argv = command(run, arm, Path("/out") / arm.name)
            self.assertTrue(argv, arm.name)

    def test_each_override_arm_sends_its_imports(self) -> None:
        scenario = scenario_by_name("experts")
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC)
        for name, targets in (
            ("titan_compiled", ()),
            ("titan_compiled_te_per_expert", (HOST_COUNT_TARGET, PER_EXPERT_TARGET)),
        ):
            with self.subTest(arm=name):
                argv = command(run, scenario.arm(name), Path("/out"))
                sent = [
                    argv[index + 1]
                    for index, token in enumerate(argv)
                    if token == "--override.imports"
                ]
                self.assertEqual(len(sent), 1 if targets else 0)
                self.assertEqual(tuple(parse_cli_imports(sent)), targets)
                self.assertNotIn("--dataloader.offset-rows", argv)

    def test_the_te_arm_replaces_its_count_of_nodes(self) -> None:
        """A subprocess counts the TE arm, so TE does not patch this process."""
        counts = _count_in_subprocess(self, "experts", ["titan_compiled_te_per_expert"])
        self.assertEqual(sorted(counts), ["titan_compiled_te_per_expert"])
        for name, sizes in counts.items():
            self.assertEqual(sorted(sizes), sorted(OVERRIDE_COUNT_SIZES))
            config = scenario_by_name("experts").arm(name).config
            for size, targets in sizes.items():
                self.assertEqual(sorted(targets), sorted(config.override_imports))
                for target, count in targets.items():
                    with self.subTest(arm=name, size=size, target=target):
                        self.assertEqual(count, PIPER_SHAPES[size].n_layers)

    def test_packed_offsets_on_the_te_arm_is_refused(self) -> None:
        arm = configured(
            scenario_by_name("experts").arm("titan_compiled_te_per_expert"),
            packed_offsets=True,
        )
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC)
        (refusal,) = engine_for(arm).check(run, arm)
        self.assertIn("titan_compiled_te_per_expert: packed_offsets is on", refusal)

    def test_the_per_expert_experts_without_the_host_count_dispatcher_is_refused(self) -> None:
        arm = configured(
            scenario_by_name("experts").arm("titan_compiled_te_per_expert"),
            overrides_per_block=1,
            override_imports=(PER_EXPERT_TARGET,),
        )
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC)
        (refusal,) = engine_for(arm).check(run, arm)
        self.assertIn(f"titan_compiled_te_per_expert: the override import {PER_EXPERT_TARGET}", refusal)
        self.assertIn("add the host count dispatcher", refusal)

    def test_the_host_count_dispatcher_without_the_per_expert_experts_is_refused(self) -> None:
        arm = configured(
            scenario_by_name("experts").arm("titan_compiled_te_per_expert"),
            overrides_per_block=1,
            override_imports=(HOST_COUNT_TARGET,),
        )
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC)
        (refusal,) = engine_for(arm).check(run, arm)
        self.assertIn(
            f"titan_compiled_te_per_expert: the override import {HOST_COUNT_TARGET}",
            refusal,
        )
        self.assertIn("add the per-expert experts", refusal)

    def test_either_host_count_override_at_ep_1_is_refused(self) -> None:
        arm = scenario_by_name("experts").arm("titan_compiled_te_per_expert")
        for spec in (TRIVIAL_SPEC, ParallelismSpec(dp=2)):
            for imports in (
                (HOST_COUNT_TARGET, PER_EXPERT_TARGET),
                (PER_EXPERT_TARGET,),
                (HOST_COUNT_TARGET,),
            ):
                with self.subTest(dp=spec.dp, imports=imports):
                    swapped = configured(arm, override_imports=imports)
                    refusals = engine_for(swapped).check(
                        run_spec(ac_mode="none", parallelism=spec), swapped
                    )
                    (refusal,) = [r for r in refusals if "ep 1" in r]
                    self.assertIn(f"the override import {imports[0]}", refusal)
                    self.assertIn("needs an expert-parallel mesh", refusal)
        self.assertEqual(
            engine_for(arm).check(run_spec(ac_mode="none", parallelism=EP2_SPEC), arm), []
        )

    def test_a_te_override_under_spmd_types_is_refused(self) -> None:
        name = "titan_compiled_te_per_expert"
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC)
        for flags in (
            ("--parallelism.spmd-backend", "spmd_types"),
            ("--parallelism.spmd-backend=spmd_types",),
            ("--parallelism.spmd_backend", "spmd_types"),
            (
                "--parallelism.spmd-backend",
                "default",
                "--parallelism.spmd-backend=spmd_types",
            ),
        ):
            with self.subTest(flags=flags):
                arm = configured(scenario_by_name("experts").arm(name), extra_flags=flags)
                (refusal,) = engine_for(arm).check(run, arm)
                self.assertIn(f"{name}: the override import {HOST_COUNT_TARGET}", refusal)
                self.assertIn("no SPMD type rule", refusal)
                self.assertIn("--parallelism.spmd-backend spmd_types", refusal)

    def test_a_te_override_under_another_backend_passes(self) -> None:
        scenario = scenario_by_name("experts")
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC)
        for name, flags in (
            ("titan_compiled_te_per_expert", ("--parallelism.spmd-backend", "full_dtensor")),
            (
                "titan_compiled_te_per_expert",
                (
                    "--parallelism.spmd-backend=spmd_types",
                    "--parallelism.spmd-backend=default",
                ),
            ),
            ("titan_compiled", ("--parallelism.spmd-backend", "spmd_types")),
        ):
            with self.subTest(arm=name, flags=flags):
                arm = configured(scenario.arm(name), extra_flags=flags)
                self.assertEqual(engine_for(arm).check(run, arm), [])


STACKED_ARM = "titan_compiled_fa3_te_per_expert"


class StackedScenarioTests(unittest.TestCase):
    def test_scenario_registration(self) -> None:
        scenario = scenario_by_name("stacked")
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            ["titan_compiled", STACKED_ARM, "megatron_stock"],
        )
        self.assertIs(scenario.data, C4_REPLAY_DATA)
        self.assertEqual(scenario.supported_ac_modes, ("none",))
        self.assertIs(scenario.arm("titan_compiled"), ENGINES.arm("titan_compiled"))
        self.assertIs(scenario.arm("megatron_stock"), ENGINES.arm("megatron_stock"))

    def test_the_stacked_arm_declares_its_imports_count_markers_and_offsets(self) -> None:
        arm = scenario_by_name("stacked").arm(STACKED_ARM)
        self.assertEqual(engine_for(arm).name, "torchtitan")
        self.assertEqual(
            arm.config,
            TorchTitanConfig(
                compile=CompileMode.TORCH,
                overrides_per_block=3,
                override_imports=(FA3_TARGET, HOST_COUNT_TARGET, PER_EXPERT_TARGET),
                trace_kernel_markers=(*FA3_MARKERS, *PER_EXPERT_MARKERS),
                packed_offsets=True,
            ),
        )

    def test_the_stacked_arm_joins_the_fa3_arm_and_the_per_expert_arm(self) -> None:
        stacked = scenario_by_name("stacked").arm(STACKED_ARM).config
        fa3 = scenario_by_name("attention").arm("titan_compiled_fa3").config
        per_expert = scenario_by_name("experts").arm("titan_compiled_te_per_expert").config
        self.assertEqual(
            stacked.override_imports, (*fa3.override_imports, *per_expert.override_imports)
        )
        self.assertEqual(
            stacked.overrides_per_block,
            fa3.overrides_per_block + per_expert.overrides_per_block,
        )
        self.assertEqual(
            stacked.trace_kernel_markers,
            (*fa3.trace_kernel_markers, *per_expert.trace_kernel_markers),
        )
        self.assertEqual(stacked.packed_offsets, fa3.packed_offsets)
        self.assertEqual(stacked.compile, per_expert.compile)

    def test_the_engine_accepts_the_stacked_arm_at_ep_4_and_refuses_it_at_ep_1(self) -> None:
        arm = scenario_by_name("stacked").arm(STACKED_ARM)
        ep4 = ParallelismSpec(dp=4, ep=4, zero=1)
        self.assertEqual(engine_for(arm).check(run_spec(ac_mode="none", parallelism=ep4), arm), [])
        for spec in (TRIVIAL_SPEC, ParallelismSpec(dp=4, zero=1)):
            with self.subTest(dp=spec.dp):
                (refusal,) = engine_for(arm).check(
                    run_spec(ac_mode="none", parallelism=spec), arm
                )
                self.assertIn(f"{STACKED_ARM}: the override import {HOST_COUNT_TARGET}", refusal)
                self.assertIn("needs an expert-parallel mesh", refusal)

    def test_the_stacked_arm_under_spmd_types_is_refused(self) -> None:
        arm = configured(
            scenario_by_name("stacked").arm(STACKED_ARM),
            extra_flags=("--parallelism.spmd-backend", "spmd_types"),
        )
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC)
        (refusal,) = engine_for(arm).check(run, arm)
        self.assertIn(f"{STACKED_ARM}: the override import {HOST_COUNT_TARGET}", refusal)
        self.assertIn("no SPMD type rule", refusal)

    def test_the_stacked_arm_without_packed_offsets_is_refused(self) -> None:
        (arm,) = apply_overrides(
            (scenario_by_name("stacked").arm(STACKED_ARM),),
            _overrides(f"{STACKED_ARM}.packed_offsets=off"),
        )
        (refusal,) = engine_for(arm).check(run_spec(ac_mode="none", parallelism=EP2_SPEC), arm)
        self.assertIn(f"{STACKED_ARM}: the override import {FA3_TARGET}", refusal)
        self.assertIn("packed_offsets is off", refusal)

    def test_every_arm_command_builds_at_ac_none(self) -> None:
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC)
        for arm in scenario_by_name("stacked").arms:
            argv = command(run, arm, Path("/out") / arm.name)
            self.assertTrue(argv, arm.name)

    def test_the_stacked_arm_sends_its_imports_and_the_rows_of_one_microbatch(self) -> None:
        scenario = scenario_by_name("stacked")
        run = run_spec(ac_mode="none", parallelism=EP2_SPEC, local_batch_size=8)
        argv = command(run, scenario.arm(STACKED_ARM), Path("/out"))
        sent = [
            argv[index + 1] for index, token in enumerate(argv) if token == "--override.imports"
        ]
        self.assertEqual(len(sent), 1)
        self.assertEqual(
            tuple(parse_cli_imports(sent)), (FA3_TARGET, HOST_COUNT_TARGET, PER_EXPERT_TARGET)
        )
        self.assertEqual(argv[argv.index("--dataloader.offset-rows") + 1], "8")

    def test_the_stacked_arm_replaces_its_count_of_nodes(self) -> None:
        """A subprocess counts the stacked arm, so TE does not patch this process."""
        counts = _count_in_subprocess(self, "stacked", [STACKED_ARM])
        self.assertEqual(sorted(counts), [STACKED_ARM])
        sizes = counts[STACKED_ARM]
        self.assertEqual(sorted(sizes), sorted(OVERRIDE_COUNT_SIZES))
        for size, targets in sizes.items():
            self.assertEqual(
                sorted(targets), sorted((FA3_TARGET, HOST_COUNT_TARGET, PER_EXPERT_TARGET))
            )
            for target, count in targets.items():
                with self.subTest(size=size, target=target):
                    self.assertEqual(count, PIPER_SHAPES[size].n_layers)
            self.assertEqual(sum(targets.values()), 3 * PIPER_SHAPES[size].n_layers)


class CompilerEnvironmentTests(unittest.TestCase):
    """The ``requires_gcc_toolset`` branch, against the synthetic arm.

    No registered arm sets the field, so the branch has no live caller.
    It stays because an override arm can need a C++ host compiler that the
    stock one is not, and a branch nothing exercises is a branch that
    breaks unseen.
    """

    metadata = {
        "requested_gpu": "0",
        "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
        "cpu_pinning": "none: test",
        "torch_version": "test",
        "torchtitan_git_rev": "titan-rev",
        "benchmarks_git_rev": "bench-rev",
        "megatron_git_rev": "megatron-rev",
    }

    def _run(self, arm: Arm, compiler_env: Path) -> dict[str, str]:
        """Execute one arm and return the environment it was launched with."""
        captured: dict[str, str] = {}

        def fake_process(command, **keywords):
            captured.update(keywords["env"])
            return SimpleNamespace(returncode=0)

        scenario = replace(scenario_by_name("engines"), arms=(arm,))
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", self.metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ), mock.patch(
            "benchmarks.e2e.runner.scenario_by_name", return_value=scenario
        ), mock.patch("benchmarks.e2e.runner.validate_arm"):
            execute_run(
                check_request(
                    RunRequest(
                        axes=RequestedAxes(ac_mode="none"),
                        gpu="0",
                        scenario_name="engines",
                        out_dir=Path(temporary) / "run",
                        compiler_env=compiler_env,
                    ),
                    environment={"PATH": os.environ["PATH"]},
                ),
                process_runner=fake_process,
            )
        return captured

    def test_the_arm_that_asks_for_the_compiler_gets_the_sourced_script(
        self,
    ) -> None:
        """The script's own exports reach the training subprocess."""
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "enable"
            script.write_text("export BENCH_TEST_TOOLSET=13\n")
            environment = self._run(OVERRIDE_ARM, script)
        self.assertEqual(environment.get("BENCH_TEST_TOOLSET"), "13")

    def test_an_arm_that_does_not_ask_for_it_runs_without_it(self) -> None:
        """The branch is per arm, so one arm cannot enable it for another."""
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "enable"
            script.write_text("export BENCH_TEST_TOOLSET=13\n")
            environment = self._run(
                configured(OVERRIDE_ARM, requires_gcc_toolset=False), script
            )
        self.assertNotIn("BENCH_TEST_TOOLSET", environment)

    def test_a_missing_script_fails_the_arm_by_name(self) -> None:
        """A silently skipped script would build the extension with the
        stock compiler and fail deep inside the training subprocess."""
        with tempfile.TemporaryDirectory() as temporary:
            absent = Path(temporary) / "no-such-enable"
            with self.assertRaisesRegex(ValueError, "does not exist"):
                self._run(OVERRIDE_ARM, absent)


class CpuPinningTests(unittest.TestCase):
    def _sysfs(self, root: Path, device: str, node: str) -> Path:
        node_dir = root / "bus/pci/devices" / device
        node_dir.mkdir(parents=True)
        (node_dir / "numa_node").write_text(node + "\n")
        cpus_dir = root / f"devices/system/node/node{node}"
        cpus_dir.mkdir(parents=True)
        (cpus_dir / "cpulist").write_text("64-127,192-255\n")
        return root

    def test_pins_to_the_gpu_numa_node(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.execution.affinity.run_text", return_value="00000000:E3:00.0\n"
        ), mock.patch("benchmarks.execution.affinity.shutil.which", return_value="/usr/bin/numactl"):
            sysfs = self._sysfs(Path(temporary), "0000:e3:00.0", "1")
            pinning = resolve_cpu_pinning(
                "7", sysfs_root=sysfs, allowed_cpus=frozenset(range(256))
            )
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
        extra = "--training.gc-freq 7"
        metadata = {**_METADATA, "requested_gpu": "3"}
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("rtx-a6000", metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning(("numactl", "--cpunodebind=0", "--membind=0"), "numactl --cpunodebind=0 --membind=0"),
        ), mock.patch("benchmarks.e2e.runner.validate_arm"):
            out_dir = Path(temporary) / "run"
            execute_run(
                check_request(
                    RunRequest(
                        axes=RequestedAxes(ac_mode="none", model_size="1b"),
                        gpu="3",
                        scenario_name="engines",
                        arm_names=("titan_eager",),
                        out_dir=out_dir,
                        overrides=_overrides(f"titan_eager.extra_flags+={extra}"),
                    ),
                    environment={"PATH": os.environ["PATH"]},
                ),
                process_runner=lambda command, **kwargs: SimpleNamespace(returncode=0),
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        self.assertEqual(manifest["schema_version"], 20)
        self.assertEqual(manifest["scenario"], "engines")
        self.assertEqual(manifest["hardware"], "rtx-a6000")
        self.assertEqual(
            manifest["hardware_metadata"],
            {**metadata, "cpu_pinning": "numactl --cpunodebind=0 --membind=0"},
        )
        self.assertEqual(manifest["run"]["ac_mode"], "none")
        self.assertEqual(manifest["run"]["shape"], PIPER_1B.describe(seq_len=4096))
        self.assertEqual(manifest["run"]["data"]["local_batch_size"], 4)
        for key in (
            "model_size",
            "megatron_p2p_sync",
            "extra_torchtitan_args",
            "extra_megatron_args",
            "execution_model",
            "selected_arms",
            "commands",
        ):
            self.assertNotIn(key, manifest)
        (arm,) = manifest["arms"]
        self.assertEqual(arm["name"], "titan_eager")
        self.assertEqual(arm["engine"], "torchtitan")
        self.assertEqual(arm["config_type"], "TorchTitanConfig")
        self.assertEqual(arm["config"]["compile"], "none")
        self.assertEqual(arm["config"]["extra_flags"], ["--training.gc-freq", "7"])
        self.assertEqual(arm["execution_model"], "single-gpu-plain-bf16-no-fsdp")
        self.assertEqual(arm["cpu_pinning"], "numactl --cpunodebind=0 --membind=0")
        self.assertEqual(arm["env_delta"]["CUDA_VISIBLE_DEVICES"], "3")
        self.assertIn("--training.gc-freq", arm["command"])
        self.assertEqual(arm["command"][:3], ["numactl", "--cpunodebind=0", "--membind=0"])
        self.assertEqual(arm["command"][-3], "--dump-folder")


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
                check_request(
                    RunRequest(
                        axes=RequestedAxes(
                            model_size="1b",
                            ac_mode="none",
                        ),
                        gpu="0",
                        scenario_name="engines",
                        arm_names=("titan_eager",),
                        out_dir=out_dir,
                    ),
                    environment={"PATH": os.environ["PATH"]},
                ),
                process_runner=fake_process,
            )
        return json.loads((out_dir / "manifest.json").read_text())

    def test_an_eager_arm_builds_no_compile_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._run(Path(temporary) / "run")

        self.assertEqual(manifest["schema_version"], 20)
        (arm,) = manifest["arms"]
        self.assertNotIn("--compile.enable", arm["command"])

    def test_the_manifest_records_the_arm_compile_treatment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._run(Path(temporary) / "run")

        recorded = {arm["name"]: arm["config"]["compile"] for arm in manifest["arms"]}
        self.assertEqual(recorded, {"titan_eager": "none"})


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
# other rule. Exported so the other test modules build the same log.
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
            validate(run_spec(), arm, root, log)

            log.write_text(completed + applied * 15)
            with self.assertRaisesRegex(RuntimeError, "expected 16 override"):
                validate(run_spec(), arm, root, log)

            log.write_text(
                completed
                + "[Override] torchtitan.overrides.other.thing: fqn ...\n" * 16
            )
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate(run_spec(), arm, root, log)

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
            validate(run_spec(), arm, root, log)

            log.write_text(_SAC_LINE + _SIZE_LINE + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "did not apply it"):
                validate(run_spec(), arm, root, log)

    def test_an_eager_arm_refuses_the_compile_line(self) -> None:
        """Rule 8 inverts on an eager arm: a run that silently compiled
        cannot be published as eager."""
        arm = ENGINES.arm("titan_eager")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root)

            log.write_text(_SAC_LINE + _SIZE_LINE + "Training completed\n")
            validate(run_spec(), arm, root, log)

            log.write_text(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "compiled the model"):
                validate(run_spec(), arm, root, log)

    def test_ac_mode_must_match_the_applied_treatment(self) -> None:
        arm = ENGINES.arm("titan_compiled")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._rope_baseline_fixture(root)

            # sac requested, SelectiveAC absent: the run measured no-AC.
            log.write_text(_compiled_line("default") + _SIZE_LINE + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "ac mode 'sac'"):
                validate(run_spec(), arm, root, log)

            # none requested, SelectiveAC applied: the run measured SAC.
            log.write_text(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "ac mode 'none'"):
                validate(run_spec(ac_mode="none"), arm, root, log)

            # none requested, SelectiveAC absent: valid.
            log.write_text(_compiled_line("default") + _SIZE_LINE + "Training completed\n")
            validate(run_spec(ac_mode="none"), arm, root, log)


class TrainingMetricsTests(unittest.TestCase):
    def test_stable_samples_exclude_compile_and_profiler_steps(self) -> None:
        # Step 2 also leaves, because the profiled rule drops the slow first step.
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "arm.log"
            log.write_text(
                "".join(
                    titan_step_line(step, tps=tps)
                    for step, tps in (
                        (1, 100),
                        (2, 9900),
                        (10, 10100),
                        (11, 8000),
                        (21, 200),
                        (22, 10000),
                    )
                )
            )
            samples = arm_steps(ENGINES.arm("titan_eager"), log)[0].samples

        self.assertEqual(
            [
                sample.tokens_per_second
                for sample in stable_samples(
                    samples, ProfileWindow(freq=20, warmup=5, active=5)
                )
            ],
            [10100, 10000],
        )

    def test_stable_samples_exclude_the_slow_first_step(self) -> None:
        samples = [
            StepSample(
                rank=0,
                step=step,
                tokens_per_second=1000,
                peak_memory_gib=3.0,
                loss=1.0,
                grad_norm=2.0,
            )
            for step in range(1, 81)
        ]
        stable = stable_samples(samples, ProfileWindow(freq=20, warmup=5, active=5))
        self.assertEqual(
            [sample.step for sample in stable],
            [
                *range(3, 11),
                *range(22, 31),
                *range(42, 51),
                *range(62, 71),
            ],
        )
        self.assertEqual(len(stable), 35)


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
                axes=RequestedAxes(
                    model_size="1b",
                    ac_mode="none",
                ),
                gpu="0",
                scenario_name="engines",
                arm_names=("titan_compiled",),
                out_dir=out_dir,
                seq_len=512,
                steps=60,
                batch=2,
                overrides=_overrides("titan_compiled.extra_flags+=--debug.deterministic"),
            )
            execute_run(
                check_request(
                    request,
                    environment=environment,
                ),
                process_runner=fake_process,
            )

            resumed = RunRequest(
                gpu="0",
                scenario_name=None,
                arm_names=("titan_compiled",),
                resume_dir=out_dir,
            )
            process = mock.Mock(side_effect=fake_process)
            execute_run(
                check_request(
                    resumed,
                    environment=environment,
                ),
                process_runner=process,
                event_handler=events.append,
            )
            process.assert_not_called()
            self.assertTrue(any(event.kind == "skip" for event in events))

            (out_dir / "titan_compiled.log").write_text("interrupted\n")
            retry_process = mock.Mock(side_effect=fake_process)
            execute_run(
                check_request(
                    resumed,
                    environment=environment,
                ),
                process_runner=retry_process,
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
            with self.assertRaisesRegex(ValueError, "existing manifest: run.data$"):
                execute_run(
                    check_request(
                        incompatible,
                        environment=environment,
                    ),
                    process_runner=fake_process,
                )

            conflicting_args = RunRequest(
                gpu="0",
                scenario_name=None,
                arm_names=("titan_compiled",),
                resume_dir=out_dir,
                overrides=_overrides("titan_compiled.extra_flags+=--training.gc-freq 9"),
            )
            with self.assertRaisesRegex(
                ValueError, "existing manifest: titan_compiled.config.extra_flags$"
            ):
                execute_run(
                    check_request(
                        conflicting_args,
                        environment=environment,
                    ),
                    process_runner=fake_process,
                )

            conflicting_size = RunRequest(
                axes=RequestedAxes(
                    model_size="huge",
                ),
                gpu="0",
                scenario_name=None,
                arm_names=("titan_compiled",),
                resume_dir=out_dir,
            )
            with self.assertRaisesRegex(ValueError, "existing manifest: run.shape$"):
                execute_run(
                    check_request(
                        conflicting_size,
                        environment=environment,
                    ),
                    process_runner=fake_process,
                )

            conflicting_profile = RunRequest(
                axes=RequestedAxes(profile=True),
                gpu="0",
                scenario_name=None,
                arm_names=("titan_compiled",),
                resume_dir=out_dir,
            )
            with self.assertRaisesRegex(
                ValueError,
                "does not match the existing manifest: run.profile, "
                "run.warmup_steps$",
            ):
                execute_run(
                    check_request(
                        conflicting_profile,
                        environment=environment,
                    ),
                    process_runner=fake_process,
                )

            conflicting_profile_and_args = RunRequest(
                axes=RequestedAxes(profile=True),
                gpu="0",
                scenario_name=None,
                arm_names=("titan_compiled",),
                resume_dir=out_dir,
                overrides=_overrides("titan_compiled.extra_flags+=--training.gc-freq 9"),
            )
            with self.assertRaisesRegex(
                ValueError,
                "does not match the existing manifest: run.profile, "
                "run.warmup_steps, titan_compiled.config.extra_flags$",
            ):
                execute_run(
                    check_request(
                        conflicting_profile_and_args,
                        environment=environment,
                    ),
                    process_runner=fake_process,
                )

            manifest_path = out_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["run"]["profile"] = True
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(
                ValueError, "warmup_steps 10 does not match profile True"
            ):
                execute_run(
                    check_request(
                        resumed,
                        environment=environment,
                    ),
                    process_runner=fake_process,
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
                check_request(
                    RunRequest(
                        axes=RequestedAxes(
                            model_size="1b",
                            ac_mode=mode,
                        ),
                        gpu="0",
                        scenario_name="engines",
                        arm_names=("titan_eager",),
                        out_dir=out_dir,
                    ),
                    environment=environment,
                ),
                process_runner=fake_process,
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["run"]["ac_mode"], mode)

            (out_dir / "titan_eager.log").write_text("interrupted\n")
            retry_process = mock.Mock(side_effect=fake_process)
            execute_run(
                check_request(
                    RunRequest(
                        gpu="0",
                        scenario_name=None,
                        arm_names=("titan_eager",),
                        resume_dir=out_dir,
                    ),
                    environment=environment,
                ),
                process_runner=retry_process,
            )
            retry_command = retry_process.call_args.args[0]
            self.assertNotIn("--compile.enable", retry_command)
            self.assertIn("activation-checkpoint:none", retry_command)
            self.assertIn(
                "Training completed", (out_dir / "titan_eager.log").read_text()
            )


if __name__ == "__main__":
    unittest.main()
