"""The engine registry, and what each engine passes to its builder and its validation.

The type of an arm's config selects its engine. These tests hold the
registry contracts: every engine class is registered once, a config type
selects one engine, and an unknown config type or name raises and names
the choices. They also hold what each engine hands on: its own command
builder and its own validation.
"""

import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.engines.api import Arm, CompileMode, Engine, EngineConfig
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.engines.megatron_stock.engine import MegatronStockEngine
from benchmarks.e2e.engines.megatron_stock.flags import (
    DRIVER_MODULE,
    MEGATRON_LM_SCHEDULES,
    stock_megatron_flags,
)
from benchmarks.e2e.engines.registry import (
    ENGINES,
    _registry,
    engine_for,
    engine_named,
)
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig
from benchmarks.e2e.engines.torchtitan.engine import TorchTitanEngine
from benchmarks.e2e.engines.torchtitan.mesh import SCHEDULES as TITAN_SCHEDULES
from benchmarks.e2e.parallelism import PP_SCHEDULES, ParallelismSpec
from benchmarks.e2e.registry import SCENARIOS
from benchmarks.e2e.engines.torchtitan.flags import TRAIN_MODULE, trainer_args
from benchmarks.e2e.engines.api import MeshObserved, RankEvidence
from benchmarks.e2e.engines.torchtitan.validate import COMPILE_LINE, mesh_markers
from benchmarks.e2e.engines.megatron_stock.validate import required_lines
from tests.engine_helpers import run_spec


def _parallelism_lines(arm, run) -> tuple[str, ...]:
    """The mesh lines that the arm's engine asks every rank to print."""
    if engine_for(arm).name == "torchtitan":
        if run.parallelism.world_size == 1:
            return ()
        return mesh_markers(run.parallelism, run.data)
    return required_lines(run, arm).get("parallelism", ())


_MESHES = (
    ParallelismSpec(dp=2),
    ParallelismSpec(pp=2, pp_schedule="1F1B"),
    ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"),
    ParallelismSpec(dp=2, ep=2, zero=1),
)
"""The meshes that arm rule 12 checks: each shape above one rank, and one with a schedule."""


def _engine_classes(base: type = Engine) -> set[type]:
    """Every concrete subclass of ``base``."""
    found = set()
    for subclass in base.__subclasses__():
        if not getattr(subclass, "__abstractmethods__", None):
            found.add(subclass)
        found |= _engine_classes(subclass)
    return found


@dataclass(frozen=True, kw_only=True)
class _UnregisteredConfig(EngineConfig):
    """A config type that no engine takes."""


class RegistryTests(unittest.TestCase):
    def test_every_engine_class_is_registered_once(self):
        self.assertEqual(
            sorted(type(engine).__name__ for engine in ENGINES.values()),
            sorted(engine.__name__ for engine in _engine_classes()),
        )

    def test_each_engine_is_keyed_by_its_own_config_type(self):
        for config_type, engine in ENGINES.items():
            with self.subTest(engine=engine.name):
                self.assertIs(engine.config_type, config_type)

    def test_the_two_engines_and_their_names(self):
        self.assertIsInstance(ENGINES[TorchTitanConfig], TorchTitanEngine)
        self.assertIsInstance(ENGINES[MegatronStockConfig], MegatronStockEngine)
        self.assertEqual(
            sorted(engine.name for engine in ENGINES.values()),
            ["megatron_stock", "torchtitan"],
        )

    def test_a_repeated_config_type_is_refused(self):
        with self.assertRaisesRegex(ValueError, "both take TorchTitanConfig"):
            _registry((TorchTitanEngine(), TorchTitanEngine()))

    def test_a_repeated_name_is_refused(self):
        renamed = mock.Mock(spec=Engine)
        renamed.name = "torchtitan"
        renamed.config_type = _UnregisteredConfig
        with self.assertRaisesRegex(ValueError, "two engines are named"):
            _registry((TorchTitanEngine(), renamed))

    def test_an_unknown_config_type_is_refused_by_name(self):
        arm = Arm(
            name="invented",
            description="an arm whose config no engine takes",
            config=_UnregisteredConfig(),
        )
        with self.assertRaisesRegex(
            ValueError,
            "no engine takes _UnregisteredConfig. Available: "
            "MegatronStockConfig, TorchTitanConfig",
        ):
            engine_for(arm)

    def test_engine_named_finds_each_engine_and_refuses_another_name(self):
        for engine in ENGINES.values():
            self.assertIs(engine_named(engine.name), engine)
        with self.assertRaisesRegex(
            ValueError, "Unknown engine 'mcore'. Available: megatron_stock, torchtitan"
        ):
            engine_named("mcore")


class ArmDeclarationTests(unittest.TestCase):
    def test_every_arm_reaches_a_registered_engine(self):
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                with self.subTest(scenario=scenario.name, arm=arm.name):
                    self.assertIn(engine_for(arm), ENGINES.values())

    def test_every_compiled_arm_runs_on_an_engine_that_can_prove_it(self):
        """Arm rule 8 reads a log line, so the validation must name one."""
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                if getattr(arm.config, "compile", None) is CompileMode.TORCH:
                    with self.subTest(arm=arm.name):
                        self.assertIsInstance(engine_for(arm), TorchTitanEngine)
                        self.assertTrue(COMPILE_LINE)

    def test_every_engine_asks_for_mesh_lines_above_one_rank(self):
        """Rule 12 needs a line for each mesh, from each engine."""
        scenario = SCENARIOS["engines"]
        for arm in scenario.arms:
            for spec in _MESHES:
                with self.subTest(arm=arm.name, spec=spec):
                    run = run_spec(
                        ac_mode="none", parallelism=spec, local_batch_size=8
                    )
                    markers = _parallelism_lines(arm, run)
                    self.assertTrue(markers)
                    for marker in markers:
                        self.assertTrue(marker.strip())

    def test_no_engine_asks_for_mesh_lines_at_one_rank(self):
        for arm in SCENARIOS["engines"].arms:
            with self.subTest(arm=arm.name):
                self.assertEqual(
                    _parallelism_lines(arm, run_spec(ac_mode="none")), ()
                )


class EngineDelegationTests(unittest.TestCase):
    """Each engine builds its own command and passes its own validation."""

    def test_the_torchtitan_launch_runs_the_trainer_with_the_arm_arguments(self):
        arm = SCENARIOS["engines"].arm("titan_compiled")
        run = run_spec(ac_mode="none")
        launch = engine_for(arm).launch(run, arm, Path("/tmp/arm"))
        self.assertEqual(
            launch.target,
            ("-m", TRAIN_MODULE, *trainer_args(run, arm.config, Path("/tmp/arm"))),
        )
        self.assertEqual((launch.processes, launch.pin), ("per_rank", True))

    def test_the_megatron_launch_runs_the_driver_with_the_passthrough_last(self):
        arm = replace(
            SCENARIOS["engines"].arm("megatron_stock"),
            config=MegatronStockConfig(extra_flags=("--moe-permute-fusion",)),
        )
        run = run_spec(ac_mode="none")
        launch = engine_for(arm).launch(run, arm, Path("/tmp/arm"))
        self.assertEqual(
            launch.target,
            (
                "-m",
                DRIVER_MODULE,
                *stock_megatron_flags(run, arm.config, arm_dir="/tmp/arm"),
                "--moe-permute-fusion",
            ),
        )
        self.assertEqual((launch.processes, launch.pin), ("per_rank", True))

    def test_each_engine_reads_the_evidence_of_its_own_log(self):
        for arm_name, log in (
            (
                "titan_compiled",
                "Building device mesh with parallelism: pp=2, dp_replicate=1, "
                "dp_shard=2, cp=1, tp=1, ep=2\n"
                "Model piper_qwen3 1b size: 1,066,241,024 total parameters\n"
                "Training completed\n",
            ),
            (
                "megatron_stock",
                "Megatron-LM stock parallelism: dp=2 pp=2 ep=2 schedule=1F1B "
                "microbatches=4 stages=2\n"
                "Megatron-LM stock data parallel: DistributedDataParallel over "
                "2 ranks (overlap_grad_reduce=False, grad_reduce_in_fp32=True, "
                "sharding_strategy=no_shard, expert_parallel=2, "
                "optimizer=ChainedOptimizer[DistributedOptimizer])\n"
                "Model qwen3_piper_1b stock-megatron size: 1,066,241,024 total "
                "parameters\n"
                "Training completed\n",
            ),
        ):
            arm = SCENARIOS["engines"].arm(arm_name)
            with self.subTest(arm=arm_name):
                self.assertEqual(
                    engine_for(arm).read_evidence(3, log),
                    RankEvidence(
                        rank=3,
                        completed=True,
                        param_count=1_066_241_024,
                        mesh=MeshObserved(dp=2, pp=2, ep=2, zero=1),
                    ),
                )

    def test_the_megatron_engine_asks_for_its_three_treatments(self):
        arm = SCENARIOS["engines"].arm("megatron_stock")
        lines = required_lines(run_spec(ac_mode="none"), arm)
        self.assertEqual(
            list(lines), ["megatron nan guard", "megatron precision"]
        )
        self.assertIn(
            "check_for_nan_in_loss_and_grad=False", lines["megatron nan guard"][0]
        )
        self.assertIn(
            "use_precision_aware_optimizer=False", lines["megatron precision"]
        )

    def test_a_launch_is_the_same_for_the_same_inputs(self):
        scenario = SCENARIOS["engines"]
        run = run_spec(ac_mode="none")
        for arm in scenario.arms:
            with self.subTest(arm=arm.name):
                engine = engine_for(arm)
                self.assertEqual(
                    engine.launch(run, arm, Path("/tmp/arm")),
                    engine.launch(run, arm, Path("/tmp/arm")),
                )


class EngineCheckTests(unittest.TestCase):
    """Each engine refuses the schedules it does not run."""

    def _refusals(self, arm_name: str, schedule: str) -> list[str]:
        arm = SCENARIOS["engines"].arm(arm_name)
        run = run_spec(
            ac_mode="none",
            parallelism=ParallelismSpec(pp=2, pp_schedule=schedule),
            local_batch_size=8,
        )
        return engine_for(arm).check(run, arm)

    def test_megatron_refuses_a_schedule_megatron_lm_does_not_implement(self):
        for name in ("InterleavedZeroBubble", "ZBVZeroBubble", "DualPipeV"):
            with self.subTest(schedule=name):
                self.assertRegex(
                    " ".join(self._refusals("megatron_stock", name)),
                    f"'{name}' is not implemented by Megatron-LM.*"
                    "choose --pp-schedule 1F1B",
                )

    def test_megatron_refuses_a_schedule_its_driver_does_not_run(self):
        """Megatron-LM implements Interleaved1F1B, and the stock driver runs 1F1B alone."""
        self.assertIn("Interleaved1F1B", MEGATRON_LM_SCHEDULES)
        self.assertRegex(
            " ".join(self._refusals("megatron_stock", "Interleaved1F1B")),
            "implements '1F1B' alone.*choose --pp-schedule 1F1B",
        )

    def test_megatron_accepts_1f1b(self):
        self.assertEqual(self._refusals("megatron_stock", "1F1B"), [])

    def test_torchtitan_refuses_a_compiled_arm_under_an_uncompiled_schedule(self):
        self.assertRegex(
            " ".join(self._refusals("titan_compiled", "ZBVZeroBubble")),
            "raises on a compiled stage module, and titan_compiled",
        )
        self.assertEqual(self._refusals("titan_eager", "ZBVZeroBubble"), [])
        self.assertEqual(self._refusals("titan_compiled", "Interleaved1F1B"), [])

    def test_each_engine_declares_every_schedule_it_runs_by_a_shared_name(self):
        self.assertLessEqual(set(TITAN_SCHEDULES), set(PP_SCHEDULES))
        self.assertLessEqual(set(MEGATRON_LM_SCHEDULES), set(PP_SCHEDULES))

    def test_an_engine_refuses_a_schedule_it_does_not_declare(self):
        arm = SCENARIOS["engines"].arm("titan_eager")
        run = run_spec(
            ac_mode="none",
            parallelism=ParallelismSpec(pp=2, pp_schedule="NoSuchSchedule"),
            local_batch_size=8,
        )
        self.assertRegex(
            " ".join(engine_for(arm).check(run, arm)),
            "TorchTitan runs no pipeline schedule 'NoSuchSchedule'; choose one of 1F1B",
        )


class ExecutionModelTests(unittest.TestCase):
    """Each engine states how its arm holds the model state."""

    def _model(self, arm_name: str, spec: ParallelismSpec, **config) -> str:
        arm = SCENARIOS["engines"].arm(arm_name)
        arm = replace(arm, config=replace(arm.config, **config))
        return engine_for(arm).execution_model(run_spec(parallelism=spec), arm)

    def test_torchtitan_keeps_the_strings_that_schema_18_recorded(self):
        for spec, expected in (
            (ParallelismSpec(), "single-gpu-plain-bf16-no-fsdp"),
            (ParallelismSpec(pp=2, pp_schedule="1F1B"), "2-gpu-plain-bf16-no-fsdp-pp2-1F1B"),
            (ParallelismSpec(dp=2), "2-gpu-plain-bf16-dp2"),
            (ParallelismSpec(dp=4, ep=4, zero=1), "4-gpu-plain-bf16-dp4-zero1-ep4"),
        ):
            with self.subTest(spec=spec):
                self.assertEqual(self._model("titan_eager", spec), expected)

    def test_megatron_names_its_fp32_master_state(self):
        for spec, precision, expected in (
            (
                ParallelismSpec(),
                "stock",
                "single-gpu-bf16-fp32-master-fp32-grads-fp32-moments-dp1",
            ),
            (
                ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"),
                "stock",
                "4-gpu-bf16-fp32-master-fp32-grads-fp32-moments-dp2-pp2-1F1B",
            ),
            (
                ParallelismSpec(dp=4, ep=4, zero=1),
                "lean",
                "4-gpu-bf16-fp32-master-bf16-grads-bf16-moments-dp4-zero1-ep4",
            ),
        ):
            with self.subTest(spec=spec, precision=precision):
                self.assertEqual(
                    self._model("megatron_stock", spec, precision=precision),
                    expected,
                )


class EngineWarningTests(unittest.TestCase):
    """TorchTitan warns that its arms hold ZeRO-2 at one pipeline rank."""

    def _warnings(self, arm_name: str, spec: ParallelismSpec) -> list[str]:
        arm = SCENARIOS["engines"].arm(arm_name)
        return engine_for(arm).warnings(run_spec(parallelism=spec), arm)

    def test_torchtitan_warns_at_zero_1_and_one_pipeline_rank(self):
        (warning,) = self._warnings("titan_eager", ParallelismSpec(dp=8, zero=1))
        self.assertIn("ZeRO-2", warning)
        self.assertIn("Megatron holds ZeRO-1", warning)

    def test_torchtitan_is_silent_under_a_pipeline_or_at_zero_0(self):
        for spec in (
            ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B", zero=1),
            ParallelismSpec(dp=8),
        ):
            with self.subTest(spec=spec):
                self.assertEqual(self._warnings("titan_eager", spec), [])

    def test_megatron_never_warns(self):
        self.assertEqual(
            self._warnings("megatron_stock", ParallelismSpec(dp=8, zero=1)), []
        )


class MegatronCheckTests(unittest.TestCase):
    """The stock Megatron engine refuses each run that its driver cannot honour."""

    def _refusals(self, config=MegatronStockConfig(), **run_fields) -> str:
        arm = replace(SCENARIOS["engines"].arm("megatron_stock"), config=config)
        run = run_spec(**{"ac_mode": "none", **run_fields})
        return " ".join(engine_for(arm).check(run, arm))

    def test_the_default_arm_passes(self):
        self.assertEqual(self._refusals(), "")

    def test_p2p_sync_on_at_one_pipeline_rank_is_refused(self):
        self.assertIn(
            "no pipeline message", self._refusals(MegatronStockConfig(p2p_sync="on"))
        )

    def test_lean_precision_at_zero_0_is_refused(self):
        self.assertIn(
            "'lean' needs --zero 1",
            self._refusals(MegatronStockConfig(precision="lean")),
        )
        self.assertEqual(
            self._refusals(
                MegatronStockConfig(precision="lean"),
                parallelism=ParallelismSpec(dp=2, zero=1),
            ),
            "",
        )

    def test_parameter_gather_overlap_at_zero_0_is_refused(self):
        self.assertIn(
            "--overlap-param-gather needs --zero 1",
            self._refusals(
                MegatronStockConfig(extra_flags=("--overlap-param-gather",))
            ),
        )

    def test_an_owned_passthrough_flag_is_refused(self):
        self.assertIn(
            "owned by --model-size",
            self._refusals(MegatronStockConfig(extra_flags=("--num-layers=4",))),
        )

    def test_selective_recompute_is_refused(self):
        self.assertIn("no Megatron parity", self._refusals(ac_mode="sac"))

    def test_an_unseeded_run_is_refused(self):
        self.assertIn("seeded workload", self._refusals(seed=None))

    def test_a_partial_profiler_cycle_is_refused_under_the_profiler_alone(self):
        self.assertIn(
            "whole number of profiler cycles",
            self._refusals(profile=True, steps=50),
        )
        self.assertEqual(self._refusals(profile=False, steps=50), "")

    def test_an_unknown_value_is_refused_and_names_the_choices(self):
        for field, label, choices in (
            ("p2p_sync", "megatron p2p sync", "'on', 'off'"),
            ("nan_guard", "megatron nan guard", "'on', 'off'"),
            ("precision", "megatron precision", "'stock', 'lean'"),
        ):
            with self.subTest(field=field):
                self.assertIn(
                    f"{label} 'bogus' is not one of {choices}",
                    self._refusals(MegatronStockConfig(**{field: "bogus"})),
                )


if __name__ == "__main__":
    unittest.main()
