"""The wiring of the stock Megatron-LM scenario: argv, profile, registry.

Three things join here, and each can fail silently:

1. The engine command must build the stock argv and must leave every other
   arm's argv alone. A stray token at the trivial spec changes what every
   existing scenario measures, and no rule reads a command line back.
2. The ``megatron_stock`` validation profile must prove the mesh. A profile
   that proves nothing passes, which is worse than no profile.
3. The scenario must decline every mode its Megatron arm cannot honor, and
   must still admit a TorchTitan-only subset.

**This module holds the golden argv tests for every engine.**
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.e2e.engines.api import Arm
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.engines.megatron_stock.engine import MegatronStockEngine
from benchmarks.e2e.engines.registry import ENGINES as ENGINE_RECORDS, engine_for
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig
from benchmarks.e2e.engines.megatron_stock.flags import DRIVER_MODULE, PP_SCHEDULE
from benchmarks.e2e.axes import RequestedAxes, RunRequest
from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.parallelism import TRIVIAL_SPEC
from benchmarks.e2e.registry import (
    SCENARIOS,
    scenario_by_name,
)
from benchmarks.e2e.runner import _resolve_run
from benchmarks.e2e.engines.megatron_stock.evidence import (
    COMPLETION_LINE,
    read_evidence,
)
from benchmarks.e2e.engines.megatron_stock.validate import (
    nan_guard_markers,
    required_lines,
    p2p_markers,
    mesh_markers as megatron_mesh_markers,
    precision_markers,
)
from benchmarks.execution.affinity import CpuPinning
from benchmarks.execution.launcher import torchrun_flags
from benchmarks.models.piper_qwen3.shape import shape_by_name
from tests.engine_helpers import command, configured, run_spec

SCENARIO_NAME = "engines"
STOCK_PACKAGE = "benchmarks.e2e.engines.megatron_stock"

# The mesh of the run matrix: two pipelines of four stages, eight GPUs.
MESH = ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B", pp_microbatch_size=4)
# The same mesh under the other zero value, plus the expert
# split that value makes legal. Every marker the profile builds moves
# between the two, so a test that reads only MESH proves half the
# contract.
SHARDED_MESH = ParallelismSpec(
    dp=2,
    pp=4,
    ep=2,
    pp_schedule="1F1B",
    pp_microbatch_size=4,
    zero=1,
)

# Every mesh the argv checks below sweep, with the batch each needs.
#
# ``MESH`` needs the lifted caps, and it has them: ``MAX_WORLD_SIZE`` and
# ``MAX_PP`` are both 8, and ``tests/test_parallelism.py`` validates this
# exact mesh. The engine command calls no validator either way, so the argv
# is buildable whatever the caps say -- which is why the sweep freezes the
# three smaller meshes beside it rather than resting on one spec.
SPECS = (
    ("trivial", TRIVIAL_SPEC, None),
    ("dp2", ParallelismSpec(dp=2), None),
    ("pp2", ParallelismSpec(pp=2, pp_schedule="1F1B"), None),
    ("dp2xpp2", ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"), None),
    ("dp2xpp4", MESH, 32),
)

_METADATA = {
    "requested_gpu": "0",
    "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
    "torch_version": "test",
    "torchtitan_git_rev": "titan-rev",
    "benchmarks_git_rev": "bench-rev",
    "megatron_git_rev": "mcore-rev",
}

# The harness's own flag group. ``flags.py`` emits it; the tests below check
# that every one of them reaches the argv. The two degrees are absent on
# purpose: the driver reads them from the arguments Megatron itself
# resolved, and a degree the engine resolved is stronger evidence than a
# degree the harness asserts.
BENCH_FLAG_NAMES = (
    "--bench-arm-dir",
    "--bench-model-size",
    "--bench-local-batch-size",
    "--bench-profile-freq",
    "--bench-profiler-warmup",
    "--bench-profiler-active",
)

# The log fragments the driver and this profile must agree on, character for
# character. They are the parts of the four marker strings that carry no
# interpolated value, so they appear as literals in the driver's own source.
#
# ``StockMarkerContractTests`` pins them in both directions: every fragment
# is a substring of a marker this profile really builds, and every fragment
# appears in the driver package's source.
STOCK_LOG_FRAGMENTS = (
    "Training completed",
    "Megatron-LM stock training loop (",
    "Megatron-LM stock parallelism: dp=",
    # ``pipelined_pattern`` needs ``dp=<n> pp=`` with one space between, so
    # the pipeline token is part of the contract and not decoration. A
    # driver printing ``dp=2, pp=4`` fails arm rule 12 on a real run.
    " pp=",
    " ep=",
    " schedule=1F1B microbatches=",
    " stages=",
    "Megatron-LM stock data parallel: ",
    " ranks (overlap_grad_reduce=",
    ", grad_reduce_in_fp32=True, ",
    # The overlap VALUE is not here: it moves with the zero
    # value. STOCK_OVERLAP_FRAGMENTS below pins it per value, so the
    # roster still says which token sits between the two above.
    "sharding_strategy=",
    "expert_parallel=",
    # The optimizer class. It is what separates zero1 from replicate,
    # because both keep the DistributedDataParallel wrapper. The VALUE
    # moves with --zero, so only the field name sits here.
    "optimizer=",
    # The p2p line, printed off the built config on every rank. The sync
    # VALUE is not here: it moves with --megatron-p2p-sync, and the
    # contract test below pins it per value.
    "Megatron-LM stock p2p: batch_p2p_comm=",
    " batch_p2p_sync=",
    # The nan guard line, printed from the parsed value on every rank at
    # every mesh. The VALUE moves with --megatron-nan-guard.
    "Megatron-LM stock nan guard: check_for_nan_in_loss_and_grad=",
)

# The wrapper class name, per ZeRO level. Megatron builds one class at
# both levels here, because this suite sends no wrapper flag. The name is
# still read off the wrapper rather than hardcoded, so a run that reached
# another class fails arm rule 12.
STOCK_WRAPPER_FRAGMENTS = {
    0: "Megatron-LM stock data parallel: DistributedDataParallel ",
    1: "Megatron-LM stock data parallel: DistributedDataParallel ",
}

# The overlap half of the same line. The whole field is pinned here,
# commas included, because a roster entry that stopped at the "=" would no
# longer say which token follows it.
STOCK_OVERLAP_FRAGMENTS = {
    0: "(overlap_grad_reduce=False, grad_reduce_in_fp32=True,",
    1: "(overlap_grad_reduce=False, grad_reduce_in_fp32=True,",
}

# The rest of the driver's own line. The profile's first precision marker
# stops at the open bracket, so no rule matches these fields twice.
#
# **Four of them carry a run-time rule now.** ``precision_markers`` is the
# ``--megatron-precision`` half of arm rule 12, and it asks every rank for
# ``use_precision_aware_optimizer``, ``main_grads_dtype``,
# ``exp_avg_dtype`` and ``exp_avg_sq_dtype`` under both values. The other
# four fields record the treatment for the reader alone.
#
# **Only the field names are pinned, never the values.** Every one of the
# eight is a documented reversal target: turning cross-entropy fusion on,
# or moving the reduction to bf16, changes a value here and must stay a
# one-line edit in ``flags.py``.
STOCK_MODE_LINE_FIELDS = (
    "main_params_dtype=",
    "main_grads_dtype=",
    "use_precision_aware_optimizer=",
    "exp_avg_dtype=",
    "exp_avg_sq_dtype=",
    "accumulate_allreduce_grads_in_fp32=",
    "cross_entropy_loss_fusion=",
    "moe_token_dispatcher_type=",
)

# The parameter lines the counting model builder prints, for arm rule 11.
# Stock Megatron prints no whole-model total of its own, so the builder is
# the only thing that satisfies that rule.
STOCK_PARAMETER_FRAGMENTS = (
    "stock-megatron stage ",
    " local size: ",
    " stock-megatron size: ",
    " total parameters",
)


def _stock_arm() -> Arm:
    return scenario_by_name(SCENARIO_NAME).arm("megatron_stock")


def _titan_arm() -> Arm:
    return scenario_by_name(SCENARIO_NAME).arm("titan_compiled")


def _command(
    arm: Arm,
    *,
    model_size: str = "1b",
    ac_mode: str = "none",
    parallelism: ParallelismSpec = TRIVIAL_SPEC,
    local_batch_size: int | None = None,
    megatron_p2p_sync: str = "off",
    megatron_nan_guard: str = "off",
    megatron_precision: str = "stock",
) -> list[str]:
    data = scenario_by_name(SCENARIO_NAME).data
    if local_batch_size is not None:
        data = replace(data, local_batch_size=local_batch_size)
    if isinstance(arm.config, MegatronStockConfig):
        arm = configured(
            arm,
            p2p_sync=megatron_p2p_sync,
            nan_guard=megatron_nan_guard,
            precision=megatron_precision,
        )
    run = run_spec(model_size, data=data, parallelism=parallelism, ac_mode=ac_mode)
    return command(run, arm, "/tmp/arm-dir")


def _flag_names(command: list[str]) -> list[str]:
    return [token for token in command if token.startswith("--")]


# --------------------------------------------------------------------------
# 1. The scenario declaration.
# --------------------------------------------------------------------------


class StockScenarioDeclarationTests(unittest.TestCase):
    def test_the_scenario_is_registered_with_three_arms(self) -> None:
        scenario = scenario_by_name(SCENARIO_NAME)
        self.assertEqual(
            [arm.name for arm in scenario.arms], ["titan_compiled", "titan_eager", "megatron_stock"]
        )

    def test_the_stock_arm_names_the_stock_engine(self) -> None:
        self.assertIsInstance(engine_for(_stock_arm()), MegatronStockEngine)

    def test_the_titan_arm_is_a_plain_torchtitan_arm(self) -> None:
        self.assertEqual(
            _titan_arm().config,
            TorchTitanConfig(compile=_titan_arm().config.compile),
        )

    def test_the_axis_the_megatron_arm_cannot_honor_is_declined(self) -> None:
        scenario = scenario_by_name(SCENARIO_NAME)
        self.assertEqual(scenario.supported_ac_modes, ("none",))

    def test_the_stock_arm_declares_no_permute_marker(self) -> None:
        """``--moe-permute-fusion`` is off, so its kernel must not be pinned.

        The tuned arm sets the fusion on and pins ``_permute_kernel``. Stock
        Megatron defaults it off, so the same marker here would fail an
        honest run.
        """
        markers = _stock_arm().config.trace_kernel_markers
        self.assertNotIn("_permute_kernel", markers)
        self.assertEqual(
            markers, ("cudnn_generated_fort_native_sdpa", "_mul_silu_split")
        )


# --------------------------------------------------------------------------
# 2. The argv. The hazard is a stray token at the trivial spec.
# --------------------------------------------------------------------------


class TitanArmArgvTests(unittest.TestCase):
    """The titan arm of this scenario, which needs no new package."""

    def test_the_titan_argv_carries_both_less_layers_flags_at_pp4(self) -> None:
        """Weight 0 is what makes the two engines split the same way.

        TorchTitan counts the embedding and the output head as layers.
        Megatron divides ``config.num_layers`` alone. At 16 layers and four
        stages the default weight splits [4, 5, 4, 3] where weight 0 splits
        [4, 4, 4, 4], so without these two flags the engines would train
        different models per rank.
        """
        command = _command(_titan_arm(), parallelism=MESH, local_batch_size=32)
        for flag in (
            "--parallelism.pipeline-parallel-first-stage-less-layers",
            "--parallelism.pipeline-parallel-last-stage-less-layers",
        ):
            with self.subTest(flag=flag):
                self.assertEqual(command[command.index(flag) + 1], "0")

    def test_the_upstream_default_is_still_the_one_those_flags_correct(
        self,
    ) -> None:
        """This test must break when TorchTitan changes the default.

        The two flags above exist because upstream defaults both fields to
        1. If a bump makes the default 0, the flags stop correcting anything
        and somebody must re-read parallelism rule 7 before trusting the
        split. The source is read as text, so this costs no torch import.
        """
        configs = (
            Path(__file__).resolve().parents[1]
            / "third_party/torchtitan/torchtitan/config/configs.py"
        )
        source = configs.read_text()
        for field in (
            "pipeline_parallel_first_stage_less_layers",
            "pipeline_parallel_last_stage_less_layers",
        ):
            with self.subTest(field=field):
                match = re.search(rf"^\s*{field}: int = (\d+)$", source, re.M)
                self.assertIsNotNone(
                    match, f"{field} is gone from {configs}"
                )
                self.assertEqual(
                    match.group(1),
                    "1",
                    f"{field} no longer defaults to 1; re-read parallelism "
                    "rule 7 before trusting the pipeline split",
                )

    def test_the_titan_argv_carries_no_bench_flag(self) -> None:
        command = _command(_titan_arm(), parallelism=MESH, local_batch_size=32)
        self.assertEqual(
            [token for token in command if token.startswith("--bench-")], []
        )


class TrivialSpecArgvTests(unittest.TestCase):
    """No arm of any scenario gains a token at the trivial spec."""

    def test_no_torchtitan_arm_gains_a_parallelism_token(self) -> None:
        """Every TorchTitan arm of every scenario, at the trivial spec.

        This covers the TorchTitan arms; ``StockArgvTests`` covers the
        Megatron arm, and skips until the flag module lands.
        """
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                if not isinstance(arm.config, TorchTitanConfig):
                    continue
                with self.subTest(scenario=scenario.name, arm=arm.name):
                    omitted = command(
                        run_spec(data=scenario.data, ac_mode="none"),
                        arm,
                        "/tmp/arm-dir",
                    )
                    self.assertEqual(
                        [
                            token
                            for token in omitted
                            if token.startswith("--parallelism.")
                        ],
                        [],
                    )

    def test_the_titan_arm_of_this_scenario_gains_none_either(self) -> None:
        command = _command(_titan_arm())
        self.assertEqual(
            [
                token
                for token in command
                if token.startswith("--parallelism.")
            ],
            [],
        )
        self.assertEqual(
            command[: command.index("torchtitan.train") + 1],
            [sys.executable, *torchrun_flags(1), "-m", "torchtitan.train"],
        )


class StockArgvTests(unittest.TestCase):
    """The stock argv, frozen as launcher plus module plus the flag list.

    The argv is not written out as one flat golden list, because every flag
    in it belongs to ``benchmarks/e2e/engines/megatron_stock/flags.py``, and that
    module freezes its own output. What is frozen here is the composition:
    the launcher prefix, the ``python -m`` target, and then exactly what
    ``stock_megatron_flags`` returns for the same three values. A token
    the engine adds outside those places fails the equality.
    """

    def _flags(
        self,
        parallelism,
        local_batch_size=None,
        model_size="1b",
        megatron_p2p_sync="off",
        megatron_nan_guard="off",
        megatron_precision="stock",
    ):
        from benchmarks.e2e.engines.megatron_stock.flags import stock_megatron_flags

        data = scenario_by_name(SCENARIO_NAME).data
        if local_batch_size is not None:
            data = replace(data, local_batch_size=local_batch_size)
        return list(
            stock_megatron_flags(
                run_spec(
                    model_size, data=data, parallelism=parallelism, ac_mode="none"
                ),
                MegatronStockConfig(
                    p2p_sync=megatron_p2p_sync,
                    nan_guard=megatron_nan_guard,
                    precision=megatron_precision,
                ),
                arm_dir="/tmp/arm-dir",
            )
        )

    def test_the_lean_precision_argv_is_exactly_its_two_parts(self) -> None:
        """The value crosses the engine untouched into ``flags.py``.

        ``lean`` needs a sharded dense value, so the mesh here is the
        sharded one. The four flags and their three dtype tokens appear
        once each, and none of them reaches the launcher head.
        """
        command = _command(
            _stock_arm(),
            parallelism=SHARDED_MESH,
            local_batch_size=32,
            megatron_precision="lean",
        )
        head = command[: command.index(DRIVER_MODULE) + 1]
        self.assertEqual(
            command,
            head
            + self._flags(
                SHARDED_MESH,
                local_batch_size=32,
                megatron_precision="lean",
            ),
        )
        self.assertEqual(command.count("--use-precision-aware-optimizer"), 1)
        self.assertNotIn("--use-precision-aware-optimizer", head)

    def test_the_titan_arm_gets_no_token_under_lean(self) -> None:
        """TorchTitan holds its own bf16 optimizer states, and this axis
        never reaches it."""
        self.assertEqual(
            _command(_titan_arm(), megatron_precision="lean"),
            _command(_titan_arm()),
        )

    def test_the_nan_guard_off_argv_is_exactly_its_two_parts(self) -> None:
        """The value crosses the engine untouched into ``flags.py``, at
        the trivial spec and at the mesh: the guard runs at every mesh."""
        for parallelism, local_batch_size in ((TRIVIAL_SPEC, None), (MESH, 32)):
            with self.subTest(pp=parallelism.pp):
                command = _command(
                    _stock_arm(),
                    parallelism=parallelism,
                    local_batch_size=local_batch_size,
                    megatron_nan_guard="off",
                )
                head = command[: command.index(DRIVER_MODULE) + 1]
                self.assertEqual(
                    command,
                    head
                    + self._flags(
                        parallelism,
                        local_batch_size=local_batch_size,
                        megatron_nan_guard="off",
                    ),
                )
                self.assertEqual(
                    command.count("--no-check-for-nan-in-loss-and-grad"), 1
                )
                self.assertNotIn("--no-check-for-nan-in-loss-and-grad", head)

    def test_the_titan_arm_gets_no_token_under_nan_guard_off(self) -> None:
        """TorchTitan has no Megatron NaN guard."""
        self.assertEqual(
            _command(_titan_arm(), megatron_nan_guard="off"),
            _command(_titan_arm()),
        )

    def test_the_p2p_sync_off_argv_is_exactly_its_two_parts(self) -> None:
        """The value crosses the engine untouched into ``flags.py``."""
        command = _command(
            _stock_arm(),
            parallelism=MESH,
            local_batch_size=32,
            megatron_p2p_sync="off",
        )
        head = command[: command.index(DRIVER_MODULE) + 1]
        self.assertEqual(
            command,
            head
            + self._flags(MESH, local_batch_size=32, megatron_p2p_sync="off"),
        )
        self.assertEqual(command[-2:], ["--bench-batch-p2p-sync", "off"])

    def test_the_titan_arm_gets_no_token_under_p2p_sync_off(self) -> None:
        """TorchTitan sends no pipeline message through Megatron."""
        self.assertEqual(
            _command(
                _titan_arm(),
                parallelism=MESH,
                local_batch_size=32,
                megatron_p2p_sync="off",
            ),
            _command(_titan_arm(), parallelism=MESH, local_batch_size=32),
        )

    def test_the_trivial_spec_argv_starts_torchrun_for_one_rank(self) -> None:
        command = _command(_stock_arm())
        self.assertEqual(
            command[: command.index(DRIVER_MODULE) + 1],
            [sys.executable, *torchrun_flags(1), "-m", DRIVER_MODULE],
        )
        # A schedule at pp 1 would name a split that does not happen.
        self.assertNotIn("--bench-pp-schedule", command)

    def test_the_trivial_spec_argv_is_exactly_its_two_parts(self) -> None:
        command = _command(_stock_arm())
        self.assertEqual(
            command,
            [sys.executable, *torchrun_flags(1), "-m", DRIVER_MODULE]
            + self._flags(TRIVIAL_SPEC),
        )

    def test_the_mesh_argv_starts_torchrun_for_eight_ranks(self) -> None:
        command = _command(
            _stock_arm(), parallelism=MESH, local_batch_size=32
        )
        self.assertEqual(
            command[:15],
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node=8",
                "--rdzv-backend",
                "c10d",
                "--rdzv-endpoint",
                "localhost:0",
                "--local-ranks-filter",
                "0,1,2,3,4,5,6,7",
                "--role",
                "rank",
                "--tee",
                "3",
                "-m",
            ],
        )
        self.assertEqual(command[15], DRIVER_MODULE)

    def test_the_mesh_argv_is_exactly_its_two_parts(self) -> None:
        command = _command(
            _stock_arm(), parallelism=MESH, local_batch_size=32
        )
        head = command[: command.index(DRIVER_MODULE) + 1]
        self.assertEqual(
            command, head + self._flags(MESH, local_batch_size=32)
        )

    def test_the_argv_repeats_no_flag_name(self) -> None:
        """Megatron's parser is last-wins, so a duplicate changes a value.

        The two halves of the argv are built by two modules. A flag that
        both emit would be parsed once and read as the second value, with
        nothing in the manifest to show it. This is the check that turns
        such a disagreement into a test failure.
        """
        for label, parallelism, batch in SPECS:
            with self.subTest(spec=label):
                names = _flag_names(
                    _command(
                        _stock_arm(),
                        parallelism=parallelism,
                        local_batch_size=batch,
                    )
                )
                duplicates = sorted(
                    {name for name in names if names.count(name) > 1}
                )
                self.assertEqual(duplicates, [])

    def test_the_argv_carries_every_harness_flag(self) -> None:
        command = _command(
            _stock_arm(), parallelism=MESH, local_batch_size=32
        )
        for flag in BENCH_FLAG_NAMES:
            with self.subTest(flag=flag):
                self.assertIn(flag, command)
        self.assertEqual(
            command[command.index("--bench-pp-schedule") + 1],
            PP_SCHEDULE,
        )

    def test_the_marker_count_matches_the_flags_the_argv_carries(self) -> None:
        """The marker rests on the two batch flags the argv carries.

        ``megatron_mesh_markers`` reads its count from
        ``microbatch_geometry``, and ``flags.py`` builds ``--micro-batch-size``
        and ``--global-batch-size`` from that same function. This test reads
        those two flags back out of the argv, applies Megatron's own
        division, and compares the answer to the marker. It fails whichever
        side moves.
        """
        scenario = scenario_by_name(SCENARIO_NAME)
        for spec, batch in (
            (TRIVIAL_SPEC, None),
            (ParallelismSpec(dp=2), None),
            (ParallelismSpec(pp=2, pp_schedule="1F1B"), None),
            (MESH, 32),
            (
                ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B"),
                8,
            ),
        ):
            with self.subTest(spec=spec, batch=batch):
                command = _command(
                    _stock_arm(), parallelism=spec, local_batch_size=batch
                )
                micro = int(command[command.index("--micro-batch-size") + 1])
                total = int(command[command.index("--global-batch-size") + 1])
                # ConstantNumMicroBatchesCalculator, megatron/core.
                megatron = total // (micro * spec.dp)
                data = scenario.data
                if batch is not None:
                    data = replace(data, local_batch_size=batch)
                marker = megatron_mesh_markers(spec, data, "stock", ())[0]
                self.assertIn(f"microbatches={megatron} ", marker)

    def test_the_argv_carries_no_torchtitan_parallelism_token(self) -> None:
        for label, parallelism, batch in SPECS:
            with self.subTest(spec=label):
                command = _command(
                    _stock_arm(),
                    parallelism=parallelism,
                    local_batch_size=batch,
                )
                self.assertEqual(
                    [
                        token
                        for token in command
                        if token.startswith("--parallelism.")
                    ],
                    [],
                )

    def test_the_model_size_reaches_the_driver_and_the_flags(self) -> None:
        command = _command(
            _stock_arm(),
            model_size="9b",
            parallelism=MESH,
            local_batch_size=32,
        )
        self.assertEqual(command[command.index("--bench-model-size") + 1], "9b")
        shape = shape_by_name("9b")
        self.assertIn(str(shape.n_layers), command)
        self.assertIn(str(shape.dim), command)


class StockArgvRefusalTests(unittest.TestCase):
    """Each refusal lands in the engine check, before the subprocess starts."""

    def _refusals(self, run, arm=None) -> str:
        arm = arm or _stock_arm()
        return " ".join(engine_for(arm).check(run, arm))

    def test_an_owned_passthrough_flag_is_refused(self) -> None:
        self.assertIn(
            "owned by --model-size",
            self._refusals(
                run_spec(ac_mode="none"),
                configured(_stock_arm(), extra_flags=("--num-layers=4",)),
            ),
        )

    def test_sac_is_refused(self) -> None:
        self.assertIn("no Megatron parity", self._refusals(run_spec(ac_mode="sac")))

    def test_an_unseeded_run_is_refused(self) -> None:
        self.assertIn(
            "seeded workload", self._refusals(run_spec(ac_mode="none", seed=None))
        )

    def test_a_schedule_this_driver_does_not_implement_is_refused(
        self,
    ) -> None:
        """Megatron-LM implements Interleaved1F1B, and the driver does not."""
        spec = replace(MESH, pp_schedule="Interleaved1F1B")
        self.assertIn(
            "implements '1F1B' alone",
            self._refusals(
                run_spec(ac_mode="none", parallelism=spec, local_batch_size=32)
            ),
        )


# --------------------------------------------------------------------------
# 3. The validation. The hazard is a validation that proves nothing.
# --------------------------------------------------------------------------


def _observed(line: str):
    """The mesh that the stock evidence reader takes from one log line."""
    return read_evidence(0, line + "\nTraining completed\n").mesh


class StockValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = replace(
            scenario_by_name(SCENARIO_NAME).data, local_batch_size=32
        )

    def test_the_engine_is_registered_and_asks_for_its_treatments(self) -> None:
        self.assertIn(MegatronStockConfig, ENGINE_RECORDS)
        refusals = " ".join(
            engine_for(_stock_arm()).validate(
                run_spec(ac_mode="none", profile=False),
                _stock_arm(),
                Path("/a"),
                {0: "Training completed\n"},
            )
        )
        self.assertIn("megatron nan guard did not apply", refusals)
        self.assertIn("megatron precision did not apply", refusals)

    def test_the_driver_line_is_the_first_precision_marker(self) -> None:
        """Rule 8 no longer holds this line, and it is still matched: the
        precision markers ask every rank for it at every mesh."""
        self.assertEqual(
            precision_markers("stock")[0],
            "Megatron-LM stock training loop (",
        )

    def test_the_engine_does_not_check_the_ac_line(self) -> None:
        lines = required_lines(run_spec(ac_mode="none"), _stock_arm())
        self.assertFalse(
            any("SelectiveAC" in line for group in lines.values() for line in group)
        )

    def test_the_profile_proves_no_compile_treatment(self) -> None:
        """``compile_marker`` is None, so rule 8 asks this engine nothing.

        megatron-core compiles no whole layer, so no log line proves the
        treatment either way. The Megatron config has no compile field.
        """
        lines = required_lines(run_spec(ac_mode="none"), _stock_arm())
        self.assertFalse(
            any("torch.compile" in line for group in lines.values() for line in group)
        )
        self.assertFalse(hasattr(_stock_arm().config, "compile"))

    # -- arm rule 12 ----------------------------------------------------

    def test_the_markers_interpolate_the_spec(self) -> None:
        markers = megatron_mesh_markers(
            MESH, self.data, "stock", ()
        )
        self.assertEqual(
            markers[0],
            "Megatron-LM stock parallelism: dp=2 pp=4 ep=1 schedule=1F1B "
            "microbatches=8 stages=4",
        )
        self.assertEqual(
            markers[1],
            "Megatron-LM stock data parallel: DistributedDataParallel over "
            "2 ranks (overlap_grad_reduce=False, grad_reduce_in_fp32=True, "
            "sharding_strategy=no_shard, expert_parallel=1, "
            "optimizer=ChainedOptimizer[Float16OptimizerWithFloat16Params])",
        )

    def test_the_sharded_markers_name_the_sharded_optimizer(self) -> None:
        """``--zero 1`` moves the optimizer field of the second line.

        A run that lost ``--use-distributed-optimizer`` prints the
        replicated line and fails arm rule 12, which is the point.
        """
        markers = megatron_mesh_markers(
            SHARDED_MESH, self.data, "stock", ()
        )
        self.assertEqual(
            markers[0],
            "Megatron-LM stock parallelism: dp=2 pp=4 ep=2 schedule=1F1B "
            "microbatches=8 stages=4",
        )
        self.assertEqual(
            markers[1],
            "Megatron-LM stock data parallel: DistributedDataParallel "
            "over 2 ranks (overlap_grad_reduce=False, "
            "grad_reduce_in_fp32=True, "
            "sharding_strategy=no_shard, expert_parallel=2, "
            "optimizer=ChainedOptimizer[DistributedOptimizer])",
        )

    def test_the_two_values_share_no_data_parallel_marker(self) -> None:
        """A replicated log must not satisfy a sharded arm's rule.

        The manifest records the zero value. If one log could
        satisfy both markers, a run that ignored the sharding flags would
        publish under the sharded label.
        """
        replicated = megatron_mesh_markers(
            MESH, self.data, "stock", ()
        )[1]
        sharded = megatron_mesh_markers(
            SHARDED_MESH, self.data, "stock", ()
        )[1]
        self.assertNotEqual(replicated, sharded)
        self.assertNotIn(replicated, sharded)
        self.assertNotIn(sharded, replicated)

    def test_the_marker_names_the_members_of_the_chain(self) -> None:
        """The standard path chains, and the marker names the members.

        ``get_megatron_optimizer`` ends its standard path with an
        unconditional ``ChainedOptimizer(optimizers)``, so ``zero 0``
        and ``zero 1`` both print a chain. The ``30b-a3b`` cell of
        2026-09-16 printed the outer class alone, and arm rule 12 refused
        it.
        """
        line = megatron_mesh_markers(
            replace(MESH, zero=1), self.data, "stock", ()
        )[1]
        self.assertIn(
            "optimizer=ChainedOptimizer[DistributedOptimizer])", line
        )

    def test_a_replicated_chain_cannot_satisfy_a_zero1_marker(self) -> None:
        """The hazard: ZeRO-0 must not publish under a ZeRO-1 label.

        Megatron keeps ``DistributedDataParallel`` at both levels, so
        the wrapper separates neither, and both chains carry the same
        outer class. A chain of ``Float16OptimizerWithFloat16Params`` is
        what a run that lost ``--use-distributed-optimizer`` builds.
        """
        zero1 = megatron_mesh_markers(
            replace(MESH, zero=1),
            self.data,
            "stock", (),
        )[1]
        replicated = megatron_mesh_markers(
            MESH, self.data, "stock", ()
        )[1]
        self.assertIn(
            "optimizer=ChainedOptimizer[DistributedOptimizer])", zero1
        )
        self.assertIn(
            "optimizer=ChainedOptimizer["
            "Float16OptimizerWithFloat16Params])",
            replicated,
        )
        self.assertNotEqual(zero1, replicated)

    def test_the_gradient_reduction_follows_the_precision(self) -> None:
        """``--megatron-precision lean`` sends ``--main-grads-dtype bf16``.

        Megatron then leaves ``accumulate_allreduce_grads_in_fp32`` off,
        and the wrapper carries ``grad_reduce_in_fp32=False``. This marker
        pinned True before 2026-09-16, and the real cell failed on it.
        """
        sharded = replace(MESH, zero=1)
        stock = megatron_mesh_markers(
            sharded, self.data, "stock", ()
        )[1]
        lean = megatron_mesh_markers(
            sharded, self.data, "lean", ()
        )[1]
        self.assertIn("grad_reduce_in_fp32=True", stock)
        self.assertIn("grad_reduce_in_fp32=False", lean)
        self.assertNotEqual(stock, lean)

    def test_the_markers_are_non_empty_above_world_size_one(self) -> None:
        """Each marker names the mesh, and the first one names the degrees.

        ``tests/test_engines.py`` pins the non-empty part over every
        engine. This test adds what that one cannot: the content of this
        driver's own line at four meshes.
        """
        for spec in (
            ParallelismSpec(pp=2, pp_schedule="1F1B"),
            ParallelismSpec(dp=2),
            ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B", pp_microbatch_size=4),
            ParallelismSpec(dp=8),
        ):
            with self.subTest(spec=spec):
                markers = megatron_mesh_markers(
                    spec, self.data, "stock", ()
                )
                self.assertTrue(markers)
                self.assertIn(
                    f"dp={spec.dp} pp={spec.pp} ep={spec.ep}", markers[0]
                )

    def test_the_data_parallel_line_appears_only_above_dp_one(self) -> None:
        pipeline_only = megatron_mesh_markers(
            ParallelismSpec(pp=4, pp_schedule="1F1B", pp_microbatch_size=4),
            self.data,
            "stock", (),
        )
        self.assertEqual(len(pipeline_only), 1)
        with_dp = megatron_mesh_markers(
            MESH, self.data, "stock", ()
        )
        self.assertEqual(len(with_dp), 2)

    def test_the_microbatch_count_is_megatrons_own_arithmetic(self) -> None:
        """The marker must state the count Megatron itself derives.

        Megatron derives ``get_num_microbatches()`` as ``global_batch_size
        // (micro_batch_size * dp)``. This test reads those two flags out of
        the argv the harness really builds, applies that division, and
        compares the answer to the marker. It fails whichever side moves.

        **The count is 1 at ``pp`` 1**, because one Megatron sample is one
        packed sequence and the harness packs the whole local batch into one
        sample without a pipeline. ``n_microbatches`` describes the split a
        pipeline would make, so it is not this number and is not read here.
        """
        for spec, local_batch_size in (
            (MESH, 32),
            (ParallelismSpec(dp=2), 4),
            (TRIVIAL_SPEC, 4),
            (ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B"), 8),
        ):
            with self.subTest(spec=spec, batch=local_batch_size):
                command = _command(
                    _stock_arm(),
                    parallelism=spec,
                    local_batch_size=local_batch_size,
                )
                micro = int(command[command.index("--micro-batch-size") + 1])
                total = int(command[command.index("--global-batch-size") + 1])
                # ConstantNumMicroBatchesCalculator, megatron/core.
                megatron = total // (micro * spec.dp)
                data = replace(
                    self.data, local_batch_size=local_batch_size
                )
                marker = megatron_mesh_markers(
                    spec, data, "stock", ()
                )[0]
                self.assertIn(f"microbatches={megatron} ", marker)
                if spec.pp == 1:
                    self.assertEqual(megatron, 1)

    # -- the two inversions ---------------------------------------------

    def test_the_pipelined_pattern_ignores_a_pipeline_degree_of_one(
        self,
    ) -> None:
        line = megatron_mesh_markers(
            ParallelismSpec(dp=2), self.data, "stock", ()
        )[0]
        self.assertIn("pp=1", line)
        self.assertEqual(_observed(line).pp, 1)

    def test_the_pipelined_pattern_sees_a_real_pipeline(self) -> None:
        for spec in (
            ParallelismSpec(pp=2, pp_schedule="1F1B"),
            MESH,
            ParallelismSpec(pp=4, pp_schedule="1F1B", pp_microbatch_size=4),
        ):
            with self.subTest(spec=spec):
                line = megatron_mesh_markers(
                    spec, self.data, "stock", ()
                )[0]
                self.assertEqual(_observed(line).pp, spec.pp)

    def test_the_data_parallel_pattern_ignores_a_degree_of_one(self) -> None:
        """This is the hazard: a pattern that matches ``dp=1`` passes always.

        Every line a ``dp`` 1 run can print is checked, including the
        pipeline-only mesh line, which carries ``dp=1`` in the same
        position.
        """
        for spec in (
            TRIVIAL_SPEC,
            ParallelismSpec(pp=2, pp_schedule="1F1B"),
            ParallelismSpec(pp=4, pp_schedule="1F1B", pp_microbatch_size=4),
        ):
            with self.subTest(spec=spec):
                for line in megatron_mesh_markers(
                    spec, self.data, "stock", ()
                ):
                    self.assertEqual(_observed(line).dp, 1, line)

    def test_the_data_parallel_pattern_sees_every_degree_above_one(
        self,
    ) -> None:
        """Two digits must match too. ``(?!1\\b)`` refuses 1, never 10."""
        for dp in (2, 4, 8, 10, 12, 16):
            with self.subTest(dp=dp):
                line = (
                    f"Megatron-LM stock parallelism: dp={dp} pp=1 ep=1 "
                    "schedule=1F1B microbatches=4 stages=1"
                )
                self.assertEqual(_observed(line).dp, dp)

    def test_the_data_parallel_pattern_sees_the_wrapper_line(self) -> None:
        for spec in (MESH, SHARDED_MESH):
            with self.subTest(zero=spec.zero):
                line = megatron_mesh_markers(
                    spec, self.data, "stock", ()
                )[1]
                observed = _observed(line)
                self.assertEqual((observed.dp, observed.zero), (spec.dp, spec.zero))

    def test_each_level_names_the_wrapper_class(self) -> None:
        """Asserted by name, from the table beside the flags that build it.

        The class name says which mechanism ran, and the driver reads it
        off the wrapper rather than from the argv. Both levels build one
        class here, so the optimizer field is what separates them; the
        test above pins that.
        """
        for spec in (MESH, SHARDED_MESH):
            with self.subTest(zero=spec.zero):
                line = megatron_mesh_markers(
                    spec, self.data, "stock", ()
                )[1]
                for roster in (
                    STOCK_WRAPPER_FRAGMENTS, STOCK_OVERLAP_FRAGMENTS
                ):
                    self.assertIn(roster[spec.zero], line)


# --------------------------------------------------------------------------
# 4. The contract between this profile and the driver the other agent owns.
# --------------------------------------------------------------------------


class _StockArgs:
    """The five ``args`` fields ``parallelism_lines`` reads.

    Megatron's own parser produces these. Building them from a
    ``ParallelismSpec`` is what lets this module compare the driver's real
    line against the profile's marker without a Megatron-LM checkout.

    ``expert_model_parallel_size`` is the flag verbatim, which is what
    Megatron's parser holds at the point the driver prints this line: no
    process group exists until ``pretrain()`` runs. The driver reads the
    built group later, in ``install_data_parallel_marker``, and refuses a
    disagreement there.
    """

    def __init__(self, spec: ParallelismSpec) -> None:
        self.world_size = spec.world_size
        self.data_parallel_size = spec.dp
        self.pipeline_model_parallel_size = spec.pp
        self.expert_model_parallel_size = spec.ep
        self.bench_pp_schedule = spec.pp_schedule


# The meshes the diff runs over. Each pair is a spec and the local batch
# size the run would carry, because the microbatch count reads both.
STOCK_MESH_CASES = (
    (ParallelismSpec(dp=2), 4),
    (ParallelismSpec(pp=2, pp_schedule="1F1B"), 4),
    (ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"), 8),
    (ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B"), 32),
    (ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B"), 8),
    # The zero control cell of the matrix, and the expert split
    # it makes legal. Both lines carry a field that moves between the two
    # values, so a diff over the replicated cells alone proves half of it.
    (ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B", zero=1), 32),
    (
        ParallelismSpec(dp=2, pp=4, ep=2, pp_schedule="1F1B", zero=1),
        32,
    ),
    # The deepest pipeline eight GPUs hold. pp 8 forces dp 1, so it prints
    # the mesh line and no data-parallel line.
    (ParallelismSpec(pp=8, pp_schedule="1F1B", pp_microbatch_size=2), 32),
)


def _geometry(data, spec):
    from benchmarks.e2e.engines.megatron_stock.flags import microbatch_geometry

    return microbatch_geometry(data, spec)


def _driver_data_parallel_line(
    zero: str,
    *,
    dp: int,
    ep: int,
    megatron_precision: str = "stock",
) -> str:
    """The driver's data-parallel line for one zero value.

    ``install_data_parallel_marker`` fills these five fields from the
    wrapper, the optimizer and the expert group. This helper states the
    values Megatron resolves for each ``--zero`` value, so the
    diff below reads the driver's own template.

    **The optimizer field names a chain**, because Megatron builds one
    optimizer for each ``(optimizer_name, is_expert)`` bucket and every
    registered shape is a mixture of experts. ``grad_reduce_in_fp32``
    follows ``--megatron-precision``, and neither value is written here.
    """
    from benchmarks.e2e.engines.megatron_stock.driver import markers
    from benchmarks.e2e.engines.megatron_stock.flags import (
        DATA_PARALLEL_WRAPPERS,
        SHARDING_STRATEGIES,
        data_parallel_optimizer,
        data_parallel_overlap,
        grad_reduce_in_fp32,
    )

    return markers.DATA_PARALLEL_LINE.format(
        wrapper=DATA_PARALLEL_WRAPPERS[zero],
        dp=dp,
        overlap=data_parallel_overlap(()),
        fp32=grad_reduce_in_fp32(megatron_precision),
        sharding=SHARDING_STRATEGIES[zero],
        expert=ep,
        optimizer=data_parallel_optimizer(zero),
    )


def _driver_lines() -> list[str]:
    """Every line the driver prints that a marker fragment can live in."""
    from benchmarks.e2e.engines.megatron_stock.driver import markers

    spec = ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B")
    return [
        markers.TRAINING_COMPLETED,
        markers.MODE_LINE.format(
            mode="default",
            main_params_dtype="torch.float32",
            main_grads_dtype="torch.float32",
            precision_aware=False,
            exp_avg_dtype="torch.float32",
            exp_avg_sq_dtype="torch.float32",
            accumulate=True,
            cross_entropy_loss_fusion=False,
            dispatcher="alltoall",
        ),
        *markers.parallelism_lines(_StockArgs(spec), microbatches=8),
        _driver_data_parallel_line(0, dp=2, ep=1),
        _driver_data_parallel_line(1, dp=2, ep=2),
        markers.STAGE_SIZE_LINE.format(stage=0, stages=4, count=1),
        markers.MODEL_SIZE_LINE.format(size="1b", total="1,066,241,024"),
        markers.P2P_LINE.format(comm=True, sync=True),
        markers.P2P_LINE.format(comm=True, sync=False),
        markers.NAN_GUARD_LINE.format(value=True),
        markers.NAN_GUARD_LINE.format(value=False),
    ]


class StockMarkerContractTests(unittest.TestCase):
    """The marker strings, pinned in both directions.

    A one-character difference between the driver's line and this profile's
    marker fails a real eight-GPU run at arm rule 12, hours after it
    started. These fragments are the parts of the four marker strings that
    carry no interpolated value.
    """

    def setUp(self) -> None:
        self.data = replace(
            scenario_by_name(SCENARIO_NAME).data, local_batch_size=32
        )

    def test_every_fragment_belongs_to_a_marker_this_profile_builds(
        self,
    ) -> None:
        """The fragments cannot drift away from the profile."""
        built = [
            COMPLETION_LINE,
            *precision_markers("stock"),
            *megatron_mesh_markers(
                MESH, self.data, "stock", ()
            ),
            *p2p_markers(MESH, "on"),
            *nan_guard_markers("on"),
        ]
        for fragment in STOCK_LOG_FRAGMENTS:
            with self.subTest(fragment=fragment):
                self.assertTrue(
                    any(fragment in marker for marker in built),
                    f"{fragment!r} is in no marker this profile builds",
                )

    def test_the_driver_prints_every_fragment(self) -> None:
        """The driver's own lines must carry each fragment verbatim.

        The lines are built from the driver's own constants rather than
        searched for in its source. A source search cannot see a fragment
        that spans an interpolated field, and the driver writes
        ``schedule={schedule} microbatches=`` rather than the literal the
        profile matches.

        Importing the driver costs no torch and no megatron: every heavy
        import in ``train.py`` sits inside ``main``.
        """
        for fragment in (
            *STOCK_LOG_FRAGMENTS,
            *STOCK_WRAPPER_FRAGMENTS.values(),
            *STOCK_OVERLAP_FRAGMENTS.values(),
            *STOCK_MODE_LINE_FIELDS,
            *STOCK_PARAMETER_FRAGMENTS,
        ):
            with self.subTest(fragment=fragment):
                self.assertTrue(
                    any(fragment in line for line in _driver_lines()),
                    f"{fragment!r} is in no line the driver prints",
                )

    def test_the_driver_mesh_line_equals_this_profile_marker(self) -> None:
        """The character-for-character diff, at every mesh this run allows.

        A one-character difference fails a real eight-GPU run at arm rule
        12, hours after it started. The driver builds its line through
        ``parallelism_lines``, which is the function a real run calls, so
        this compares the two strings and not two descriptions of them.
        """
        from benchmarks.e2e.engines.megatron_stock.driver import markers

        for spec, batch in STOCK_MESH_CASES:
            with self.subTest(spec=spec, batch=batch):
                data = replace(
                    scenario_by_name(SCENARIO_NAME).data,
                    local_batch_size=batch,
                )
                _, microbatches, _ = _geometry(data, spec)
                printed = markers.parallelism_lines(
                    _StockArgs(spec), microbatches=microbatches
                )
                expected = megatron_mesh_markers(
                    spec, data, "stock", ()
                )
                self.assertEqual(printed[0], expected[0])

    def test_the_driver_data_parallel_line_equals_this_profile_marker(
        self,
    ) -> None:
        """The other half of the same diff, above ``dp`` 1.

        The driver formats this line with the wrapper's own
        ``ddp_config``, so every value in it is resolved rather than
        declared.

        ``grad_reduce_in_fp32`` is True under both parities: ``--bf16``
        with the default ``--main-grads-dtype fp32`` sets
        ``accumulate_allreduce_grads_in_fp32``, which
        ``get_megatron_ddp_config`` copies in.

        **``overlap_grad_reduce`` MOVES with the parity, and the argv is not
        why.** The flag list omits ``--overlap-grad-reduce`` under both
        values, so reading the argument would give False under both.
        ``MegatronFSDP.__init__`` then sets it True on the config object it
        was handed -- the reference, not a copy -- whenever the sharding
        strategy is one Megatron overlaps. ``data_parallel_overlap`` gives
        the expected value.
        """
        from benchmarks.e2e.engines.megatron_stock.driver import markers

        for spec, batch in STOCK_MESH_CASES:
            if spec.dp == 1:
                continue
            with self.subTest(spec=spec, batch=batch):
                data = replace(
                    scenario_by_name(SCENARIO_NAME).data,
                    local_batch_size=batch,
                )
                printed = _driver_data_parallel_line(
                    spec.zero, dp=spec.dp, ep=spec.ep
                )
                markers = megatron_mesh_markers(
                    spec, data, "stock", ()
                )
                self.assertEqual(printed, markers[1])

    def test_the_driver_p2p_line_equals_this_profile_marker(self) -> None:
        """The p2p half of arm rule 12, character for character.

        The driver formats the line off the config ``gpt_config_from_args``
        built, so the two fields are observations. ``batch_p2p_comm`` reads
        True on this arm: stock Megatron derives it as ``not
        overlap_p2p_comm`` and forces the overlap off for the
        non-interleaved schedule.
        """
        from benchmarks.e2e.engines.megatron_stock.driver import markers

        for value, sync in (("on", True), ("off", False)):
            with self.subTest(value=value):
                self.assertEqual(
                    p2p_markers(MESH, value),
                    (markers.P2P_LINE.format(comm=True, sync=sync),),
                )

    def test_no_p2p_line_is_asked_below_a_pipeline(self) -> None:
        """Below ``pp`` 1 there is no message to synchronize."""
        for spec in (TRIVIAL_SPEC, ParallelismSpec(dp=2)):
            for value in ("on", "off"):
                with self.subTest(spec=spec, value=value):
                    self.assertEqual(p2p_markers(spec, value), ())

    def test_the_driver_nan_guard_line_equals_this_profile_marker(
        self,
    ) -> None:
        """The nan guard half of arm rule 12, character for character.

        The driver formats Megatron's own bool, so the two tokens are
        ``True`` and ``False``.
        """
        from benchmarks.e2e.engines.megatron_stock.driver import markers

        for value, parsed in (("on", True), ("off", False)):
            with self.subTest(value=value):
                self.assertEqual(
                    nan_guard_markers(value),
                    (markers.NAN_GUARD_LINE.format(value=parsed),),
                )

    def test_the_nan_guard_line_is_asked_at_every_mesh(self) -> None:
        """Unlike the p2p line: the guard runs at pp 1 and at dp 1, so the
        callable takes no spec and the same line is asked everywhere."""
        (line,) = nan_guard_markers("on")
        self.assertIn("stock", line)

    def test_the_driver_mode_line_starts_with_this_profile_marker(
        self,
    ) -> None:
        """The first precision marker is the prefix up to the bracket."""
        from benchmarks.e2e.engines.megatron_stock.driver import markers

        printed = markers.MODE_LINE.format(
            mode="default",
            main_params_dtype="torch.float32",
            main_grads_dtype="torch.float32",
            precision_aware=False,
            exp_avg_dtype="torch.float32",
            exp_avg_sq_dtype="torch.float32",
            accumulate=True,
            cross_entropy_loss_fusion=False,
            dispatcher="alltoall",
        )
        self.assertTrue(
            printed.startswith(precision_markers("stock")[0])
        )

    def test_the_driver_prints_the_data_parallel_line_once(self) -> None:
        """``install_data_parallel_marker`` is its only source.

        A copy derived from ``args`` would satisfy arm rule 12 on its own,
        so a run that lost the wrapper shim would pass the rule the shim
        exists to enforce.
        """
        from benchmarks.e2e.engines.megatron_stock.driver import markers

        spec = ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B")
        printed = markers.parallelism_lines(_StockArgs(spec), microbatches=8)
        self.assertEqual(len(printed), 1)
        self.assertNotIn("stock data parallel", printed[0])

    def test_the_driver_package_prints_no_tuned_marker(self) -> None:
        """The stock driver must not print an earlier driver's marker lines."""
        source = self._package_source()
        for fragment in (
            "Megatron-LM training loop (mode=",
            "Megatron-LM parallelism: dp=",
            "Megatron-LM data parallel: DistributedDataParallel over ",
        ):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, source)

    def _package_source(self) -> str:
        spec = find_spec(STOCK_PACKAGE)
        assert spec is not None and spec.submodule_search_locations
        root = Path(list(spec.submodule_search_locations)[0])
        return "\n".join(
            path.read_text() for path in sorted(root.rglob("*.py"))
        )


# --------------------------------------------------------------------------
# 5. The run axes, resolved the way a real command line resolves them.
# --------------------------------------------------------------------------


class StockRunResolutionTests(unittest.TestCase):
    """What ``run`` accepts for this scenario.

    The compiled and the eager treatment are arm properties now, so the
    roster resolves without a compile axis. Every other combination must be
    refused before a GPU is claimed.
    """

    def _resolve(self, names: tuple[str, ...], ac_mode: str):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", dict(_METADATA)),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return _resolve_run(
                RunRequest(
                    axes=RequestedAxes(
                        ac_mode=ac_mode,
                    ),
                    gpu="0",
                    scenario_name=SCENARIO_NAME,
                    arm_names=names,
                    out_dir=Path(temporary) / "run",
                ),
                {"PATH": os.environ["PATH"]},
            )

    def test_sac_is_refused_for_the_whole_scenario(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support ac mode"):
            self._resolve((), "sac")

    def test_every_subset_resolves_at_ac_none(self) -> None:
        for names in (
            (),
            ("megatron_stock",),
            ("megatron_stock", "titan_compiled"),
            ("titan_compiled",),
            ("titan_eager",),
        ):
            with self.subTest(names=names):
                self.assertTrue(self._resolve(names, "none").arms)

    def test_the_eager_arm_resolves_on_its_own(self) -> None:
        """Cell 2 of the run matrix: the eager arm needs no Megatron
        opponent, and it names its own treatment."""
        resolved = self._resolve(("titan_eager",), "none")
        self.assertEqual([arm.name for arm in resolved.arms], ["titan_eager"])
        self.assertNotIn("--compile.enable", resolved.commands["titan_eager"])


if __name__ == "__main__":
    unittest.main()
