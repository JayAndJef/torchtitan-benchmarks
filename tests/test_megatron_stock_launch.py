"""The wiring of the stock Megatron-LM scenario: argv, profile, registry.

Three things join here, and each can fail silently:

1. ``command_for_arm`` must build the stock argv and must leave every other
   arm's argv alone. A stray token at the trivial spec changes what every
   existing scenario measures, and no rule reads a command line back.
2. The ``megatron_stock`` validation profile must prove the mesh. A profile
   that proves nothing passes, which is worse than no profile.
3. The scenario must decline every mode its Megatron arm cannot honor, and
   must still admit a TorchTitan-only subset at ``--compile-mode none``.

**The golden argv tests live here and not in
``tests/test_migration_contract.py``.** That module holds the goldens for
the two engines that already existed, and three agents wrote this scenario
in parallel; a new golden there would have collided.

**Some tests here need ``benchmarks/e2e/megatron_stock/``, which the driver
agent owns.** Each such test skips with a named reason until that package
lands. The skip states the missing module, so a reader cannot take it for a
passing test.
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

from benchmarks.e2e.launch import (
    STOCK_MEGATRON_DRIVER_MODULE,
    STOCK_MEGATRON_PP_SCHEDULE,
    command_for_arm,
)
from benchmarks.e2e.parallelism import (
    ParallelismSpec,
    TRIVIAL_SPEC,
    n_microbatches,
)
from benchmarks.e2e.registry import SCENARIOS, Arm, scenario_by_name
from benchmarks.e2e.runner import RunRequest, _resolve_run
from benchmarks.e2e.validation import (
    VALIDATION_PROFILES,
    _megatron_stock_parallelism_markers,
    validate_arm,
)
from benchmarks.execution.affinity import CpuPinning
from benchmarks.models.piper_qwen3.shape import shape_by_name


SCENARIO_NAME = "piper_megatron_stock"
STOCK_PACKAGE = "benchmarks.e2e.megatron_stock"
STOCK_FLAGS_MODULE = f"{STOCK_PACKAGE}.flags"
# The driver itself, which prints the marker strings. It is a separate
# gate from the flag module, because the package lands in more than one
# commit and the flag module comes first.
STOCK_DRIVER_MODULE = f"{STOCK_PACKAGE}.train"

# The mesh of the run matrix: two pipelines of four stages, eight GPUs.
MESH = ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B", pp_microbatch_size=4)

# Every mesh the argv checks below sweep, with the batch each needs.
#
# ``MESH`` needs the lifted caps: ``MAX_WORLD_SIZE`` is 4 and ``MAX_PP`` is 2
# until the parallelism agent's commit merges, so ``validate_parallelism``
# refuses it today. ``command_for_arm`` calls no validator, so the argv is
# still buildable and still worth freezing -- it is the mesh all four cells
# of the run matrix use. The three meshes above it are the ones a run can
# reach at this rev, and they are swept so the checks are not all taken on a
# spec nothing can start yet.
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
    "--bench-mode",
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
    "Megatron-LM stock training loop (mode=",
    "Megatron-LM stock parallelism: dp=",
    # ``pipelined_pattern`` needs ``dp=<n> pp=`` with one space between, so
    # the pipeline token is part of the contract and not decoration. A
    # driver printing ``dp=2, pp=4`` fails arm rule 12 on a real run.
    " pp=",
    " schedule=1F1B microbatches=",
    " stages=",
    "Megatron-LM stock data parallel: DistributedDataParallel over ",
    " ranks (overlap_grad_reduce=False, grad_reduce_in_fp32=True)",
)

# The rest of the mode line. ``ValidationProfile.mode_line`` stops at the
# comma after the mode, so no run-time rule reads these five fields; the
# plan puts them on the line so the log records the treatment per run.
#
# **Only the field names are pinned, never the values.** Every one of the
# five is a documented reversal target: turning cross-entropy fusion on, or
# moving the reduction to bf16, changes a value here and must stay a
# one-line edit in ``flags.py``.
STOCK_MODE_LINE_FIELDS = (
    "main_params_dtype=",
    "main_grads_dtype=",
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


def _missing(module: str) -> str | None:
    """The reason ``module`` cannot be imported, or None when it can."""
    try:
        if find_spec(module) is None:
            return f"{module} does not exist yet"
    except ModuleNotFoundError as error:
        return f"{module} does not exist yet ({error})"
    return None


def _skip_without_stock_package(module: str):
    reason = _missing(module)
    return unittest.skipIf(
        reason is not None,
        f"pending the stock Megatron driver package: {reason}",
    )


def _stock_arm() -> Arm:
    return scenario_by_name(SCENARIO_NAME).arm("baseline")


def _titan_arm() -> Arm:
    return scenario_by_name(SCENARIO_NAME).arm("titan_stock")


def _command(
    arm: Arm,
    *,
    model_size: str = "1b",
    compile_mode: str = "default",
    ac_mode: str = "none",
    parallelism: ParallelismSpec = TRIVIAL_SPEC,
    local_batch_size: int | None = None,
) -> list[str]:
    scenario = scenario_by_name(SCENARIO_NAME)
    workload = scenario.workload
    if local_batch_size is not None:
        workload = replace(workload, local_batch_size=local_batch_size)
    return command_for_arm(
        workload,
        arm,
        Path("/tmp/arm-dir"),
        (),
        compile_mode,
        ac_mode,
        model_size=model_size,
        parallelism=parallelism,
    )


def _flag_names(command: list[str]) -> list[str]:
    return [token for token in command if token.startswith("--")]


# --------------------------------------------------------------------------
# 1. The scenario declaration.
# --------------------------------------------------------------------------


class StockScenarioDeclarationTests(unittest.TestCase):
    def test_the_scenario_is_registered_with_two_arms(self) -> None:
        scenario = scenario_by_name(SCENARIO_NAME)
        self.assertEqual(
            [arm.name for arm in scenario.arms], ["baseline", "titan_stock"]
        )

    def test_the_stock_arm_names_the_stock_launcher_and_profile(self) -> None:
        arm = _stock_arm()
        self.assertEqual(arm.launcher, "megatron_stock")
        self.assertEqual(arm.validation, "megatron_stock")

    def test_the_titan_arm_is_a_plain_torchtitan_arm(self) -> None:
        arm = _titan_arm()
        self.assertEqual(arm.launcher, "torchtitan")
        self.assertEqual(arm.validation, "torchtitan")
        # It reuses the scenario workload's pre-tokenized config, exactly as
        # the arm of the same name in piper1b_megatron does.
        self.assertIsNone(arm.config)
        self.assertEqual(
            scenario_by_name(SCENARIO_NAME).workload,
            scenario_by_name("piper1b_megatron").workload,
        )

    def test_the_scenario_declares_no_regions(self) -> None:
        """Rule 7 guards nothing here, so it must not claim to.

        Region pooling reads Inductor's whole-block annotations. Stock
        Megatron has none, and a pipelined run of either engine reaches a
        different invocation count per rank.
        """
        self.assertEqual(scenario_by_name(SCENARIO_NAME).regions, ())

    def test_the_axes_the_megatron_arm_cannot_honor_are_declined(self) -> None:
        scenario = scenario_by_name(SCENARIO_NAME)
        self.assertEqual(scenario.supported_ac_modes, ("none",))
        self.assertEqual(scenario.supported_compile_modes, ("default",))

    def test_the_stock_arm_declares_no_permute_marker(self) -> None:
        """``--moe-permute-fusion`` is off, so its kernel must not be pinned.

        The tuned arm sets the fusion on and pins ``_permute_kernel``. Stock
        Megatron defaults it off, so the same marker here would fail an
        honest run.
        """
        markers = _stock_arm().trace_kernel_markers
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

        ``tests/test_migration_contract.py`` asks the same question over the
        whole registry, and its ``subTest`` for the new Megatron arm now
        errors on the ac mode, so that arm alone is unasserted there. This
        covers the arms that already existed; ``StockArgvTests`` covers the
        new one, and skips until the flag module lands.
        """
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                if arm.launcher != "torchtitan":
                    continue
                with self.subTest(scenario=scenario.name, arm=arm.name):
                    omitted = command_for_arm(
                        scenario.workload,
                        arm,
                        Path("/tmp/arm-dir"),
                        (),
                        "default",
                        "none",
                    )
                    named = command_for_arm(
                        scenario.workload,
                        arm,
                        Path("/tmp/arm-dir"),
                        (),
                        "default",
                        "none",
                        parallelism=TRIVIAL_SPEC,
                    )
                    self.assertEqual(omitted, named)
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
        self.assertEqual(command[0], "./run_train.sh")


@_skip_without_stock_package(STOCK_FLAGS_MODULE)
class StockArgvTests(unittest.TestCase):
    """The stock argv, frozen as launcher plus module plus the flag list.

    The argv is not written out as one flat golden list, because every flag
    in it belongs to ``benchmarks/e2e/megatron_stock/flags.py``, and that
    module freezes its own output. What is frozen here is the composition:
    the launcher prefix, the ``python -m`` target, and then exactly what
    ``stock_megatron_flags`` returns for the same three values. A token
    ``launch.py`` adds outside those places fails the equality.
    """

    def _flags(
        self,
        parallelism,
        local_batch_size=None,
        model_size="1b",
        compile_mode="default",
    ):
        from benchmarks.e2e.megatron_stock.flags import stock_megatron_flags

        workload = scenario_by_name(SCENARIO_NAME).workload
        if local_batch_size is not None:
            workload = replace(workload, local_batch_size=local_batch_size)
        return list(
            stock_megatron_flags(
                shape_by_name(model_size),
                workload,
                parallelism,
                arm_dir="/tmp/arm-dir",
                model_size=model_size,
                compile_mode=compile_mode,
            )
        )

    def test_the_trivial_spec_argv_starts_no_launcher(self) -> None:
        command = _command(_stock_arm())
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1:3], ["-m", STOCK_MEGATRON_DRIVER_MODULE])
        self.assertNotIn("torch.distributed.run", command)
        # A schedule at pp 1 would name a split that does not happen.
        self.assertNotIn("--bench-pp-schedule", command)

    def test_the_trivial_spec_argv_is_exactly_its_two_parts(self) -> None:
        command = _command(_stock_arm())
        self.assertEqual(
            command,
            [sys.executable, "-m", STOCK_MEGATRON_DRIVER_MODULE]
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
        self.assertEqual(command[15], STOCK_MEGATRON_DRIVER_MODULE)

    def test_the_mesh_argv_is_exactly_its_two_parts(self) -> None:
        command = _command(
            _stock_arm(), parallelism=MESH, local_batch_size=32
        )
        head = command[: command.index(STOCK_MEGATRON_DRIVER_MODULE) + 1]
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
            STOCK_MEGATRON_PP_SCHEDULE,
        )

    def test_the_marker_count_matches_the_flags_the_argv_carries(self) -> None:
        """The removed ``pp`` 1 exception rests on one flag mapping.

        ``_megatron_stock_parallelism_markers`` writes ``n_microbatches`` at
        every pipeline degree, and that is right only while the flag list
        sends ``--micro-batch-size pp_microbatch_size`` and
        ``--global-batch-size local_batch_size * dp``. Under the other
        mapping -- one pack of every row at ``pp`` 1 -- Megatron would
        derive 1 and arm rule 12 would fail an honest run.

        So this reads the two flags out of the argv the harness really
        builds, applies Megatron's own division, and compares the answer to
        the marker. It fails whichever side moves.
        """
        profile = VALIDATION_PROFILES["megatron_stock"]
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
                workload = scenario.workload
                if batch is not None:
                    workload = replace(workload, local_batch_size=batch)
                marker = profile.parallelism_markers(spec, workload)[0]
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
    """Every refusal lands in the parent, before the subprocess starts.

    None of these needs the driver package: each raises before the flag
    builder is reached, which is why they all run today.
    """

    def test_a_torchtitan_passthrough_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "passthrough"):
            command_for_arm(
                scenario_by_name(SCENARIO_NAME).workload,
                _stock_arm(),
                Path("/tmp/arm-dir"),
                ("--training.steps", "5"),
                "default",
                "none",
            )

    def test_sac_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "no Megatron parity"):
            _command(_stock_arm(), ac_mode="sac")

    def test_the_uncompiled_mode_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot apply to this arm"):
            _command(_stock_arm(), compile_mode="none")

    def test_the_cuda_graph_mode_is_refused(self) -> None:
        """This driver captures no graph, so the mode names no treatment."""
        with self.assertRaisesRegex(ValueError, "captures no CUDA graph"):
            _command(_stock_arm(), compile_mode="cuda-graph")

    def test_an_unseeded_workload_is_refused(self) -> None:
        scenario = scenario_by_name(SCENARIO_NAME)
        with self.assertRaisesRegex(ValueError, "seeded workload"):
            command_for_arm(
                replace(scenario.workload, seed=None),
                _stock_arm(),
                Path("/tmp/arm-dir"),
                (),
                "default",
                "none",
            )

    def test_a_schedule_this_driver_does_not_implement_is_refused(
        self,
    ) -> None:
        """Parallelism rule 5 does not see this launcher.

        Rule 5 reads ``"megatron" in engines``, so a schedule Megatron-LM
        implements and this driver does not would otherwise reach the
        training subprocess. The refusal lands here instead.
        """
        spec = replace(MESH, pp_schedule="Interleaved1F1B")
        with self.assertRaisesRegex(ValueError, "implements '1F1B' alone"):
            _command(_stock_arm(), parallelism=spec, local_batch_size=32)


# --------------------------------------------------------------------------
# 3. The validation profile. The hazard is a profile that proves nothing.
# --------------------------------------------------------------------------


class StockValidationProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = VALIDATION_PROFILES["megatron_stock"]
        self.workload = replace(
            scenario_by_name(SCENARIO_NAME).workload, local_batch_size=32
        )

    def test_the_profile_is_registered_and_selected_by_the_arm(self) -> None:
        self.assertIn("megatron_stock", VALIDATION_PROFILES)
        self.assertIs(
            VALIDATION_PROFILES[_stock_arm().validation], self.profile
        )

    def test_the_mode_line_names_the_requested_mode(self) -> None:
        self.assertEqual(
            self.profile.mode_line("default"),
            "Megatron-LM stock training loop (mode=default,",
        )

    def test_the_profile_checks_neither_the_ac_line_nor_the_regions(
        self,
    ) -> None:
        self.assertFalse(self.profile.check_ac_line)
        self.assertFalse(self.profile.check_regions)

    def test_the_profile_cannot_prove_an_uncompiled_run(self) -> None:
        """``compiled_marker`` is None, so ``--compile-mode none`` is refused.

        megatron-core binds ``jit_fuser = torch.compile`` at import, so no
        log line proves this engine ran uncompiled. The absence of a marker
        must refuse the run rather than skip the check.
        """
        self.assertIsNone(self.profile.compiled_marker)
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "baseline.log"
            log.write_text("Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "cannot prove"):
                validate_arm(
                    _stock_arm(),
                    Path(temporary),
                    log,
                    self.workload,
                    compile_mode="none",
                    ac_mode="none",
                    model_size="1b",
                )

    # -- arm rule 12 ----------------------------------------------------

    def test_the_markers_interpolate_the_spec(self) -> None:
        markers = _megatron_stock_parallelism_markers(MESH, self.workload)
        self.assertEqual(
            markers[0],
            "Megatron-LM stock parallelism: dp=2 pp=4 schedule=1F1B "
            "microbatches=8 stages=4",
        )
        self.assertEqual(
            markers[1],
            "Megatron-LM stock data parallel: DistributedDataParallel over "
            "2 ranks (overlap_grad_reduce=False, grad_reduce_in_fp32=True)",
        )

    def test_the_markers_are_non_empty_above_world_size_one(self) -> None:
        """An empty tuple would make ``validate_arm`` refuse the run.

        Every mesh this scenario can run must produce at least the
        parallelism line, or the arm cannot be published at all.
        """
        for spec in (
            ParallelismSpec(pp=2, pp_schedule="1F1B"),
            ParallelismSpec(dp=2),
            ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B", pp_microbatch_size=4),
            ParallelismSpec(dp=8),
        ):
            with self.subTest(spec=spec):
                markers = self.profile.parallelism_markers(spec, self.workload)
                self.assertTrue(markers)
                self.assertIn(f"dp={spec.dp} pp={spec.pp}", markers[0])

    def test_the_data_parallel_line_appears_only_above_dp_one(self) -> None:
        pipeline_only = self.profile.parallelism_markers(
            ParallelismSpec(pp=4, pp_schedule="1F1B", pp_microbatch_size=4),
            self.workload,
        )
        self.assertEqual(len(pipeline_only), 1)
        with_dp = self.profile.parallelism_markers(MESH, self.workload)
        self.assertEqual(len(with_dp), 2)

    def test_the_microbatch_count_is_megatrons_own_arithmetic(self) -> None:
        """Stock Megatron has no ``pp`` 1 exception, and the tuned arm does.

        Megatron derives ``get_num_microbatches()`` as ``global_batch_size /
        (micro_batch_size * dp)``. The harness sends ``--global-batch-size
        local_batch_size * dp`` and ``--micro-batch-size
        pp_microbatch_size``, so the ``dp`` term cancels at every pipeline
        degree. A profile that wrote 1 at ``pp`` 1, as the tuned profile
        does for its own driver, would fail an honest run.
        """
        for spec, local_batch_size in (
            (MESH, 32),
            (ParallelismSpec(dp=2), 4),
            (TRIVIAL_SPEC, 4),
            (ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B"), 8),
        ):
            with self.subTest(spec=spec, batch=local_batch_size):
                megatron = (local_batch_size * spec.dp) // (
                    spec.pp_microbatch_size * spec.dp
                )
                self.assertEqual(
                    n_microbatches(spec, local_batch_size=local_batch_size),
                    megatron,
                )
                workload = replace(
                    self.workload, local_batch_size=local_batch_size
                )
                marker = self.profile.parallelism_markers(spec, workload)[0]
                self.assertIn(f"microbatches={megatron} ", marker)

    # -- the two inversions ---------------------------------------------

    def test_the_pipelined_pattern_ignores_a_pipeline_degree_of_one(
        self,
    ) -> None:
        line = self.profile.parallelism_markers(
            ParallelismSpec(dp=2), self.workload
        )[0]
        self.assertIn("pp=1", line)
        self.assertIsNone(self.profile.pipelined_pattern.search(line))

    def test_the_pipelined_pattern_sees_a_real_pipeline(self) -> None:
        for spec in (
            ParallelismSpec(pp=2, pp_schedule="1F1B"),
            MESH,
            ParallelismSpec(pp=4, pp_schedule="1F1B", pp_microbatch_size=4),
        ):
            with self.subTest(spec=spec):
                line = self.profile.parallelism_markers(spec, self.workload)[0]
                self.assertIsNotNone(
                    self.profile.pipelined_pattern.search(line)
                )

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
                for line in self.profile.parallelism_markers(
                    spec, self.workload
                ):
                    self.assertIsNone(
                        self.profile.data_parallel_pattern.search(line),
                        line,
                    )

    def test_the_data_parallel_pattern_sees_every_degree_above_one(
        self,
    ) -> None:
        """Two digits must match too. ``(?!1\\b)`` refuses 1, never 10."""
        for dp in (2, 4, 8, 10, 12, 16):
            with self.subTest(dp=dp):
                line = (
                    f"Megatron-LM stock parallelism: dp={dp} pp=1 "
                    "schedule=1F1B microbatches=4 stages=1"
                )
                self.assertIsNotNone(
                    self.profile.data_parallel_pattern.search(line)
                )

    def test_the_data_parallel_pattern_sees_the_wrapper_line(self) -> None:
        line = self.profile.parallelism_markers(MESH, self.workload)[1]
        self.assertIsNotNone(self.profile.data_parallel_pattern.search(line))

    def test_neither_pattern_matches_the_tuned_arms_lines(self) -> None:
        """The two drivers must not satisfy each other's rules.

        A tuned log reaching this profile, or the reverse, would publish one
        engine configuration under the other's label.
        """
        tuned = VALIDATION_PROFILES["megatron"]
        tuned_lines = tuned.parallelism_markers(MESH, self.workload)
        for line in tuned_lines:
            with self.subTest(line=line):
                self.assertIsNone(
                    self.profile.pipelined_pattern.search(line), line
                )
                self.assertIsNone(
                    self.profile.data_parallel_pattern.search(line), line
                )
        for line in self.profile.parallelism_markers(MESH, self.workload):
            with self.subTest(line=line):
                self.assertIsNone(tuned.pipelined_pattern.search(line), line)
                self.assertIsNone(
                    tuned.data_parallel_pattern.search(line), line
                )

    def test_the_two_profiles_share_no_marker_string(self) -> None:
        tuned = VALIDATION_PROFILES["megatron"]
        self.assertNotEqual(
            tuned.mode_line("default"), self.profile.mode_line("default")
        )
        self.assertEqual(
            set(tuned.parallelism_markers(MESH, self.workload))
            & set(self.profile.parallelism_markers(MESH, self.workload)),
            set(),
        )


# --------------------------------------------------------------------------
# 4. The contract between this profile and the driver the other agent owns.
# --------------------------------------------------------------------------


class StockMarkerContractTests(unittest.TestCase):
    """The marker strings, pinned in both directions.

    A one-character difference between the driver's line and this profile's
    marker fails a real eight-GPU run at arm rule 12, hours after it
    started. These fragments are the parts of the four marker strings that
    carry no interpolated value.
    """

    def setUp(self) -> None:
        self.profile = VALIDATION_PROFILES["megatron_stock"]
        self.workload = replace(
            scenario_by_name(SCENARIO_NAME).workload, local_batch_size=32
        )

    def test_every_fragment_belongs_to_a_marker_this_profile_builds(
        self,
    ) -> None:
        """The fragments cannot drift away from the profile."""
        built = [
            self.profile.completion_marker,
            self.profile.mode_line("default"),
            *self.profile.parallelism_markers(MESH, self.workload),
        ]
        for fragment in STOCK_LOG_FRAGMENTS:
            with self.subTest(fragment=fragment):
                self.assertTrue(
                    any(fragment in marker for marker in built),
                    f"{fragment!r} is in no marker this profile builds",
                )

    @_skip_without_stock_package(STOCK_DRIVER_MODULE)
    def test_the_driver_package_prints_every_fragment(self) -> None:
        """The driver's own source must carry each fragment verbatim.

        The package is read as text and not imported: its ``train.py``
        imports Megatron-LM, which needs a GPU host and a ``sys.path``
        edit.
        A text search finds a literal fragment, and the interpolated
        values are deliberately outside every fragment.
        """
        source = self._package_source()
        for fragment in (
            *STOCK_LOG_FRAGMENTS,
            *STOCK_MODE_LINE_FIELDS,
            *STOCK_PARAMETER_FRAGMENTS,
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

    @_skip_without_stock_package(STOCK_DRIVER_MODULE)
    def test_the_driver_package_prints_no_tuned_marker(self) -> None:
        """The stock driver must not print the tuned driver's lines."""
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
            path.read_text() for path in sorted(root.glob("*.py"))
        )


# --------------------------------------------------------------------------
# 5. The run axes, resolved the way a real command line resolves them.
# --------------------------------------------------------------------------


class StockRunResolutionTests(unittest.TestCase):
    """What ``run`` and ``run-all`` accept for this scenario.

    Cell 1 of the run matrix runs both arms at ``--compile-mode default``.
    Cell 2 runs the titan arm alone at ``--compile-mode none``. Both must
    resolve, and every other combination must be refused before a GPU is
    claimed.
    """

    def _resolve(self, names: tuple[str, ...], compile_mode: str, ac_mode: str):
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", dict(_METADATA)),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return _resolve_run(
                RunRequest(
                    gpu="0",
                    scenario_name=SCENARIO_NAME,
                    arm_names=names,
                    out_dir=Path(temporary) / "run",
                    compile_mode=compile_mode,
                    ac_mode=ac_mode,
                ),
                {"PATH": os.environ["PATH"]},
            )

    def test_sac_is_refused_for_the_whole_scenario(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support ac mode"):
            self._resolve((), "default", "sac")

    def test_the_cuda_graph_mode_is_refused_for_the_whole_scenario(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError, "does not support compile mode"
        ):
            self._resolve((), "cuda-graph", "none")

    def test_the_uncompiled_mode_is_refused_for_the_whole_roster(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "does not support compile mode"
        ):
            self._resolve((), "none", "none")

    def test_the_uncompiled_mode_is_refused_for_the_megatron_arm(self) -> None:
        """The subset exception is for TorchTitan-only selections alone."""
        for names in (("baseline",), ("baseline", "titan_stock")):
            with self.subTest(names=names):
                with self.assertRaisesRegex(
                    ValueError, "does not support compile mode"
                ):
                    self._resolve(names, "none", "none")

    def test_the_uncompiled_mode_is_accepted_for_the_titan_arm_alone(
        self,
    ) -> None:
        """Cell 2 of the run matrix. It needs no Megatron opponent."""
        resolved = self._resolve(("titan_stock",), "none", "none")
        self.assertEqual([arm.name for arm in resolved[2]], ["titan_stock"])
        self.assertEqual(resolved[7], "none")


if __name__ == "__main__":
    unittest.main()
