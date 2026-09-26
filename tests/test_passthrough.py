"""The passthrough tables: disjoint, complete, and owned by real options."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import typing
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cli.e2e import run_command
from benchmarks.e2e.engines.api import Arm, CompileMode
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig
from benchmarks.e2e.launch import megatron_stock_command, titan_command
from benchmarks.e2e.megatron_stock.flags import (
    ALWAYS_OMITTED_FLAGS,
    stock_megatron_flags,
)
from benchmarks.e2e.parallelism import TRIVIAL_SPEC, ParallelismSpec
from benchmarks.e2e.passthrough import (
    OWNED_FLAGS,
    PERF_FLAGS,
    PINNED_FLAGS,
    flag_name,
    matches,
    refusal,
    refuse_passthrough,
    row_for,
)
from benchmarks.e2e.registry import scenario_by_name
from benchmarks.e2e.runner import reach_refusal
from tests.engine_helpers import configured, run_spec

PARALLEL_SPEC = ParallelismSpec(dp=2, pp=2, ep=2, zero=1, pp_schedule="1F1B")
ENGINES = scenario_by_name("engines")
OVERRIDE_ARM = Arm(
    name="override_arm",
    description="an arm with an override import",
    config=TorchTitanConfig(
        compile=CompileMode.NONE, override_imports=("some.override",)
    ),
)


def _run_option_names() -> set[str]:
    return {
        name for param in run_command.params for name in getattr(param, "opts", ())
    }


def _owner_options() -> set[str]:
    return {
        name
        for owner in OWNED_FLAGS
        if owner.startswith("--")
        for name in owner.split("/")
    }


def _classes(side: str, name: str) -> list[str]:
    found = []
    if row_for(side, name, OWNED_FLAGS) is not None:
        found.append("owned")
    if row_for(side, name, PINNED_FLAGS) is not None:
        found.append("pinned")
    if any(matches(name, pattern) for pattern in getattr(PERF_FLAGS, side)):
        found.append("perf")
    return found


def _repeated_patterns(side: str) -> list[str]:
    patterns = [
        pattern
        for table in (OWNED_FLAGS, PINNED_FLAGS)
        for flags in table.values()
        for pattern in getattr(flags, side)
    ] + list(getattr(PERF_FLAGS, side))
    return sorted({p for p in patterns if patterns.count(p) > 1})


def _megatron_argvs() -> list[list[str]]:
    return [
        stock_megatron_flags(
            run_spec(profile=False, local_batch_size=8),
            MegatronStockConfig(),
            arm_dir="/x",
        ),
        stock_megatron_flags(
            run_spec(parallelism=PARALLEL_SPEC, local_batch_size=8),
            MegatronStockConfig(precision="lean"),
            arm_dir="/x",
        ),
    ]


def _titan_argvs() -> list[list[str]]:
    argvs = []
    for arm in (*ENGINES.arms, OVERRIDE_ARM):
        if engine_for(arm).name != "torchtitan":
            continue
        for spec, profile, ac_mode in (
            (TRIVIAL_SPEC, False, "none"),
            (PARALLEL_SPEC, True, "sac"),
        ):
            run = run_spec(
                parallelism=spec,
                profile=profile,
                ac_mode=ac_mode,
                local_batch_size=8,
            )
            argvs.append(titan_command(run, arm, Path("/x")))
    return argvs


MEGATRON_ARM = ENGINES.arm("megatron_stock")
TITAN_ARM = ENGINES.arm("titan_eager")


class TableTests(unittest.TestCase):
    def test_no_pattern_sits_in_two_rows(self) -> None:
        for side in ("torchtitan", "megatron"):
            with self.subTest(side=side):
                self.assertEqual(_repeated_patterns(side), [])

    def test_every_owner_is_a_run_option(self) -> None:
        self.assertLessEqual(_owner_options(), _run_option_names())


class MegatronTableTests(unittest.TestCase):
    def test_every_emitted_flag_has_exactly_one_class(self) -> None:
        for argv in _megatron_argvs():
            for token in argv:
                name = flag_name("megatron", token)
                if name is None:
                    continue
                with self.subTest(name=name):
                    classes = _classes("megatron", name)
                    self.assertEqual(len(classes), 1, classes)

    def test_the_perf_members_of_the_omitted_tuple_pass(self) -> None:
        perf = (
            "--moe-permute-fusion",
            "--cross-entropy-loss-fusion",
            "--use-flash-attn",
            "--overlap-grad-reduce",
            "--overlap-param-gather",
        )
        for flag in perf:
            self.assertIn(flag, ALWAYS_OMITTED_FLAGS)
        refuse_passthrough("megatron", MEGATRON_ARM.name, perf, zero=1)

    def test_the_other_omitted_members_are_refused(self) -> None:
        for flag in (
            "--grad-reduce-in-bf16",
            "--profile-ranks",
            "--data-path",
            "--mock-data",
            "--tensorboard-dir",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, ALWAYS_OMITTED_FLAGS)
                with self.assertRaisesRegex(ValueError, flag):
                    refuse_passthrough(
                        "megatron", MEGATRON_ARM.name, (flag,), zero=1
                    )

    def test_the_equals_form_and_a_prefix_are_refused(self) -> None:
        for tokens, reason in (
            (("--num-layers=4",), "owned by --model-size"),
            (("--bench-seq-len", "8"), "pinned by the harness driver"),
            (("--fp8-format", "hybrid"), "owned by --megatron-precision"),
            (("--recompute-granularity", "full"), "owned by --ac"),
        ):
            with self.subTest(tokens=tokens):
                with self.assertRaisesRegex(ValueError, reason):
                    refuse_passthrough(
                        "megatron", MEGATRON_ARM.name, tokens, zero=1
                    )

    def test_an_unlisted_flag_passes(self) -> None:
        self.assertIsNone(refusal("megatron", "--attention-backend"))

    def test_param_gather_overlap_needs_zero_1(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs --zero 1"):
            refuse_passthrough(
                "megatron",
                MEGATRON_ARM.name,
                ("--overlap-param-gather",),
                zero=0,
            )

    def test_the_passthrough_lands_last(self) -> None:
        command = megatron_stock_command(
            run_spec(ac_mode="none", profile=False),
            configured(
                MEGATRON_ARM,
                extra_flags=(
                    "--moe-token-dispatcher-type",
                    "flex",
                    "--moe-permute-fusion",
                ),
            ),
            Path("/x"),
        )
        self.assertEqual(
            command[-3:],
            ["--moe-token-dispatcher-type", "flex", "--moe-permute-fusion"],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "reading Megatron's parser needs torch"
    )
    def test_every_spelling_of_a_refused_setting_is_refused(self) -> None:
        from benchmarks.models.piper_qwen3.megatron_bootstrap import (
            add_megatron_to_path,
        )

        add_megatron_to_path()
        if not hasattr(typing, "override"):
            import typing_extensions

            typing.override = typing_extensions.override
        from megatron.training.arguments import add_megatron_arguments

        parser = argparse.ArgumentParser(allow_abbrev=False)
        add_megatron_arguments(parser)
        missing = []
        for action in parser._actions:
            refused = [
                name
                for name in action.option_strings
                if refusal("megatron", name) is not None
            ]
            if refused:
                missing.extend(
                    name for name in action.option_strings if name not in refused
                )
        self.assertEqual(missing, [])


class TitanTableTests(unittest.TestCase):
    def test_every_emitted_flag_is_owned_or_pinned(self) -> None:
        for argv in _titan_argvs():
            for token in argv[1:]:
                name = flag_name("torchtitan", token)
                if name is None:
                    continue
                with self.subTest(name=name):
                    self.assertIsNotNone(refusal("torchtitan", token))
                    self.assertEqual(len(_classes("torchtitan", name)), 1)

    @unittest.skipUnless(
        importlib.util.find_spec("torchtitan"), "reading the config needs torchtitan"
    )
    def test_every_config_field_has_exactly_one_class(self) -> None:
        from torchtitan.trainer import Trainer

        unclassified = []
        for section in dataclasses.fields(Trainer.Config):
            if section.name == "model_spec":
                continue
            prefix = "--" + section.name.replace("_", "-")
            if section.default_factory is dataclasses.MISSING:
                names = [prefix]
            else:
                value = section.default_factory()
                fields = (
                    dataclasses.fields(value)
                    if dataclasses.is_dataclass(value)
                    else ()
                )
                names = [
                    f"{prefix}.{field.name.replace('_', '-')}" for field in fields
                ] or [f"{prefix}.any-field"]
            for name in names:
                classes = _classes("torchtitan", name)
                if len(classes) != 1:
                    unclassified.append((name, classes))
        self.assertEqual(unclassified, [])

    def test_perf_flags_pass_in_every_spelling(self) -> None:
        refuse_passthrough(
            "torchtitan",
            TITAN_ARM.name,
            (
                "--compile.mode",
                "max-autotune",
                "--training.gc_freq=10",
                "--parallelism.no-enable-sequence-parallel",
                "--comm.init-timeout-seconds",
                "600",
            ),
            zero=0,
        )

    def test_owned_and_pinned_flags_are_refused(self) -> None:
        for token, reason in (
            ("--training.seq_len=8", "owned by --seq-len"),
            ("--compile.no-enable", "owned by the arm's compile value"),
            ("--parallelism.data-parallel-shard-degree", "owned by --dp/--pp/--ep"),
            ("activation-checkpoint:full", "owned by --ac"),
            ("--profiler.enable-profiling", "owned by --profile"),
            ("--debug.seed", "pinned by the shared data stream"),
            ("--training.dtype", "pinned by the precision recipe"),
        ):
            with self.subTest(token=token):
                self.assertEqual(refusal("torchtitan", token), reason)

    def test_an_unlisted_flag_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not classified"):
            refuse_passthrough(
                "torchtitan", TITAN_ARM.name, ("--training.new-field",), zero=0
            )


class ReachTests(unittest.TestCase):
    def test_a_list_must_reach_an_arm_of_its_engine(self) -> None:
        self.assertIsNone(reach_refusal(ENGINES.arms, ("--a",), ("--b",)))
        self.assertIn(
            "--megatron-arg reaches no arm",
            reach_refusal((TITAN_ARM,), (), ("--moe-permute-fusion",)),
        )
        self.assertIn(
            "--torchtitan-arg reaches no arm",
            reach_refusal((MEGATRON_ARM,), ("--compile.mode",), ()),
        )


if __name__ == "__main__":
    unittest.main()
