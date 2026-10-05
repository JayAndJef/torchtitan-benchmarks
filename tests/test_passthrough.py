"""The passthrough tables: disjoint, complete, and owned by real options."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cli.e2e import run_command
from benchmarks.e2e.engines.api import Arm, CompileMode
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig
from benchmarks.e2e.engines.torchtitan import flags as titan_flags
from benchmarks.e2e.engines.megatron_stock import flags as megatron_flags
from benchmarks.e2e.engines.megatron_stock.flags import (
    ALWAYS_OMITTED_FLAGS,
    passthrough_refusals,
    stock_megatron_flags,
)
from benchmarks.e2e.parallelism import TRIVIAL_SPEC, ParallelismSpec
from benchmarks.e2e.passthrough import matches, row_for
from benchmarks.e2e.registry import scenario_by_name
from tests.engine_helpers import configured, run_spec

PARALLEL_SPEC = ParallelismSpec(dp=2, pp=2, ep=2, zero=1, pp_schedule="1F1B")
ENGINES = scenario_by_name("engines")
OVERRIDES = scenario_by_name("overrides")
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


TABLES = {
    "torchtitan": (
        titan_flags.OWNED_FLAGS,
        titan_flags.PINNED_FLAGS,
        titan_flags.PERF_FLAGS,
    ),
    "megatron": (
        megatron_flags.OWNED_FLAGS,
        megatron_flags.PINNED_FLAGS,
        megatron_flags.PERF_FLAGS,
    ),
}
"""The owned, pinned and perf tables of each engine."""


def _owner_options() -> set[str]:
    return {
        name
        for owned, _, _ in TABLES.values()
        for owner in owned
        if owner.startswith("--")
        for name in owner.split()[0].split("/")
    }


def _classes(side: str, name: str) -> list[str]:
    owned, pinned, perf = TABLES[side]
    found = []
    if row_for(name, owned) is not None:
        found.append("owned")
    if row_for(name, pinned) is not None:
        found.append("pinned")
    if any(matches(name, pattern) for pattern in perf):
        found.append("perf")
    return found


def _repeated_patterns(side: str) -> list[str]:
    owned, pinned, perf = TABLES[side]
    patterns = [
        pattern
        for table in (owned, pinned)
        for row in table.values()
        for pattern in row
    ] + list(perf)
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
    override_arms = (
        OVERRIDES.arm(name)
        for name in (
            "titan_compiled_fa3",
            "titan_compiled_te_gemm",
            "titan_compiled_fa3_te_gemm",
        )
    )
    for arm in (*ENGINES.arms, *override_arms, OVERRIDE_ARM):
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
            argvs.append(list(titan_flags.trainer_args(run, arm.config, Path("/x"))))
    return argvs


def _megatron_parser() -> argparse.ArgumentParser:
    """Megatron's own parser, imported through the driver's bootstrap."""
    from benchmarks.e2e.engines.megatron_stock.driver import bootstrap
    from benchmarks.models.piper_qwen3.megatron_bootstrap import (
        add_megatron_to_path,
    )

    bootstrap.install_typing_override()
    add_megatron_to_path()
    from megatron.training.arguments import add_megatron_arguments

    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_megatron_arguments(parser)
    return parser


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
                name = megatron_flags.flag_name(token)
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
        self.assertEqual(passthrough_refusals(MEGATRON_ARM.name, perf, zero=1), [])

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
                self.assertIn(
                    flag,
                    " ".join(passthrough_refusals(MEGATRON_ARM.name, (flag,), zero=1)),
                )

    def test_the_equals_form_and_a_prefix_are_refused(self) -> None:
        for tokens, reason in (
            (("--num-layers=4",), "owned by --model-size"),
            (("--bench-seq-len", "8"), "pinned by the harness driver"),
            (("--fp8-format", "hybrid"), "owned by --set megatron_stock.precision"),
            (("--recompute-granularity", "full"), "owned by --ac"),
        ):
            with self.subTest(tokens=tokens):
                self.assertIn(
                    reason,
                    " ".join(passthrough_refusals(MEGATRON_ARM.name, tokens, zero=1)),
                )

    def test_an_unlisted_flag_passes(self) -> None:
        self.assertIsNone(megatron_flags.refusal("--attention-backend"))

    def test_param_gather_overlap_needs_zero_1(self) -> None:
        self.assertIn(
            "needs --zero 1",
            " ".join(
                passthrough_refusals(
                    MEGATRON_ARM.name, ("--overlap-param-gather",), zero=0
                )
            ),
        )

    def test_the_passthrough_lands_last(self) -> None:
        command = engine_for(MEGATRON_ARM).launch(
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
        ).target
        self.assertEqual(
            command[-3:],
            ("--moe-token-dispatcher-type", "flex", "--moe-permute-fusion"),
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "reading Megatron's parser needs torch"
    )
    def test_every_spelling_of_a_refused_setting_is_refused(self) -> None:
        parser = _megatron_parser()
        missing = []
        for action in parser._actions:
            refused = [
                name
                for name in action.option_strings
                if megatron_flags.refusal(name) is not None
            ]
            if refused:
                missing.extend(
                    name for name in action.option_strings if name not in refused
                )
        self.assertEqual(missing, [])

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "reading Megatron's parser needs torch"
    )
    def test_every_emitted_megatron_flag_is_in_the_pinned_parser(self) -> None:
        parser = _megatron_parser()
        known = {name for action in parser._actions for name in action.option_strings}
        unknown = sorted(
            {
                name
                for argv in _megatron_argvs()
                for token in argv
                if (name := megatron_flags.flag_name(token)) is not None
                and not name.startswith("--bench-")
                and name not in known
            }
        )
        self.assertEqual(unknown, [])


class TitanTableTests(unittest.TestCase):
    def test_every_emitted_flag_is_owned_or_pinned(self) -> None:
        for argv in _titan_argvs():
            for token in argv:
                name = titan_flags.flag_name(token)
                if name is None:
                    continue
                with self.subTest(name=name):
                    self.assertIsNotNone(titan_flags.refusal(token))
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
        self.assertEqual(
            titan_flags.passthrough_refusals(
                TITAN_ARM.name,
                (
                    "--compile.mode",
                    "max-autotune",
                    "--training.gc_freq=10",
                    "--parallelism.no-enable-sequence-parallel",
                    "--comm.init-timeout-seconds",
                    "600",
                ),
            ),
            [],
        )

    def test_owned_and_pinned_flags_are_refused(self) -> None:
        for token, reason in (
            ("--training.seq_len=8", "owned by --seq-len"),
            ("--compile.no-enable", "owned by the arm's compile value"),
            ("--parallelism.data-parallel-shard-degree", "owned by --dp/--pp/--ep"),
            ("activation-checkpoint:full", "owned by --ac"),
            ("--profiler.enable-profiling", "owned by --profile"),
            ("--debug.seed", "pinned by the shared data stream"),
            ("--dataloader.replay-steps", "pinned by the shared data stream"),
            ("--dataloader.offset-rows=2", "owned by the arm's packed_offsets value"),
            ("--training.dtype", "pinned by the precision recipe"),
        ):
            with self.subTest(token=token):
                self.assertEqual(titan_flags.refusal(token), reason)

    def test_an_unlisted_flag_is_refused(self) -> None:
        (refusal,) = titan_flags.passthrough_refusals(
            TITAN_ARM.name, ("--training.new-field",)
        )
        self.assertIn("not classified", refusal)
        self.assertIn("cannot pass through titan_eager.extra_flags", refusal)

    def test_an_unlisted_loader_flag_is_refused(self) -> None:
        (refusal,) = titan_flags.passthrough_refusals(
            TITAN_ARM.name, ("--dataloader.foo=1",)
        )
        self.assertIn("--dataloader.foo=1 (not classified", refusal)

    def test_the_engine_check_refuses_an_unlisted_flag(self) -> None:
        arm = configured(TITAN_ARM, extra_flags=("--training.new-field",))
        (refusal,) = engine_for(arm).check(run_spec(ac_mode="none"), arm)
        self.assertIn("--training.new-field (not classified", refusal)


if __name__ == "__main__":
    unittest.main()
