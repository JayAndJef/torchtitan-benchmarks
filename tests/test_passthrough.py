"""The passthrough tables: disjoint, complete, and owned by real options."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import typing
import unittest
from dataclasses import replace
from pathlib import Path

from benchmarks.cli.e2e import run_command
from benchmarks.e2e.launch import megatron_stock_command, titan_command
from benchmarks.e2e.megatron_stock.flags import (
    ALWAYS_OMITTED_FLAGS,
    OWNED_FLAGS,
    PERF_FLAGS,
    PINNED_FLAGS,
    refuse_megatron_passthrough,
    stock_megatron_flags,
)
from benchmarks.e2e.parallelism import TRIVIAL_SPEC, ParallelismSpec
from benchmarks.e2e.passthrough import (
    TITAN_OWNED_FLAGS,
    TITAN_PERF_FLAGS,
    TITAN_PINNED_FLAGS,
    flag_names,
    matches,
    refuse_titan_passthrough,
    table_entry,
    titan_flag_name,
    titan_refusal,
)
from benchmarks.e2e.registry import C4_REPLAY_WORKLOAD, scenario_by_name
from benchmarks.e2e.schema import Arm
from benchmarks.models.piper_qwen3.shape import PIPER_1B

PARALLEL_SPEC = ParallelismSpec(dp=2, pp=2, ep=2, zero=1, pp_schedule="1F1B")
WORKLOAD = replace(C4_REPLAY_WORKLOAD, local_batch_size=8, steps=40)
ENGINES = scenario_by_name("engines")
OVERRIDE_ARM = Arm(
    name="override_arm",
    description="an arm with an override import",
    compile="none",
    override_imports=("some.override",),
)


def _run_option_names() -> set[str]:
    return {
        name for param in run_command.params for name in getattr(param, "opts", ())
    }


def _owner_options(table: dict[str, tuple[str, ...]]) -> set[str]:
    return {
        name
        for owner in table
        if owner.startswith("--")
        for name in owner.split("/")
    }


def _megatron_argvs() -> list[list[str]]:
    return [
        stock_megatron_flags(
            PIPER_1B,
            WORKLOAD,
            TRIVIAL_SPEC,
            arm_dir="/x",
            model_size="1b",
            profile=False,
        ),
        stock_megatron_flags(
            PIPER_1B,
            WORKLOAD,
            PARALLEL_SPEC,
            arm_dir="/x",
            model_size="1b",
            megatron_p2p_sync="off",
            megatron_nan_guard="off",
            megatron_precision="lean",
            profile=True,
        ),
    ]


def _titan_argvs() -> list[list[str]]:
    argvs = []
    for arm in (*ENGINES.arms, OVERRIDE_ARM):
        if arm.engine != "torchtitan":
            continue
        for spec, profile, ac_mode in (
            (TRIVIAL_SPEC, False, "none"),
            (PARALLEL_SPEC, True, "sac"),
        ):
            argvs.append(
                titan_command(
                    WORKLOAD,
                    arm,
                    Path("/x"),
                    (),
                    ac_mode,
                    parallelism=spec,
                    profile=profile,
                )
            )
    return argvs


def _pattern_rows(*tables) -> list[tuple[str, str]]:
    rows = []
    for table in tables:
        if isinstance(table, dict):
            rows.extend(
                (key, pattern) for key, patterns in table.items() for pattern in patterns
            )
        else:
            rows.extend(("perf", pattern) for pattern in table)
    return rows


class MegatronTableTests(unittest.TestCase):
    def _classes(self, name: str) -> list[str]:
        found = []
        if table_entry(name, OWNED_FLAGS) is not None:
            found.append("owned")
        if table_entry(name, PINNED_FLAGS) is not None:
            found.append("pinned")
        if any(matches(name, pattern) for pattern in PERF_FLAGS):
            found.append("perf")
        return found

    def test_no_pattern_sits_in_two_rows(self) -> None:
        rows = _pattern_rows(OWNED_FLAGS, PINNED_FLAGS, PERF_FLAGS)
        patterns = [pattern for _, pattern in rows]
        repeated = sorted({p for p in patterns if patterns.count(p) > 1})
        self.assertEqual(repeated, [])

    def test_every_emitted_flag_has_exactly_one_class(self) -> None:
        for argv in _megatron_argvs():
            for name in flag_names(argv):
                with self.subTest(name=name):
                    self.assertEqual(len(self._classes(name)), 1, self._classes(name))

    def test_every_owner_is_a_run_option(self) -> None:
        self.assertLessEqual(_owner_options(OWNED_FLAGS), _run_option_names())

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
        refuse_megatron_passthrough("arm", perf, zero=1)

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
                    refuse_megatron_passthrough("arm", (flag,), zero=1)

    def test_the_equals_form_and_a_prefix_are_refused(self) -> None:
        for tokens, owner in (
            (("--num-layers=4",), "--model-size"),
            (("--bench-seq-len", "8"), "--megatron-p2p-sync"),
            (("--fp8-format", "hybrid"), "--megatron-precision"),
            (("--recompute-granularity", "full"), "--ac"),
        ):
            with self.subTest(tokens=tokens):
                with self.assertRaisesRegex(ValueError, f"owned by {owner}"):
                    refuse_megatron_passthrough("arm", tokens, zero=1)

    def test_param_gather_overlap_needs_zero_1(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs --zero 1"):
            refuse_megatron_passthrough(
                "arm", ("--overlap-param-gather",), zero=0
            )

    def test_the_passthrough_lands_last(self) -> None:
        command = megatron_stock_command(
            WORKLOAD,
            ENGINES.arm("megatron_stock"),
            Path("/x"),
            ("--moe-token-dispatcher-type", "flex", "--moe-permute-fusion"),
            "none",
            profile=False,
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
                if table_entry(name, OWNED_FLAGS) or table_entry(name, PINNED_FLAGS)
            ]
            if refused:
                missing.extend(
                    name for name in action.option_strings if name not in refused
                )
        self.assertEqual(missing, [])


class TitanTableTests(unittest.TestCase):
    def _classes(self, name: str) -> list[str]:
        found = []
        if table_entry(name, TITAN_OWNED_FLAGS) is not None:
            found.append("owned")
        if table_entry(name, TITAN_PINNED_FLAGS) is not None:
            found.append("pinned")
        if any(matches(name, pattern) for pattern in TITAN_PERF_FLAGS):
            found.append("perf")
        return found

    def test_no_pattern_sits_in_two_rows(self) -> None:
        rows = _pattern_rows(TITAN_OWNED_FLAGS, TITAN_PINNED_FLAGS, TITAN_PERF_FLAGS)
        patterns = [pattern for _, pattern in rows]
        repeated = sorted({p for p in patterns if patterns.count(p) > 1})
        self.assertEqual(repeated, [])

    def test_every_emitted_flag_is_owned_or_pinned(self) -> None:
        for argv in _titan_argvs():
            for token in argv[1:]:
                if not token.startswith("--") and ":" not in token:
                    continue
                with self.subTest(token=token):
                    self.assertIsNotNone(titan_refusal(token))
                    self.assertEqual(
                        len(self._classes(titan_flag_name(token))), 1
                    )

    def test_every_owner_is_a_run_option(self) -> None:
        self.assertLessEqual(_owner_options(TITAN_OWNED_FLAGS), _run_option_names())

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
                fields = dataclasses.fields(value) if dataclasses.is_dataclass(value) else ()
                names = [
                    f"{prefix}.{field.name.replace('_', '-')}" for field in fields
                ] or [f"{prefix}.any-field"]
            for name in names:
                if len(self._classes(name)) != 1:
                    unclassified.append((name, self._classes(name)))
        self.assertEqual(unclassified, [])

    def test_perf_flags_pass_in_every_spelling(self) -> None:
        refuse_titan_passthrough(
            "arm",
            (
                "--compile.mode",
                "max-autotune",
                "--training.gc_freq=10",
                "--parallelism.no-enable-sequence-parallel",
                "--comm.init-timeout-seconds",
                "600",
            ),
        )

    def test_owned_and_pinned_flags_are_refused(self) -> None:
        for token, reason in (
            ("--training.seq_len=8", "owned by --seq-len"),
            ("--compile.no-enable", "owned by the arm's compile value"),
            ("--parallelism.data-parallel-shard-degree", "owned by --dp/--pp/--ep"),
            ("activation-checkpoint:full", "owned by --ac"),
            ("--profiler.enable-profiling", "owned by --profile"),
            ("--debug.seed", "pinned by the shared data stream"),
            ("--training.dtype", "pinned by the bf16 recipe"),
        ):
            with self.subTest(token=token):
                self.assertEqual(titan_refusal(token), reason)

    def test_an_unknown_flag_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not classified"):
            refuse_titan_passthrough("arm", ("--training.new-field",))


if __name__ == "__main__":
    unittest.main()
