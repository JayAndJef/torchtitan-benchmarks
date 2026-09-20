"""The stock Megatron-LM driver package, checked without a GPU.

Every test here runs on the CPU. The module imports torch, because
``benchmarks/e2e/megatron_stock/data.py`` and ``profiling.py`` do, but it
allocates nothing on a device and it starts no training.

The tests are grouped by the file they guard, and each group names the one
way that file can produce a wrong number:

* ``flags`` -- a flag that does not exist in this Megatron rev, or a
  geometry value that does not come from ``PiperShape``;
* ``bootstrap`` -- a shim that changes behaviour on Python 3.12, or that
  hides a missing ``typing_extensions``;
* ``data`` -- a ``cu_seqlens`` row Megatron's merge misreads, or a stream
  that wraps instead of raising;
* ``profiling`` -- a run that writes one window and still passes, or a shim
  that silently did not install;
* ``train`` -- a marker string that differs by one character from the
  validation profile's;
* ``model_builder`` -- a printed parameter total that is not backed by a
  real count.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import pathlib
import sys
import sysconfig
import tempfile
import time
import types
import unittest.mock
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from benchmarks.e2e.megatron_stock import (  # noqa: E402
    bootstrap,
    data,
    profiling,
    train,
)
from benchmarks.e2e.megatron_stock.flags import (  # noqa: E402
    ALWAYS_OMITTED_FLAGS,
    BENCH_ARM_DIR,
    BENCH_BATCH_P2P_SYNC,
    BENCH_FLAGS,
    BENCH_FLAGS_OMITTED_BY_DEFAULT,
    BENCH_PP_SCHEDULE,
    DATA_PARALLEL_OPTIMIZERS,
    DATA_PARALLEL_OVERLAP,
    DATA_PARALLEL_WRAPPERS,
    MEGATRON_CHECKPOINT_FORMAT,
    MEGATRON_FSDP_GRAD_OVERLAP_STRATEGIES,
    MEGATRON_FSDP_VERSION,
    MEGATRON_SHARDING_STRATEGY,
    LEAN_PRECISION_FLAGS,
    NO_CHECK_FOR_NAN_FLAG,
    SHARDING_STRATEGIES,
    ZERO1_FLAGS,
    ZERO3_FLAGS,
    microbatch_geometry,
    omitted_flags,
    refuse_unknown_nan_guard,
    refuse_unknown_p2p_sync,
    stock_megatron_flags,
)
from benchmarks.e2e.parallelism import (  # noqa: E402
    TRIVIAL_SPEC,
    ParallelismSpec,
)
from benchmarks.e2e.registry import PIPER_1B_MEGATRON_WORKLOAD  # noqa: E402
from benchmarks.e2e.results import (  # noqa: E402
    GRAD_NORM_METRIC,
    LOSS_METRIC,
    STEP_METRICS,
)
from benchmarks.models.piper_qwen3.shape import (  # noqa: E402
    PIPER_SHAPES,
    shape_by_name,
)

# The mesh cell 1 and cell 3 of the run matrix use.
PP4_SPEC = ParallelismSpec(
    dp=2, pp=4, pp_schedule="1F1B", pp_microbatch_size=4
)
# The same mesh with the dense parameters sharded on both engines, and then
# with the expert split that value makes legal. Named rather than numbered:
# two run matrices use overlapping cell numbers for different cells.
SHARDED_PP4_SPEC = dataclasses.replace(PP4_SPEC, dense_sharding="zero3")
EXPERT_PP4_SPEC = dataclasses.replace(SHARDED_PP4_SPEC, ep=2)
BATCH_32 = dataclasses.replace(PIPER_1B_MEGATRON_WORKLOAD, local_batch_size=32)
# Read as text rather than imported, so the check needs no megatron import
# and no GPU. DATA_PARALLEL_OPTIMIZERS is a claim about this file.
MEGATRON_OPTIMIZER_SOURCE = (
    Path(__file__).resolve().parent.parent
    / "third_party"
    / "Megatron-LM"
    / "megatron"
    / "core"
    / "optimizer"
    / "__init__.py"
)


def flags_for(shape_name, spec, workload=BATCH_32, **keywords):
    return stock_megatron_flags(
        shape_by_name(shape_name),
        workload,
        spec,
        arm_dir="/tmp/arm",
        model_size=shape_name,
        **keywords,
    )


def value_after(emitted, flag):
    """The token after ``flag``, or None when the flag is absent."""
    if flag not in emitted:
        return None
    return emitted[emitted.index(flag) + 1]


# --------------------------------------------------------------------------
# flags.py
# --------------------------------------------------------------------------

# Section 7 of PIPER_STOCK_MEGATRON_PLAN.md, transcribed. See the note at
# ZERO3_FLAGS: the expert degree and the five sharding flags supersede
# what that section declares, so read this tuple as the replicated roster.
#
# **What is checked, and what is not.** Megatron's argparse knows every
# name here, and knows the five sharding flags too. Nothing in this suite
# calls parse_and_validate_args, which is where four asserts live that stop
# a sharded run at parsing: CUDA_DEVICE_MAX_CONNECTIONS, the checkpoint
# format, the optimizer, and the two moe_single_grouped_* fields. Only
# train.py calls it, at run time. The connection-limit assert is the one
# this repo now guards parent-side, in
# benchmarks/execution/environment.py. The other three are unexercised
# until an eight-rank cell runs.
SECTION_7_FLAGS = (
    "--num-layers",
    "--hidden-size",
    "--num-attention-heads",
    "--group-query-attention",
    "--num-query-groups",
    "--kv-channels",
    "--ffn-hidden-size",
    "--moe-ffn-hidden-size",
    "--num-experts",
    "--moe-router-topk",
    "--moe-layer-freq",
    "--seq-length",
    "--max-position-embeddings",
    "--position-embedding-type",
    "--use-rotary-position-embeddings",
    "--rotary-percent",
    "--rotary-base",
    "--normalization",
    "--norm-epsilon",
    "--swiglu",
    "--disable-bias-linear",
    "--untie-embeddings-and-output-weights",
    "--qk-layernorm",
    "--attention-dropout",
    "--hidden-dropout",
    "--init-method-std",
    "--bf16",
    "--transformer-impl",
    "--use-mcore-models",
    "--no-gradient-accumulation-fusion",
    "--moe-token-dispatcher-type",
    "--moe-grouped-gemm",
    "--moe-router-load-balancing-type",
    "--moe-aux-loss-coeff",
    "--moe-router-dtype",
    "--micro-batch-size",
    "--global-batch-size",
    "--train-iters",
    "--lr",
    "--lr-decay-style",
    "--lr-decay-iters",
    "--lr-warmup-iters",
    "--min-lr",
    "--adam-beta1",
    "--adam-beta2",
    "--adam-eps",
    "--weight-decay",
    "--clip-grad",
    "--tensor-model-parallel-size",
    "--context-parallel-size",
    "--expert-model-parallel-size",
    "--pipeline-model-parallel-size",
    "--tokenizer-type",
    "--vocab-size",
    "--padded-vocab-size",
    "--dataloader-type",
    "--dataloader-inter-document-masking",
    "--no-create-attention-mask-in-dataloader",
    "--num-workers",
    "--eval-iters",
    "--eval-interval",
    "--seed",
    "--rerun-mode",
    "--log-interval",
    "--log-throughput",
    "--profile",
    "--use-pytorch-profiler",
    "--profile-step-start",
    "--profile-step-end",
)


class FlagListTest(unittest.TestCase):
    """What the command line says, for both shapes the suite runs."""

    def test_every_declared_flag_is_emitted(self) -> None:
        for size in ("1b", "9b"):
            for spec in (TRIVIAL_SPEC, PP4_SPEC):
                emitted = set(flags_for(size, spec))
                for flag in SECTION_7_FLAGS:
                    with self.subTest(size=size, pp=spec.pp, flag=flag):
                        self.assertIn(flag, emitted)

    def test_the_harness_group_is_emitted(self) -> None:
        """Every ``--bench-`` flag except the two a default argv omits.

        The schedule is absent at pipeline degree 1, where the driver
        refuses it: a schedule there names a split that does not happen.
        The p2p value is absent at ``on``, which restates Megatron's own
        default. ``BENCH_FLAGS_OMITTED_BY_DEFAULT`` names both.
        """
        self.assertEqual(
            BENCH_FLAGS_OMITTED_BY_DEFAULT,
            (BENCH_PP_SCHEDULE, BENCH_BATCH_P2P_SYNC),
        )
        trivial = set(flags_for("1b", TRIVIAL_SPEC))
        pipelined = set(flags_for("1b", PP4_SPEC))
        for flag in BENCH_FLAGS:
            with self.subTest(flag=flag):
                if flag == BENCH_BATCH_P2P_SYNC:
                    self.assertNotIn(flag, pipelined)
                else:
                    self.assertIn(flag, pipelined)
                if flag not in BENCH_FLAGS_OMITTED_BY_DEFAULT:
                    self.assertIn(flag, trivial)
        for flag in BENCH_FLAGS_OMITTED_BY_DEFAULT:
            self.assertNotIn(flag, trivial)
        self.assertEqual(
            value_after(flags_for("1b", PP4_SPEC), BENCH_PP_SCHEDULE), "1F1B"
        )

    def test_the_default_p2p_value_changes_no_argv(self) -> None:
        """``on`` is Megatron's own default, and every published cell's.

        Passing it by name must build the argv a caller that passes nothing
        builds, token for token, with no p2p flag in it.
        """
        for spec in (TRIVIAL_SPEC, PP4_SPEC, SHARDED_PP4_SPEC):
            with self.subTest(pp=spec.pp, dense_sharding=spec.dense_sharding):
                emitted = flags_for("1b", spec)
                self.assertEqual(
                    emitted, flags_for("1b", spec, megatron_p2p_sync="on")
                )
                self.assertNotIn(BENCH_BATCH_P2P_SYNC, emitted)

    def test_p2p_sync_off_adds_exactly_one_flag(self) -> None:
        """The off argv is the default argv plus one harness pair.

        Megatron has no flag for the field, so nothing else in the argv
        may move: the driver sets the field on ``args`` from this one pair.
        """
        for spec in (PP4_SPEC, SHARDED_PP4_SPEC, EXPERT_PP4_SPEC):
            with self.subTest(dense_sharding=spec.dense_sharding, ep=spec.ep):
                default = flags_for("1b", spec)
                off = flags_for("1b", spec, megatron_p2p_sync="off")
                self.assertEqual(
                    off, default + [BENCH_BATCH_P2P_SYNC, "off"]
                )
                self.assertEqual(off.count(BENCH_BATCH_P2P_SYNC), 1)

    def test_p2p_sync_off_at_pipeline_degree_one_is_refused(self) -> None:
        """The field is inert without a pipeline message."""
        for spec in (TRIVIAL_SPEC, ParallelismSpec(dp=2)):
            with self.subTest(dp=spec.dp):
                with self.assertRaisesRegex(
                    ValueError, "no pipeline message"
                ):
                    flags_for("1b", spec, megatron_p2p_sync="off")

    def test_an_unknown_p2p_value_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not one of"):
            flags_for("1b", PP4_SPEC, megatron_p2p_sync="maybe")
        with self.assertRaisesRegex(ValueError, "not one of"):
            refuse_unknown_p2p_sync("false")

    def test_the_default_nan_guard_changes_no_argv(self) -> None:
        """``on`` is Megatron's own default, and every published cell's."""
        for spec in (TRIVIAL_SPEC, PP4_SPEC, SHARDED_PP4_SPEC):
            with self.subTest(pp=spec.pp, dense_sharding=spec.dense_sharding):
                emitted = flags_for("1b", spec)
                self.assertEqual(
                    emitted, flags_for("1b", spec, megatron_nan_guard="on")
                )
                self.assertNotIn(NO_CHECK_FOR_NAN_FLAG, emitted)

    def test_nan_guard_off_adds_exactly_the_one_megatron_flag(self) -> None:
        """The off argv is the default argv plus Megatron's own token.

        Legal at every mesh, because the guard runs at every mesh. The
        token sits ahead of the harness group, whose tail a sibling test
        pins, and it is a Megatron flag rather than a ``--bench-`` one: a
        stock user can type the same argv.
        """
        for spec in (TRIVIAL_SPEC, ParallelismSpec(dp=2), PP4_SPEC, EXPERT_PP4_SPEC):
            with self.subTest(dp=spec.dp, pp=spec.pp, ep=spec.ep):
                default = flags_for("1b", spec)
                off = flags_for("1b", spec, megatron_nan_guard="off")
                self.assertEqual(off.count(NO_CHECK_FOR_NAN_FLAG), 1)
                self.assertEqual(
                    [token for token in off if token != NO_CHECK_FOR_NAN_FLAG],
                    default,
                )
                self.assertLess(
                    off.index(NO_CHECK_FOR_NAN_FLAG), off.index(BENCH_ARM_DIR)
                )
                self.assertFalse(NO_CHECK_FOR_NAN_FLAG.startswith("--bench-"))

    def test_an_unknown_nan_guard_value_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not one of"):
            flags_for("1b", PP4_SPEC, megatron_nan_guard="maybe")
        with self.assertRaisesRegex(ValueError, "not one of"):
            refuse_unknown_nan_guard("false")

    def test_the_stock_argv_for_30b_a3b_carries_the_written_geometry(
        self,
    ) -> None:
        """The two written fields reach Megatron, not their derivations.

        --num-attention-heads must read 32 where dim // kv-channels is 16,
        and both expert widths must read 768 where 3.5x dim is 7168. The
        other seven geometry flags are checked beside them, so a shape that
        drifted in any field fails here rather than in a GPU cell.
        """
        for spec in (TRIVIAL_SPEC, PP4_SPEC):
            with self.subTest(pp=spec.pp):
                emitted = flags_for("30b-a3b", spec)
                expected = {
                    "--num-layers": "48",
                    "--hidden-size": "2048",
                    "--num-attention-heads": "32",
                    "--num-query-groups": "4",
                    "--kv-channels": "128",
                    "--ffn-hidden-size": "768",
                    "--moe-ffn-hidden-size": "768",
                    "--num-experts": "128",
                    "--moe-router-topk": "8",
                    "--vocab-size": "151936",
                    "--padded-vocab-size": "151936",
                }
                for flag, value in expected.items():
                    self.assertEqual(value_after(emitted, flag), value, flag)
                self.assertIn("--group-query-attention", emitted)
                self.assertEqual(
                    value_after(emitted, "--bench-model-size"), "30b-a3b"
                )

    def test_no_declined_flag_is_emitted(self) -> None:
        """Every flag the value declines, asserted by name.

        Each is declined for a reason section 7 of the plan states. An
        emitted one would change what the arm measures without changing
        anything the manifest records. The roster is a function of the
        dense-sharding value, because all five sharding flags move from
        declined to required under ``zero3``.
        """
        for size in ("1b", "9b"):
            for spec in (
                TRIVIAL_SPEC,
                PP4_SPEC,
                SHARDED_PP4_SPEC,
                EXPERT_PP4_SPEC,
            ):
                emitted = set(flags_for(size, spec))
                for flag in omitted_flags(spec.dense_sharding):
                    with self.subTest(
                        size=size,
                        pp=spec.pp,
                        dense_sharding=spec.dense_sharding,
                        flag=flag,
                    ):
                        self.assertNotIn(flag, emitted)

    def test_replicate_declines_every_sharding_flag(self) -> None:
        """The five flags, asserted by name against the roster.

        ``replicate`` is stock Megatron's own default, and it is what every
        published cell of this scenario ran.
        """
        self.assertEqual(
            omitted_flags("replicate"),
            ALWAYS_OMITTED_FLAGS + ZERO3_FLAGS,
        )
        for flag in ZERO3_FLAGS:
            with self.subTest(flag=flag):
                self.assertNotIn(flag, set(flags_for("1b", PP4_SPEC)))

    def test_shard_emits_every_sharding_flag(self) -> None:
        """The other half of the same roster, and the values it carries.

        ``--use-distributed-optimizer`` is one of the five. Megatron-FSDP
        v1 turns it on itself, so an argv that omitted it would deny a fact
        the run has.
        """
        self.assertEqual(omitted_flags("zero3"), ALWAYS_OMITTED_FLAGS)
        for spec in (SHARDED_PP4_SPEC, EXPERT_PP4_SPEC):
            emitted = flags_for("1b", spec)
            for flag in ZERO3_FLAGS:
                with self.subTest(ep=spec.ep, flag=flag):
                    self.assertIn(flag, emitted)
            self.assertEqual(
                value_after(emitted, "--megatron-fsdp-version"),
                MEGATRON_FSDP_VERSION,
            )
            self.assertEqual(
                value_after(emitted, "--data-parallel-sharding-strategy"),
                MEGATRON_SHARDING_STRATEGY,
            )
            self.assertEqual(
                value_after(emitted, "--ckpt-format"),
                MEGATRON_CHECKPOINT_FORMAT,
            )

    def test_megatron_fsdp_runs_at_version_one(self) -> None:
        """Version 2 refuses every shape in this suite.

        ``FullyShardedDataParallelV2._validate_config`` raises on a
        pipeline degree, an expert degree and on any config whose
        ``num_moe_experts`` is set. Every registered shape is a mixture of
        experts, so v2 refuses this suite even at pp 1 and ep 1.
        """
        self.assertEqual(MEGATRON_FSDP_VERSION, "1")
        self.assertEqual(
            DATA_PARALLEL_WRAPPERS["zero3"], "FullyShardedDataParallelV1"
        )
        # The other two values never reach that wrapper: training.py picks
        # it on --use-megatron-fsdp alone, and neither value sends it.
        self.assertEqual(
            DATA_PARALLEL_WRAPPERS["zero1"], "DistributedDataParallel"
        )

    def test_the_expert_degree_reaches_the_argv(self) -> None:
        """A degree the harness holds and the argv drops trains a
        different model.
        """
        for spec, expected in (
            (TRIVIAL_SPEC, "1"),
            (PP4_SPEC, "1"),
            (SHARDED_PP4_SPEC, "1"),
            (EXPERT_PP4_SPEC, "2"),
        ):
            with self.subTest(ep=spec.ep):
                self.assertEqual(
                    value_after(
                        flags_for("1b", spec),
                        "--expert-model-parallel-size",
                    ),
                    expected,
                )

    def test_the_trivial_spec_argv_does_not_move(self) -> None:
        """The default value must change no published command line.

        Every cell under ``out/`` ran at ``replicate``. The expert degree
        was already sent as a literal ``1``, and the sharding flags are
        empty there, so the trivial argv is what it was.
        """
        emitted = flags_for("1b", TRIVIAL_SPEC)
        self.assertEqual(
            value_after(emitted, "--expert-model-parallel-size"), "1"
        )
        for flag in ZERO3_FLAGS:
            with self.subTest(flag=flag):
                self.assertNotIn(flag, emitted)

    def test_an_expert_split_without_sharding_is_refused(self) -> None:
        """TorchTitan cannot split experts and replicate the dense
        parameters.

        A replicated expert row would compare two memory strategies, which
        is two changes rather than one.
        """
        with self.assertRaisesRegex(ValueError, "--dense-sharding zero3"):
            flags_for("1b", dataclasses.replace(PP4_SPEC, ep=2))

    def test_an_unknown_dense_sharding_value_is_refused(self) -> None:
        """A silent fall through would send the replicated argv under the
        other label.

        ``shard`` is the retired spelling of ``zero3``. It is not a declared
        value any more, so this module must refuse it rather than build the
        sharded argv for it.
        """
        with self.assertRaisesRegex(ValueError, "dense sharding"):
            flags_for(
                "1b", dataclasses.replace(PP4_SPEC, dense_sharding="shard")
            )
        with self.assertRaisesRegex(ValueError, "dense sharding"):
            omitted_flags("shard")

    def test_zero1_emits_the_distributed_optimizer_alone(self) -> None:
        """One flag, and the four Megatron-FSDP flags stay out.

        ``--use-distributed-optimizer`` alone gives Megatron a
        ``DistributedOptimizer`` beside a plain ``DistributedDataParallel``,
        which shards the optimizer states and nothing else. It builds no
        device mesh, which is why spec rule 17 lets this value hold a
        pipeline where ``zero3`` cannot.
        """
        emitted = flags_for(
            "1b", dataclasses.replace(PP4_SPEC, dense_sharding="zero1")
        )
        self.assertEqual(ZERO1_FLAGS, ("--use-distributed-optimizer",))
        self.assertIn("--use-distributed-optimizer", emitted)
        for flag in ZERO3_FLAGS:
            if flag in ZERO1_FLAGS:
                continue
            with self.subTest(flag=flag):
                self.assertNotIn(flag, emitted)

    def test_zero1_declines_the_four_flags_it_does_not_send(self) -> None:
        """``omitted_flags`` subtracts what the value sends.

        ``--use-distributed-optimizer`` is the flag that moves between the
        two sharded values, so a hand-written roster could declare it
        declined while this value's own argv carries it.
        """
        declined = omitted_flags("zero1")
        self.assertNotIn("--use-distributed-optimizer", declined)
        for flag in ZERO3_FLAGS:
            if flag in ZERO1_FLAGS:
                continue
            with self.subTest(flag=flag):
                self.assertIn(flag, declined)

    def test_the_optimizer_table_separates_replicate_from_zero1(self) -> None:
        """The wrapper table cannot separate them, and that is why this
        table exists.

        Both values build a plain ``DistributedDataParallel``, because
        ``training.py`` picks the Megatron-FSDP wrapper on
        ``--use-megatron-fsdp`` alone and ``zero1`` does not send it. The
        optimizer class is what says which of the two ran.
        """
        self.assertEqual(
            DATA_PARALLEL_WRAPPERS["replicate"],
            DATA_PARALLEL_WRAPPERS["zero1"],
        )
        self.assertNotEqual(
            DATA_PARALLEL_OPTIMIZERS["replicate"],
            DATA_PARALLEL_OPTIMIZERS["zero1"],
        )
        self.assertEqual(
            DATA_PARALLEL_OPTIMIZERS["replicate"],
            "Float16OptimizerWithFloat16Params",
        )
        self.assertEqual(
            DATA_PARALLEL_OPTIMIZERS["zero1"], "DistributedOptimizer"
        )
        self.assertEqual(
            DATA_PARALLEL_OPTIMIZERS["zero3"], "DistributedOptimizer"
        )

    def test_the_optimizer_table_reads_megatrons_own_branch(self) -> None:
        """The table is a claim about Megatron's source, so check it.

        ``get_megatron_optimizer`` builds ``DistributedOptimizer`` under
        ``use_distributed_optimizer`` and
        ``Float16OptimizerWithFloat16Params`` otherwise. A submodule bump
        that renamed either class would leave the table naming a class no
        run can print, and arm rule 12 would fail hours into a real cell.
        """
        if not MEGATRON_OPTIMIZER_SOURCE.exists():
            raise unittest.SkipTest(
                f"{MEGATRON_OPTIMIZER_SOURCE} is absent; init the submodule"
            )
        source = MEGATRON_OPTIMIZER_SOURCE.read_text()
        self.assertIn("if config.use_distributed_optimizer:", source)
        self.assertIn("optimizer = DistributedOptimizer(", source)
        self.assertIn(
            "optimizer = Float16OptimizerWithFloat16Params(", source
        )
        for name in DATA_PARALLEL_OPTIMIZERS.values():
            with self.subTest(optimizer=name):
                self.assertIn(f"optimizer = {name}(", source)

    def test_the_sharding_strategy_table_reads_no_shard_under_replicate(
        self,
    ) -> None:
        """Megatron's argparse default is not the value the run acts on.

        ``--data-parallel-sharding-strategy`` defaults to
        ``optim_grads_params`` and reaches every ``ddp_config``, but
        ``megatron/core/optimizer/__init__.py`` reads it only under
        ``use_megatron_fsdp``. A marker built from the raw field would say
        a replicated run sharded.
        """
        self.assertEqual(SHARDING_STRATEGIES["replicate"], "no_shard")
        # zero1 reads no_shard for the same reason: the field is read only
        # under use_megatron_fsdp, which this value does not send. It shards
        # the optimizer states through the DistributedOptimizer instead.
        self.assertEqual(SHARDING_STRATEGIES["zero1"], "no_shard")
        self.assertEqual(
            SHARDING_STRATEGIES["zero3"], MEGATRON_SHARDING_STRATEGY
        )

    def test_lean_sends_the_four_precision_flags(self) -> None:
        """The whole recipe, and the dtype each flag carries.

        The four together take the optimizer state from 18 bytes per
        parameter to 10: the master becomes a 2-byte remainder, the
        gradients become bf16, and the two Adam moments become bf16.
        """
        emitted = flags_for(
            "1b",
            dataclasses.replace(PP4_SPEC, dense_sharding="zero1"),
            megatron_precision="lean",
        )
        self.assertIn("--use-precision-aware-optimizer", emitted)
        for flag in (
            "--main-grads-dtype",
            "--exp-avg-dtype",
            "--exp-avg-sq-dtype",
        ):
            with self.subTest(flag=flag):
                self.assertEqual(value_after(emitted, flag), "bf16")

    def test_stock_sends_no_precision_flag(self) -> None:
        """The default value must change no published command line."""
        for spec in (TRIVIAL_SPEC, PP4_SPEC, SHARDED_PP4_SPEC):
            emitted = flags_for("1b", spec)
            for flag in LEAN_PRECISION_FLAGS:
                with self.subTest(dense_sharding=spec.dense_sharding, flag=flag):
                    self.assertNotIn(flag, emitted)

    def test_lean_under_the_replicated_parity_is_refused(self) -> None:
        """Megatron asserts use_distributed_optimizer under the
        precision-aware optimizer, and the dense-sharding value is the one
        owner of that flag. Unrefused, the run dies inside Megatron's own
        config validation and names neither axis.
        """
        with self.assertRaisesRegex(
            ValueError, "needs --dense-sharding zero1 or"
        ):
            flags_for("1b", PP4_SPEC, megatron_precision="lean")

    def test_an_unknown_precision_value_is_refused(self) -> None:
        """A silent fall through would send the stock argv under the lean
        label, and record 10 bytes per parameter for a run that held 18."""
        with self.assertRaisesRegex(ValueError, "megatron precision"):
            flags_for(
                "1b",
                dataclasses.replace(PP4_SPEC, dense_sharding="zero1"),
                megatron_precision="bf16",
            )

    def test_the_two_flags_the_recipe_never_sends(self) -> None:
        """Both would be wrong, and each for its own reason.

        ``--main-params-dtype`` accepts fp32 and fp16 only, and the master
        is already 2 bytes through ``store_param_remainders`` while staying
        exactly fp32. ``--grad-reduce-in-bf16`` would state one fact twice:
        under ``--bf16`` Megatron turns fp32 accumulation on only when the
        main-grad dtype is fp32, so ``--main-grads-dtype bf16`` leaves it
        off by itself.
        """
        for precision in ("stock", "lean"):
            emitted = flags_for(
                "1b",
                dataclasses.replace(PP4_SPEC, dense_sharding="zero1"),
                megatron_precision=precision,
            )
            with self.subTest(megatron_precision=precision):
                self.assertNotIn("--main-params-dtype", emitted)
                self.assertNotIn("--grad-reduce-in-bf16", emitted)
        # The one that moved: it is the precision axis's flag now, so a
        # roster entry would say a lean run declines what its argv carries.
        self.assertIn("--grad-reduce-in-bf16", ALWAYS_OMITTED_FLAGS)
        self.assertNotIn("--use-precision-aware-optimizer", ALWAYS_OMITTED_FLAGS)

    def test_geometry_comes_from_the_shape(self) -> None:
        """Every registered shape, field by field.

        A hardcoded width would build one model and publish it under the
        requested size, and no validation rule reads the command line.
        """
        for name, shape in PIPER_SHAPES.items():
            emitted = flags_for(name, TRIVIAL_SPEC)
            expected = {
                "--num-layers": shape.n_layers,
                "--hidden-size": shape.dim,
                "--num-attention-heads": shape.n_heads,
                "--num-query-groups": shape.n_kv_heads,
                "--kv-channels": shape.head_dim,
                "--ffn-hidden-size": shape.moe_hidden_dim,
                "--moe-ffn-hidden-size": shape.moe_hidden_dim,
                "--num-experts": shape.num_experts,
                "--moe-router-topk": shape.top_k,
                "--vocab-size": shape.vocab_size,
                "--padded-vocab-size": shape.vocab_size,
                "--rotary-base": int(shape.rope_theta),
            }
            for flag, want in expected.items():
                with self.subTest(size=name, flag=flag):
                    self.assertEqual(value_after(emitted, flag), str(want))

    def test_rotary_base_is_an_integer_string(self) -> None:
        """Megatron's parser reads ``--rotary-base`` as an int.

        ``1e6`` is a float literal and argparse rejects it, so a run would
        die at parse time rather than measure anything.
        """
        for name in PIPER_SHAPES:
            with self.subTest(size=name):
                rendered = value_after(
                    flags_for(name, TRIVIAL_SPEC), "--rotary-base"
                )
                self.assertEqual(int(rendered), int(rendered.lstrip("-")))
                self.assertNotIn("e", rendered)
                self.assertNotIn(".", rendered)

    def test_one_sample_is_one_packed_sequence(self) -> None:
        """The micro batch size is 1, and the harness packs the rows.

        Megatron flattens a ``(m, S)`` microbatch to ``(1, m*S)`` and then
        sizes its pipeline receive buffer as ``(S, m, H)``, so any micro
        batch size above 1 delivers a permuted activation and raises
        nothing. The packing keeps every shape equal.
        """
        for spec in (TRIVIAL_SPEC, PP4_SPEC):
            with self.subTest(pp=spec.pp):
                emitted = flags_for("1b", spec)
                rows, microbatches, packed = microbatch_geometry(
                    BATCH_32, spec
                )
                self.assertEqual(
                    value_after(emitted, "--micro-batch-size"), "1"
                )
                self.assertEqual(
                    value_after(emitted, "--global-batch-size"),
                    str(microbatches * spec.dp),
                )
                self.assertEqual(
                    value_after(emitted, "--seq-length"), str(packed)
                )
                self.assertEqual(
                    rows * BATCH_32.seq_len, packed
                )
                self.assertEqual(
                    rows * microbatches, BATCH_32.local_batch_size
                )

    def test_the_microbatch_count_matches_the_other_engine(self) -> None:
        """Both engines split one batch the same number of ways.

        Under a pipeline the count is the harness's own
        ``local_batch_size // pp_microbatch_size``. Without one neither
        engine splits, which is what the tuned megatron driver's
        ``pipeline_settings`` returns at pipeline degree 1.
        """
        self.assertEqual(microbatch_geometry(BATCH_32, PP4_SPEC)[1], 8)
        self.assertEqual(microbatch_geometry(BATCH_32, TRIVIAL_SPEC)[1], 1)

    def test_the_position_ceiling_covers_the_packed_sample(self) -> None:
        """Megatron asserts max_position_embeddings >= seq_length."""
        for spec in (TRIVIAL_SPEC, PP4_SPEC):
            for name in PIPER_SHAPES:
                with self.subTest(size=name, pp=spec.pp):
                    emitted = flags_for(name, spec)
                    self.assertGreaterEqual(
                        int(value_after(emitted, "--max-position-embeddings")),
                        int(value_after(emitted, "--seq-length")),
                    )

    def test_the_profiler_stops_on_the_last_step(self) -> None:
        """No iteration may follow ``prof.stop()``.

        ``post_training_step_callbacks`` stops the profiler at
        ``--profile-step-end``, and the top of Megatron's loop keeps calling
        ``prof.step()`` regardless. Every iteration after the stop therefore
        transits a dead Kineto session. A whole number of cycles makes
        ``--profile-step-end`` equal ``--train-iters``, so there is no such
        iteration.
        """
        for steps in (40, 60, 80):
            with self.subTest(steps=steps):
                workload = dataclasses.replace(
                    BATCH_32, steps=steps, local_batch_size=32
                )
                emitted = flags_for("1b", PP4_SPEC, workload)
                end = int(value_after(emitted, "--profile-step-end"))
                self.assertEqual(end % workload.profile_freq, 0)
                self.assertEqual(end, steps)
                self.assertEqual(
                    end, int(value_after(emitted, "--train-iters"))
                )

    def test_a_partial_profiler_cycle_is_refused(self) -> None:
        """The refusal that makes the test above an invariant.

        Flooring the end to the last whole cycle does not help: it only
        delays the first dead-session transit, measured at step 50 for a
        50-step run. Ending at ``--train-iters`` instead writes a truncated
        window, which ``assert_windows_written`` does not catch because it
        refuses a count below the requirement rather than a short window.
        """
        for steps in (41, 45, 50, 55, 59, 99):
            with self.subTest(steps=steps):
                workload = dataclasses.replace(
                    BATCH_32, steps=steps, local_batch_size=32
                )
                with self.assertRaisesRegex(
                    ValueError, r"whole number of profiler cycles"
                ):
                    flags_for("1b", PP4_SPEC, workload)

    def test_the_workload_supplies_the_run_lengths(self) -> None:
        emitted = flags_for("1b", TRIVIAL_SPEC)
        self.assertEqual(
            value_after(emitted, "--train-iters"), str(BATCH_32.steps)
        )
        self.assertEqual(
            value_after(emitted, "--lr-decay-iters"), str(BATCH_32.steps)
        )
        self.assertEqual(
            value_after(emitted, "--profile-step-end"), str(BATCH_32.steps)
        )
        self.assertEqual(value_after(emitted, "--profile-step-start"), "1")
        self.assertEqual(
            value_after(emitted, "--bench-min-trace-windows"),
            str(BATCH_32.min_trace_windows),
        )
        self.assertEqual(value_after(emitted, "--seed"), str(BATCH_32.seed))
        # --seq-length is the packed sample, and --bench-seq-len is the
        # titan row the workload declares.
        self.assertEqual(
            value_after(emitted, "--bench-seq-len"), str(BATCH_32.seq_len)
        )

    def test_a_refused_request_names_its_reason(self) -> None:
        shape = shape_by_name("1b")
        cases = {
            "seeded": (
                dataclasses.replace(BATCH_32, seed=None),
                TRIVIAL_SPEC,
                "default",
                "seeded",
            ),
            "mode": (BATCH_32, TRIVIAL_SPEC, "none", "compile mode"),
            "expert": (
                BATCH_32,
                ParallelismSpec(dp=2, ep=2),
                "default",
                "expert-parallel",
            ),
            "schedule": (
                BATCH_32,
                ParallelismSpec(pp=2, pp_schedule="Interleaved1F1B"),
                "default",
                "pipeline schedule",
            ),
            "divides": (
                dataclasses.replace(BATCH_32, local_batch_size=6),
                PP4_SPEC,
                "default",
                "does not divide",
            ),
            "divides_trivial": (
                dataclasses.replace(BATCH_32, local_batch_size=6),
                ParallelismSpec(
                    pp=4, pp_schedule="1F1B", pp_microbatch_size=4
                ),
                "default",
                "does not divide",
            ),
        }
        for label, (workload, spec, mode, phrase) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError) as caught:
                    stock_megatron_flags(
                        shape,
                        workload,
                        spec,
                        arm_dir="/tmp/arm",
                        model_size="1b",
                        compile_mode=mode,
                    )
                self.assertIn(phrase, str(caught.exception))


# --------------------------------------------------------------------------
# bootstrap.py
# --------------------------------------------------------------------------


class TypingOverrideShimTest(unittest.TestCase):
    """The one name Python 3.10 lacks, and the three ways to get it wrong."""

    def test_it_changes_nothing_when_the_name_exists(self) -> None:
        """A Python 3.12 interpreter must be untouched."""
        def interpreters_own(function):
            return function

        def the_shims(function):
            return function

        stand_in = types.SimpleNamespace(override=interpreters_own)
        extensions = types.SimpleNamespace(override=the_shims)
        added = bootstrap.install_typing_override(
            typing_module=stand_in, extensions=extensions
        )
        self.assertFalse(added)
        self.assertIs(stand_in.override, interpreters_own)

    def test_it_adds_the_name_once(self) -> None:
        def the_shims(function):
            return function

        stand_in = types.SimpleNamespace()
        extensions = types.SimpleNamespace(override=the_shims)
        self.assertTrue(
            bootstrap.install_typing_override(
                typing_module=stand_in, extensions=extensions
            )
        )
        self.assertIs(stand_in.override, the_shims)
        self.assertFalse(
            bootstrap.install_typing_override(
                typing_module=stand_in, extensions=extensions
            )
        )

    def test_a_non_callable_name_is_not_trusted(self) -> None:
        """``hasattr`` is not enough. ``typing.override = None`` breaks
        Megatron inside a class body, so the shim replaces it; a
        non-callable value that is not ``None`` belongs to somebody else,
        so the shim raises rather than shadowing it."""

        def the_shims(function):
            return function

        placeholder = types.SimpleNamespace(override=None)
        self.assertTrue(
            bootstrap.install_typing_override(
                typing_module=placeholder,
                extensions=types.SimpleNamespace(override=the_shims),
            )
        )
        self.assertIs(placeholder.override, the_shims)
        with self.assertRaises(RuntimeError):
            bootstrap.install_typing_override(
                typing_module=types.SimpleNamespace(override="not mine"),
                extensions=types.SimpleNamespace(override=the_shims),
            )

    def test_a_non_callable_replacement_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            bootstrap.install_typing_override(
                typing_module=types.SimpleNamespace(),
                extensions=types.SimpleNamespace(override="not callable"),
            )

    def test_a_missing_module_raises(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            bootstrap.install_typing_override(
                typing_module=types.SimpleNamespace(), extensions=None
            )
        self.assertIn("typing_extensions", str(caught.exception))

    def test_a_module_without_the_name_raises(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            bootstrap.install_typing_override(
                typing_module=types.SimpleNamespace(),
                extensions=types.SimpleNamespace(),
            )
        self.assertIn("override", str(caught.exception))

    def test_a_fresh_interpreter_gets_the_name_from_the_shim(self) -> None:
        """Run in a subprocess, because the name is process-global.

        Another test in this suite calls ``bootstrap.prepare()``, which
        installs the shim. An in-process ``hasattr`` after that asserts
        nothing about the shim: it passes whoever set the name. A fresh
        interpreter is the only place the claim can be tested.
        """
        import subprocess

        probe = (
            "import typing, sys;"
            "before = hasattr(typing, 'override');"
            "sys.path.insert(0, %r);"
            "from benchmarks.e2e.megatron_stock import bootstrap;"
            "added = bootstrap.install_typing_override();"
            "print(before, added, callable(typing.override))"
            % str(Path(__file__).resolve().parent.parent)
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        before, added, is_callable = result.stdout.split()
        # On Python 3.12 and above the interpreter supplies the name and
        # the shim declines. Below it, the shim installs one.
        self.assertEqual(added, "False" if before == "True" else "True")
        self.assertEqual(is_callable, "True")


# --------------------------------------------------------------------------
# data.py
# --------------------------------------------------------------------------


def synthetic_samples(count, seq_len, document_lengths):
    """``(input, positions, label)`` triples with known document splits.

    ``document_lengths`` is one list per sample, and each must sum to
    ``seq_len``.
    """
    samples = []
    for index in range(count):
        lengths = document_lengths[index % len(document_lengths)]
        if sum(lengths) != seq_len:
            raise ValueError("document lengths must sum to the sequence")
        positions = torch.cat(
            [torch.arange(length) for length in lengths]
        ).to(torch.int64)
        tokens = torch.arange(
            index * seq_len, (index + 1) * seq_len, dtype=torch.int64
        )
        labels = tokens + 1
        samples.append((tokens, positions, labels))
    return samples


class MicrobatchContractTest(unittest.TestCase):
    """What Megatron's ``get_batch`` and its merge read out of one dict."""

    def setUp(self) -> None:
        self.seq_len = 16
        self.rows = 2
        self.packed = self.rows * self.seq_len
        self.samples = synthetic_samples(
            8, self.seq_len, [[16], [4, 12], [8, 4, 4]]
        )
        self.iterator = data.StockReplayIterator(
            self.samples, rows_per_sample=self.rows, seq_len=self.seq_len
        )

    def test_the_dict_holds_the_declared_keys(self) -> None:
        microbatch = next(self.iterator)
        self.assertEqual(
            sorted(microbatch), sorted(data.MICROBATCH_KEYS)
        )
        self.assertIsNone(microbatch["attention_mask"])
        self.assertIsNone(microbatch["cu_seqlens_padded"])

    def test_the_dtypes_and_shapes_are_megatron_s(self) -> None:
        """One microbatch is one packed row, so the batch dimension is 1.

        Above 1 Megatron flattens the rows and then reads the pipeline
        activation back through a differently shaped buffer.
        """
        microbatch = next(self.iterator)
        for key in ("tokens", "labels", "position_ids"):
            with self.subTest(key=key):
                self.assertEqual(microbatch[key].dtype, torch.int64)
                self.assertEqual(
                    tuple(microbatch[key].shape), (1, self.packed)
                )
        self.assertEqual(microbatch["loss_mask"].dtype, torch.float32)
        self.assertEqual(
            tuple(microbatch["loss_mask"].shape), (1, self.packed)
        )
        self.assertTrue(bool((microbatch["loss_mask"] == 1).all()))
        self.assertEqual(microbatch["cu_seqlens"].dtype, torch.int32)
        self.assertEqual(tuple(microbatch["cu_seqlens"].shape)[0], 1)
        self.assertEqual(microbatch["max_seqlen"].dtype, torch.int32)
        self.assertEqual(tuple(microbatch["max_seqlen"].shape), (1,))

    def test_every_cu_seqlens_row_starts_at_zero_and_ends_padded(self) -> None:
        """The shape Megatron's merge reads.

        ``_merge_cu_seqlens_across_micro_batch`` keeps the entries up to the
        **first** value equal to ``seq_length`` and drops the rest. A row
        must therefore start at 0, rise strictly, reach ``seq_length``, and
        carry only copies of ``seq_length`` after that.
        """
        for _ in range(self.iterator.microbatch_count):
            rows = next(self.iterator)["cu_seqlens"]
            for row in rows:
                values = [int(entry) for entry in row]
                self.assertEqual(values[0], 0)
                end = values.index(self.packed)
                real = values[: end + 1]
                self.assertEqual(
                    real, sorted(set(real)), "offsets must rise strictly"
                )
                self.assertEqual(real[-1], self.packed)
                self.assertEqual(
                    values[end + 1 :],
                    [self.packed] * (len(values) - end - 1),
                )

    def test_the_padded_width_is_the_widest_pack_of_this_rank(self) -> None:
        # The pack of rows [8,4,4] and [16] holds five documents, so six
        # entries: 0, 8, 12, 16, 32.  Wait -- that pack is rows 2 and 3.
        widest = self.iterator.padded_documents
        for _ in range(self.iterator.microbatch_count):
            self.assertEqual(
                tuple(next(self.iterator)["cu_seqlens"].shape), (1, widest)
            )

    def test_max_seqlen_is_the_longest_document_in_the_pack(self) -> None:
        microbatch = next(self.iterator)
        # The pack holds rows [16] and [4, 12], so five documents whose
        # longest is 16.
        self.assertEqual([int(v) for v in microbatch["max_seqlen"]], [16])

    def test_exhaustion_raises_rather_than_wrapping(self) -> None:
        """A wrap trains a second epoch under the first epoch's label."""
        for _ in range(self.iterator.microbatch_count):
            next(self.iterator)
        with self.assertRaises(RuntimeError) as caught:
            next(self.iterator)
        self.assertIn("exhausted", str(caught.exception))

    def test_the_pack_keeps_the_stream_order(self) -> None:
        """Microbatch i holds samples 2i and 2i+1, concatenated in order."""
        for index in range(self.iterator.microbatch_count):
            tokens = next(self.iterator)["tokens"][0]
            expected = torch.cat(
                [
                    self.samples[self.rows * index + row][0]
                    for row in range(self.rows)
                ]
            )
            self.assertTrue(bool((tokens == expected).all()))

    def test_an_indivisible_sample_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            data.StockReplayIterator(
                self.samples[:7], rows_per_sample=2, seq_len=self.seq_len
            )

    def test_a_sample_that_does_not_start_a_document_raises(self) -> None:
        positions = torch.arange(1, 17, dtype=torch.int64)
        with self.assertRaises(ValueError) as caught:
            data.document_offsets(positions, 16)
        self.assertIn("document start", str(caught.exception))

    def test_the_provider_is_distributed(self) -> None:
        """Every rank builds its own stream.

        Under ``--dataloader-inter-document-masking`` the middle pipeline
        stages read the batch too, so a provider that built on tensor rank 0
        alone would leave them with no iterator.
        """
        self.assertIs(
            data.train_valid_test_datasets_provider.is_distributed, True
        )


# --------------------------------------------------------------------------
# profiling.py
# --------------------------------------------------------------------------


class FakeProfiler:
    """Enough of a ``torch.profiler.profile`` for the trace handler."""

    def __init__(self, step_num, **kwargs):
        self.step_num = step_num
        self.kwargs = kwargs
        self.exported = None

    def export_chrome_trace(self, path):
        self.exported = path
        Path(path).write_text("{}")


class ProfilerShimTest(unittest.TestCase):
    """The schedule, the path, and the two guards."""

    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.arm_dir = Path(self.tmp.name) / "baseline"
        self.built = []
        self.original = torch.profiler.profile

        def recorder(*args, **kwargs):
            self.built.append(kwargs)
            return FakeProfiler(step_num=20, **kwargs)

        torch.profiler.profile = recorder
        self.shim = profiling.install_profiler_shim(
            arm_dir=self.arm_dir,
            rank=3,
            profile_freq=20,
            profiler_warmup=5,
            profiler_active=5,
        )

    def tearDown(self) -> None:
        torch.profiler.profile = self.original
        self.tmp.cleanup()

    def test_it_replaces_the_attribute_and_restores_it(self) -> None:
        self.assertIs(torch.profiler.profile, self.shim.replacement)
        self.shim.uninstall()
        self.assertIsNot(torch.profiler.profile, self.shim.replacement)

    def test_it_overrides_the_schedule_and_the_handler(self) -> None:
        """Megatron's own schedule carries repeat=1 and writes one window."""
        torch.profiler.profile(
            schedule="megatron's",
            on_trace_ready="megatron's",
            record_shapes=False,
        )
        self.assertEqual(self.shim.calls, 1)
        passed = self.built[-1]
        self.assertNotEqual(passed["schedule"], "megatron's")
        self.assertNotEqual(passed["on_trace_ready"], "megatron's")
        self.assertIs(passed["record_shapes"], False)

    def test_the_handler_writes_the_layout_the_harness_reads(self) -> None:
        torch.profiler.profile(schedule=None, on_trace_ready=None)
        handler = self.built[-1]["on_trace_ready"]
        handler(FakeProfiler(step_num=20))
        handler(FakeProfiler(step_num=40))
        written = sorted(
            path.relative_to(self.arm_dir).as_posix()
            for path in self.arm_dir.rglob("*.json.gz")
        )
        self.assertEqual(
            written,
            [
                "profiling/traces/iteration_20/rank3_trace.json.gz",
                "profiling/traces/iteration_40/rank3_trace.json.gz",
            ],
        )

    def test_a_shim_that_never_ran_fails_the_run(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            profiling.assert_windows_written(self.shim, min_trace_windows=2)
        self.assertIn("0 time(s)", str(caught.exception))

    def test_one_window_fails_the_run(self) -> None:
        torch.profiler.profile(schedule=None, on_trace_ready=None)
        self.built[-1]["on_trace_ready"](FakeProfiler(step_num=20))
        with self.assertRaises(RuntimeError) as caught:
            profiling.assert_windows_written(self.shim, min_trace_windows=2)
        self.assertIn("1 profiler window", str(caught.exception))

    def test_two_windows_pass(self) -> None:
        torch.profiler.profile(schedule=None, on_trace_ready=None)
        handler = self.built[-1]["on_trace_ready"]
        handler(FakeProfiler(step_num=20))
        handler(FakeProfiler(step_num=40))
        windows = profiling.assert_windows_written(
            self.shim, min_trace_windows=2
        )
        self.assertEqual(len(windows), 2)

    def test_another_rank_s_windows_do_not_count(self) -> None:
        """The rule is per rank, so the glob names the rank."""
        window = self.arm_dir / "profiling/traces/iteration_20"
        window.mkdir(parents=True)
        (window / "rank0_trace.json.gz").write_text("{}")
        self.shim.calls = 1
        with self.assertRaises(RuntimeError):
            profiling.assert_windows_written(self.shim, min_trace_windows=1)

    def test_the_installed_schedule_writes_one_window_per_cycle(
        self,
    ) -> None:
        """Drive the REAL schedule, not a fabricated step number.

        Every other test here calls the handler by hand, so none of them
        would notice a ``repeat=1`` -- which is the stock Megatron
        behaviour this shim exists to correct, and which writes one window.

        The indices carry ``PROFILER_STEP_OFFSET``, because the schedule is
        shifted by that much to put Megatron's steps on TorchTitan's
        profiler actions. ``ProfilerAlignmentTest`` owns the offset itself;
        this test owns the cycle.
        """
        from torch.profiler import ProfilerAction

        offset = profiling.PROFILER_STEP_OFFSET
        torch.profiler.profile(schedule=None, on_trace_ready=None)
        schedule = self.built[-1]["schedule"]
        saves = [
            step
            for step in range(1, 61 + offset)
            if schedule(step) is ProfilerAction.RECORD_AND_SAVE
        ]
        # One per 20-step cycle, and never the single window repeat=1 gives.
        self.assertEqual(saves, [19 + offset, 39 + offset, 59 + offset])
        # A window flushes on the transition out of RECORD_AND_SAVE, so the
        # flush lands on the next step.
        for step in saves:
            self.assertIs(schedule(step + 1), ProfilerAction.NONE)
        self.assertEqual(len([s for s in saves if s <= 40]), 40 // 20)

    def test_a_non_positive_requirement_is_refused(self) -> None:
        """A guard that accepts zero windows guards nothing."""
        self.shim.calls = 1
        with self.assertRaises(ValueError):
            profiling.assert_windows_written(self.shim, min_trace_windows=0)

    def test_a_cycle_that_cannot_hold_the_windows_raises(self) -> None:
        with self.assertRaises(ValueError):
            profiling.install_profiler_shim(
                arm_dir=self.arm_dir,
                rank=0,
                profile_freq=8,
                profiler_warmup=5,
                profiler_active=5,
            )


# --------------------------------------------------------------------------
# train.py
# --------------------------------------------------------------------------

# Section 8.4 of PIPER_STOCK_MEGATRON_PLAN.md, transcribed. The validation
# profile carries the same strings, and a one-character difference fails a
# real run at validation time.
PLAN_MODE_PREFIX = "Megatron-LM stock training loop ("
PLAN_PARALLELISM_LINE = (
    "Megatron-LM stock parallelism: dp={dp} pp={pp} ep={ep} "
    "schedule={schedule} microbatches={microbatches} stages={stages}"
)
PLAN_DATA_PARALLEL_LINE = (
    "Megatron-LM stock data parallel: {wrapper} over {dp} "
    "ranks (overlap_grad_reduce={overlap}, grad_reduce_in_fp32={fp32}, "
    "sharding_strategy={sharding}, expert_parallel={expert}, "
    "optimizer={optimizer})"
)
PLAN_STAGE_SIZE_LINE = (
    "stock-megatron stage {stage}/{stages} local size: {count} parameters"
)
PLAN_MODEL_SIZE_LINE = (
    "Model qwen3_piper_{size} stock-megatron size: {total} total parameters"
)
# The p2p line the driver prints from its built config. The validation
# profile carries the same string; both fields are in it because megatron
# guards its per-message sync on both.
PLAN_P2P_LINE = (
    "Megatron-LM stock p2p: batch_p2p_comm={comm} batch_p2p_sync={sync}"
)
# The NaN-guard line, from the value Megatron parsed. The validation profile
# carries the same string, at every mesh.
PLAN_NAN_GUARD_LINE = (
    "Megatron-LM stock nan guard: check_for_nan_in_loss_and_grad={value}"
)


def stock_args(**overrides):
    """An argument namespace shaped like the one Megatron resolves."""
    base = dict(
        bench_pp_schedule=None,
        bench_batch_p2p_sync="on",
        check_for_nan_in_loss_and_grad=True,
        bench_local_batch_size=32,
        bench_model_size="1b",
        bench_seq_len=1024,
        bench_rows_per_sample=32,
        bench_min_trace_windows=2,
        seq_length=32 * 1024,
        micro_batch_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        expert_model_parallel_size=1,
        data_parallel_size=1,
        world_size=1,
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        # Megatron maps every dtype string to a torch.dtype before the
        # optimizer exists, so the resolved arguments carry these and the
        # log prints "torch.float32" where the flag says "fp32".
        use_precision_aware_optimizer=False,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        accumulate_allreduce_grads_in_fp32=True,
        cross_entropy_loss_fusion=False,
        moe_token_dispatcher_type="alltoall",
        overlap_grad_reduce=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class ProfilerAlignmentTest(unittest.TestCase):
    """The two engines must be sampled in the same profiler state.

    Megatron calls ``prof.step()`` at the top of its training loop and
    TorchTitan calls it at the bottom, so the same training step runs under
    schedule indices that differ by one. ``skip_first`` removes the
    difference. Without it the ``NONE -> WARMUP`` transition, which runs
    torch's ``prepare_trace``, lands on a step ``stable_tps`` samples on one
    engine only, and the published cross-engine ratio carries that bias.

    Every assertion below runs against the **real**
    ``torch.profiler.schedule`` of the pinned torch, not a model of it.
    """

    WAIT, WARMUP, ACTIVE, FREQ = 10, 5, 5, 20

    # torch's own action map, restricted to the pairs that write a window.
    # Read from torch.profiler.profiler at this rev.
    def _writes(self):
        action = torch.profiler.ProfilerAction
        save = action.RECORD_AND_SAVE
        return {
            (save, action.NONE),
            (save, action.WARMUP),
            (save, action.RECORD),
            (save, save),
            (save, None),
            (action.RECORD, None),
        }

    def _schedule(self, *, skip_first):
        return torch.profiler.schedule(
            wait=self.WAIT,
            warmup=self.WARMUP,
            active=self.ACTIVE,
            repeat=0,
            skip_first=skip_first,
        )

    def _walk(self, schedule, steps, *, steps_first, stop_at=None):
        """Replay a training loop to its END. Returns actions, windows, deaths.

        ``steps_first`` True is Megatron: ``prof.step()`` runs before the
        training step. False is TorchTitan: it runs after.

        **The loop runs to ``steps``, never to ``stop_at``.** Megatron's
        ``prof.stop()`` is guarded on ``iteration == --profile-step-end``
        and its ``prof.step()`` is not, so the loop keeps stepping a stopped
        profiler. A model that stopped iterating at ``stop_at`` would report
        a clean run over exactly the workloads where the real one is not,
        which is how the first version of this test passed while the arm
        would have crashed.

        ``deaths`` counts transits taken after the stop that are not the
        ``(NONE, NONE)`` no-op. Any of them touches a dead Kineto session.
        """
        action = torch.profiler.ProfilerAction
        writes = self._writes()
        step_num, current = 0, schedule(0)
        per_step, windows, recorded, deaths = {}, [], 0, 0
        stopped = False
        for step in range(1, steps + 1):
            if steps_first:
                step_num += 1
                previous, current = current, schedule(step_num)
                if stopped:
                    if (previous, current) != (action.NONE, action.NONE):
                        deaths += 1
                elif (previous, current) in writes:
                    windows.append(recorded)
                    recorded = 0
            per_step[step] = current
            if not stopped and current in (
                action.RECORD,
                action.RECORD_AND_SAVE,
            ):
                recorded += 1
            if not steps_first:
                step_num += 1
                previous, current = current, schedule(step_num)
                if (previous, current) in writes:
                    windows.append(recorded)
                    recorded = 0
            if step == stop_at and not stopped:
                if (current, None) in writes:
                    windows.append(recorded)
                    recorded = 0
                stopped = True
        return per_step, windows, deaths

    def _titan(self, steps):
        return self._walk(
            self._schedule(skip_first=0), steps, steps_first=False
        )

    def _stock(self, steps, *, skip_first):
        # flags.py refuses a partial cycle, so --profile-step-end is
        # --train-iters and the stop lands on the last iteration.
        return self._walk(
            self._schedule(skip_first=skip_first),
            steps,
            steps_first=True,
            stop_at=steps,
        )

    def test_the_shim_declares_the_offset(self) -> None:
        """One named constant, so the reason is not spread over the code."""
        self.assertEqual(profiling.PROFILER_STEP_OFFSET, 1)

    def test_the_two_engines_share_every_step_s_profiler_action(self) -> None:
        titan, _, _ = self._titan(40)
        stock, _, _ = self._stock(40, skip_first=profiling.PROFILER_STEP_OFFSET)
        for step in range(1, 41):
            with self.subTest(step=step):
                self.assertEqual(stock[step], titan[step])

    def test_without_the_offset_a_sampled_step_disagrees(self) -> None:
        """The defect this offset repairs, pinned so it cannot come back.

        ``stable_tps`` keeps steps 2 to 10 of every 20-step cycle. Step 10
        is where the two engines part without the offset.
        """
        titan, _, _ = self._titan(40)
        unshifted, _, _ = self._stock(40, skip_first=0)
        sampled = [s for s in range(1, 41) if 2 <= ((s - 1) % 20) + 1 <= 10]
        disagreeing = [s for s in sampled if unshifted[s] != titan[s]]
        self.assertEqual(disagreeing, [10, 30])
        self.assertEqual(
            unshifted[10], torch.profiler.ProfilerAction.WARMUP
        )
        self.assertEqual(titan[10], torch.profiler.ProfilerAction.NONE)

    def test_no_sampled_step_carries_a_transition(self) -> None:
        """With the offset, every sampled step is NONE on both engines."""
        titan, _, _ = self._titan(40)
        stock, _, _ = self._stock(40, skip_first=profiling.PROFILER_STEP_OFFSET)
        for step in range(1, 41):
            if not 2 <= ((step - 1) % 20) + 1 <= 10:
                continue
            with self.subTest(step=step):
                self.assertEqual(stock[step], torch.profiler.ProfilerAction.NONE)
                self.assertEqual(titan[step], torch.profiler.ProfilerAction.NONE)

    def test_every_window_still_holds_a_full_active_phase(self) -> None:
        """The offset must not truncate a window, lose one, or outlive one.

        ``assert_windows_written`` refuses a run below the declared count,
        so a lost window fails loudly. A **short** window would not: it
        would be pooled with the full ones and would move every per-step
        figure. A transit after ``prof.stop()`` touches a dead Kineto
        session. All three are checked, at every step count ``flags.py``
        accepts.
        """
        for steps in (40, 60, 80, 100, 200):
            with self.subTest(steps=steps):
                _, windows, deaths = self._stock(
                    steps, skip_first=profiling.PROFILER_STEP_OFFSET
                )
                self.assertEqual(
                    windows, [self.ACTIVE] * (steps // self.FREQ)
                )
                self.assertEqual(deaths, 0)

    def _deaths_at(self, steps, skip_first):
        """Dead-session transits when the end is floored to a whole cycle."""
        _, _, deaths = self._walk(
            self._schedule(skip_first=skip_first),
            steps,
            steps_first=True,
            stop_at=(steps // self.FREQ) * self.FREQ,
        )
        return deaths

    def test_a_partial_cycle_would_step_a_stopped_profiler(self) -> None:
        """Why ``flags.py`` refuses one. This is the reason, not the rule.

        The rule lives in ``stock_megatron_flags``, and
        ``FlagListTest.test_a_partial_profiler_cycle_is_refused`` owns it.
        A reader who removes the refusal must first make this test pass.

        **The hazard predates the offset.** Flooring the end to the last
        whole cycle leaves the loop running past the stop either way, so
        both schedules reach a dead Kineto session.
        """
        for steps in (50, 57):
            for skip_first in (0, profiling.PROFILER_STEP_OFFSET):
                with self.subTest(steps=steps, skip_first=skip_first):
                    self.assertGreater(self._deaths_at(steps, skip_first), 0)

    def test_the_offset_makes_a_partial_cycle_fail_sooner(self) -> None:
        """The offset widens the hazard, which is why the refusal landed.

        At 41 steps the unshifted schedule is idle when the stop fires and
        stays idle, so nothing transits. The shifted one is still in
        ``RECORD_AND_SAVE``, and the next step writes a window through a
        dead session. Neither is acceptable; the refusal removes both.
        """
        self.assertEqual(self._deaths_at(41, 0), 0)
        self.assertGreater(
            self._deaths_at(41, profiling.PROFILER_STEP_OFFSET), 0
        )

    def test_the_stock_windows_match_the_titan_windows(self) -> None:
        """Same count, same recorded-step count, at every step count."""
        for steps in (40, 60, 80):
            with self.subTest(steps=steps):
                _, titan, _ = self._titan(steps)
                _, stock, _ = self._stock(
                    steps, skip_first=profiling.PROFILER_STEP_OFFSET
                )
                self.assertEqual(stock, titan)


class MarkerStringTest(unittest.TestCase):
    """The driver's own strings against the plan's, character for character."""

    def test_the_mode_line_carries_the_declared_prefix(self) -> None:
        line = train.mode_line(stock_args())
        self.assertTrue(line.startswith(PLAN_MODE_PREFIX))
        for field in (
            "main_params_dtype=torch.float32",
            "main_grads_dtype=torch.float32",
            "use_precision_aware_optimizer=False",
            "exp_avg_dtype=torch.float32",
            "exp_avg_sq_dtype=torch.float32",
            "accumulate_allreduce_grads_in_fp32=True",
            "cross_entropy_loss_fusion=False",
            "moe_token_dispatcher_type=alltoall",
        ):
            with self.subTest(field=field):
                self.assertIn(field, line)
        self.assertEqual(line.count("\n"), 0)

    def test_the_mode_line_carries_the_lean_precision(self) -> None:
        """The four fields at the other value.

        Megatron maps every dtype string to a ``torch.dtype`` before the
        optimizer exists, so the line prints ``torch.bfloat16`` where the
        flag says ``bf16``.
        """
        line = train.mode_line(
            stock_args(
                use_precision_aware_optimizer=True,
                main_grads_dtype=torch.bfloat16,
                exp_avg_dtype=torch.bfloat16,
                exp_avg_sq_dtype=torch.bfloat16,
            )
        )
        for field in (
            "use_precision_aware_optimizer=True",
            "main_grads_dtype=torch.bfloat16",
            "exp_avg_dtype=torch.bfloat16",
            "exp_avg_sq_dtype=torch.bfloat16",
        ):
            with self.subTest(field=field):
                self.assertIn(field, line)
        # The master copy stays fp32 under both values, because the recipe
        # never sends --main-params-dtype.
        self.assertIn("main_params_dtype=torch.float32", line)

    def test_the_precision_fields_are_the_profile_markers(self) -> None:
        """The driver's line against the profile's markers, both values.

        A one-character difference fails a real run at arm rule 12.
        """
        from benchmarks.e2e.validation import VALIDATION_PROFILES

        profile = VALIDATION_PROFILES["megatron_stock"]
        cases = {
            "stock": stock_args(),
            "lean": stock_args(
                use_precision_aware_optimizer=True,
                main_grads_dtype=torch.bfloat16,
                exp_avg_dtype=torch.bfloat16,
                exp_avg_sq_dtype=torch.bfloat16,
            ),
        }
        for value, args in cases.items():
            line = train.mode_line(args)
            for marker in profile.precision_markers(value):
                with self.subTest(value=value, marker=marker):
                    self.assertIn(marker, line)

    def test_the_parallelism_templates_match_the_plan(self) -> None:
        self.assertEqual(train.PARALLELISM_LINE, PLAN_PARALLELISM_LINE)
        self.assertEqual(train.DATA_PARALLEL_LINE, PLAN_DATA_PARALLEL_LINE)
        self.assertEqual(
            train.MODE_LINE[: len(PLAN_MODE_PREFIX)], PLAN_MODE_PREFIX
        )
        self.assertEqual(train.TRAINING_COMPLETED, "Training completed")

    def test_the_parameter_templates_match_the_plan(self) -> None:
        self.assertEqual(train.STAGE_SIZE_LINE, PLAN_STAGE_SIZE_LINE)
        self.assertEqual(train.MODEL_SIZE_LINE, PLAN_MODEL_SIZE_LINE)

    def test_the_p2p_template_matches_the_plan(self) -> None:
        self.assertEqual(train.P2P_LINE, PLAN_P2P_LINE)

    def test_the_nan_guard_template_matches_the_plan(self) -> None:
        self.assertEqual(train.NAN_GUARD_LINE, PLAN_NAN_GUARD_LINE)

    def test_the_model_size_line_carries_a_thousands_separator(self) -> None:
        """Arm rule 11 builds its target with ``f"{param_count:,}"``."""
        for name, shape in PIPER_SHAPES.items():
            with self.subTest(size=name):
                line = train.MODEL_SIZE_LINE.format(
                    size=name, total=f"{shape.param_count:,}"
                )
                self.assertIn(
                    f"size: {shape.param_count:,} total parameters", line
                )

    def test_no_mesh_line_at_world_size_one(self) -> None:
        """Arm rule 12 is consulted only above one rank."""
        self.assertEqual(
            train.parallelism_lines(stock_args(), microbatches=1), []
        )

    def test_a_pipeline_prints_one_line(self) -> None:
        lines = train.parallelism_lines(
            stock_args(
                world_size=4,
                pipeline_model_parallel_size=4,
                bench_pp_schedule="1F1B",
            ),
            microbatches=8,
        )
        self.assertEqual(
            lines,
            [
                "Megatron-LM stock parallelism: dp=1 pp=4 ep=1 "
                "schedule=1F1B microbatches=8 stages=4"
            ],
        )

    def test_a_data_parallel_mesh_prints_the_mesh_line_alone(self) -> None:
        """The data-parallel line comes from the wrapper, not from here.

        ``install_data_parallel_marker`` prints it after
        ``setup_model_and_optimizer`` returns, and raises when no chunk
        carries a wrapper. A copy derived from ``args`` would satisfy arm
        rule 12 without the wrapper, so this function must not print one.
        """
        lines = train.parallelism_lines(
            stock_args(
                world_size=8,
                pipeline_model_parallel_size=4,
                data_parallel_size=2,
                bench_pp_schedule="1F1B",
            ),
            microbatches=8,
        )
        self.assertEqual(
            lines,
            [
                "Megatron-LM stock parallelism: dp=2 pp=4 ep=1 "
                "schedule=1F1B microbatches=8 stages=4"
            ],
        )

    def test_the_mesh_line_carries_the_expert_degree(self) -> None:
        """A degree the harness holds and no line names cannot be read.

        This field restates the argv, which is what ``pp`` already does:
        the line prints before ``pretrain()`` runs, so no process group
        exists yet. ``install_data_parallel_marker`` reads the built group
        later and refuses a disagreement.
        """
        lines = train.parallelism_lines(
            stock_args(
                world_size=8,
                pipeline_model_parallel_size=4,
                data_parallel_size=2,
                expert_model_parallel_size=2,
                bench_pp_schedule="1F1B",
            ),
            microbatches=8,
        )
        self.assertEqual(
            lines,
            [
                "Megatron-LM stock parallelism: dp=2 pp=4 ep=2 "
                "schedule=1F1B microbatches=8 stages=4"
            ],
        )


class DriverRefusalTest(unittest.TestCase):
    """A run this driver cannot honour must fail before it builds anything."""

    def test_a_schedule_other_than_1f1b_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            train.refuse_unsupported_run(
                stock_args(
                    pipeline_model_parallel_size=4,
                    bench_pp_schedule="Interleaved1F1B",
                )
            )
        self.assertIn("--bench-pp-schedule", str(caught.exception))

    def test_a_schedule_at_pipeline_degree_one_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            train.refuse_unsupported_run(stock_args(bench_pp_schedule="1F1B"))

    def test_a_virtual_pipeline_degree_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            train.refuse_unsupported_run(
                stock_args(virtual_pipeline_model_parallel_size=2)
            )
        self.assertIn("virtual pipeline", str(caught.exception))

    def test_a_batched_microbatch_is_refused(self) -> None:
        """It would send the next pipeline stage a permuted activation."""
        with self.assertRaises(ValueError) as caught:
            train.refuse_unsupported_run(stock_args(micro_batch_size=4))
        self.assertIn("permuted", str(caught.exception))

    def test_a_packing_that_does_not_match_seq_length_is_refused(
        self,
    ) -> None:
        with self.assertRaises(ValueError) as caught:
            train.refuse_unsupported_run(
                stock_args(bench_rows_per_sample=8)
            )
        self.assertIn("--seq-length", str(caught.exception))

    def test_the_declared_run_is_accepted(self) -> None:
        train.refuse_unsupported_run(stock_args())
        train.refuse_unsupported_run(
            stock_args(
                world_size=8,
                pipeline_model_parallel_size=4,
                data_parallel_size=2,
                bench_pp_schedule="1F1B",
                bench_rows_per_sample=4,
                seq_length=4096,
            )
        )

    def test_p2p_sync_off_at_pipeline_degree_one_is_refused(self) -> None:
        """The field is inert without a pipeline message, and the run would
        print a treatment it did not have."""
        with self.assertRaises(ValueError) as caught:
            train.refuse_unsupported_run(stock_args(bench_batch_p2p_sync="off"))
        self.assertIn(BENCH_BATCH_P2P_SYNC, str(caught.exception))
        self.assertIn("no pipeline message", str(caught.exception))

    def test_p2p_sync_off_under_a_pipeline_is_accepted(self) -> None:
        train.refuse_unsupported_run(
            stock_args(
                world_size=4,
                pipeline_model_parallel_size=4,
                bench_pp_schedule="1F1B",
                bench_rows_per_sample=8,
                seq_length=8192,
                bench_batch_p2p_sync="off",
            )
        )


class P2pSyncMappingTest(unittest.TestCase):
    """How ``--bench-batch-p2p-sync`` reaches a field Megatron has no flag for.

    ``core_transformer_config_from_args`` copies every ``args`` attribute
    whose name is a config field. Megatron's own parser never sets
    ``batch_p2p_sync``, so the attribute exists only when this driver puts
    it there.
    """

    def test_off_sets_the_field_false_on_args(self) -> None:
        args = train.apply_p2p_sync(stock_args(bench_batch_p2p_sync="off"))
        self.assertIs(args.batch_p2p_sync, False)

    def test_on_leaves_the_attribute_absent(self) -> None:
        """Megatron's dataclass default must rule, exactly as it did before
        the option existed. An attribute set to True would be copied too,
        and a Megatron bump that moved the default would then be masked."""
        args = train.apply_p2p_sync(stock_args())
        self.assertFalse(hasattr(args, "batch_p2p_sync"))

    def test_the_line_reads_the_built_transformer_config(self) -> None:
        """Both fields off the config, never off ``args``.

        ``batch_p2p_comm`` is derived inside
        ``core_transformer_config_from_args`` from ``overlap_p2p_comm``, so
        no argument carries it at all.
        """
        for sync in (True, False):
            with self.subTest(sync=sync):
                model_cfg = SimpleNamespace(
                    transformer=SimpleNamespace(
                        batch_p2p_comm=True, batch_p2p_sync=sync
                    )
                )
                self.assertEqual(
                    train.p2p_line(model_cfg),
                    "Megatron-LM stock p2p: batch_p2p_comm=True "
                    f"batch_p2p_sync={sync}",
                )

    def test_main_maps_after_the_refusal_and_prints_the_built_config(
        self,
    ) -> None:
        """Read off the source: the assignment sits between the refusal and
        the config build, and the print reads ``model_cfg``."""
        import inspect

        source = inspect.getsource(train.main)
        refusal = source.index("refuse_unsupported_run(args)")
        mapping = source.index("apply_p2p_sync(args)")
        build = source.index("model_cfg = gpt_config_from_args(")
        printed = source.index("print(p2p_line(model_cfg), flush=True)")
        self.assertLess(refusal, mapping)
        self.assertLess(mapping, build)
        self.assertLess(build, printed)
        self.assertLess(printed, source.index("\n    pretrain(\n"))


class NanGuardLineTest(unittest.TestCase):
    """The driver prints Megatron's PARSED value, and nothing sets it here.

    ``--megatron-nan-guard off`` reaches the argv as Megatron's own
    ``--no-check-for-nan-in-loss-and-grad``. The driver reads no harness
    value for it; the line is an observation of what Megatron resolved.
    """

    def test_the_line_reads_the_parsed_value(self) -> None:
        for value in (True, False):
            with self.subTest(value=value):
                self.assertEqual(
                    train.nan_guard_line(
                        stock_args(check_for_nan_in_loss_and_grad=value)
                    ),
                    "Megatron-LM stock nan guard: "
                    f"check_for_nan_in_loss_and_grad={value}",
                )

    def test_main_prints_it_after_the_refusal_and_before_pretrain(
        self,
    ) -> None:
        """Read off the source, as the p2p order is: parsed, refused, then
        printed on every rank beside the mode line."""
        import inspect

        source = inspect.getsource(train.main)
        parsed = source.index("parse_and_validate_args(")
        refusal = source.index("refuse_unsupported_run(args)")
        printed = source.index("print(nan_guard_line(args), flush=True)")
        self.assertLess(parsed, refusal)
        self.assertLess(refusal, printed)
        self.assertLess(printed, source.index("\n    pretrain(\n"))
        # Unconditional: the print sits at the body's own indentation, so
        # no rank check and no mesh check guards it.
        line_start = source.rindex("\n", 0, printed) + 1
        self.assertEqual(source[line_start:printed], "    ")


def _megatron_source(relative: str) -> str:
    from benchmarks.models.piper_qwen3.megatron_bootstrap import megatron_dir

    return (megatron_dir() / relative).read_text()


class MegatronNanGuardSourceTest(unittest.TestCase):
    """What the one Megatron field gates, pinned against the submodule.

    The registry comment above ``MEGATRON_NAN_GUARD_MODES`` states these
    four facts about Megatron-LM 59b72fa5. A submodule bump that moves any
    of them must fail here rather than leave the option gating something
    else under the same name.
    """

    def test_the_flag_spelling_and_its_dest_are_megatron_s_own(self) -> None:
        """A misspelt flag fails at parse time; a moved dest parses and
        gates nothing, which is the case this test exists for."""
        source = _megatron_source("megatron/training/arguments.py")
        self.assertIn(f"'{NO_CHECK_FOR_NAN_FLAG}'", source)
        flag_at = source.index(f"'{NO_CHECK_FOR_NAN_FLAG}'")
        declaration = source[flag_at : flag_at + 300]
        self.assertIn("action='store_false'", declaration)
        self.assertIn("dest='check_for_nan_in_loss_and_grad'", declaration)

    def test_the_loss_check_reads_the_field_directly(self) -> None:
        """pretrain_gpt.py's loss_func gates both validate_result calls."""
        source = _megatron_source("pretrain_gpt.py")
        gate = source.index("if args.check_for_nan_in_loss_and_grad:")
        block = source[gate : gate + 600]
        self.assertEqual(block.count("rerun_state_machine.validate_result("), 2)
        self.assertIn("rejection_func=torch.isnan", block)
        self.assertIn("rejection_func=torch.isinf", block)

    def test_the_gradient_check_descends_from_the_same_field(self) -> None:
        """training.py copies it into ddp_config, and check_grads reads
        that copy for every bucket."""
        self.assertIn(
            'kwargs["check_for_nan_in_grad"] = '
            "args.check_for_nan_in_loss_and_grad",
            _megatron_source("megatron/training/training.py"),
        )
        buffer = _megatron_source(
            "megatron/core/distributed/param_and_grad_buffer.py"
        )
        self.assertIn("if self.ddp_config.check_for_nan_in_grad or", buffer)
        self.assertIn(
            "check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad",
            buffer,
        )
        self.assertIn("def check_grads(self, check_for_nan_or_inf", buffer)

    def test_rerun_mode_disabled_still_evaluates_the_rejection(self) -> None:
        """``--rerun-mode disabled``, which the argv sends, removes neither
        check: validate_result evaluates rejection_func under DISABLED and
        raises when fatal. Only the field above skips the calls."""
        self.assertIn("disabled", flags_for("1b", TRIVIAL_SPEC))
        source = _megatron_source("megatron/core/rerun_state_machine.py")
        start = source.index("def validate_result(")
        # The next method at the class's own indentation. The docstring
        # holds an example ``def train_step`` at a deeper one.
        body = source[start : source.index("\n    def ", start + 10)]
        disabled = body.index("if self.mode == RerunMode.DISABLED:")
        branch = body[disabled : disabled + 800]
        self.assertIn("result_rejected: bool = rejection_func(result)", branch)
        self.assertIn("raise RuntimeError(full_message)", branch)


class RendezvousDefaultsTest(unittest.TestCase):
    """At one rank the driver fills the rendezvous; under torchrun it does not."""

    def test_an_empty_environment_gets_both_variables(self) -> None:
        env: dict[str, str] = {}
        train.install_rendezvous_defaults(env)
        self.assertEqual(env["MASTER_ADDR"], "127.0.0.1")
        self.assertTrue(1024 <= int(env["MASTER_PORT"]) <= 65535)

    def test_the_launcher_values_are_kept(self) -> None:
        env = {"MASTER_ADDR": "10.0.0.7", "MASTER_PORT": "29500"}
        train.install_rendezvous_defaults(env)
        self.assertEqual(
            env, {"MASTER_ADDR": "10.0.0.7", "MASTER_PORT": "29500"}
        )

    def test_a_set_port_is_kept_when_only_the_address_is_absent(self) -> None:
        env = {"MASTER_PORT": "29500"}
        train.install_rendezvous_defaults(env)
        self.assertEqual(env["MASTER_PORT"], "29500")
        self.assertEqual(env["MASTER_ADDR"], "127.0.0.1")


class StepLineTest(unittest.TestCase):
    """The line ``benchmarks/e2e/results.py`` parses."""

    def line(self, **overrides):
        fields = dict(
            step=7,
            loss=2.5,
            grad_norm=0.75,
            memory_bytes=12 * 2**30,
            device_total_bytes=140 * 2**30,
            tps=41234,
            tflops=123.4,
            mfu=12.5,
        )
        fields.update(overrides)
        return train.step_log_line(**fields)

    def test_the_three_regexes_read_it(self) -> None:
        line = self.line()
        match = STEP_METRICS.search(line)
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)), 7)
        self.assertAlmostEqual(float(match.group(2)), 12.0, places=2)
        self.assertEqual(int(match.group(3).replace(",", "")), 41234)
        self.assertAlmostEqual(
            float(LOSS_METRIC.search(line).group(2)), 2.5, places=4
        )
        self.assertAlmostEqual(
            float(GRAD_NORM_METRIC.search(line).group(2)), 0.75, places=4
        )

    def test_a_rank_without_a_loss_prints_no_loss_field(self) -> None:
        """Only the last pipeline stage computes one, and nothing broadcasts.

        ``results.py`` reads the trajectory from ``loss_visible_rank``,
        which is a last-stage rank, so an absent field costs nothing and a
        sentinel would add a constant that is not a loss.
        """
        line = self.line(loss=None)
        self.assertIsNone(LOSS_METRIC.search(line))
        self.assertIsNotNone(STEP_METRICS.search(line))
        self.assertIsNotNone(GRAD_NORM_METRIC.search(line))

    def test_a_skipped_step_prints_nan(self) -> None:
        line = self.line(grad_norm=None)
        self.assertEqual(GRAD_NORM_METRIC.search(line).group(2), "nan")

    def test_tokens_per_second_divides_by_the_pipeline_degree(self) -> None:
        """The published figure is per device, as TorchTitan's is."""
        self.assertEqual(train.tokens_per_second(32768, 1.0, 1), 32768)
        self.assertEqual(train.tokens_per_second(32768, 1.0, 4), 8192)
        with self.assertRaises(ValueError):
            train.tokens_per_second(32768, 1.0, 0)

    def test_loss_value_reads_megatron_s_dict(self) -> None:
        self.assertIsNone(train.loss_value({}))
        self.assertAlmostEqual(
            train.loss_value({"lm loss": torch.tensor(1.25)}), 1.25
        )
        self.assertIsNone(train.loss_value({"skipped iterations": 0}))


class LossBroadcastTest(unittest.TestCase):
    """The loss every rank prints, and the path it comes through.

    Only the last pipeline stage computes a loss. A rank without one would
    print no loss field, and ``benchmarks/e2e/results.py`` reads the
    trajectory from whichever rank it selects. So the driver broadcasts.
    """

    def test_the_loss_is_unchanged_without_a_process_group(self) -> None:
        """One rank is the whole pipeline, so there is nobody to ask."""
        self.assertEqual(train.broadcast_pipeline_loss(1.25), 1.25)
        self.assertIsNone(train.broadcast_pipeline_loss(None))

    def test_the_step_line_takes_its_loss_from_the_broadcast(self) -> None:
        """The shim must not print this rank's own empty ``loss_dict``.

        A rank that is not the last stage holds no loss. Printing
        ``loss_value(loss_dict)`` directly gives that rank a line with no
        loss field, which ``LOSS_METRIC`` does not match.
        """
        inner = [
            const
            for const in train.install_step_log_shim.__code__.co_consts
            if isinstance(const, types.CodeType)
            and const.co_name == "replacement"
        ]
        self.assertEqual(len(inner), 1)
        self.assertIn("broadcast_pipeline_loss", inner[0].co_names)


class ResolvedDegreeTest(unittest.TestCase):
    """The provider refuses a mesh Megatron built two different ways.

    Megatron derives the data-parallel degree twice: its parser writes
    ``args.data_parallel_size``, and ``initialize_model_parallel`` builds
    the group ``mpu.get_data_parallel_world_size()`` reports. The marker
    line arm rule 12 matches states the first, and the token slice uses the
    second. A disagreement puts the wrong shard on this rank while the log
    names the right mesh, and every validation rule passes.
    """

    def _provider(self, *, parser_degree, group_degree, group_rank=0):
        """Call the provider with both degrees stubbed."""
        megatron_core = types.ModuleType("megatron.core")
        megatron_core.mpu = SimpleNamespace(
            get_data_parallel_rank=lambda: group_rank,
            get_data_parallel_world_size=lambda: group_degree,
        )
        megatron_training = types.ModuleType("megatron.training")
        megatron_training.get_args = lambda: SimpleNamespace(
            data_parallel_size=parser_degree,
            bench_seq_len=8,
            train_iters=1,
            bench_local_batch_size=1,
            bench_rows_per_sample=1,
        )
        saved = {
            name: sys.modules.get(name)
            for name in ("megatron", "megatron.core", "megatron.training")
        }
        sys.modules["megatron"] = types.ModuleType("megatron")
        sys.modules["megatron.core"] = megatron_core
        sys.modules["megatron.training"] = megatron_training
        try:
            return data.train_valid_test_datasets_provider(None)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_a_disagreement_raises_before_a_token_is_read(self) -> None:
        with self.assertRaises(RuntimeError) as caught:
            self._provider(parser_degree=2, group_degree=1)
        message = str(caught.exception)
        self.assertIn("data-parallel group of 1", message)
        self.assertIn("say 2", message)

    def test_the_other_direction_raises_too(self) -> None:
        """A group wider than the arguments is equally wrong."""
        with self.assertRaises(RuntimeError) as caught:
            self._provider(parser_degree=1, group_degree=4)
        self.assertIn("data-parallel group of 4", str(caught.exception))

    def test_agreement_reaches_the_iterator(self) -> None:
        """The check must not refuse an honest mesh."""
        try:
            result = self._provider(parser_degree=1, group_degree=1)
        except Exception as error:  # pragma: no cover - dataset dependent
            raise unittest.SkipTest(f"the c4_test stream is unavailable: {error}")
        self.assertEqual(len(result), 3)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])


class DataParallelMarkerTest(unittest.TestCase):
    """Arm rule 12's data-parallel half, against the real wrapper.

    Arm rule 13 cannot carry this axis alone. Stock Megatron all-reduces
    the loss over its data-parallel group on every step, so the NCCL
    marker appears whether or not a gradient was reduced.
    """

    def megatron_symbols(self):
        """Megatron's own module and class, or a skip."""
        try:
            bootstrap.prepare()
            import megatron.training.training as megatron_training
            from megatron.core.distributed import DistributedDataParallel
        except Exception as error:  # pragma: no cover - host dependent
            raise unittest.SkipTest(f"megatron is not importable: {error}")
        return megatron_training, DistributedDataParallel

    def wrapper_classes(self):
        """The two classes Megatron picks between, or a skip.

        ``FullyShardedDataParallel`` is a **factory function**, not a
        class, so it is not one of them: its own docstring says to use the
        version classes for a type check.
        """
        try:
            bootstrap.prepare()
            from megatron.core.distributed import DistributedDataParallel
            from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
                FullyShardedDataParallelV1,
            )
        except Exception as error:  # pragma: no cover - host dependent
            raise unittest.SkipTest(f"megatron is not importable: {error}")
        return DistributedDataParallel, FullyShardedDataParallelV1

    @contextlib.contextmanager
    def expert_group_of(self, size):
        """Report ``size`` from ``mpu.get_expert_model_parallel_world_size``.

        The accessor prefers the module global over the process group, so
        the setter reaches it without ``torch.distributed``.

        **The group path is therefore unexercised here.** A real run has
        no module global -- nothing in Megatron's training path calls the
        setter -- so the accessor reads the group
        ``initialize_model_parallel`` built. This proves the shim calls the
        accessor and compares its answer, not that the group is right.
        """
        import megatron.core.parallel_state as mpu

        original = mpu.get_expert_model_parallel_world_size()
        mpu.set_expert_model_parallel_world_size(size)
        try:
            yield
        finally:
            mpu.set_expert_model_parallel_world_size(original or None)

    def run_shim(
        self,
        chunk,
        *,
        dp=2,
        ep=1,
        built_ep=None,
        optimizer="DistributedOptimizer",
        members=None,
    ):
        """Install the shim over a stub and return what it printed.

        ``optimizer`` is the class NAME the stub optimizer carries, because
        the shim reads ``type(optimizer).__name__``. Megatron builds
        ``DistributedOptimizer`` under ``--use-distributed-optimizer`` and
        ``Float16OptimizerWithFloat16Params`` without it, and that class is
        the one observation that separates ``zero1`` from ``replicate``.

        ``members`` gives the stub Megatron's own ``chained_optimizers``
        attribute, which is what makes it a chain. A mixture of experts
        gets one, because Megatron builds one optimizer for each
        ``(optimizer_name, is_expert)`` bucket. ``None`` leaves the
        attribute absent, which is the bare optimizer of one bucket.
        """
        megatron_training, _ = self.megatron_symbols()
        original = megatron_training.setup_model_and_optimizer
        built_optimizer = type(optimizer, (), {})()
        if members is not None:
            built_optimizer.chained_optimizers = [
                type(member, (), {})() for member in members
            ]
        megatron_training.setup_model_and_optimizer = (
            lambda *args, **keywords: ([chunk], built_optimizer, None)
        )
        stream = io.StringIO()
        try:
            with self.expert_group_of(
                ep if built_ep is None else built_ep
            ):
                train.install_data_parallel_marker(
                    data_parallel_size=dp, expert_parallel_size=ep
                )
                with contextlib.redirect_stdout(stream):
                    megatron_training.setup_model_and_optimizer()
        finally:
            megatron_training.setup_model_and_optimizer = original
        return stream.getvalue().strip()

    @staticmethod
    def ddp_config(dense_sharding="replicate", **overrides):
        """The ``ddp_config`` the shim reads, as Megatron leaves it.

        **``overlap_grad_reduce`` is True under ``shard``, and the argv
        does not say so.** ``MegatronFSDP.__init__`` sets it on the object
        ``get_megatron_ddp_config`` built, because the wrapper holds the
        reference rather than a copy. A helper that defaulted this to
        False would feed the shim the value the test wants back, and the
        marker diff would then compare two copies of one assumption.

        ``data_parallel_sharding_strategy`` is ``optim_grads_params`` under
        both values, because Megatron's own argparse default puts it on
        every config. That is measured, not assumed: see
        ``test_a_replicated_run_reports_no_shard``.
        """
        base = dict(
            overlap_grad_reduce=(dense_sharding == "zero3"),
            grad_reduce_in_fp32=True,
            use_megatron_fsdp=(dense_sharding == "zero3"),
            data_parallel_sharding_strategy="optim_grads_params",
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_the_marker_is_silent_at_one_data_parallel_rank(self) -> None:
        """Below dp 2 there is no wrapper, and rule 12 asks for no line."""
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            train.install_data_parallel_marker(data_parallel_size=1)
        self.assertEqual(stream.getvalue(), "")

    def test_a_missing_wrapper_raises(self) -> None:
        """The hazard: two ranks that reduce nothing train two models.

        They report about twice the true throughput, and every other
        validation rule passes.
        """
        megatron_training, _ = self.megatron_symbols()
        original = megatron_training.setup_model_and_optimizer
        megatron_training.setup_model_and_optimizer = (
            lambda *args, **keywords: ([torch.nn.Linear(2, 2)], None, None)
        )
        try:
            train.install_data_parallel_marker(data_parallel_size=2)
            with self.assertRaises(RuntimeError) as caught:
                megatron_training.setup_model_and_optimizer()
        finally:
            megatron_training.setup_model_and_optimizer = original
        self.assertIn("no gradient is reduced", str(caught.exception))
        self.assertIn("no model chunk carries a data-parallel wrapper",
                      str(caught.exception))

    def test_the_line_reads_the_wrapper_and_not_the_arguments(self) -> None:
        """The values come from the live ``ddp_config``."""
        ddp_cls, _ = self.wrapper_classes()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = self.ddp_config()
        self.assertEqual(
            self.run_shim(chunk),
            train.DATA_PARALLEL_LINE.format(
                wrapper="DistributedDataParallel",
                dp=2,
                overlap=False,
                fp32=True,
                sharding="no_shard",
                expert=1,
                optimizer="DistributedOptimizer",
            ),
        )

    def test_a_chain_names_its_members(self) -> None:
        """A mixture of experts gets a chain, and the line names it.

        Megatron builds one optimizer for each
        ``(optimizer_name, is_expert)`` bucket, so every shape this suite
        runs has two. The ``30b-a3b`` cell of 2026-09-16 printed the outer
        class alone, and arm rule 12 refused it.
        """
        ddp_cls, _ = self.wrapper_classes()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = self.ddp_config()
        printed = self.run_shim(
            chunk,
            optimizer="ChainedOptimizer",
            members=("DistributedOptimizer", "DistributedOptimizer"),
        )
        self.assertIn(
            "optimizer=ChainedOptimizer[DistributedOptimizer])", printed
        )

    def test_a_chain_of_the_replicated_optimizer_reads_differently(
        self,
    ) -> None:
        """The hazard: ZeRO-0 must not pass under a ZeRO-1 label.

        A chain of ``Float16OptimizerWithFloat16Params`` is ZeRO-0. It
        carries the same outer class as a ZeRO-1 chain, so the members are
        what refuse it.
        """
        ddp_cls, _ = self.wrapper_classes()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = self.ddp_config()
        zero0 = self.run_shim(
            chunk,
            optimizer="ChainedOptimizer",
            members=("Float16OptimizerWithFloat16Params",) * 2,
        )
        zero1 = self.run_shim(
            chunk,
            optimizer="ChainedOptimizer",
            members=("DistributedOptimizer",) * 2,
        )
        self.assertIn(
            "optimizer=ChainedOptimizer[Float16OptimizerWithFloat16Params])",
            zero0,
        )
        self.assertNotEqual(zero0, zero1)
        self.assertNotIn("ChainedOptimizer[DistributedOptimizer]", zero0)

    def test_members_that_disagree_are_sorted_and_joined(self) -> None:
        """One dense shape covers both cases, and the order is stable."""
        ddp_cls, _ = self.wrapper_classes()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = self.ddp_config()
        printed = self.run_shim(
            chunk,
            optimizer="ChainedOptimizer",
            members=(
                "Float16OptimizerWithFloat16Params",
                "DistributedOptimizer",
            ),
        )
        self.assertIn(
            "optimizer=ChainedOptimizer[DistributedOptimizer+"
            "Float16OptimizerWithFloat16Params])",
            printed,
        )

    def test_an_empty_chain_raises(self) -> None:
        """A chain with no member names no class.

        ``ChainedOptimizer`` accepts an empty list, for a rank that holds
        no trainable parameter. The line would then state a ZeRO level
        nothing observed.
        """
        ddp_cls, _ = self.wrapper_classes()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = self.ddp_config()
        with self.assertRaises(RuntimeError) as caught:
            self.run_shim(chunk, optimizer="ChainedOptimizer", members=())
        self.assertIn("no member optimizer", str(caught.exception))

    def test_a_bare_optimizer_keeps_its_own_name(self) -> None:
        """The Megatron-FSDP branch returns one, so ``zero3`` reads it.

        This needs no megatron, because it reads the function alone.
        """
        self.assertEqual(
            train.optimizer_class_name(
                type("DistributedOptimizer", (), {})()
            ),
            "DistributedOptimizer",
        )

    def test_one_member_and_two_members_read_alike(self) -> None:
        """The expert degree must not move the printed string.

        ``get_megatron_optimizer`` appends a second member only where the
        model carries expert parameter groups, which this argv reaches
        above expert degree 1 alone. So a dense shape gives a chain of one
        and ``30b-a3b`` at ``ep`` 2 gives a chain of two. Deduplication
        collapses both to one name, so one expected string covers every
        shape.

        This needs no megatron, because it reads the function alone.
        """
        from benchmarks.e2e.megatron_stock.flags import (
            data_parallel_optimizer,
        )

        def chain(count):
            optimizer = type("ChainedOptimizer", (), {})()
            optimizer.chained_optimizers = [
                type("DistributedOptimizer", (), {})()
                for _ in range(count)
            ]
            return train.optimizer_class_name(optimizer)

        self.assertEqual(chain(1), chain(2))
        self.assertEqual(chain(1), data_parallel_optimizer("zero1"))

    def test_the_megatron_fsdp_wrapper_is_accepted_and_named(self) -> None:
        """The repair, stated as one assertion.

        ``DistributedDataParallel`` and ``FullyShardedDataParallelV1`` are
        siblings under ``_BaseDataParallel``, so the old narrow isinstance
        raised on an honest sharded run. Widening it alone would prove less
        than the narrow check did; the class name restores that, because it
        says which mechanism ran.
        """
        _, fsdp_cls = self.wrapper_classes()
        chunk = object.__new__(fsdp_cls)
        chunk.ddp_config = self.ddp_config("zero3")
        printed = self.run_shim(chunk, ep=2)
        self.assertIn(
            "Megatron-LM stock data parallel: FullyShardedDataParallelV1 "
            "over 2 ranks",
            printed,
        )
        self.assertIn("sharding_strategy=optim_grads_params", printed)
        self.assertIn("expert_parallel=2", printed)
        self.assertIn("overlap_grad_reduce=True", printed)

    def test_the_two_wrappers_are_siblings_and_not_one_a_subclass(
        self,
    ) -> None:
        """Read off Megatron itself. This is why the narrow check broke."""
        ddp_cls, fsdp_cls = self.wrapper_classes()
        from megatron.core.distributed.data_parallel_base import (
            _BaseDataParallel,
        )

        self.assertTrue(issubclass(ddp_cls, _BaseDataParallel))
        self.assertTrue(issubclass(fsdp_cls, _BaseDataParallel))
        self.assertFalse(issubclass(fsdp_cls, ddp_cls))
        self.assertFalse(issubclass(ddp_cls, fsdp_cls))

    def test_the_fsdp_name_megatron_exports_is_a_function(self) -> None:
        """``isinstance`` against it raises ``TypeError``. Never write it."""
        try:
            bootstrap.prepare()
            from megatron.core.distributed import FullyShardedDataParallel
        except Exception as error:  # pragma: no cover - host dependent
            raise unittest.SkipTest(f"megatron is not importable: {error}")
        self.assertFalse(isinstance(FullyShardedDataParallel, type))
        with self.assertRaises(TypeError):
            isinstance(object(), FullyShardedDataParallel)

    def test_a_replicated_run_reports_no_shard(self) -> None:
        """Megatron's argparse default is not the value the run acts on.

        ``data_parallel_sharding_strategy`` reaches every ``ddp_config``,
        but the optimizer reads it only under ``use_megatron_fsdp``. A line
        built from the raw field would say this run sharded.
        """
        ddp_cls, _ = self.wrapper_classes()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = self.ddp_config("replicate")
        printed = self.run_shim(chunk)
        self.assertIn("sharding_strategy=no_shard", printed)
        self.assertNotIn("optim_grads_params", printed)

    def test_the_line_names_the_optimizer_class_megatron_built(self) -> None:
        """The field that separates ``zero1`` from ``replicate``.

        ``--use-distributed-optimizer`` alone gives ZeRO-1, and it leaves
        the wrapper at ``DistributedDataParallel``. So the two values share
        a wrapper name, and only the optimizer class tells them apart.
        """
        from benchmarks.e2e.megatron_stock.flags import (
            DATA_PARALLEL_OPTIMIZERS,
            DATA_PARALLEL_WRAPPERS,
        )

        self.assertEqual(
            DATA_PARALLEL_WRAPPERS["zero1"],
            DATA_PARALLEL_WRAPPERS["replicate"],
        )
        self.assertNotEqual(
            DATA_PARALLEL_OPTIMIZERS["zero1"],
            DATA_PARALLEL_OPTIMIZERS["replicate"],
        )
        ddp_cls, _ = self.wrapper_classes()
        for value in ("replicate", "zero1"):
            with self.subTest(dense_sharding=value):
                chunk = object.__new__(ddp_cls)
                chunk.ddp_config = self.ddp_config(value)
                printed = self.run_shim(
                    chunk, optimizer=DATA_PARALLEL_OPTIMIZERS[value]
                )
                self.assertIn(
                    f"optimizer={DATA_PARALLEL_OPTIMIZERS[value]}", printed
                )

    def test_a_missing_optimizer_raises(self) -> None:
        """An absent optimizer is no observation of the ZeRO level.

        The line would then name no class, and the run would record a ZeRO
        level nothing observed. It raises instead, exactly as the absent
        wrapper does.
        """
        ddp_cls, _ = self.wrapper_classes()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = self.ddp_config()
        megatron_training, _ = self.megatron_symbols()
        original = megatron_training.setup_model_and_optimizer
        megatron_training.setup_model_and_optimizer = (
            lambda *args, **keywords: ([chunk], None, None)
        )
        try:
            train.install_data_parallel_marker(data_parallel_size=2)
            with self.assertRaises(RuntimeError) as caught:
                megatron_training.setup_model_and_optimizer()
        finally:
            megatron_training.setup_model_and_optimizer = original
        self.assertIn("no line can name the class", str(caught.exception))

    def test_an_expert_group_that_contradicts_the_arguments_raises(
        self,
    ) -> None:
        """The argv states the degree twice and proves it neither time.

        ``initialize_model_parallel`` has run by now, so the group is the
        one the run really has. This is the cross-check ``data.py``
        already applies to the data-parallel degree.
        """
        ddp_cls, _ = self.wrapper_classes()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = self.ddp_config()
        with self.assertRaises(RuntimeError) as caught:
            self.run_shim(chunk, ep=2, built_ep=1)
        message = str(caught.exception)
        self.assertIn("expert-model-parallel group of 1", message)
        self.assertIn("say 2", message)

    def test_the_matching_profile_marker_is_the_line_the_shim_prints(
        self,
    ) -> None:
        """The character-for-character diff, for every value.

        A one-character difference fails a real eight-GPU run at arm rule
        12, hours after it started. The ``30b-a3b`` cell of 2026-09-16
        failed on two fields at once, so this loop covers the precision
        axis beside the dense-sharding one.

        **The stub optimizer follows the dense-sharding value.**
        ``get_megatron_optimizer`` ends its standard path with an
        unconditional ``ChainedOptimizer(optimizers)``, and ``replicate``
        and ``zero1`` take that path. ``zero3`` takes the Megatron-FSDP
        branch, which returns its one optimizer bare at a single model
        chunk.
        """
        from benchmarks.e2e.megatron_stock.flags import (
            CHAINED_OPTIMIZER,
            DATA_PARALLEL_OPTIMIZERS,
            grad_reduce_in_fp32,
        )
        from benchmarks.e2e.validation import VALIDATION_PROFILES

        profile = VALIDATION_PROFILES["megatron_stock"]
        ddp_cls, fsdp_cls = self.wrapper_classes()
        for spec, cls in (
            (PP4_SPEC, ddp_cls),
            (SHARDED_PP4_SPEC, fsdp_cls),
            (EXPERT_PP4_SPEC, fsdp_cls),
        ):
            # ``lean`` needs a sharded dense value: Megatron asserts
            # use_distributed_optimizer under the precision-aware
            # optimizer, and ``_resolve_run`` refuses the other pair.
            precisions = (
                ("stock",)
                if spec.dense_sharding == "replicate"
                else ("stock", "lean")
            )
            # Megatron-FSDP returns its single optimizer bare. Every other
            # value reaches the standard path, which always chains.
            chained = spec.dense_sharding != "zero3"
            inner = DATA_PARALLEL_OPTIMIZERS[spec.dense_sharding]
            for precision in precisions:
                with self.subTest(
                    dense_sharding=spec.dense_sharding,
                    ep=spec.ep,
                    precision=precision,
                ):
                    chunk = object.__new__(cls)
                    chunk.ddp_config = self.ddp_config(
                        spec.dense_sharding,
                        grad_reduce_in_fp32=grad_reduce_in_fp32(precision),
                    )
                    printed = self.run_shim(
                        chunk,
                        dp=spec.dp,
                        ep=spec.ep,
                        optimizer=CHAINED_OPTIMIZER if chained else inner,
                        members=(inner, inner) if chained else None,
                    )
                    markers = profile.parallelism_markers(
                        spec, BATCH_32, precision
                    )
                    self.assertEqual(printed, markers[1])

    def test_an_expert_group_that_does_not_exist_raises(self) -> None:
        """``get_expert_model_parallel_world_size`` returns 0 without a
        group.

        ``initialize_model_parallel`` has run by the time this replacement
        fires, so 0 means the mesh Megatron built is not the mesh it says
        it built. The comparison against the argument catches it.
        """
        ddp_cls, _ = self.wrapper_classes()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = self.ddp_config()
        with self.assertRaisesRegex(RuntimeError, "group of 0 rank"):
            self.run_shim(chunk, ep=1, built_ep=0)

    def fsdp_constructor_source(self):
        """``MegatronFSDP.__init__``'s own source, or a skip.

        **The source of that one function, not of the module.** A bump
        that moved either statement below into another method would still
        satisfy a whole-file search, and the mutation only reaches the
        object the shim reads when it runs in the constructor.

        Building a real wrapper needs a process group and a device, so
        the text is what a CPU host can check.
        """
        import inspect

        try:
            bootstrap.prepare()
            from megatron.core.distributed.fsdp.src.megatron_fsdp import (
                megatron_fsdp,
            )
        except Exception as error:  # pragma: no cover - host dependent
            raise unittest.SkipTest(f"megatron is not importable: {error}")
        return inspect.getsource(megatron_fsdp.MegatronFSDP.__init__)

    def adapter_constructor_source(self):
        """``FullyShardedDataParallelV1.__init__``'s own source, or a skip.

        The middle link of the chain below. Megatron's training loop hands
        its ``ddp_config`` to THIS constructor, which then hands it on.
        """
        import inspect

        try:
            bootstrap.prepare()
            from megatron.core.distributed.fsdp import mcore_fsdp_adapter
        except Exception as error:  # pragma: no cover - host dependent
            raise unittest.SkipTest(f"megatron is not importable: {error}")
        return inspect.getsource(
            mcore_fsdp_adapter.FullyShardedDataParallelV1.__init__
        )

    def test_megatron_fsdp_turns_the_grad_overlap_on_in_place(self) -> None:
        """The fact ``DATA_PARALLEL_OVERLAP`` states, read off Megatron.

        ``MegatronFSDP.__init__`` mutates the ``ddp_config`` it was handed
        rather than a copy, so the wrapper reports True where the argv
        says nothing. A marker that pinned the argument would fail every
        honest sharded run.
        """
        source = self.fsdp_constructor_source()
        self.assertIn(
            "self.ddp_config.overlap_grad_reduce = True", source
        )
        # Every trailing newline below is the point. Without it the match
        # also accepts ``= ddp_config.copy()``, which is exactly the edit
        # that would break the marker while the assertion stayed green.
        self.assertIn("self.ddp_config = ddp_config\n", source)

    def test_the_mutated_config_reaches_the_object_the_shim_reads(
        self,
    ) -> None:
        """**Three links carry the mutation, and the test above pins one.**

        ``install_data_parallel_marker`` reads
        ``FullyShardedDataParallelV1.ddp_config``. The mutation happens in
        ``MegatronFSDP.__init__``, which is a different object. It reaches
        the shim only because the adapter keeps the reference it was given
        AND hands that same object on:

        1. the adapter stores it -- ``self.ddp_config = ddp_config``;
        2. the adapter forwards it -- ``ddp_config=ddp_config`` into
           ``MegatronFSDP``;
        3. ``MegatronFSDP`` stores it, then mutates it. The test above
           pins link 3.

        A submodule bump that copied at link 1 or link 2 leaves that test
        green, leaves ``DATA_PARALLEL_OVERLAP`` saying True, and fails arm
        rule 12 hours into the first sharded eight-GPU cell. This pins the
        other two.
        """
        source = self.adapter_constructor_source()
        self.assertIn("self.ddp_config = ddp_config\n", source)
        # Anchored to the call, not to the keyword. A bare search for
        # ``ddp_config=ddp_config`` would be satisfied by any other
        # constructor this method happens to call with that keyword.
        self.assertRegex(
            source, r"MegatronFSDP\(\s*ddp_config=ddp_config,"
        )

    def test_the_overlap_table_reads_megatrons_own_guard(self) -> None:
        """The table is derived, and this pins what it derives from.

        Megatron keys the mutation on the sharding **strategy**, not on
        this repo's dense-sharding value. ``MEGATRON_SHARDING_STRATEGY``
        is a documented reversal target, so a hand-written table would
        keep saying True after somebody moved that constant to a strategy
        the guard does not hold. This asserts the guard's own list.
        """
        source = self.fsdp_constructor_source()
        self.assertIn(
            'data_parallel_sharding_strategy in '
            '["optim_grads_params", "optim_grads"]',
            source,
        )
        self.assertEqual(
            set(MEGATRON_FSDP_GRAD_OVERLAP_STRATEGIES),
            {"optim_grads_params", "optim_grads"},
        )
        # Derived, so it moves with the strategy rather than beside it.
        for value, strategy in SHARDING_STRATEGIES.items():
            with self.subTest(dense_sharding=value):
                self.assertIs(
                    DATA_PARALLEL_OVERLAP[value],
                    strategy in MEGATRON_FSDP_GRAD_OVERLAP_STRATEGIES,
                )
        self.assertTrue(DATA_PARALLEL_OVERLAP["zero3"])
        # False for two reasons: no_shard is not in the guard's list, and
        # no Megatron-FSDP wrapper exists under this value at all.
        self.assertFalse(DATA_PARALLEL_OVERLAP["replicate"])
        # zero1 is False for both of those reasons too. It sends no
        # --use-megatron-fsdp, so nothing mutates the config in place.
        self.assertFalse(DATA_PARALLEL_OVERLAP["zero1"])
        self.assertNotIn(
            SHARDING_STRATEGIES["replicate"],
            MEGATRON_FSDP_GRAD_OVERLAP_STRATEGIES,
        )
        self.assertNotIn("--use-megatron-fsdp", flags_for("1b", PP4_SPEC))

    def test_the_shim_puts_megatron_back(self) -> None:
        """One wrap, then the module holds Megatron's own function again."""
        megatron_training, ddp_cls = self.megatron_symbols()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = self.ddp_config(
            overlap_grad_reduce=True, grad_reduce_in_fp32=False
        )
        # The shim reads the optimizer beside the model and refuses an
        # absent one, so the stub returns the class Megatron would build.
        optimizer = type("DistributedOptimizer", (), {})()
        stub = lambda *args, **keywords: (  # noqa: E731
            [chunk],
            optimizer,
            None,
        )
        original = megatron_training.setup_model_and_optimizer
        megatron_training.setup_model_and_optimizer = stub
        try:
            with self.expert_group_of(1):
                train.install_data_parallel_marker(data_parallel_size=2)
                with contextlib.redirect_stdout(io.StringIO()):
                    megatron_training.setup_model_and_optimizer()
            self.assertIs(
                megatron_training.setup_model_and_optimizer, stub
            )
        finally:
            megatron_training.setup_model_and_optimizer = original


class HarnessArgumentTest(unittest.TestCase):
    """The harness group parses the flags ``flags.py`` emits."""

    def test_the_group_accepts_the_emitted_flags(self) -> None:
        import argparse

        parser = train.add_bench_args(
            argparse.ArgumentParser(allow_abbrev=False)
        )
        emitted = flags_for("1b", PP4_SPEC)
        bench_only, index = [], 0
        while index < len(emitted):
            token = emitted[index]
            if token.startswith("--bench-"):
                bench_only.extend(emitted[index : index + 2])
                index += 2
            else:
                index += 1
        parsed = parser.parse_args(bench_only)
        self.assertEqual(parsed.bench_arm_dir, Path("/tmp/arm"))
        self.assertEqual(parsed.bench_model_size, "1b")
        self.assertEqual(parsed.bench_local_batch_size, 32)
        self.assertEqual(parsed.bench_profile_freq, 20)
        self.assertEqual(parsed.bench_profiler_warmup, 5)
        self.assertEqual(parsed.bench_profiler_active, 5)
        self.assertEqual(parsed.bench_pp_schedule, "1F1B")
        self.assertEqual(parsed.bench_seq_len, BATCH_32.seq_len)
        self.assertEqual(parsed.bench_rows_per_sample, 4)
        self.assertEqual(
            parsed.bench_min_trace_windows, BATCH_32.min_trace_windows
        )
        # The default argv omits the p2p flag, and the parser fills in
        # the value the flag list stands for.
        self.assertEqual(parsed.bench_batch_p2p_sync, "on")

    def test_the_group_parses_the_p2p_flag_and_refuses_another_value(
        self,
    ) -> None:
        import argparse

        parser = train.add_bench_args(
            argparse.ArgumentParser(allow_abbrev=False)
        )
        emitted = flags_for("1b", PP4_SPEC, megatron_p2p_sync="off")
        bench_only, index = [], 0
        while index < len(emitted):
            token = emitted[index]
            if token.startswith("--bench-"):
                bench_only.extend(emitted[index : index + 2])
                index += 2
            else:
                index += 1
        parsed = parser.parse_args(bench_only)
        self.assertEqual(parsed.bench_batch_p2p_sync, "off")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                bench_only[:-2] + [BENCH_BATCH_P2P_SYNC, "maybe"]
            )


# --------------------------------------------------------------------------
# model_builder.py
# --------------------------------------------------------------------------


class PipelineShapeAgreementTest(unittest.TestCase):
    """The invariant the packing exists to hold, against Megatron itself.

    Megatron flattens a microbatch that carries ``cu_seqlens`` into one
    row, and then allocates its pipeline receive buffer from
    ``get_tensor_shapes``, which reads ``--seq-length`` and
    ``--micro-batch-size`` and validates nothing. The two shapes hold the
    same number of elements, so a mismatch is a silent permutation across
    the stage boundary, not an error.
    """

    def megatron_functions(self):
        """Megatron's own merge and shape functions, or a skip.

        **Called from inside a test, never at module scope**, for the
        reason ``ModelBuilderTest.megatron_symbols`` gives.
        """
        try:
            bootstrap.prepare()
            from megatron.core.pipeline_parallel.schedules import (
                get_tensor_shapes,
            )
            from megatron.core.utils import (
                flatten_batch_for_packed_sequences,
            )
        except Exception as error:  # pragma: no cover - host dependent
            raise unittest.SkipTest(f"megatron is not importable: {error}")
        return flatten_batch_for_packed_sequences, get_tensor_shapes

    def test_the_activation_and_the_receive_buffer_agree(self) -> None:
        flatten, tensor_shapes = self.megatron_functions()

        class Config:
            variable_seq_lengths = False
            sequence_parallel = False
            hidden_size = 8

        class Group:
            def size(self):
                return 1

        seq_len = 16
        for rows in (1, 2, 4):
            with self.subTest(rows_per_sample=rows):
                samples = synthetic_samples(
                    4 * rows, seq_len, [[16], [4, 12], [8, 4, 4]]
                )
                iterator = data.StockReplayIterator(
                    samples, rows_per_sample=rows, seq_len=seq_len
                )
                microbatch = next(iterator)
                flat = flatten(
                    {
                        key: (
                            value.clone()
                            if torch.is_tensor(value)
                            else value
                        )
                        for key, value in microbatch.items()
                    }
                )
                # The first stage emits [s, b, h] of the flattened tokens.
                activation = (
                    flat["tokens"].shape[1],
                    1,
                    Config.hidden_size,
                )
                buffers = tensor_shapes(
                    seq_length=rows * seq_len,
                    micro_batch_size=1,
                    decoder_seq_length=None,
                    config=Config,
                    tp_group=Group(),
                    cp_group=Group(),
                )
                self.assertEqual(list(buffers), [activation])

    def test_the_merge_returns_the_offsets_unchanged(self) -> None:
        """At one row there is nothing to merge, and nothing may move."""
        flatten, _ = self.megatron_functions()
        seq_len = 16
        samples = synthetic_samples(8, seq_len, [[16], [4, 12], [8, 4, 4]])
        iterator = data.StockReplayIterator(
            samples, rows_per_sample=2, seq_len=seq_len
        )
        for _ in range(iterator.microbatch_count):
            microbatch = next(iterator)
            before = microbatch["cu_seqlens"][0]
            flat = flatten(
                {
                    key: (
                        value.clone() if torch.is_tensor(value) else value
                    )
                    for key, value in microbatch.items()
                }
            )
            after = flat["cu_seqlens"][0]
            # The merge strips the padding and keeps every real offset.
            self.assertEqual(
                after.tolist(),
                [
                    int(entry)
                    for entry in before[
                        : (before == iterator.packed_len)
                        .nonzero()[0]
                        .item()
                        + 1
                    ]
                ],
            )


class ModelBuilderTest(unittest.TestCase):
    """The builder path Megatron imports, and the count it must agree with."""

    def megatron_symbols(self):
        """Megatron's own classes, or a skip.

        **Called from inside a test, never at module scope.**
        ``add_megatron_to_path`` puts the checkout at ``sys.path[0]``, and
        Megatron-LM ships a ``tests/`` directory of its own, so a
        module-scope call would shadow this suite's own package for every
        module discovered after it.
        """
        try:
            bootstrap.prepare()
            from megatron.training.models.gpt import (
                GPTModelBuilder,
                GPTModelConfig,
            )
        except Exception as error:  # pragma: no cover - host dependent
            raise unittest.SkipTest(f"megatron is not importable: {error}")
        return GPTModelBuilder, GPTModelConfig

    def test_the_builder_path_resolves_to_the_counting_builder(self) -> None:
        """``ModelConfig.get_builder_cls`` imports that dotted string.

        A typo there fails at model build, after the run claimed a GPU.
        """
        import importlib

        builder_cls, config_cls = self.megatron_symbols()
        from benchmarks.e2e.megatron_stock import model_builder

        module_path, _, class_name = (
            model_builder.BenchGPTModelConfig.builder.rpartition(".")
        )
        resolved = getattr(importlib.import_module(module_path), class_name)
        self.assertIs(resolved, model_builder.CountingGPTModelBuilder)
        self.assertTrue(
            issubclass(model_builder.BenchGPTModelConfig, config_cls)
        )
        self.assertTrue(
            issubclass(model_builder.CountingGPTModelBuilder, builder_cls)
        )

    def test_the_stage_counts_sum_to_the_model(self) -> None:
        """The identity the printed total rests on.

        The builder asserts one stage's count and then prints the whole
        model's. That is only honest because every stage runs the same
        assertion and the counts sum to ``param_count``.
        """
        for name, shape in PIPER_SHAPES.items():
            for degree in (1, 2, 4):
                if shape.n_layers % degree:
                    continue
                with self.subTest(size=name, pp=degree):
                    total = sum(
                        shape.stage_param_count(
                            pipeline_degree=degree, stage_index=stage
                        )
                        for stage in range(degree)
                    )
                    self.assertEqual(total, shape.param_count)


class DatasetHelperBuildTest(unittest.TestCase):
    """``ensure_dataset_helpers`` against the hazard it exists for.

    Megatron's ``compile_helpers`` calls ``sys.exit(1)`` when its ``make``
    fails, and no flag skips it. The Makefile asks the **system**
    ``python3-config`` for the output name, so the harness cannot predict
    that name from this interpreter alone. The function therefore writes
    every suffix a caller might ask for.
    """

    def _fake_checkout(self, tmp: pathlib.Path) -> pathlib.Path:
        datasets = tmp / "megatron" / "core" / "datasets"
        datasets.mkdir(parents=True)
        (datasets / "helpers.cpp").write_text("int main() { return 0; }\n")
        return tmp

    def test_a_checkout_without_the_source_asks_for_no_build(self) -> None:
        """A tree with no ``helpers.cpp`` is not an error. It is a no-op."""
        with tempfile.TemporaryDirectory() as raw:
            root = pathlib.Path(raw)
            (root / "megatron" / "core" / "datasets").mkdir(parents=True)
            self.assertEqual(bootstrap.ensure_dataset_helpers(root), ())

    def test_it_writes_the_name_this_interpreter_imports(self) -> None:
        """The venv's own suffix must be among the names it guarantees."""
        wanted = sysconfig.get_config_var("EXT_SUFFIX")
        with tempfile.TemporaryDirectory() as raw:
            root = self._fake_checkout(pathlib.Path(raw))
            with unittest.mock.patch.object(
                bootstrap, "_compile_dataset_helper", _fake_compile
            ):
                built = bootstrap.ensure_dataset_helpers(root)
        names = {path.name for path in built}
        self.assertIn(f"helpers_cpp{wanted}", names)

    def test_it_compiles_once_and_copies_the_rest(self) -> None:
        """Two names cost one compile. A second compile would be waste."""
        calls = []

        def counting(source, target):
            calls.append(target)
            return _fake_compile(source, target)

        with tempfile.TemporaryDirectory() as raw:
            root = self._fake_checkout(pathlib.Path(raw))
            with unittest.mock.patch.object(
                bootstrap, "_compile_dataset_helper", counting
            ):
                built = bootstrap.ensure_dataset_helpers(root)
            # Assert inside the block. The directory goes away at its end.
            self.assertEqual(len(calls), 1)
            for path in built:
                self.assertTrue(path.is_file())

    def test_a_current_target_asks_for_no_second_build(self) -> None:
        """The second call does nothing, so a run pays the build once."""
        with tempfile.TemporaryDirectory() as raw:
            root = self._fake_checkout(pathlib.Path(raw))
            with unittest.mock.patch.object(
                bootstrap, "_compile_dataset_helper", _fake_compile
            ):
                first = bootstrap.ensure_dataset_helpers(root)
                second = bootstrap.ensure_dataset_helpers(root)
        self.assertNotEqual(first, ())
        self.assertEqual(second, ())

    def test_a_source_newer_than_the_target_rebuilds(self) -> None:
        """A submodule bump changes helpers.cpp. A stale object must go."""
        with tempfile.TemporaryDirectory() as raw:
            root = self._fake_checkout(pathlib.Path(raw))
            source = root / "megatron" / "core" / "datasets" / "helpers.cpp"
            with unittest.mock.patch.object(
                bootstrap, "_compile_dataset_helper", _fake_compile
            ):
                bootstrap.ensure_dataset_helpers(root)
                os.utime(source, (time.time() + 60, time.time() + 60))
                again = bootstrap.ensure_dataset_helpers(root)
        self.assertNotEqual(again, ())


def _fake_compile(source: pathlib.Path, target: pathlib.Path) -> pathlib.Path:
    """Stand in for the compiler. The test checks names, not machine code."""
    target.write_bytes(b"\x7fELF fake")
    return target


if __name__ == "__main__":
    unittest.main()
