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
    BENCH_FLAGS,
    BENCH_PP_SCHEDULE,
    OMITTED_FLAGS,
    microbatch_geometry,
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
BATCH_32 = dataclasses.replace(PIPER_1B_MEGATRON_WORKLOAD, local_batch_size=32)


def flags_for(shape_name, spec, workload=BATCH_32):
    return stock_megatron_flags(
        shape_by_name(shape_name),
        workload,
        spec,
        arm_dir="/tmp/arm",
        model_size=shape_name,
    )


def value_after(emitted, flag):
    """The token after ``flag``, or None when the flag is absent."""
    if flag not in emitted:
        return None
    return emitted[emitted.index(flag) + 1]


# --------------------------------------------------------------------------
# flags.py
# --------------------------------------------------------------------------

# Section 7 of PIPER_STOCK_MEGATRON_PLAN.md, transcribed. Every one of these
# was checked against the pinned Megatron-LM parser: argparse knows all of
# them, and parse_and_validate_args accepted the whole list at the trivial
# spec and at dp 2 x pp 4, for the 1b and the 9b shape.
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
        """Every ``--bench-`` flag except the pipeline schedule.

        The schedule is absent at pipeline degree 1, where the driver
        refuses it: a schedule there names a split that does not happen.
        """
        trivial = set(flags_for("1b", TRIVIAL_SPEC))
        pipelined = set(flags_for("1b", PP4_SPEC))
        for flag in BENCH_FLAGS:
            with self.subTest(flag=flag):
                self.assertIn(flag, pipelined)
                if flag != BENCH_PP_SCHEDULE:
                    self.assertIn(flag, trivial)
        self.assertNotIn(BENCH_PP_SCHEDULE, trivial)
        self.assertEqual(
            value_after(flags_for("1b", PP4_SPEC), BENCH_PP_SCHEDULE), "1F1B"
        )

    def test_no_declined_flag_is_emitted(self) -> None:
        """The five sharding flags, the two fusions and the rest stay out.

        Each is declined for a reason section 7 of the plan states. An
        emitted one would change what the arm measures without changing
        anything the manifest records.
        """
        for size in ("1b", "9b"):
            for spec in (TRIVIAL_SPEC, PP4_SPEC):
                emitted = set(flags_for(size, spec))
                for flag in OMITTED_FLAGS:
                    with self.subTest(size=size, pp=spec.pp, flag=flag):
                        self.assertNotIn(flag, emitted)

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
            "mode": (BATCH_32, TRIVIAL_SPEC, "cuda-graph", "compile mode"),
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
PLAN_MODE_PREFIX = "Megatron-LM stock training loop (mode="
PLAN_PARALLELISM_LINE = (
    "Megatron-LM stock parallelism: dp={dp} pp={pp} schedule={schedule} "
    "microbatches={microbatches} stages={stages}"
)
PLAN_DATA_PARALLEL_LINE = (
    "Megatron-LM stock data parallel: DistributedDataParallel over {dp} "
    "ranks (overlap_grad_reduce={overlap}, grad_reduce_in_fp32={fp32})"
)
PLAN_STAGE_SIZE_LINE = (
    "stock-megatron stage {stage}/{stages} local size: {count} parameters"
)
PLAN_MODEL_SIZE_LINE = (
    "Model qwen3_piper_{size} stock-megatron size: {total} total parameters"
)


def stock_args(**overrides):
    """An argument namespace shaped like the one Megatron resolves."""
    base = dict(
        bench_mode="default",
        bench_pp_schedule=None,
        bench_local_batch_size=32,
        bench_model_size="1b",
        bench_seq_len=1024,
        bench_rows_per_sample=32,
        bench_min_trace_windows=2,
        seq_length=32 * 1024,
        micro_batch_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        data_parallel_size=1,
        world_size=1,
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
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
        self.assertTrue(line.startswith(PLAN_MODE_PREFIX + "default,"))
        for field in (
            "main_params_dtype=torch.float32",
            "main_grads_dtype=torch.float32",
            "accumulate_allreduce_grads_in_fp32=True",
            "cross_entropy_loss_fusion=False",
            "moe_token_dispatcher_type=alltoall",
        ):
            with self.subTest(field=field):
                self.assertIn(field, line)
        self.assertEqual(line.count("\n"), 0)

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
                "Megatron-LM stock parallelism: dp=1 pp=4 schedule=1F1B "
                "microbatches=8 stages=4"
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
                "Megatron-LM stock parallelism: dp=2 pp=4 schedule=1F1B "
                "microbatches=8 stages=4"
            ],
        )


class DriverRefusalTest(unittest.TestCase):
    """A run this driver cannot honour must fail before it builds anything."""

    def test_a_mode_other_than_default_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            train.refuse_unsupported_run(stock_args(bench_mode="cuda-graph"))
        self.assertIn("--bench-mode", str(caught.exception))

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

    def test_the_line_reads_the_wrapper_and_not_the_arguments(self) -> None:
        """The values come from the live ``ddp_config``."""
        megatron_training, ddp_cls = self.megatron_symbols()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = SimpleNamespace(
            overlap_grad_reduce=False, grad_reduce_in_fp32=True
        )
        original = megatron_training.setup_model_and_optimizer
        megatron_training.setup_model_and_optimizer = (
            lambda *args, **keywords: ([chunk], None, None)
        )
        stream = io.StringIO()
        try:
            train.install_data_parallel_marker(data_parallel_size=2)
            with contextlib.redirect_stdout(stream):
                megatron_training.setup_model_and_optimizer()
        finally:
            megatron_training.setup_model_and_optimizer = original
        self.assertEqual(
            stream.getvalue().strip(),
            train.DATA_PARALLEL_LINE.format(
                dp=2, overlap=False, fp32=True
            ),
        )

    def test_the_shim_puts_megatron_back(self) -> None:
        """One wrap, then the module holds Megatron's own function again."""
        megatron_training, ddp_cls = self.megatron_symbols()
        chunk = object.__new__(ddp_cls)
        chunk.ddp_config = SimpleNamespace(
            overlap_grad_reduce=True, grad_reduce_in_fp32=False
        )
        stub = lambda *args, **keywords: ([chunk], None, None)  # noqa: E731
        original = megatron_training.setup_model_and_optimizer
        megatron_training.setup_model_and_optimizer = stub
        try:
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
        self.assertEqual(parsed.bench_mode, "default")
        self.assertEqual(parsed.bench_pp_schedule, "1F1B")
        self.assertEqual(parsed.bench_seq_len, BATCH_32.seq_len)
        self.assertEqual(parsed.bench_rows_per_sample, 4)
        self.assertEqual(
            parsed.bench_min_trace_windows, BATCH_32.min_trace_windows
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
