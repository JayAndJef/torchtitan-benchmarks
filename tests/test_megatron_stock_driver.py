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

import dataclasses
import sys
import types
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
                "--max-position-embeddings": shape.max_seq_len,
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

    def test_batch_sizes_follow_the_spec_and_the_workload(self) -> None:
        emitted = flags_for("1b", PP4_SPEC)
        self.assertEqual(value_after(emitted, "--micro-batch-size"), "4")
        self.assertEqual(value_after(emitted, "--global-batch-size"), "64")
        self.assertEqual(
            value_after(emitted, "--pipeline-model-parallel-size"), "4"
        )
        trivial = flags_for("1b", TRIVIAL_SPEC)
        self.assertEqual(value_after(trivial, "--micro-batch-size"), "1")
        self.assertEqual(value_after(trivial, "--global-batch-size"), "32")

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
            value_after(emitted, "--seq-length"), str(BATCH_32.seq_len)
        )
        self.assertEqual(value_after(emitted, "--seed"), str(BATCH_32.seed))

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
        self.samples = synthetic_samples(
            8, self.seq_len, [[16], [4, 12], [8, 4, 4]]
        )
        self.iterator = data.StockReplayIterator(
            self.samples, micro_batch_size=2, seq_len=self.seq_len
        )

    def test_the_dict_holds_the_declared_keys(self) -> None:
        microbatch = next(self.iterator)
        self.assertEqual(
            sorted(microbatch), sorted(data.MICROBATCH_KEYS)
        )
        self.assertIsNone(microbatch["attention_mask"])
        self.assertIsNone(microbatch["cu_seqlens_padded"])

    def test_the_dtypes_and_shapes_are_megatron_s(self) -> None:
        microbatch = next(self.iterator)
        for key in ("tokens", "labels", "position_ids"):
            with self.subTest(key=key):
                self.assertEqual(microbatch[key].dtype, torch.int64)
                self.assertEqual(
                    tuple(microbatch[key].shape), (2, self.seq_len)
                )
        self.assertEqual(microbatch["loss_mask"].dtype, torch.float32)
        self.assertEqual(
            tuple(microbatch["loss_mask"].shape), (2, self.seq_len)
        )
        self.assertTrue(bool((microbatch["loss_mask"] == 1).all()))
        self.assertEqual(microbatch["cu_seqlens"].dtype, torch.int32)
        self.assertEqual(microbatch["max_seqlen"].dtype, torch.int32)
        self.assertEqual(tuple(microbatch["max_seqlen"].shape), (2,))

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
                end = values.index(self.seq_len)
                real = values[: end + 1]
                self.assertEqual(
                    real, sorted(set(real)), "offsets must rise strictly"
                )
                self.assertEqual(real[-1], self.seq_len)
                self.assertEqual(
                    values[end + 1 :],
                    [self.seq_len] * (len(values) - end - 1),
                )

    def test_the_padded_width_is_the_widest_pack_of_this_rank(self) -> None:
        # [8, 4, 4] is the widest split, so four entries: 0, 8, 12, 16.
        self.assertEqual(self.iterator.padded_documents, 4)
        for _ in range(self.iterator.microbatch_count):
            self.assertEqual(
                tuple(next(self.iterator)["cu_seqlens"].shape), (2, 4)
            )

    def test_max_seqlen_is_the_longest_document_of_each_row(self) -> None:
        microbatch = next(self.iterator)
        # Rows 0 and 1 are the [16] and the [4, 12] splits.
        self.assertEqual(
            [int(v) for v in microbatch["max_seqlen"]], [16, 12]
        )

    def test_exhaustion_raises_rather_than_wrapping(self) -> None:
        """A wrap trains a second epoch under the first epoch's label."""
        for _ in range(self.iterator.microbatch_count):
            next(self.iterator)
        with self.assertRaises(RuntimeError) as caught:
            next(self.iterator)
        self.assertIn("exhausted", str(caught.exception))

    def test_the_rows_keep_the_stream_order(self) -> None:
        """Microbatch i holds samples 2i and 2i+1, in that order."""
        for index in range(self.iterator.microbatch_count):
            microbatch = next(self.iterator)
            for row in range(2):
                self.assertTrue(
                    bool(
                        (
                            microbatch["tokens"][row]
                            == self.samples[2 * index + row][0]
                        ).all()
                    )
                )

    def test_an_indivisible_sample_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            data.StockReplayIterator(
                self.samples[:7], micro_batch_size=2, seq_len=self.seq_len
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

    def test_a_data_parallel_mesh_prints_both_lines(self) -> None:
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
                "microbatches=8 stages=4",
                "Megatron-LM stock data parallel: DistributedDataParallel "
                "over 2 ranks (overlap_grad_reduce=False, "
                "grad_reduce_in_fp32=True)",
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

    def test_the_declared_run_is_accepted(self) -> None:
        train.refuse_unsupported_run(stock_args())
        train.refuse_unsupported_run(
            stock_args(
                world_size=8,
                pipeline_model_parallel_size=4,
                data_parallel_size=2,
                bench_pp_schedule="1F1B",
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


# --------------------------------------------------------------------------
# model_builder.py
# --------------------------------------------------------------------------


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


if __name__ == "__main__":
    unittest.main()
