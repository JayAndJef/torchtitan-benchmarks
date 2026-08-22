"""CPU tests for the ``embedding_stage`` kernel scenario.

More of this scenario is testable without a GPU than of its neighbours,
because the titan arm is eager: ``build_embedding_stage_titan`` adds no
``torch.compile`` wrapper, so the tests below exercise the builder itself
rather than a hand-assembled stand-in.

``LayoutFreedomTests`` is the reason that matters. The scenario's first
version charged the titan arm a ``transpose(0, 1).contiguous()`` to match
megatron's, and the charge was wrong: at the THD packing our driver runs,
megatron's transpose is a **free view**, because PyTorch's contiguity test
skips size-1 dimensions. Those tests reproduce both sides of that fact on a
real module -- free at ``[1, T]``, a real allocation at ``[batch, seq_len]``
-- so the correction cannot be undone silently.

The mcore arm is **not** covered here and cannot be. It needs CUDA,
TransformerEngine and a megatron process group, and TE does not even import on
a host without the cuda-compat stack. Its guard is covered instead: the guard
reads its module by class *name*, so a stand-in with the same attributes
exercises every branch on any host.

The gate constants below mirror the ones the scenario declares. A test that
invented its own tolerance would pass while the run failed.
"""

import gc
import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from benchmarks.kernel.engine.run import resolve_symbol
from benchmarks.kernel.operations.embedding_stage import (
    MCORE_ARM_NAME,
    MCORE_MODULE_CLASS,
    EmbeddingStageInputs,
    _assert_layout_conversion_is_free,
    _assert_mcore_embedding,
    _assert_output_layout,
    _assert_parameters_released,
    _assert_titan_embedding,
    _embedding_stage_arm,
    _watch_released_parameters,
    build_embedding_stage_copy_floor,
    build_embedding_stage_mcore_base,
    build_embedding_stage_titan,
    embedding_stage_inputs,
    embedding_stage_reference,
    gathered_and_output,
    titan_embedding_module,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.shape import PiperShape

# Small enough to run in a second, and every mechanism under test is the one a
# GPU arm uses. A 64-row vocabulary against 32 tokens guarantees repeated ids,
# which is what makes the backward a real scatter-add rather than a permutation.
TINY = PiperShape.derived(name="tiny", dim=256, n_layers=2, vocab_size=64)
WORKLOAD = KernelWorkload(batch=2, seq_len=16)

BATCH, SEQ, DIM = WORKLOAD.batch, WORKLOAD.seq_len, TINY.dim
TOKENS = BATCH * SEQ

# The canonical layout both arms are compared in, and megatron's native one.
CANONICAL_SHAPE = (BATCH, SEQ, DIM)
MCORE_NATIVE_SHAPE = (TOKENS, 1, DIM)

# The rel_l2 the declaration gates on. The forward is an exact gather and lands
# at 0.0; the gradient is a bf16 reduction and lands near 2e-3 at this shape.
DECLARED_GATE = 2e-2

# The dotted paths the registry declaration names. The engine resolves these
# strings inside the worker, so a renamed builder is a run-time failure on the
# GPU rather than an import error here.
DECLARED_BUILDER_PATHS = (
    "benchmarks.kernel.operations.embedding_stage:embedding_stage_inputs",
    "benchmarks.kernel.operations.embedding_stage:embedding_stage_reference",
    "benchmarks.kernel.operations.embedding_stage"
    ":build_embedding_stage_copy_floor",
    "benchmarks.kernel.operations.embedding_stage"
    ":build_embedding_stage_mcore_base",
    "benchmarks.kernel.operations.embedding_stage:build_embedding_stage_titan",
)

# The modes both arms declare. There is deliberately no isolated "backward":
# the first backward allocates and zero-fills a dense [V, D] gradient and every
# later one accumulates into it, so a retained-graph backward would time two
# different amounts of traffic under one label.
DECLARED_MODES = {"forward", "forward_backward"}

# The gated outputs, named once. ``weight_grad_rows`` is the compaction the
# reference can afford; ``weight_grad_norm`` bounds a gross write outside those
# rows and, as the module docstring states with the arithmetic, nothing finer.
DECLARED_OUTPUTS = {"out", "weight_grad_rows", "weight_grad_norm"}


def make_inputs(seed: int = 0) -> EmbeddingStageInputs:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return embedding_stage_inputs(
        TINY, WORKLOAD, torch.device("cpu"), generator
    )


def rel_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    delta = value.double() - truth.double()
    norm = truth.double().norm()
    return (delta.norm() / norm).item() if norm else delta.norm().item()


class InputsTests(unittest.TestCase):
    def test_the_shared_tensors_have_the_shapes_both_engines_need(self) -> None:
        inputs = make_inputs()
        self.assertEqual(tuple(inputs.tokens.shape), (BATCH, SEQ))
        self.assertEqual(inputs.tokens.dtype, torch.int64)
        self.assertEqual(tuple(inputs.grad_out.shape), CANONICAL_SHAPE)
        self.assertEqual(inputs.grad_out.dtype, torch.bfloat16)
        self.assertEqual(tuple(inputs.weight.shape), (TINY.vocab_size, DIM))
        self.assertEqual(inputs.weight.dtype, torch.bfloat16)

    def test_every_token_id_addresses_a_row_of_the_table(self) -> None:
        """An out-of-range id is an index error on the GPU and nowhere here."""
        tokens = make_inputs().tokens
        self.assertGreaterEqual(int(tokens.min()), 0)
        self.assertLess(int(tokens.max()), TINY.vocab_size)

    def test_the_draw_selects_many_distinct_rows(self) -> None:
        """A degenerate draw would make the gather one broadcast row.

        The absolute number this scenario reports depends on how scattered the
        gather is, so a draw that collapsed onto a handful of rows would
        measure something else entirely.
        """
        inputs = make_inputs()
        self.assertGreater(inputs.touched_ids.numel(), TINY.vocab_size // 4)

    def test_touched_ids_are_the_sorted_unique_tokens(self) -> None:
        """The reference indexes a compact gradient by ``searchsorted``, which
        is exact only against a sorted, deduplicated id list."""
        inputs = make_inputs()
        self.assertEqual(inputs.touched_ids.dtype, torch.int64)
        self.assertTrue(
            torch.equal(inputs.touched_ids, torch.unique(inputs.tokens))
        )
        self.assertTrue(
            torch.equal(inputs.touched_ids, inputs.touched_ids.sort().values)
        )

    def test_the_floor_byte_count_is_the_gathers_read_and_write(self) -> None:
        """Two passes, not four. The scenario charges no layout copy to either
        engine, so a floor at twice an arm's traffic would flatter it."""
        inputs = make_inputs()
        self.assertEqual(
            inputs.copy_bytes,
            2 * inputs.grad_out.numel() * inputs.grad_out.element_size(),
        )

    def test_one_seed_gives_one_set_of_tensors(self) -> None:
        """Every worker rebuilds the inputs, so they must not drift."""
        first, second = make_inputs(3), make_inputs(3)
        self.assertTrue(torch.equal(first.tokens, second.tokens))
        self.assertTrue(torch.equal(first.grad_out, second.grad_out))
        self.assertTrue(torch.equal(first.weight, second.weight))


class ReferenceTests(unittest.TestCase):
    """The fp64 truth both arms are gated against.

    An error here fails both arms identically and invisibly, so the reference
    is checked against an independent autograd computation rather than against
    itself.
    """

    def test_the_forward_is_the_exact_gather(self) -> None:
        """A gather copies rows, so fp64 promotion after the fact loses
        nothing and the reference is bit-exact, not merely close."""
        inputs = make_inputs()
        reference = embedding_stage_reference(TINY, WORKLOAD, inputs)
        expected = inputs.weight.double()[inputs.tokens]
        self.assertTrue(torch.equal(reference["out"], expected))

    def test_the_gradient_matches_a_full_table_autograd_backward(self) -> None:
        """The compaction is checked against the thing it compacts."""
        inputs = make_inputs()
        reference = embedding_stage_reference(TINY, WORKLOAD, inputs)
        weight = inputs.weight.double().detach().requires_grad_()
        out = weight[inputs.tokens]
        torch.autograd.backward(out, inputs.grad_out.double())
        expected = weight.grad.index_select(0, inputs.touched_ids)
        self.assertLess(rel_l2(reference["weight_grad_rows"], expected), 1e-12)

    def test_the_untouched_rows_of_that_gradient_are_exactly_zero(self) -> None:
        """What makes the compaction lossless. It is *not* what makes the
        whole-table norm sensitive; see the next test."""
        inputs = make_inputs()
        weight = inputs.weight.double().detach().requires_grad_()
        torch.autograd.backward(
            weight[inputs.tokens], inputs.grad_out.double()
        )
        untouched = torch.ones(
            TINY.vocab_size, dtype=torch.bool
        ).index_fill_(0, inputs.touched_ids, False)
        self.assertEqual(float(weight.grad[untouched].abs().sum()), 0.0)

    def test_the_whole_table_norm_moves_as_the_docstring_says_it_does(
        self,
    ) -> None:
        """The honest limit of ``weight_grad_norm``, pinned as arithmetic.

        Contaminating ``k`` untouched rows with rows of typical magnitude
        moves a Frobenius norm over ``U`` touched rows by
        ``sqrt(1 + k/U) - 1``. Two consequences the module docstring states
        and this pins: the sensitivity is a function of ``U`` alone, and at
        the production ``U`` one stray row is three orders below the gate.
        """
        inputs = make_inputs()
        rows = embedding_stage_reference(TINY, WORKLOAD, inputs)[
            "weight_grad_rows"
        ]
        clean = torch.linalg.vector_norm(rows)
        stray = rows[0].norm()
        measured = float(
            (torch.sqrt(clean**2 + stray**2) - clean) / clean
        )
        touched = rows.shape[0]
        self.assertAlmostEqual(
            measured, (1.0 + 1.0 / touched) ** 0.5 - 1.0, places=2
        )
        # The production draw touches ~4041 of 151936 rows at the default
        # workload. One stray row is invisible; ~163 are needed to reach 2e-2.
        production = 4041
        self.assertLess(
            (1.0 + 1.0 / production) ** 0.5 - 1.0, DECLARED_GATE / 100
        )
        self.assertGreater(
            (1.0 + 163.0 / production) ** 0.5 - 1.0, DECLARED_GATE * 0.9
        )

    def test_the_reference_names_every_gated_output_in_fp64(self) -> None:
        reference = embedding_stage_reference(TINY, WORKLOAD, make_inputs())
        self.assertEqual(set(reference), DECLARED_OUTPUTS)
        for tensor in reference.values():
            self.assertEqual(tensor.dtype, torch.float64)
        self.assertEqual(tuple(reference["weight_grad_norm"].shape), (1,))


class _FakeLanguageModelEmbedding(nn.Module):
    """megatron's forward, reduced to the two lines this scenario turns on.

    ``word_embeddings`` is a real ``nn.Embedding`` so the forward hook
    ``gathered_and_output`` installs has something to fire on, and the
    ``transpose(0, 1).contiguous()`` is copied from
    ``language_model_embedding.py:124`` verbatim.
    """

    def __init__(self, rows: int = TINY.vocab_size, dim: int = DIM):
        super().__init__()
        self.word_embeddings = nn.Embedding(rows, dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(ids).transpose(0, 1).contiguous()


class LayoutFreedomTests(unittest.TestCase):
    """megatron's transpose is free at the packing our driver runs.

    This is the fact the scenario rests on, and the fact its first version got
    backwards. ``benchmarks/e2e/megatron/data.py:54`` packs ``[1, T]``, so the
    embedding produces ``[1, T, D]`` and the transposed view is contiguous --
    PyTorch's contiguity test skips size-1 dimensions -- so ``.contiguous()``
    returns ``self`` and no bytes move.
    """

    def test_a_size_one_leading_dimension_makes_the_transpose_a_view(
        self,
    ) -> None:
        gathered = torch.zeros(1, TOKENS, DIM)
        view = gathered.transpose(0, 1)
        self.assertEqual(tuple(view.shape), MCORE_NATIVE_SHAPE)
        self.assertTrue(view.is_contiguous())
        self.assertIs(view.contiguous(), view)

    def test_the_titan_packing_would_make_the_same_line_allocate(self) -> None:
        gathered = torch.zeros(BATCH, SEQ, DIM)
        view = gathered.transpose(0, 1)
        self.assertFalse(view.is_contiguous())
        self.assertNotEqual(
            view.contiguous().untyped_storage().data_ptr(),
            gathered.untyped_storage().data_ptr(),
        )

    def test_the_gradient_transposes_back_without_a_copy_too(self) -> None:
        """Backward is free for the same reason, so neither direction charges
        one engine a copy the other does not pay."""
        grad = torch.zeros(MCORE_NATIVE_SHAPE)
        self.assertTrue(grad.transpose(0, 1).is_contiguous())

    def test_the_guard_passes_on_the_thd_packing(self) -> None:
        module = _FakeLanguageModelEmbedding()
        ids = torch.zeros(1, TOKENS, dtype=torch.int64)
        gathered, out = gathered_and_output(module, module, ids)
        self.assertEqual(tuple(out.shape), MCORE_NATIVE_SHAPE)
        _assert_layout_conversion_is_free(MCORE_ARM_NAME, gathered, out)

    def test_the_guard_fires_when_the_transpose_really_allocates(self) -> None:
        """Change the packing to [batch, seq_len] and megatron starts paying a
        copy the titan arm has no counterpart for. The guard refuses rather
        than publishing the difference as a kernel result."""
        module = _FakeLanguageModelEmbedding()
        ids = torch.zeros(BATCH, SEQ, dtype=torch.int64)
        gathered, out = gathered_and_output(module, module, ids)
        with self.assertRaisesRegex(RuntimeError, "layout copy"):
            _assert_layout_conversion_is_free(MCORE_ARM_NAME, gathered, out)

    def test_a_contiguity_test_would_return_green_on_both(self) -> None:
        """Why the guard compares storage rather than calling
        ``is_contiguous()``: the free view passes a contiguity test, so the
        first version of this guard was green on exactly the case it existed
        to detect."""
        free = _FakeLanguageModelEmbedding()(
            torch.zeros(1, TOKENS, dtype=torch.int64)
        )
        paid = _FakeLanguageModelEmbedding()(
            torch.zeros(BATCH, SEQ, dtype=torch.int64)
        )
        self.assertTrue(free.is_contiguous())
        self.assertTrue(paid.is_contiguous())

    def test_a_module_that_never_gathers_is_refused(self) -> None:
        class Silent(nn.Module):
            def __init__(self):
                super().__init__()
                self.word_embeddings = nn.Embedding(TINY.vocab_size, DIM)

            def forward(self, ids):
                return torch.zeros(ids.shape + (DIM,))

        module = Silent()
        with self.assertRaisesRegex(RuntimeError, "word_embeddings"):
            gathered_and_output(
                module, module, torch.zeros(1, TOKENS, dtype=torch.int64)
            )


class TitanArmTests(unittest.TestCase):
    """The titan arm, built by its own builder. It is eager, so this is it."""

    def test_the_module_is_the_node_the_production_config_carries(self) -> None:
        from benchmarks.models.piper_qwen3.config_registry import (
            _piper_1b_model,
        )

        declared = _piper_1b_model(fuse_qkv=True, shape=TINY).tok_embeddings
        module = titan_embedding_module(TINY, torch.device("cpu"))
        self.assertIsInstance(module, nn.Embedding)
        self.assertEqual(module.num_embeddings, declared.num_embeddings)
        self.assertEqual(module.embedding_dim, declared.embedding_dim)
        self.assertEqual(module.weight.dtype, torch.bfloat16)

    def test_the_arm_agrees_with_the_fp64_reference(self) -> None:
        inputs = make_inputs()
        arm = build_embedding_stage_titan(TINY, WORKLOAD, inputs)
        reference = embedding_stage_reference(TINY, WORKLOAD, inputs)
        outputs = arm.correctness_outputs()
        self.assertEqual(set(outputs), DECLARED_OUTPUTS)
        for name in sorted(DECLARED_OUTPUTS):
            with self.subTest(output=name):
                self.assertLess(
                    rel_l2(outputs[name], reference[name]), DECLARED_GATE
                )

    def test_the_forward_is_bitwise_the_gather_the_reference_names(
        self,
    ) -> None:
        """The scenario records a bitwise cross-engine check on ``out``. It
        rests on this: a gather is a copy, so the bf16 output holds the table's
        own bytes and no arithmetic can move them."""
        inputs = make_inputs()
        arm = build_embedding_stage_titan(TINY, WORKLOAD, inputs)
        got = arm.correctness_outputs()["out"]
        self.assertTrue(torch.equal(got, inputs.weight[inputs.tokens]))

    def test_the_timed_call_is_the_lookup_and_no_layout_conversion(
        self,
    ) -> None:
        """The correction. An earlier version appended
        ``transpose(0, 1).contiguous()`` here to match a megatron copy that
        does not exist at our packing, which cost the arm a second pass over
        the whole output on a scenario with no arithmetic in it."""
        inputs = make_inputs()
        arm = build_embedding_stage_titan(TINY, WORKLOAD, inputs)
        out = arm.calls["forward"]()
        self.assertEqual(tuple(out.shape), CANONICAL_SHAPE)
        self.assertTrue(out.is_contiguous())
        self.assertTrue(torch.equal(out, inputs.weight[inputs.tokens]))

    def test_the_arm_exposes_the_declared_modes_and_no_third(self) -> None:
        """``_seeded_build`` raises when the builder disagrees, so pin both."""
        arm = build_embedding_stage_titan(TINY, WORKLOAD, make_inputs())
        self.assertEqual(set(arm.calls), DECLARED_MODES)

    def test_forward_backward_clears_the_gradient_between_calls(self) -> None:
        """Two calls must cost the same. Without the reset the second would
        accumulate into an existing buffer instead of filling a fresh one."""
        inputs = make_inputs()
        arm = build_embedding_stage_titan(TINY, WORKLOAD, inputs)
        arm.calls["forward_backward"]()
        first = arm.correctness_outputs()["weight_grad_rows"].clone()
        arm.calls["forward_backward"]()
        second = arm.correctness_outputs()["weight_grad_rows"]
        self.assertTrue(torch.equal(first, second))


class ArmContractTests(unittest.TestCase):
    """What ``_embedding_stage_arm`` refuses, on a module either arm could
    hand it."""

    def build(self, **overrides):
        inputs = make_inputs()
        module = titan_embedding_module(TINY, torch.device("cpu"))
        kwargs = dict(
            module=module,
            weight=module.weight,
            call=module,
            tokens_native=inputs.tokens,
            grad_native=inputs.grad_out,
            native_shape=CANONICAL_SHAPE,
            canonical_shape=CANONICAL_SHAPE,
            to_canonical=lambda out: out,
            touched_ids=inputs.touched_ids,
        )
        kwargs.update(overrides)
        return _embedding_stage_arm("titan", **kwargs)

    def test_a_well_formed_arm_builds(self) -> None:
        self.assertEqual(set(self.build().calls), DECLARED_MODES)

    def test_a_table_that_takes_no_gradient_is_refused(self) -> None:
        """Backward would skip the scatter-add that dominates this stage."""
        module = titan_embedding_module(TINY, torch.device("cpu"))
        module.weight.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "require grad"):
            self.build(module=module, weight=module.weight, call=module)

    def test_a_strided_upstream_gradient_is_refused(self) -> None:
        """ATen materializes it inside ``embedding_dense_backward``, so the
        arm would pay a copy its opponent does not."""
        strided = make_inputs().grad_out.transpose(0, 1)
        self.assertFalse(strided.is_contiguous())
        with self.assertRaisesRegex(RuntimeError, "strided view"):
            self.build(grad_native=strided)


class OutputLayoutGuardTests(unittest.TestCase):
    """``_assert_output_layout`` runs in every worker, not only in the gates.

    Both native layouts are exercised: titan's ``[B, L, D]`` and megatron's
    ``[T, 1, D]``. The first version of these tests only ever built the titan
    shape, which is how a guard that returns green on the ``[T, 1, D]`` case
    went unnoticed.
    """

    def check_titan(self, tensor: torch.Tensor) -> None:
        _assert_output_layout(
            "titan", tensor, CANONICAL_SHAPE, CANONICAL_SHAPE, lambda o: o
        )

    def check_mcore(self, tensor: torch.Tensor) -> None:
        _assert_output_layout(
            MCORE_ARM_NAME,
            tensor,
            MCORE_NATIVE_SHAPE,
            CANONICAL_SHAPE,
            lambda o: o.reshape(CANONICAL_SHAPE),
        )

    def test_the_titan_native_layout_passes(self) -> None:
        self.check_titan(torch.zeros(CANONICAL_SHAPE, dtype=torch.bfloat16))

    def test_the_mcore_native_layout_passes(self) -> None:
        """Including the free transposed view megatron really returns."""
        gathered = torch.zeros(1, TOKENS, DIM, dtype=torch.bfloat16)
        self.check_mcore(gathered.transpose(0, 1))

    def test_a_strided_output_is_refused(self) -> None:
        view = torch.zeros(
            (SEQ, BATCH, DIM), dtype=torch.bfloat16
        ).transpose(0, 1)
        self.assertEqual(tuple(view.shape), CANONICAL_SHAPE)
        with self.assertRaisesRegex(RuntimeError, "strided view"):
            self.check_titan(view)

    def test_an_output_of_another_shape_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected"):
            self.check_mcore(
                torch.zeros(CANONICAL_SHAPE, dtype=torch.bfloat16)
            )

    def test_an_upcast_output_is_refused(self) -> None:
        """``fp32_residual_connection`` would double the bytes on one side."""
        with self.assertRaisesRegex(RuntimeError, "bfloat16"):
            self.check_titan(torch.zeros(CANONICAL_SHAPE))

    def test_a_canonical_mapping_that_reorders_tokens_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "canonical"):
            _assert_output_layout(
                MCORE_ARM_NAME,
                torch.zeros(MCORE_NATIVE_SHAPE, dtype=torch.bfloat16),
                MCORE_NATIVE_SHAPE,
                CANONICAL_SHAPE,
                lambda o: o.reshape(SEQ, BATCH, DIM),
            )


class _StubWordEmbeddings:
    def __init__(self, rows: int, dim: int, deterministic_mode: bool = False):
        self.weight = torch.zeros(rows, dim)
        self.deterministic_mode = deterministic_mode


class _StubDropout:
    def __init__(self, p: float):
        self.p = p


class _StubConfig:
    def __init__(
        self,
        use_mup: bool = False,
        mup_embedding_mult: float = 1.0,
        fp32_residual_connection: bool = False,
    ):
        self.use_mup = use_mup
        self.mup_embedding_mult = mup_embedding_mult
        self.fp32_residual_connection = fp32_residual_connection


class LanguageModelEmbedding:
    """A stand-in for megatron's module, matched by class name.

    ``_assert_mcore_embedding`` reads the class *name* rather than calling
    ``isinstance``, precisely so its branches are reachable on a host with no
    megatron. This is that host.
    """

    def __init__(
        self,
        *,
        rows: int = TINY.vocab_size,
        dim: int = DIM,
        add_position_embedding: bool = False,
        num_tokentypes: int = 0,
        reduce_scatter_embeddings: bool = False,
        dropout: float = 0.0,
        deterministic_mode: bool = False,
        config: object | None = None,
    ):
        self.add_position_embedding = add_position_embedding
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = None if num_tokentypes == 0 else object()
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.embedding_dropout = _StubDropout(dropout)
        self.config = _StubConfig() if config is None else config
        self.word_embeddings = _StubWordEmbeddings(
            rows, dim, deterministic_mode
        )


class OtherModule(LanguageModelEmbedding):
    pass


class McoreGuardTests(unittest.TestCase):
    def test_the_configuration_this_repo_builds_passes(self) -> None:
        _assert_mcore_embedding(LanguageModelEmbedding(), TINY)

    def test_a_module_of_another_class_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, MCORE_MODULE_CLASS):
            _assert_mcore_embedding(OtherModule(), TINY)

    def test_a_learned_position_embedding_is_refused(self) -> None:
        """It adds a second lookup and an add that titan has no counterpart
        for; this model is built with ``position_embedding_type='rope'``."""
        with self.assertRaisesRegex(RuntimeError, "position"):
            _assert_mcore_embedding(
                LanguageModelEmbedding(add_position_embedding=True), TINY
            )

    def test_token_type_embeddings_are_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "token types"):
            _assert_mcore_embedding(
                LanguageModelEmbedding(num_tokentypes=2), TINY
            )

    def test_reduce_scatter_embeddings_is_refused(self) -> None:
        """At ``reduce_scatter_embeddings`` the transpose moves down into
        ``VocabParallelEmbedding`` and a collective follows it, which this
        scenario does not declare."""
        with self.assertRaisesRegex(RuntimeError, "reduce_scatter"):
            _assert_mcore_embedding(
                LanguageModelEmbedding(reduce_scatter_embeddings=True), TINY
            )

    def test_a_mup_embedding_multiplier_is_refused(self) -> None:
        """Inert at every profile here, so this is defence in depth: it adds
        an elementwise multiply titan has no counterpart for."""
        with self.assertRaisesRegex(RuntimeError, "use_mup"):
            _assert_mcore_embedding(
                LanguageModelEmbedding(
                    config=_StubConfig(
                        use_mup=True, mup_embedding_mult=2.0
                    )
                ),
                TINY,
            )

    def test_mup_without_a_multiplier_is_allowed(self) -> None:
        """megatron guards the same product, so a multiplier of 1.0 adds no
        kernel and must not fail the arm."""
        _assert_mcore_embedding(
            LanguageModelEmbedding(config=_StubConfig(use_mup=True)), TINY
        )

    def test_an_fp32_residual_connection_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fp32_residual_connection"):
            _assert_mcore_embedding(
                LanguageModelEmbedding(
                    config=_StubConfig(fp32_residual_connection=True)
                ),
                TINY,
            )

    def test_a_module_with_no_config_is_refused(self) -> None:
        module = LanguageModelEmbedding()
        module.config = None
        with self.assertRaisesRegex(RuntimeError, "no config"):
            _assert_mcore_embedding(module, TINY)

    def test_a_nonzero_embedding_dropout_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "hidden_dropout"):
            _assert_mcore_embedding(
                LanguageModelEmbedding(dropout=0.1), TINY
            )

    def test_a_sharded_vocabulary_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected"):
            _assert_mcore_embedding(
                LanguageModelEmbedding(rows=TINY.vocab_size // 2), TINY
            )

    def test_deterministic_mode_is_refused(self) -> None:
        """It swaps ``F.embedding`` for ``weight[ids]``, a different backward
        kernel, and titan exposes no matching switch."""
        with self.assertRaisesRegex(RuntimeError, "deterministic_mode"):
            _assert_mcore_embedding(
                LanguageModelEmbedding(deterministic_mode=True), TINY
            )

    def test_the_guard_names_the_arm_it_would_fail(self) -> None:
        with self.assertRaisesRegex(RuntimeError, MCORE_ARM_NAME):
            _assert_mcore_embedding(OtherModule(), TINY)


class TitanGuardTests(unittest.TestCase):
    def test_the_production_module_passes(self) -> None:
        _assert_titan_embedding(
            titan_embedding_module(TINY, torch.device("cpu")), TINY
        )

    def test_a_vocab_parallel_group_is_refused(self) -> None:
        """``Embedding.forward`` takes a masked, clamped, multiplied branch
        when a tp_group is set, and megatron runs at tp_size 1 here."""
        module = titan_embedding_module(TINY, torch.device("cpu"))
        module.tp_group = object()
        with self.assertRaisesRegex(RuntimeError, "tp_group"):
            _assert_titan_embedding(module, TINY)

    def test_a_sparse_gradient_is_refused(self) -> None:
        module = titan_embedding_module(TINY, torch.device("cpu"))
        module.sparse = True
        with self.assertRaisesRegex(RuntimeError, "sparse"):
            _assert_titan_embedding(module, TINY)

    def test_a_table_of_the_wrong_shape_is_refused(self) -> None:
        other = PiperShape.derived(name="other", dim=256, n_layers=2, vocab_size=128)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            _assert_titan_embedding(
                titan_embedding_module(TINY, torch.device("cpu")), other
            )


class ArmShapeTests(unittest.TestCase):
    def test_the_floor_copies_once_and_declares_its_bytes(self) -> None:
        inputs = make_inputs()
        floor = build_embedding_stage_copy_floor(TINY, WORKLOAD, inputs)
        self.assertEqual(set(floor.calls), {"forward"})
        self.assertEqual(floor.bytes_moved, inputs.copy_bytes)
        floor.calls["forward"]()
        self.assertEqual(floor.correctness_outputs(), {})

    def test_every_declared_builder_path_resolves(self) -> None:
        for path in DECLARED_BUILDER_PATHS:
            with self.subTest(path=path):
                self.assertTrue(callable(resolve_symbol(path)))


class WeightMapTests(unittest.TestCase):
    """The cross-engine transfer this scenario needs, taken from the map.

    ``benchmarks/models/piper_qwen3/megatron_weights.py`` owns the
    titan-to-megatron parameter map. For this scenario the map says the
    transfer is the identity, which is why both builders load one shared table
    instead of converting anything. These tests pin that reading, so a map
    change cannot leave the builders silently wrong.
    """

    def setUp(self) -> None:
        from benchmarks.models.piper_qwen3.titan_model import build_titan_model

        self.model = build_titan_model(
            shape=TINY, device="cpu", dtype=torch.float32
        )
        self.state = dict(self.model.state_dict())

    def embedding_transfers(self):
        from benchmarks.models.piper_qwen3.megatron_weights import (
            weight_transfers,
        )

        return [
            transfer
            for transfer in weight_transfers(self.state, TINY)
            if transfer[0] == "embedding"
        ]

    def test_the_map_holds_exactly_one_embedding_transfer(self) -> None:
        self.assertEqual(len(self.embedding_transfers()), 1)

    def test_the_transfer_names_the_parameter_the_builder_loads(self) -> None:
        """``build_embedding_stage_mcore_base`` writes into
        ``model.embedding.word_embeddings.weight``."""
        _, name, _ = self.embedding_transfers()[0]
        self.assertEqual(name, "embedding.word_embeddings.weight")

    def test_the_transfer_is_the_identity_on_titan_tok_embeddings(self) -> None:
        _, _, tensor = self.embedding_transfers()[0]
        self.assertTrue(
            torch.equal(tensor, self.state["tok_embeddings.weight"])
        )
        self.assertEqual(
            tuple(tensor.shape), (TINY.vocab_size, TINY.dim)
        )


class ModelReleaseTests(unittest.TestCase):
    """The mcore arm drops its ``GPTModel`` and keeps the table alone.

    The builder needs CUDA, megatron and TransformerEngine, so the release is
    proved in two pieces. The two helpers run here on a stand-in model, and
    the builder's own source pins the order it calls them in. Neither piece
    is a measurement, and neither claims to be one.
    """

    @staticmethod
    def stand_in() -> tuple[nn.Module, nn.Module, torch.Tensor]:
        """A model shaped like the one the builder walks.

        ``LanguageModelEmbedding`` owns ``word_embeddings`` and, at
        ``position_embedding_type='rope'`` with no token types, nothing else.
        ``_assert_mcore_embedding`` refuses every other case, so the stand-in
        has the property the helpers are written against.
        """
        model = nn.Module()
        model.embedding = nn.Module()
        model.embedding.word_embeddings = nn.Embedding(
            TINY.vocab_size, TINY.dim
        )
        model.decoder = nn.Linear(TINY.dim, TINY.dim)
        module = model.embedding
        return model, module, module.word_embeddings.weight

    def test_the_watch_covers_every_parameter_except_the_kept_table(
        self,
    ) -> None:
        model, module, weight = self.stand_in()
        watched = _watch_released_parameters(model, module, weight)
        self.assertEqual(len(watched), len(list(model.parameters())) - 1)
        self.assertFalse(any(reference() is weight for reference in watched))

    def test_dropping_the_model_releases_every_watched_parameter(self) -> None:
        model, module, weight = self.stand_in()
        watched = _watch_released_parameters(model, module, weight)
        del model
        gc.collect()
        self.assertIsNone(_assert_parameters_released(watched))
        self.assertTrue(all(reference() is None for reference in watched))
        # The one tensor the arm keeps is untouched by the release.
        self.assertIs(module.word_embeddings.weight, weight)

    def test_a_surviving_parameter_fails_the_arm_and_names_the_count(
        self,
    ) -> None:
        """The model stays alive here, which is the defect under test."""
        model, module, weight = self.stand_in()
        watched = _watch_released_parameters(model, module, weight)
        with self.assertRaises(RuntimeError) as caught:
            _assert_parameters_released(watched)
        message = str(caught.exception)
        self.assertIn(f"{len(watched)} of {len(watched)} megatron", message)
        self.assertIn("MiB still resident", message)
        self.assertIsNotNone(model)

    def test_the_watch_refuses_a_kept_module_that_owns_a_second_parameter(
        self,
    ) -> None:
        """A second kept tensor would read as a leak, so the watch says so."""
        model, module, weight = self.stand_in()
        module.position_embeddings = nn.Embedding(TINY.vocab_size, TINY.dim)
        with self.assertRaises(RuntimeError) as caught:
            _watch_released_parameters(model, module, weight)
        self.assertIn("besides the table", str(caught.exception))

    def test_the_builder_collects_and_checks_after_it_drops_the_model(
        self,
    ) -> None:
        """The bare ``del model`` left the parameters resident until the
        next collection, which is the transient window a large shape cannot
        afford."""
        self.assertIn(
            "    watched = _watch_released_parameters(model, module, weight)\n"
            "    del model\n"
            "    gc.collect()\n"
            "    torch.cuda.empty_cache()\n"
            "    _assert_parameters_released(watched)\n",
            inspect.getsource(build_embedding_stage_mcore_base),
        )


if __name__ == "__main__":
    unittest.main()
