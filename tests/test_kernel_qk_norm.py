"""CPU tests for the ``qk_norm`` cross-engine kernel scenario.

The scenario needs a GPU, megatron-core and TransformerEngine to measure
anything. It does not need any of the three to be *wrong*, and the four
failures that would ship a wrong number are all reachable on a CPU:

* the two engines drift apart on the epsilon or on the layout, so the ratio
  compares two different computations;
* the megatron arm reads a memory layout megatron never produces, which
  deletes a copy megatron really pays and flatters the anchor;
* the titan arm builds a lookalike norm instead of the one the production
  config puts on the block;
* the fp64 reference is wrong, so every gate passes against a wrong truth;
* the megatron attribute names drift away from the cross-engine weight map,
  so the arm navigates to a module the map never checked.

The tests below cover those five. They do not cover the megatron arm's own
build, which needs a device, and they do not compile anything: the titan
builder wraps its pair in ``torch.compile``, which compiles on the first call,
so building the arm stays cheap and the arithmetic is checked on the
uncompiled pair. Strides need no device either, so the layout of every tensor
the megatron arm reads is checked here in full -- shape, stride and storage
offset, against megatron's own arithmetic.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from benchmarks.kernel.operations.qk_norm import (
    MCORE_K_NORM_ATTR,
    MCORE_Q_NORM_ATTR,
    NORM_EPS,
    TITAN_K_NORM_ATTR,
    TITAN_Q_NORM_ATTR,
    WEIGHT_COMPONENT,
    ACTIVATION_OUTPUTS,
    WEIGHT_GRAD_OUTPUTS,
    _NormPair,
    _build_titan_norms,
    _mcore_leaves,
    _titan_leaves,
    _to_blnh,
    build_qk_norm_copy_floor,
    build_qk_norm_titan,
    qk_norm_inputs,
    qk_norm_reference,
)
from benchmarks.kernel.schema import KernelWorkload, shape_summary
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.megatron_weights import (
    COMPONENTS,
    weight_transfers,
)
from benchmarks.models.piper_qwen3.shape import PiperShape, shape_by_name

# head_dim stays 64, the real value, because it is the width the norm reduces
# over and the only geometry this scenario reads. Everything else shrinks:
# 4 query heads over 2 kv groups keeps q and k different sizes, which is the
# reason the scenario times a pair.
TINY = PiperShape(name="tiny", dim=256, n_layers=1, vocab_size=64)
TINY_WORKLOAD = KernelWorkload(batch=2, seq_len=8)

# The gate the scenario declaration carries, applied to all six outputs. One
# number covers the weight gradients too, which is not the usual case: a weight
# gradient accumulates over every row and normally sits far above the
# activations. Here it does not, and the reason is the width. The reduction
# runs over 65,536 rows but the weight holds only head_dim = 64 values, and
# torch accumulates that sum in fp32. Measured on CPU at the real normal shape
# (batch 4, seq 1024): 1.66e-3 on the four activations, 1.36e-3 and 1.64e-3 on
# the two weight gradients. The gate keeps an order of magnitude of headroom.
GATE = 2e-2


def cpu_inputs(seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return qk_norm_inputs(
        TINY, TINY_WORKLOAD, torch.device("cpu"), generator
    )


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.double() - expected.double()).norm()
    return float(difference / expected.double().norm())


def megatron_source() -> str:
    """The pinned megatron ``attention.py``, read as text.

    Read rather than imported, and read at run time rather than at module
    scope. ``add_megatron_to_path`` inserts the checkout at ``sys.path[0]``
    and Megatron-LM ships a ``tests/`` package of its own, so a module-scope
    call makes every later ``from tests.<x> import ...`` in this suite
    resolve to megatron's tests instead. Reading the file needs neither the
    path entry nor an import.
    """
    from benchmarks.models.piper_qwen3.megatron_bootstrap import megatron_dir

    try:
        root = Path(megatron_dir())
    except RuntimeError as error:  # pragma: no cover - host dependent
        raise unittest.SkipTest(str(error)) from error
    return (root / "megatron/core/transformer/attention.py").read_text()


def transformer_engine_source(*parts: str) -> str:
    """One TransformerEngine source file, located without importing TE.

    ``find_spec`` on the top-level name executes no module, so this runs on a
    host with no GPU and no working TE runtime.
    """
    spec = importlib.util.find_spec("transformer_engine")
    if spec is None or not spec.submodule_search_locations:
        raise unittest.SkipTest("transformer_engine is not installed")
    return Path(spec.submodule_search_locations[0]).joinpath(*parts).read_text()


class InputsTest(unittest.TestCase):
    def test_both_layouts_hold_the_same_rows(self):
        """The megatron form is a permutation of the titan form, not new data.

        The two arms must read one q and one k. A separate draw per layout
        would make the ratio a comparison of two inputs.
        """
        inputs = cpu_inputs()
        self.assertTrue(torch.equal(_to_blnh(inputs.q_SBNH), inputs.q_BLNH))
        self.assertTrue(torch.equal(_to_blnh(inputs.k_SBNH), inputs.k_BLNH))
        self.assertTrue(torch.equal(_to_blnh(inputs.gq_SBNH), inputs.gq_BLNH))
        self.assertTrue(torch.equal(_to_blnh(inputs.gk_SBNH), inputs.gk_BLNH))

    def test_every_tensor_but_the_megatron_key_is_contiguous(self):
        """Neither engine may pay for a transpose inside a timed closure.

        ``k_SBNH`` is the one exception and it has its own tests below: it is
        strided because megatron's is, not because anything here transposes.
        Both gradient seeds stay contiguous, because both norms write a
        contiguous output.
        """
        for name in (
            "q_BLNH",
            "k_BLNH",
            "gq_BLNH",
            "gk_BLNH",
            "qkv_fused_SBGR",
            "q_SBNH",
            "gq_SBNH",
            "gk_SBNH",
        ):
            with self.subTest(tensor=name):
                self.assertTrue(getattr(cpu_inputs(), name).is_contiguous())

    def test_shapes_follow_the_geometry(self):
        inputs = cpu_inputs()
        batch, seq = TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len
        self.assertEqual(
            tuple(inputs.q_BLNH.shape),
            (batch, seq, TINY.n_heads, TINY.head_dim),
        )
        self.assertEqual(
            tuple(inputs.k_BLNH.shape),
            (batch, seq, TINY.n_kv_heads, TINY.head_dim),
        )
        self.assertEqual(
            tuple(inputs.q_SBNH.shape),
            (seq, batch, TINY.n_heads, TINY.head_dim),
        )
        self.assertEqual(inputs.q_BLNH.dtype, torch.bfloat16)
        self.assertNotEqual(TINY.n_heads, TINY.n_kv_heads)

    def test_the_two_weights_differ(self):
        """q and k hold separate weights on both engines.

        An all-ones weight, or one weight shared by both norms, hides a
        swapped or dropped parameter from every gate.
        """
        inputs = cpu_inputs()
        self.assertEqual(tuple(inputs.q_weight.shape), (TINY.head_dim,))
        self.assertEqual(tuple(inputs.k_weight.shape), (TINY.head_dim,))
        self.assertEqual(inputs.q_weight.dtype, torch.bfloat16)
        self.assertFalse(torch.equal(inputs.q_weight, inputs.k_weight))
        self.assertFalse(
            torch.equal(inputs.q_weight, torch.ones_like(inputs.q_weight))
        )

    def test_the_seed_decides_the_inputs(self):
        one, two = cpu_inputs(seed=7), cpu_inputs(seed=7)
        self.assertTrue(torch.equal(one.q_BLNH, two.q_BLNH))
        self.assertTrue(torch.equal(one.q_weight, two.q_weight))
        self.assertFalse(torch.equal(one.q_BLNH, cpu_inputs(seed=8).q_BLNH))

    def test_bytes_moved_counts_the_forward_traffic(self):
        """One read and one write of q and k, which is what the floor copies."""
        inputs = cpu_inputs()
        expected = 2 * (inputs.q_BLNH.numel() + inputs.k_BLNH.numel()) * 2
        self.assertEqual(inputs.qk_bytes, expected)
        floor = build_qk_norm_copy_floor(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(floor.bytes_moved, inputs.qk_bytes)

    def test_the_megatron_count_adds_the_copy_of_the_strided_key(self):
        """TE materializes the key inside op_forward, so the arm moves more.

        Following ``moe_router``: a per-arm ``bytes_moved`` makes the GB/s
        column show the asymmetry instead of absorbing it. ``x_floor`` is a
        ratio of times and is untouched either way.
        """
        inputs = cpu_inputs()
        one_pass_over_the_key = inputs.k_BLNH.numel() * 2
        self.assertEqual(
            inputs.mcore_qk_bytes, inputs.qk_bytes + 2 * one_pass_over_the_key
        )
        self.assertGreater(inputs.mcore_qk_bytes, inputs.qk_bytes)


class MegatronLayoutTest(unittest.TestCase):
    """The layout the megatron arm reads, asserted against megatron's own.

    Shape, stride and storage offset, not merely "it is not contiguous". A
    tensor can be non-contiguous for the wrong reason -- a transpose, a
    permutation, a slice of the wrong axis -- and every one of those would
    still measure a layout megatron never produces.
    """

    def row_width(self) -> int:
        """``(Q + 2) * H``: megatron's fused row, per key/value group."""
        return (TINY.heads_per_group + 2) * TINY.head_dim

    def test_the_fused_buffer_carries_megatrons_row(self):
        inputs = cpu_inputs()
        seq, batch = TINY_WORKLOAD.seq_len, TINY_WORKLOAD.batch
        self.assertEqual(
            tuple(inputs.qkv_fused_SBGR.shape),
            (seq, batch, TINY.n_kv_heads, self.row_width()),
        )
        self.assertTrue(inputs.qkv_fused_SBGR.is_contiguous())
        self.assertEqual(inputs.qkv_fused_SBGR.dtype, torch.bfloat16)

    def test_the_key_reaches_megatron_as_a_strided_view(self):
        """The measurand, not a detail.

        ``get_query_key_value_tensors`` splits the fused buffer and hands
        ``k_layernorm`` the split result unchanged. Storage offset ``Q * H``
        and a group stride of ``(Q + 2) * H`` are megatron's arithmetic, and
        a tensor that fails either is not the one the engine norms.
        """
        inputs = cpu_inputs()
        seq, batch = TINY_WORKLOAD.seq_len, TINY_WORKLOAD.batch
        groups, head_dim = TINY.n_kv_heads, TINY.head_dim
        row = self.row_width()
        self.assertEqual(
            tuple(inputs.k_SBNH.shape), (seq, batch, groups, head_dim)
        )
        self.assertFalse(inputs.k_SBNH.is_contiguous())
        self.assertEqual(
            inputs.k_SBNH.stride(), (batch * groups * row, groups * row, row, 1)
        )
        self.assertEqual(
            inputs.k_SBNH.storage_offset(), TINY.heads_per_group * head_dim
        )
        self.assertEqual(
            inputs.k_SBNH.untyped_storage().data_ptr(),
            inputs.qkv_fused_SBGR.untyped_storage().data_ptr(),
        )

    def test_the_query_reaches_megatron_contiguous(self):
        """Megatron's own reshape allocates, so ``q_layernorm`` reads a copy.

        ``query.reshape(s, b, -1, H)`` merges the group dimension with the
        query heads inside a group, and the two are not adjacent while the
        row is wider than ``Q * H``. That copy is megatron's work at
        ``:1908`` and it belongs to ``qkv_prep``, so it must already have
        happened before this scenario starts.
        """
        inputs = cpu_inputs()
        self.assertTrue(inputs.q_SBNH.is_contiguous())
        self.assertNotEqual(
            inputs.q_SBNH.untyped_storage().data_ptr(),
            inputs.qkv_fused_SBGR.untyped_storage().data_ptr(),
        )

    def test_the_strided_key_holds_the_canonical_values(self):
        """A layout change may not become a value change.

        The two arms must read one q and one k, or the ratio compares two
        inputs. This is the same assertion ``test_both_layouts_hold_the_same
        _rows`` makes, restated against the buffer the values now travel
        through.
        """
        inputs = cpu_inputs()
        self.assertTrue(
            torch.equal(_to_blnh(inputs.k_SBNH).contiguous(), inputs.k_BLNH)
        )
        self.assertTrue(
            torch.equal(_to_blnh(inputs.q_SBNH).contiguous(), inputs.q_BLNH)
        )

    def test_cloning_the_strided_key_returns_a_contiguous_tensor(self):
        """Why the leaf sets use ``detach`` and not ``clone``.

        ``clone`` carries a layout only for a tensor that is non-overlapping
        AND dense. The key view is neither, so ``preserve_format`` gives up.
        A cloned leaf would hand TE a layout it recognizes and would delete
        the copy this arm exists to measure, without any test failing.
        """
        inputs = cpu_inputs()
        self.assertFalse(inputs.k_SBNH.is_contiguous())
        self.assertTrue(inputs.k_SBNH.clone().is_contiguous())

    def test_megatron_still_norms_the_key_it_split(self):
        """Self-invalidating. A submodule bump can move this layout.

        Three facts on the pinned rev decide it: the split returns views, the
        query is reshaped out of the buffer before its norm, and the key is
        not. If megatron starts reshaping the key too, the inputs builder is
        modelling an engine that no longer exists.
        """
        source = megatron_source()
        self.assertIn(
            "query, key, value = torch.split(mixed_qkv, split_arg_list, dim=3)",
            source,
        )
        self.assertIn(
            "query = query.reshape(query.size(0), query.size(1), -1, "
            "self.hidden_size_per_attention_head)",
            source,
        )
        self.assertIn("key = apply_module(self.k_layernorm)(key)", source)
        self.assertNotIn("key = key.reshape(", source)
        self.assertTrue(
            BASE.config_overrides["qk_layernorm"],
            "with qk_layernorm off there is no k_layernorm to hand the "
            "strided key to",
        )

    def test_transformer_engine_still_copies_a_noncontiguous_norm_input(self):
        """Self-invalidating, and it is what makes the extra bytes real.

        TE's RMSNorm materializes its input inside ``op_forward``, so the
        strided key costs a read and a write before the norm kernel runs.
        This test FAILS if a TE upgrade drops the call -- at which point
        ``mcore_qk_bytes`` and the module docstring must be rewritten rather
        than carried.
        """
        source = transformer_engine_source(
            "pytorch", "ops", "basic", "rmsnorm.py"
        )
        self.assertIn("input_.contiguous()", source)


class LeafSetTest(unittest.TestCase):
    """The layout must survive into the tensors the timed closures read."""

    def test_a_megatron_leaf_set_keeps_megatron_strides(self):
        inputs = cpu_inputs()
        query, key = _mcore_leaves(TINY, inputs)()
        self.assertTrue(query.is_contiguous())
        self.assertFalse(key.is_contiguous())
        self.assertEqual(key.stride(), inputs.k_SBNH.stride())
        self.assertEqual(
            key.storage_offset(), inputs.k_SBNH.storage_offset()
        )
        self.assertTrue(query.is_leaf and query.requires_grad)
        self.assertTrue(key.is_leaf and key.requires_grad)
        self.assertTrue(torch.equal(query, inputs.q_SBNH))
        self.assertTrue(torch.equal(key, inputs.k_SBNH))

    def test_a_titan_leaf_set_is_contiguous(self):
        inputs = cpu_inputs()
        query, key = _titan_leaves(inputs)()
        for leaf, source in ((query, inputs.q_BLNH), (key, inputs.k_BLNH)):
            with self.subTest(shape=tuple(leaf.shape)):
                self.assertTrue(leaf.is_contiguous())
                self.assertTrue(leaf.is_leaf and leaf.requires_grad)
                self.assertTrue(torch.equal(leaf, source))

    def test_each_leaf_set_owns_its_own_storage(self):
        """Three sets per arm, and none of them may share a buffer.

        The forward mode, the forward+backward mode and the correctness pass
        each hold one set. A shared buffer would let one mode's gradient
        reach another mode's leaf.
        """
        inputs = cpu_inputs()
        for factory in (_mcore_leaves(TINY, inputs), _titan_leaves(inputs)):
            one, two = factory(), factory()
            pointers = {
                leaf.untyped_storage().data_ptr() for leaf in one + two
            }
            with self.subTest(factory=factory.__qualname__):
                self.assertEqual(len(pointers), len(one) + len(two))

    def test_a_megatron_leaf_set_does_not_alias_the_inputs(self):
        """A timed closure may not write through to the shared inputs."""
        inputs = cpu_inputs()
        _query, key = _mcore_leaves(TINY, inputs)()
        self.assertNotEqual(
            key.untyped_storage().data_ptr(),
            inputs.qkv_fused_SBGR.untyped_storage().data_ptr(),
        )


class ReferenceTest(unittest.TestCase):
    def test_forward_matches_aten_rms_norm(self):
        """An independent implementation, so the truth is not self-checked."""
        inputs = cpu_inputs()
        reference = qk_norm_reference(TINY, TINY_WORKLOAD, inputs)
        for tag, x, weight in (
            ("q", inputs.q_BLNH, inputs.q_weight),
            ("k", inputs.k_BLNH, inputs.k_weight),
        ):
            with self.subTest(tensor=tag):
                aten = F.rms_norm(
                    x.double(),
                    (TINY.head_dim,),
                    weight.double(),
                    eps=inputs.eps,
                )
                self.assertLess(rel_l2(reference[f"{tag}_out"], aten), 1e-12)

    def test_gradients_match_aten_autograd(self):
        inputs = cpu_inputs()
        reference = qk_norm_reference(TINY, TINY_WORKLOAD, inputs)
        for tag, x, grad, weight in (
            ("q", inputs.q_BLNH, inputs.gq_BLNH, inputs.q_weight),
            ("k", inputs.k_BLNH, inputs.gk_BLNH, inputs.k_weight),
        ):
            with self.subTest(tensor=tag):
                leaf = x.double().detach().requires_grad_()
                gamma = weight.double().detach().requires_grad_()
                out = F.rms_norm(
                    leaf, (TINY.head_dim,), gamma, eps=inputs.eps
                )
                torch.autograd.backward(out, grad.double())
                self.assertLess(rel_l2(reference[f"d{tag}"], leaf.grad), 1e-12)
                self.assertLess(
                    rel_l2(reference[f"{tag}_weight_grad"], gamma.grad), 1e-12
                )

    def test_the_reference_returns_every_declared_output(self):
        """The scenario declaration repeats these names as literals."""
        reference = qk_norm_reference(TINY, TINY_WORKLOAD, cpu_inputs())
        self.assertEqual(
            set(reference), set(ACTIVATION_OUTPUTS) | set(WEIGHT_GRAD_OUTPUTS)
        )
        self.assertEqual(
            tuple(reference["q_weight_grad"].shape), (TINY.head_dim,)
        )


class TitanArmTest(unittest.TestCase):
    def test_the_builder_uses_the_production_config_node(self):
        """The arm must build the norm the piper config puts on a block.

        This is what makes a ``Trainer.Config`` extraction unnecessary. The
        node is a pure function of the shape, so the builder calls the same
        one-line constructor the model registry calls. The test is what keeps
        that true: an eps, an init or an affine flag that moves in torchtitan
        fails here rather than in a published ratio.
        """
        from torchtitan.models.qwen3 import _qwen3_norm

        from benchmarks.models.piper_qwen3 import config_registry

        trainer = config_registry.qwen3_piper_1b(size="normal")
        production = trainer.model_spec.model.layers[0].attention.qk_norm
        self.assertEqual(production, _qwen3_norm(64))

    def test_both_engines_read_one_epsilon(self):
        """The fp64 reference holds one eps and gates both arms with it."""
        from torchtitan.models.qwen3 import _EPS

        self.assertEqual(float(_EPS), NORM_EPS)
        self.assertEqual(
            float(BASE.config_overrides["layernorm_epsilon"]), NORM_EPS
        )

    def test_the_base_profile_still_asks_for_this_norm(self):
        """A profile with qk_layernorm off builds IdentityOp on the mcore side.

        The arm refuses that at build time. This test refuses it one step
        earlier, without a GPU.
        """
        self.assertIs(BASE.config_overrides["qk_layernorm"], True)
        self.assertEqual(BASE.config_overrides["normalization"], "RMSNorm")

    def test_the_pair_matches_the_reference_in_bf16(self):
        """The real arithmetic, uncompiled, against the fp64 truth.

        ``build_qk_norm_titan`` wraps this pair in ``torch.compile``. Inductor
        changes the kernels and not the mathematics, so running the pair eager
        checks the weights, the layout and the two gate values on a CPU.
        """
        inputs = cpu_inputs()
        q_module, k_module = _build_titan_norms(TINY, inputs)
        self.assertIsNot(q_module, k_module)
        pair = _NormPair(q_module, k_module)

        q_leaf = inputs.q_BLNH.clone().requires_grad_()
        k_leaf = inputs.k_BLNH.clone().requires_grad_()
        q_out, k_out = pair(q_leaf, k_leaf)
        torch.autograd.backward((q_out, k_out), (inputs.gq_BLNH, inputs.gk_BLNH))

        reference = qk_norm_reference(TINY, TINY_WORKLOAD, inputs)
        measured = {
            "q_out": q_out.detach(),
            "k_out": k_out.detach(),
            "dq": q_leaf.grad,
            "dk": k_leaf.grad,
            "q_weight_grad": q_module.weight.grad,
            "k_weight_grad": k_module.weight.grad,
        }
        for name in ACTIVATION_OUTPUTS + WEIGHT_GRAD_OUTPUTS:
            with self.subTest(output=name):
                self.assertLess(rel_l2(measured[name], reference[name]), GATE)

    def test_the_builder_exposes_the_declared_modes(self):
        """``_seeded_build`` compares this set against the declaration."""
        inputs = cpu_inputs()
        arm = build_qk_norm_titan(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(arm.name, "titan")
        self.assertEqual(set(arm.calls), {"forward", "forward_backward"})
        self.assertEqual(arm.bytes_moved, inputs.qk_bytes)

    def test_the_floor_exposes_forward_only(self):
        floor = build_qk_norm_copy_floor(TINY, TINY_WORKLOAD, cpu_inputs())
        self.assertEqual(floor.name, "copy_floor")
        self.assertEqual(set(floor.calls), {"forward"})
        floor.calls["forward"]()
        self.assertEqual(floor.correctness_outputs(), {})


class ShapeSummaryTest(unittest.TestCase):
    def test_the_manifest_branch_is_exactly_this_mapping(self):
        """Every key, not a sample of them.

        The manifest is how a reader without this repo learns what the arms
        consumed, and this branch is where it learns that the two engines do
        not read the same tensors. The fused row width is what makes the
        megatron key strided, so it is recorded beside the key itself.
        """
        workload = KernelWorkload()
        batch, seq = workload.batch, workload.seq_len
        normal = shape_by_name("normal")
        heads, groups = normal.n_heads, normal.n_kv_heads
        head_dim = normal.head_dim
        self.assertEqual(
            shape_summary("qk_norm", normal, workload),
            {
                "q_titan_BLNH": [batch, seq, heads, head_dim],
                "k_titan_BLNH": [batch, seq, groups, head_dim],
                "qkv_mcore_fused_SBGR": [
                    seq,
                    batch,
                    groups,
                    (normal.heads_per_group + 2) * head_dim,
                ],
                "q_mcore_SBNH": [seq, batch, heads, head_dim],
                "k_mcore_SBNH": [seq, batch, groups, head_dim],
                "k_mcore_is_a_strided_view": True,
                "weight": [head_dim],
                "rows": batch * seq * (heads + groups),
                "reduction_length": head_dim,
            },
        )


class CrossEngineNameTest(unittest.TestCase):
    def test_the_attribute_names_match_the_shared_weight_map(self):
        """The arm navigates to the module the weight map already transfers.

        ``tools/megatron_parity_check.py`` proves the two engines agree
        numerically, and it proves it through this map. An arm that reads a
        different attribute measures a module nothing checked.
        """
        self.assertIn(WEIGHT_COMPONENT, COMPONENTS)
        found = []
        for component, mega_name, _tensor in weight_transfers(
            _partial_titan_state(TINY), TINY
        ):
            if component == WEIGHT_COMPONENT:
                found.append(mega_name)
            if len(found) == 2:
                break
        self.assertEqual(
            found,
            [
                f"decoder.layers.0.self_attention.{MCORE_Q_NORM_ATTR}.weight",
                f"decoder.layers.0.self_attention.{MCORE_K_NORM_ATTR}.weight",
            ],
        )

    def test_the_titan_attribute_names_match_the_map_sources(self):
        """The map reads ``attention.q_norm`` / ``attention.k_norm``.

        Those are the two attributes ``GQAttention`` calls before RoPE, so the
        constants this module publishes are the call site the scenario claims.
        """
        state = _partial_titan_state(TINY)
        tagged = {}
        for component, mega_name, tensor in weight_transfers(state, TINY):
            if component == WEIGHT_COMPONENT:
                tagged[mega_name] = tensor
            if len(tagged) == 2:
                break
        self.assertIs(
            tagged[f"decoder.layers.0.self_attention.{MCORE_Q_NORM_ATTR}.weight"],
            state[f"layers.0.attention.{TITAN_Q_NORM_ATTR}.weight"],
        )
        self.assertIs(
            tagged[f"decoder.layers.0.self_attention.{MCORE_K_NORM_ATTR}.weight"],
            state[f"layers.0.attention.{TITAN_K_NORM_ATTR}.weight"],
        )


def _partial_titan_state(shape: PiperShape) -> dict[str, torch.Tensor]:
    """Only the entries ``weight_transfers`` reads before the qk_norm pair.

    ``weight_transfers`` is a generator, so a caller that stops at the pair
    never touches a later key. Building the full state dict here would copy
    ``tests/test_megatron_weights.py``, which owns that job.
    """
    kv_out = shape.n_kv_heads * shape.head_dim
    state = {
        "tok_embeddings.weight": torch.zeros(shape.vocab_size, shape.dim),
        "lm_head.weight": torch.zeros(shape.vocab_size, shape.dim),
        "norm.weight": torch.zeros(shape.dim),
    }
    for layer in range(shape.n_layers):
        prefix = f"layers.{layer}"
        state.update(
            {
                f"{prefix}.attention.qkv_linear.wq.weight": torch.zeros(
                    shape.n_heads * shape.head_dim, shape.dim
                ),
                f"{prefix}.attention.qkv_linear.wk.weight": torch.zeros(
                    kv_out, shape.dim
                ),
                f"{prefix}.attention.qkv_linear.wv.weight": torch.zeros(
                    kv_out, shape.dim
                ),
                f"{prefix}.attention_norm.weight": torch.zeros(shape.dim),
                f"{prefix}.attention.wo.weight": torch.zeros(
                    shape.dim, shape.n_heads * shape.head_dim
                ),
                f"{prefix}.attention.{TITAN_Q_NORM_ATTR}.weight": torch.zeros(
                    shape.head_dim
                ),
                f"{prefix}.attention.{TITAN_K_NORM_ATTR}.weight": torch.zeros(
                    shape.head_dim
                ),
            }
        )
    return state


if __name__ == "__main__":
    unittest.main()
