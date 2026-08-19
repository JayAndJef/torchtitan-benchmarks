"""CPU tests for the ``qk_norm`` cross-engine kernel scenario.

The scenario needs a GPU, megatron-core and TransformerEngine to measure
anything. It does not need any of the three to be *wrong*, and the four
failures that would ship a wrong number are all reachable on a CPU:

* the two engines drift apart on the epsilon or on the layout, so the ratio
  compares two different computations;
* the titan arm builds a lookalike norm instead of the one the production
  config puts on the block;
* the fp64 reference is wrong, so every gate passes against a wrong truth;
* the megatron attribute names drift away from the cross-engine weight map,
  so the arm navigates to a module the map never checked.

The tests below cover those four. They do not cover the megatron arm's own
build, which needs a device, and they do not compile anything: the titan
builder wraps its pair in ``torch.compile``, which compiles on the first call,
so building the arm stays cheap and the arithmetic is checked on the
uncompiled pair.
"""

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
    _to_blnh,
    build_qk_norm_copy_floor,
    build_qk_norm_titan,
    qk_norm_inputs,
    qk_norm_reference,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.megatron_weights import (
    COMPONENTS,
    weight_transfers,
)
from benchmarks.models.piper_qwen3.shape import PiperShape

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

    def test_every_timed_tensor_is_contiguous(self):
        """Neither engine may pay for a transpose inside a timed closure."""
        for name in (
            "q_BLNH",
            "k_BLNH",
            "gq_BLNH",
            "gk_BLNH",
            "q_SBNH",
            "k_SBNH",
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
        """The floor and the norms must declare one number, or x_floor lies."""
        inputs = cpu_inputs()
        expected = 2 * (inputs.q_BLNH.numel() + inputs.k_BLNH.numel()) * 2
        self.assertEqual(inputs.qk_bytes, expected)
        floor = build_qk_norm_copy_floor(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(floor.bytes_moved, inputs.qk_bytes)


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
