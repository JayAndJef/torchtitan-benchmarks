"""The cross-engine weight map, exercised on CPU at a toy shape.

``tools/megatron_parity_check.py`` is the numerics check for this map, and it
needs two real models, a GPU and about a minute. That is the right tool for
"do the engines agree", and the wrong one for "did a reshape literal move".
These tests run the real arithmetic against stand-in modules at a shape small
enough to build in milliseconds, so a layout bug costs one CPU second.

The stand-ins are deliberately dumb: a state dict and a named-parameter dict
of the right names and shapes. The map only ever reads the first and writes
the second, so nothing here is a mock of behaviour -- it is the same code path
a real transfer takes.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.models.piper_qwen3.megatron_weights import (
    COMPONENTS,
    assert_qkv_roundtrip,
    grouped_qkv,
    transfer_weights,
    weight_transfers,
)
from benchmarks.models.piper_qwen3.shape import PiperShape

# Small enough to build instantly, and still exercising every reshape: 2 kv
# groups of 2 query heads each, 2 layers, 4 experts. The real shapes differ
# only in the numbers; vocab is cut because the embedding dominates otherwise.
TINY = PiperShape(name="tiny", dim=256, n_layers=2, vocab_size=64)


def titan_state(shape: PiperShape) -> dict[str, torch.Tensor]:
    """Titan's state-dict names and shapes, filled with distinct values."""
    counter = iter(range(10**6))

    def tensor(*size: int) -> torch.Tensor:
        # arange, not randn: a transposed or mis-sliced copy of random data
        # can still look plausible, while distinct integers cannot.
        total = 1
        for extent in size:
            total *= extent
        base = next(counter) * total
        return (torch.arange(total, dtype=torch.float32) + base).view(*size)

    state = {
        "tok_embeddings.weight": tensor(shape.vocab_size, shape.dim),
        "lm_head.weight": tensor(shape.vocab_size, shape.dim),
        "norm.weight": tensor(shape.dim),
    }
    for layer in range(shape.n_layers):
        prefix = f"layers.{layer}"
        state.update(
            {
                f"{prefix}.attention.qkv_linear.wq.weight": tensor(
                    shape.n_heads * shape.head_dim, shape.dim
                ),
                f"{prefix}.attention.qkv_linear.wk.weight": tensor(
                    shape.n_kv_heads * shape.head_dim, shape.dim
                ),
                f"{prefix}.attention.qkv_linear.wv.weight": tensor(
                    shape.n_kv_heads * shape.head_dim, shape.dim
                ),
                f"{prefix}.attention_norm.weight": tensor(shape.dim),
                f"{prefix}.attention.wo.weight": tensor(shape.dim, shape.dim),
                f"{prefix}.attention.q_norm.weight": tensor(shape.head_dim),
                f"{prefix}.attention.k_norm.weight": tensor(shape.head_dim),
                f"{prefix}.ffn_norm.weight": tensor(shape.dim),
                f"{prefix}.moe.router.gate.weight": tensor(
                    shape.num_experts, shape.dim
                ),
                f"{prefix}.moe.routed_experts.inner_experts.w1_EFD": tensor(
                    shape.num_experts, shape.moe_hidden_dim, shape.dim
                ),
                f"{prefix}.moe.routed_experts.inner_experts.w2_EDF": tensor(
                    shape.num_experts, shape.dim, shape.moe_hidden_dim
                ),
                f"{prefix}.moe.routed_experts.inner_experts.w3_EFD": tensor(
                    shape.num_experts, shape.moe_hidden_dim, shape.dim
                ),
            }
        )
    return state


class _FakeTitan:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        self._state = state

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._state)


class _FakeMegatron:
    """Zeroed parameters at the names and shapes the map writes."""

    def __init__(self, state: dict[str, torch.Tensor], shape: PiperShape) -> None:
        self._parameters = {
            name: torch.zeros_like(tensor)
            for _, name, tensor in weight_transfers(state, shape)
        }

    def named_parameters(self):
        return self._parameters.items()


class QkvInterleaveTests(unittest.TestCase):
    def test_the_interleave_is_invertible(self) -> None:
        """The one mapping that is proved bitwise rather than numerically.

        A wrong interleave shows up in a full forward pass as rel_l2 of order
        1, which is a slow and ambiguous way to find a five-literal reshape
        bug.
        """
        state = titan_state(TINY)
        wq = state["layers.0.attention.qkv_linear.wq.weight"]
        wk = state["layers.0.attention.qkv_linear.wk.weight"]
        wv = state["layers.0.attention.qkv_linear.wv.weight"]
        grouped = grouped_qkv(wq, wk, wv, TINY)
        self.assertEqual(
            tuple(grouped.shape), (TINY.qkv_out_features, TINY.dim)
        )
        # grouped_qkv already asserts this; calling it directly proves the
        # check itself runs rather than that it happens to be called.
        assert_qkv_roundtrip(grouped, wq, wk, wv, TINY)

    def test_a_broken_interleave_is_caught(self) -> None:
        """The guard must fail on a wrong layout, not merely pass on a right
        one. A plain q/k/v concatenation is the natural wrong answer: it has
        the right shape and the wrong grouping."""
        state = titan_state(TINY)
        wq = state["layers.0.attention.qkv_linear.wq.weight"]
        wk = state["layers.0.attention.qkv_linear.wk.weight"]
        wv = state["layers.0.attention.qkv_linear.wv.weight"]
        flat = torch.cat([wq, wk, wv], dim=0)
        self.assertEqual(tuple(flat.shape), (TINY.qkv_out_features, TINY.dim))
        with self.assertRaisesRegex(AssertionError, "not invertible"):
            assert_qkv_roundtrip(flat, wq, wk, wv, TINY)

    def test_the_grouping_puts_k_and_v_after_each_query_group(self) -> None:
        """The layout itself, stated independently of the roundtrip.

        The roundtrip proves the map is invertible. It does not prove the
        order is megatron's, because an interleave that swapped k and v would
        invert perfectly too.
        """
        state = titan_state(TINY)
        wq = state["layers.0.attention.qkv_linear.wq.weight"]
        wk = state["layers.0.attention.qkv_linear.wk.weight"]
        wv = state["layers.0.attention.qkv_linear.wv.weight"]
        view = grouped_qkv(wq, wk, wv, TINY).view(
            TINY.n_kv_heads, TINY.heads_per_group + 2, TINY.head_dim, TINY.dim
        )
        per_group = wq.view(
            TINY.n_kv_heads, TINY.heads_per_group, TINY.head_dim, TINY.dim
        )
        self.assertTrue(torch.equal(view[:, : TINY.heads_per_group], per_group))
        self.assertTrue(
            torch.equal(
                view[:, TINY.heads_per_group],
                wk.view(TINY.n_kv_heads, TINY.head_dim, TINY.dim),
            )
        )
        self.assertTrue(
            torch.equal(
                view[:, TINY.heads_per_group + 1],
                wv.view(TINY.n_kv_heads, TINY.head_dim, TINY.dim),
            )
        )


class TransferTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = titan_state(TINY)
        self.titan = _FakeTitan(self.state)
        self.megatron = _FakeMegatron(self.state, TINY)

    def test_a_full_transfer_writes_every_parameter(self) -> None:
        written = transfer_weights(self.titan, self.megatron, TINY)
        expected = sum(1 for _ in weight_transfers(self.state, TINY))
        self.assertEqual(written, expected)
        for name, parameter in self.megatron.named_parameters():
            with self.subTest(parameter=name):
                self.assertTrue(parameter.any(), f"{name} was left at zero")

    def test_the_gated_fc1_puts_the_gate_above_the_up_projection(self) -> None:
        """megatron's fc1 rows are [gate (titan w1); up (titan w3)].

        Swapping them is silent: the shapes match and the model still runs,
        it just computes silu(up) * gate.
        """
        transfer_weights(self.titan, self.megatron, TINY)
        fc1 = dict(self.megatron.named_parameters())[
            "decoder.layers.0.mlp.experts.linear_fc1.weight0"
        ]
        w1 = self.state["layers.0.moe.routed_experts.inner_experts.w1_EFD"][0]
        w3 = self.state["layers.0.moe.routed_experts.inner_experts.w3_EFD"][0]
        self.assertTrue(torch.equal(fc1[: TINY.moe_hidden_dim], w1))
        self.assertTrue(torch.equal(fc1[TINY.moe_hidden_dim :], w3))

    def test_a_component_slice_writes_only_that_component(self) -> None:
        """The property the cross-engine arms need.

        A scenario that compares one component must not pay a whole model's
        memory to load the rest.
        """
        written = transfer_weights(
            self.titan, self.megatron, TINY, components=("experts",)
        )
        self.assertEqual(written, TINY.n_layers * TINY.num_experts * 2)
        parameters = dict(self.megatron.named_parameters())
        self.assertTrue(
            parameters["decoder.layers.0.mlp.experts.linear_fc2.weight0"].any()
        )
        self.assertFalse(
            parameters["decoder.layers.0.self_attention.linear_qkv.weight"].any()
        )
        self.assertFalse(parameters["embedding.word_embeddings.weight"].any())

    def test_every_declared_component_is_reachable(self) -> None:
        """A tag nothing yields is a slice that silently transfers nothing."""
        produced = {
            component for component, _, _ in weight_transfers(self.state, TINY)
        }
        self.assertEqual(produced, set(COMPONENTS))

    def test_an_unknown_component_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown weight component"):
            transfer_weights(
                self.titan, self.megatron, TINY, components=("attention",)
            )

    def test_a_slice_that_matches_nothing_is_refused(self) -> None:
        """An empty selection must fail rather than report success.

        A caller that mistyped its filter would otherwise compare two models
        whose weights never met.
        """
        with self.assertRaisesRegex(ValueError, "transferred nothing"):
            transfer_weights(self.titan, self.megatron, TINY, components=())

    def test_a_missing_megatron_parameter_is_refused(self) -> None:
        """A layout divergence must not leave a parameter at its init.

        This is what a megatron bump can do: rename a submodule and every
        weight under it silently stops being transferred.
        """
        del self.megatron._parameters["decoder.layers.1.mlp.router.weight"]
        with self.assertRaisesRegex(KeyError, "has no parameter"):
            transfer_weights(self.titan, self.megatron, TINY)

    def test_a_shape_disagreement_is_refused(self) -> None:
        name = "decoder.layers.0.self_attention.linear_proj.weight"
        self.megatron._parameters[name] = torch.zeros(TINY.dim, TINY.dim + 8)
        with self.assertRaisesRegex(ValueError, "shape"):
            transfer_weights(self.titan, self.megatron, TINY)


if __name__ == "__main__":
    unittest.main()
