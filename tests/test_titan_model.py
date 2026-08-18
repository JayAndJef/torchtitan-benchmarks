"""The in-process titan build, and the override count that guards it.

Builds a real model at a toy shape on CPU, so the whole path -- registry
config, override application, ``build()``, ``init_states`` -- runs in about a
second. The shapes are the only thing small here; every mechanism under test is
the one a GPU arm uses.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.models.piper_qwen3.config_registry import _piper_1b_model
from benchmarks.models.piper_qwen3.shape import PiperShape
from benchmarks.models.piper_qwen3.titan_model import (
    apply_config_overrides,
    build_titan_model,
)

TINY = PiperShape(name="tiny", dim=256, n_layers=2, vocab_size=64)

SWIGLU_OVERRIDE = (
    "benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu."
    "piper_optimized_inductor_fused_grouped_experts"
)


class OverrideCountTests(unittest.TestCase):
    """The kernel-side equivalent of validation rule 2.

    The e2e run proves an override landed by counting ``[Override]`` log
    lines. In process there are no log lines, so the replacement count is the
    proof, and it uses the same arithmetic: one per block.
    """

    def test_an_override_replaces_one_node_per_block(self) -> None:
        config = _piper_1b_model(fuse_qkv=True, shape=TINY)
        lines = apply_config_overrides(
            config, [SWIGLU_OVERRIDE], expected=TINY.n_layers
        )
        self.assertEqual(len(lines), TINY.n_layers)
        for layer, line in enumerate(lines):
            self.assertIn(
                f"layers.{layer}.moe.routed_experts.inner_experts", line
            )
            self.assertIn("GroupedExperts.Config ->", line)

    def test_a_wrong_expected_count_raises(self) -> None:
        """The check has to bite, or it is decoration.

        A count that does not match means the override claimed a different
        set of nodes than the arm intended -- which is exactly the state the
        arms with no distinctive kernel name cannot otherwise detect.
        """
        config = _piper_1b_model(fuse_qkv=True, shape=TINY)
        with self.assertRaisesRegex(RuntimeError, "expected 4 override"):
            apply_config_overrides(config, [SWIGLU_OVERRIDE], expected=4)

    def test_no_overrides_expects_none_and_touches_nothing(self) -> None:
        config = _piper_1b_model(fuse_qkv=True, shape=TINY)
        self.assertEqual(apply_config_overrides(config, [], expected=0), [])
        model = config.build()
        self.assertEqual(
            type(model.layers["0"].moe.routed_experts.inner_experts).__name__,
            "GroupedExperts",
        )


class BuildTests(unittest.TestCase):
    def test_the_build_produces_the_registry_module_tree(self) -> None:
        """The names the cross-engine weight map reads on the titan side."""
        model = build_titan_model(shape=TINY, device="cpu", dtype=torch.float32)
        state = model.state_dict()
        for name in (
            "tok_embeddings.weight",
            "lm_head.weight",
            "norm.weight",
            "layers.0.attention.qkv_linear.wq.weight",
            "layers.0.attention.q_norm.weight",
            "layers.0.moe.router.gate.weight",
            "layers.0.moe.routed_experts.inner_experts.w1_EFD",
        ):
            with self.subTest(parameter=name):
                self.assertIn(name, state)

    def test_the_dtype_is_delivered_during_construction(self) -> None:
        """Not by a cast afterwards, which is how the run itself gets bf16.

        There is no autocast and no mixed-precision wrapper in this execution
        model, so the default dtype at build time is the whole dtype story. A
        cast after the fact would produce the same parameters here and a
        different init RNG stream.
        """
        model = build_titan_model(
            shape=TINY, device="cpu", dtype=torch.bfloat16
        )
        self.assertIs(model.tok_embeddings.weight.dtype, torch.bfloat16)
        # The default dtype must be restored whatever happens.
        self.assertIs(torch.get_default_dtype(), torch.float32)

    def test_the_seed_makes_two_builds_identical(self) -> None:
        """The correctness pass and the timing pass are different processes.

        A gate that checked different weights than the timing measured would
        prove nothing about the arm that ran.
        """
        first = build_titan_model(shape=TINY, device="cpu", dtype=torch.float32)
        second = build_titan_model(shape=TINY, device="cpu", dtype=torch.float32)
        self.assertTrue(
            torch.equal(
                first.tok_embeddings.weight, second.tok_embeddings.weight
            )
        )
        self.assertTrue(
            torch.equal(
                first.layers["0"].attention.wo.weight,
                second.layers["0"].attention.wo.weight,
            )
        )

    def test_an_override_reaches_the_built_module(self) -> None:
        """The count proves the config changed; this proves the model did."""
        model = build_titan_model(
            shape=TINY,
            device="cpu",
            dtype=torch.float32,
            overrides=[SWIGLU_OVERRIDE],
            overrides_per_block=1,
        )
        experts = model.layers["0"].moe.routed_experts.inner_experts
        self.assertEqual(
            type(experts).__name__, "InductorSwiGLUFusedGroupedExperts"
        )

    def test_an_override_without_a_declared_count_is_refused(self) -> None:
        """Passing overrides and leaving overrides_per_block at 0 is a bug.

        It would apply them and then assert that nothing was replaced, which
        is the wrong direction of failure to leave to chance.
        """
        with self.assertRaises(RuntimeError):
            build_titan_model(
                shape=TINY,
                device="cpu",
                dtype=torch.float32,
                overrides=[SWIGLU_OVERRIDE],
            )

    def test_the_unfused_qkv_variant_builds_a_different_module(self) -> None:
        """fuse_qkv is the structural axis piper1b_qkv's two arms differ on."""
        fused = build_titan_model(
            shape=TINY, device="cpu", dtype=torch.float32
        )
        unfused = build_titan_model(
            shape=TINY, device="cpu", dtype=torch.float32, fuse_qkv=False
        )
        self.assertNotEqual(
            type(fused.layers["0"].attention.qkv_linear).__name__,
            type(unfused.layers["0"].attention.qkv_linear).__name__,
        )


if __name__ == "__main__":
    unittest.main()
