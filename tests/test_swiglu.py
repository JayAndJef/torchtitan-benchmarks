"""Tests for the Piper combined-layout grouped-expert SwiGLU override."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from piper1b.swiglu.combined_swiglu import (
    combined_silu_and_mul_op,
    CombinedSwiGLUFusedGroupedExperts,
    piper_optimized_fused_grouped_experts,
)
from torchtitan.models.common.moe import GroupedExperts


class CombinedSwiGLUConfigTests(unittest.TestCase):
    def test_override_builds_local_fused_grouped_experts(self) -> None:
        config = GroupedExperts.Config(dim=16, hidden_dim=32, num_experts=4)

        replacement = piper_optimized_fused_grouped_experts(config)
        module = replacement.build()

        self.assertIsInstance(
            replacement,
            CombinedSwiGLUFusedGroupedExperts.Config,
        )
        self.assertIsInstance(module, CombinedSwiGLUFusedGroupedExperts)
        self.assertEqual(
            {name for name, _ in module.named_parameters(recurse=False)},
            {"w13", "w2_EDF"},
        )

    def test_checkpoint_uses_stock_grouped_expert_layout(self) -> None:
        config = CombinedSwiGLUFusedGroupedExperts.Config(
            dim=16,
            hidden_dim=32,
            num_experts=4,
        )
        source = config.build()
        with torch.no_grad():
            source.w13.copy_(torch.randn_like(source.w13))
            source.w2_EDF.copy_(torch.randn_like(source.w2_EDF))

        state = source.state_dict()
        destination = config.build()
        destination.load_state_dict(state)

        self.assertEqual(set(state), {"w1_EFD", "w2_EDF", "w3_EFD"})
        self.assertTrue(torch.equal(destination.w13, source.w13))
        self.assertTrue(torch.equal(destination.w2_EDF, source.w2_EDF))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CombinedSwiGLUCUDATests(unittest.TestCase):
    def test_compiled_forward_and_combined_gradient_match_upstream_op(self) -> None:
        from torchtitan.overrides.fused_swiglu import silu_and_mul_op

        torch.manual_seed(42)
        combined_reference = torch.randn(
            8,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        combined_actual = combined_reference.detach().clone().requires_grad_()
        offsets = torch.tensor([2, 5, 8], device="cuda", dtype=torch.int32)
        grad_out = torch.randn(
            8,
            64,
            device="cuda",
            dtype=torch.bfloat16,
        )

        gate, up = combined_reference.reshape(8, 64, 2).unbind(-1)
        expected = silu_and_mul_op(gate, up, offsets)

        def apply(combined, expert_offsets):
            return combined_silu_and_mul_op(combined, expert_offsets)

        actual = torch.compile(apply, fullgraph=True)(combined_actual, offsets)
        expected.backward(grad_out)
        actual.backward(grad_out)

        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(
            torch.equal(combined_actual.grad, combined_reference.grad)
        )

    def test_grouped_expert_module_matches_upstream_fused_override(self) -> None:
        from torchtitan.overrides.fused_swiglu import FusedGroupedExperts

        config_kwargs = {"dim": 64, "hidden_dim": 128, "num_experts": 4}
        upstream = FusedGroupedExperts.Config(**config_kwargs).build().cuda()
        optimized = CombinedSwiGLUFusedGroupedExperts.Config(
            **config_kwargs
        ).build().cuda()
        with torch.no_grad():
            upstream.w13.copy_(torch.randn_like(upstream.w13))
            upstream.w2_EDF.copy_(torch.randn_like(upstream.w2_EDF))
            optimized.w13.copy_(upstream.w13)
            optimized.w2_EDF.copy_(upstream.w2_EDF)

        num_tokens = torch.tensor([3, 2, 1, 2], device="cuda")
        input_upstream = torch.randn(
            8,
            64,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        input_optimized = input_upstream.detach().clone().requires_grad_()
        grad_out = torch.randn_like(input_upstream)

        output_upstream = upstream(input_upstream, num_tokens)
        output_optimized = optimized(input_optimized, num_tokens)
        output_upstream.backward(grad_out)
        output_optimized.backward(grad_out)

        self.assertTrue(torch.equal(output_optimized, output_upstream))
        self.assertTrue(torch.equal(input_optimized.grad, input_upstream.grad))
        self.assertTrue(torch.equal(optimized.w13.grad, upstream.w13.grad))
        self.assertTrue(torch.equal(optimized.w2_EDF.grad, upstream.w2_EDF.grad))


if __name__ == "__main__":
    unittest.main()
