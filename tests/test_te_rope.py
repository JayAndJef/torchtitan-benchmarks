"""CUDA correctness tests for the TransformerEngine-derived RoPE override."""

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def _has_gcc_13() -> bool:
    try:
        version = subprocess.check_output(
            ["g++", "-dumpfullversion"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return int(version.split(".", 1)[0]) >= 13


@unittest.skipUnless(
    torch.cuda.is_available() and _has_gcc_13(),
    "CUDA and a C++20-capable GCC are required for TE RoPE tests",
)
class TransformerEngineRoPETests(unittest.TestCase):
    def test_compiled_packed_positions_match_stock_forward_and_backward(self) -> None:
        from piper1b.rope.te_rope_override import TECosSinRoPE
        from torchtitan.models.common.rope import CosSinRoPE

        config_kwargs = {
            "dim": 64,
            "max_seq_len": 128,
            "theta": 1_000_000.0,
        }
        reference = CosSinRoPE(CosSinRoPE.Config(**config_kwargs)).cuda()
        optimized = TECosSinRoPE(TECosSinRoPE.Config(**config_kwargs)).cuda()

        positions = torch.tensor(
            [
                [0, 1, 2, 0, 1, 2, 3, 0],
                [0, 1, 0, 1, 2, 0, 1, 2],
            ],
            device="cuda",
            dtype=torch.int64,
        )
        torch.manual_seed(42)
        query_ref = torch.randn(
            2,
            8,
            16,
            64,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        key_ref = torch.randn(
            2,
            8,
            8,
            64,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        query_te = query_ref.detach().clone().requires_grad_()
        key_te = key_ref.detach().clone().requires_grad_()
        grad_query = torch.randn_like(query_ref)
        grad_key = torch.randn_like(key_ref)

        query_expected, key_expected = reference(query_ref, key_ref, positions)

        def apply_te(query, key, token_positions):
            return optimized(query, key, token_positions)

        query_actual, key_actual = torch.compile(
            apply_te,
            fullgraph=True,
        )(query_te, key_te, positions)

        torch.autograd.backward(
            (query_expected, key_expected),
            (grad_query, grad_key),
        )
        torch.autograd.backward(
            (query_actual, key_actual),
            (grad_query, grad_key),
        )

        self.assertTrue(torch.equal(query_actual, query_expected))
        self.assertTrue(torch.equal(key_actual, key_expected))
        self.assertTrue(torch.equal(query_te.grad, query_ref.grad))
        self.assertTrue(torch.equal(key_te.grad, key_ref.grad))


if __name__ == "__main__":
    unittest.main()
