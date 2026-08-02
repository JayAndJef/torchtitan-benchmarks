"""Numerical tests for the piper-1B LM-head benchmark losses."""

import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from piper1b.fused_losses import FusedLinearCrossEntropyLoss
from piper1b.te_cross_entropy import parallel_cross_entropy


class TransformerEngineSourceTests(unittest.TestCase):
    def test_vendored_sources_only_change_import_paths(self) -> None:
        root = Path(__file__).resolve().parent.parent / "piper1b"
        sources = {
            "te_common_cross_entropy.py": (
                "1af41ec1dca60268887daf279219873cbb34df6d1e8b1c3c64dc75e70cbb45c9",
                (),
            ),
            "te_triton_cross_entropy.py": (
                "1c7eec23f612e022303bb11ff33d2558ec9e10abd7aaefc34fbedebadcd3dea5",
                (
                    (
                        "from piper1b.te_common_cross_entropy import (",
                        "from transformer_engine.common.triton.cross_entropy import (",
                    ),
                ),
            ),
            "te_cross_entropy.py": (
                "0a1670475dc40f62b3c52163967b6fe97679172f253424b28492699c5cbfd563",
                (
                    (
                        "from piper1b import te_triton_cross_entropy as triton_cross_entropy",
                        "import transformer_engine.pytorch.triton.cross_entropy as triton_cross_entropy",
                    ),
                ),
            ),
        }
        for filename, (expected_hash, replacements) in sources.items():
            source = (root / filename).read_text()
            for local_import, upstream_import in replacements:
                source = source.replace(local_import, upstream_import)
            actual_hash = hashlib.sha256(source.encode()).hexdigest()
            self.assertEqual(actual_hash, expected_hash, filename)


class FusedLinearCrossEntropyTests(unittest.TestCase):
    def test_loss_and_gradients_match_full_logits(self) -> None:
        torch.manual_seed(42)
        hidden_ref = torch.randn(2, 4, 3, requires_grad=True)
        hidden_fused = hidden_ref.detach().clone().requires_grad_()
        weight_ref = torch.randn(7, 3, requires_grad=True)
        head_fused = torch.nn.Linear(3, 7, bias=False)
        with torch.no_grad():
            head_fused.weight.copy_(weight_ref)
        labels = torch.tensor([[0, 1, -100, 3], [4, 5, 6, 0]])
        num_valid = float((labels != -100).sum())

        ref_loss = F.cross_entropy(
            F.linear(hidden_ref, weight_ref).flatten(0, 1).float(),
            labels.flatten(),
            reduction="sum",
            ignore_index=-100,
        ) / num_valid
        ref_loss.backward()

        fused = FusedLinearCrossEntropyLoss.Config(
            batch_chunk_size=2,
            chunking_method=None,
        ).build(compile_config=None)
        fused.set_lm_head(head_fused)
        fused_loss, metrics = fused(hidden_fused, labels, num_valid)
        fused_loss.backward()

        self.assertEqual(metrics, {})
        torch.testing.assert_close(
            fused_loss,
            ref_loss.detach(),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            hidden_fused.grad,
            hidden_ref.grad,
            rtol=1e-4,
            atol=1e-5,
        )
        torch.testing.assert_close(
            head_fused.weight.grad,
            weight_ref.grad,
            rtol=1e-4,
            atol=1e-5,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for TE Triton tests")
class TECrossEntropyTests(unittest.TestCase):
    def _check(self, dtype: torch.dtype, vocab_size: int) -> None:
        torch.manual_seed(42)
        logits = torch.randn(
            1,
            4,
            vocab_size,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
        labels = torch.tensor([[0, vocab_size - 1, -100, 17]], device="cuda")
        external_grad = torch.tensor([[0.1, 0.2, 0.7, 0.4]], device="cuda")

        ref_logits = logits.detach().float().requires_grad_()
        ref_loss = F.cross_entropy(
            ref_logits.flatten(0, 1),
            labels.flatten(),
            reduction="none",
            ignore_index=-100,
        ).reshape_as(labels)
        ref_loss.backward(external_grad)

        te_loss = parallel_cross_entropy(logits, labels, reduce_loss=False)
        te_loss.backward(external_grad)

        torch.testing.assert_close(te_loss, ref_loss.detach(), rtol=2e-5, atol=2e-5)
        expected_grad = ref_logits.grad.to(dtype)
        torch.testing.assert_close(
            logits.grad,
            expected_grad,
            rtol=2e-2 if dtype == torch.bfloat16 else 2e-5,
            atol=2e-3 if dtype == torch.bfloat16 else 2e-6,
        )
        self.assertEqual(float(te_loss[0, 2].detach()), 0.0)
        self.assertTrue(torch.count_nonzero(logits.grad[0, 2]) == 0)

    def test_float32(self) -> None:
        self._check(torch.float32, 257)

    def test_bfloat16(self) -> None:
        self._check(torch.bfloat16, 257)

    def test_piper_non_power_of_two_vocab(self) -> None:
        self._check(torch.bfloat16, 151936)


if __name__ == "__main__":
    unittest.main()
