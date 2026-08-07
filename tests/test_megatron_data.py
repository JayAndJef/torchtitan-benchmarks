"""Data-stream parity between the Megatron driver and the titan replay loader.

The full 40-step bitwise comparison happens implicitly by construction (both
sides drain the same torchtitan dataset class) and explicitly in
tools/megatron_parity_check.py; here a smaller sample keeps the CPU suite
fast while still catching divergent construction arguments.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from megatron_baseline.data import (
    C4_TEST_PATH,
    TOKENIZER_PATH,
    ThdBatch,
    thd_batches,
)
from megatron_baseline.train import lr_lambda_for


def _hf_cache_writable() -> bool:
    cache = os.environ.get("HF_DATASETS_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "datasets"
    )
    probe = Path(cache)
    try:
        probe.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=probe):
            return True
    except OSError:
        return False


class PackingParityTests(unittest.TestCase):
    @unittest.skipUnless(
        _hf_cache_writable(),
        "HF datasets cache is not writable; export HF_DATASETS_CACHE",
    )
    def test_streams_are_bitwise_identical(self) -> None:
        from megatron_baseline.data import materialize_titan_samples
        from piper1b.pretokenized_data import PretokenizedReplayDataset
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

        num_samples = 12
        megatron_side = materialize_titan_samples(
            seq_len=1024, num_samples=num_samples
        )
        tokenizer = HuggingFaceTokenizer(tokenizer_path=str(TOKENIZER_PATH))
        inner = HuggingFaceTextDataset(
            dataset_name="c4_test",
            dataset_path=str(C4_TEST_PATH),
            tokenizer=tokenizer,
            seq_len=1024,
            infinite=True,
        )
        titan_side = PretokenizedReplayDataset(inner, num_samples=num_samples)
        # islice, not list(): draining past the end trips the loud-exhaustion
        # guard by design.
        from itertools import islice

        replayed = list(islice(iter(titan_side), num_samples))
        for (m_input, m_pos, m_label), (t_inputs, t_label) in zip(
            megatron_side, replayed
        ):
            self.assertTrue(torch.equal(m_input, t_inputs["input"]))
            self.assertTrue(torch.equal(m_pos, t_inputs["positions"]))
            self.assertTrue(torch.equal(m_label, t_label))

    def test_replay_exhaustion_is_loud(self) -> None:
        from piper1b.pretokenized_data import PretokenizedReplayDataset

        class TwoSamples:
            def __iter__(self):
                sample = ({"input": torch.zeros(4)}, torch.zeros(4))
                yield sample
                yield sample

        dataset = PretokenizedReplayDataset(TwoSamples(), num_samples=2)
        iterator = iter(dataset)
        next(iterator)
        next(iterator)
        with self.assertRaisesRegex(RuntimeError, "replay exhausted"):
            next(iterator)


class ThdConversionTests(unittest.TestCase):
    def _sample(self, doc_lens: list[int]):
        tokens = torch.arange(sum(doc_lens), dtype=torch.long)
        positions = torch.cat(
            [torch.arange(n, dtype=torch.long) for n in doc_lens]
        )
        labels = tokens + 1
        return tokens, positions, labels

    def test_cu_seqlens_mark_document_boundaries(self) -> None:
        samples = [
            self._sample([3, 5]),
            self._sample([8]),
        ]
        (batch,) = thd_batches(samples, batch_size=2)
        self.assertIsInstance(batch, ThdBatch)
        self.assertEqual(batch.tokens.shape, (1, 16))
        self.assertEqual(batch.cu_seqlens.tolist(), [0, 3, 8, 16])
        self.assertEqual(batch.cu_seqlens.dtype, torch.int32)
        self.assertEqual(batch.max_seqlen, 8)

    def test_batch_size_must_divide_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not divide"):
            thd_batches([self._sample([4])], batch_size=2)


class LrScheduleTests(unittest.TestCase):
    def test_matches_titan_profile_at_40_steps(self) -> None:
        lr_lambda = lr_lambda_for(40)
        # LambdaLR calls with 0-based epochs; training step = epoch + 1.
        factors = {step: lr_lambda(step - 1) for step in (1, 2, 3, 4, 40, 41)}
        self.assertAlmostEqual(factors[1], 0.5)
        self.assertAlmostEqual(factors[2], 1.0)
        self.assertAlmostEqual(factors[3], 1.0)
        self.assertAlmostEqual(factors[4], 1.0 - 1.0 / 38.0)
        self.assertAlmostEqual(factors[40], 1.0 / 38.0)
        self.assertEqual(factors[41], 0.0)


if __name__ == "__main__":
    unittest.main()
