"""The tuned Megatron driver's THD packing and its learning-rate schedule.

The data-stream parity tests live in tests/test_megatron_stock_data.py,
beside the materializer both engines call.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.megatron.data import (
    ThdBatch,
    padded_microbatches,
    thd_batches,
)
from benchmarks.e2e.megatron.train import lr_lambda_for


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


class PaddedMicrobatchTests(unittest.TestCase):
    """One static ``cu_seqlens`` length, per rank and across ranks.

    A varying document count re-records a captured graph every step, and two
    data-parallel ranks that padded to their own maxima would run different
    static shapes under one label.
    """

    def _packs(self, doc_lens_per_pack: list[list[int]]) -> list[ThdBatch]:
        packs = []
        for doc_lens in doc_lens_per_pack:
            tokens = torch.arange(sum(doc_lens), dtype=torch.long)
            positions = torch.cat(
                [torch.arange(n, dtype=torch.long) for n in doc_lens]
            )
            packs.extend(
                thd_batches(
                    [(tokens, positions, tokens + 1)], batch_size=1
                )
            )
        return packs

    def test_every_pack_of_one_rank_gets_one_length(self) -> None:
        packs = self._packs([[8], [4, 4], [2, 2, 2, 2]])
        widest = max(pack.cu_seqlens.numel() for pack in packs)
        padded = padded_microbatches(
            packs, max_documents=widest, tokens_per_microbatch=8
        )
        self.assertEqual(
            {row["cu_seqlens"].numel() for row in padded}, {widest}
        )

    def test_two_ranks_pad_to_one_length_when_the_max_is_global(self) -> None:
        """The property the driver's all-reduce exists to give.

        The two shards below have different local maxima on purpose: taking
        each rank's own would give 2 and 5 entries, so a test that used the
        local maximum would pass while the ranks disagreed.
        """
        first = self._packs([[8]])
        second = self._packs([[2, 2, 2, 2]])
        self.assertNotEqual(
            first[0].cu_seqlens.numel(), second[0].cu_seqlens.numel()
        )
        widest = max(
            pack.cu_seqlens.numel() for pack in (*first, *second)
        )
        lengths = {
            row["cu_seqlens"].numel()
            for shard in (first, second)
            for row in padded_microbatches(
                shard, max_documents=widest, tokens_per_microbatch=8
            )
        }
        self.assertEqual(lengths, {widest})

    def test_the_padding_adds_no_document(self) -> None:
        """A trailing entry equal to the last offset is a zero-length
        segment."""
        (padded,) = padded_microbatches(
            self._packs([[4, 4]]), max_documents=5, tokens_per_microbatch=8
        )
        self.assertEqual(padded["cu_seqlens"].tolist(), [0, 4, 8, 8, 8])

    def test_a_maximum_below_a_pack_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "below a pack's own"):
            padded_microbatches(
                self._packs([[2, 2, 2, 2]]),
                max_documents=2,
                tokens_per_microbatch=8,
            )


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
