"""Data-stream parity between the stock Megatron driver and the replay loader.

The full 40-step bitwise comparison happens implicitly by construction: both
sides drain the same torchtitan dataset class. Here a smaller sample keeps
the CPU suite fast while still catching divergent construction arguments.
"""

import sys
import unittest
from itertools import islice
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.megatron_stock.data import (
    C4_TEST_PATH,
    TOKENIZER_PATH,
)


class PackingParityTests(unittest.TestCase):
    def test_streams_are_bitwise_identical(self) -> None:
        from benchmarks.e2e.data.piper_qwen3 import PretokenizedReplayDataset
        from benchmarks.e2e.megatron_stock.data import materialize_titan_samples
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

        num_samples = 12
        try:
            megatron_side = materialize_titan_samples(
                seq_len=1024, num_samples=num_samples
            )
        except PermissionError as error:
            # The resolved HF datasets cache (possibly a shared read-only
            # tree with root-owned subdirs) rejects the builder lock; only
            # the real load attempt can detect this reliably.
            self.skipTest(
                f"HF datasets cache is not writable ({error}); "
                "export HF_DATASETS_CACHE"
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
        replayed = list(islice(iter(titan_side), num_samples))
        for (m_input, m_pos, m_label), (t_inputs, t_label) in zip(
            megatron_side, replayed
        ):
            self.assertTrue(torch.equal(m_input, t_inputs["input"]))
            self.assertTrue(torch.equal(m_pos, t_inputs["positions"]))
            self.assertTrue(torch.equal(m_label, t_label))

    def _megatron_shard(self, *, dp_rank: int, dp_world_size: int, count: int):
        from benchmarks.e2e.megatron_stock.data import materialize_titan_samples

        try:
            return materialize_titan_samples(
                seq_len=1024,
                num_samples=count,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
            )
        except PermissionError as error:
            self.skipTest(
                f"HF datasets cache is not writable ({error}); "
                "export HF_DATASETS_CACHE"
            )

    def test_two_data_parallel_ranks_read_different_tokens(self) -> None:
        """The hazard of the data-parallel axis, stated as one assertion.

        Two ranks that read the same tokens are not data parallelism. They
        are one step run twice, reported as twice the throughput, and every
        other check passes.
        """
        count = 6
        first = self._megatron_shard(dp_rank=0, dp_world_size=2, count=count)
        second = self._megatron_shard(dp_rank=1, dp_world_size=2, count=count)
        self.assertEqual(len(first), len(second))
        matching = sum(
            1
            for (a_input, _, _), (b_input, _, _) in zip(first, second)
            if torch.equal(a_input, b_input)
        )
        self.assertEqual(matching, 0)

    def test_each_engine_reads_the_same_shard_on_the_same_rank(self) -> None:
        """The parity claim, per rank rather than per run.

        Both sides call the same dataset class with the same ``dp_rank`` and
        ``dp_world_size``, so the split is torchtitan's own on both.
        """
        from benchmarks.e2e.data.piper_qwen3 import PretokenizedReplayDataset
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

        count = 6
        for dp_rank in (0, 1):
            with self.subTest(dp_rank=dp_rank):
                megatron_side = self._megatron_shard(
                    dp_rank=dp_rank, dp_world_size=2, count=count
                )
                tokenizer = HuggingFaceTokenizer(
                    tokenizer_path=str(TOKENIZER_PATH)
                )
                inner = HuggingFaceTextDataset(
                    dataset_name="c4_test",
                    dataset_path=str(C4_TEST_PATH),
                    tokenizer=tokenizer,
                    seq_len=1024,
                    dp_rank=dp_rank,
                    dp_world_size=2,
                    infinite=True,
                )
                titan_side = PretokenizedReplayDataset(
                    inner, num_samples=count
                )
                replayed = list(islice(iter(titan_side), count))
                for (m_input, m_pos, m_label), (t_inputs, t_label) in zip(
                    megatron_side, replayed
                ):
                    self.assertTrue(torch.equal(m_input, t_inputs["input"]))
                    self.assertTrue(
                        torch.equal(m_pos, t_inputs["positions"])
                    )
                    self.assertTrue(torch.equal(m_label, t_label))

    def test_replay_exhaustion_is_loud(self) -> None:
        from benchmarks.e2e.data.piper_qwen3 import PretokenizedReplayDataset

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


if __name__ == "__main__":
    unittest.main()
