"""Data-stream parity: both engines read the shared c4_test stream through ``materialize``.

A small sample keeps the CPU suite fast and still catches a divergent
construction argument.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.data.c4_replay import C4_TEST_PATH, TOKENIZER_PATH, materialize
from benchmarks.e2e.engines.api import REPLAY_DATASET, DataSpec

SEQ_LEN = 1024
BATCH = 4


def _spec(steps: int) -> DataSpec:
    return DataSpec(
        dataset=REPLAY_DATASET, seq_len=SEQ_LEN, local_batch_size=BATCH, steps=steps
    )


class PackingParityTests(unittest.TestCase):
    def _skip_on_a_read_only_cache(self, action):
        """Run ``action``, or skip when the HF datasets cache rejects the builder lock."""
        try:
            return action()
        except PermissionError as error:
            self.skipTest(
                f"HF datasets cache is not writable ({error}); "
                "export HF_DATASETS_CACHE"
            )

    def _reference(self, *, count: int, dp_rank: int, dp_world_size: int):
        """The first ``count`` samples of TorchTitan's own dataset class, built here."""
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

        dataset = HuggingFaceTextDataset(
            dataset_name="c4_test",
            dataset_path=str(C4_TEST_PATH),
            tokenizer=HuggingFaceTokenizer(tokenizer_path=str(TOKENIZER_PATH)),
            seq_len=SEQ_LEN,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=True,
        )
        iterator = iter(dataset)
        return [next(iterator) for _ in range(count)]

    def _titan_rows(self, *, steps: int, dp_rank: int, dp_world_size: int):
        """The (input, positions, label) rows of the TorchTitan replay loader, in batch order."""
        from benchmarks.e2e.engines.torchtitan.plugins.replay import (
            PretokenizedReplayDataLoader,
        )

        loader = self._skip_on_a_read_only_cache(
            lambda: PretokenizedReplayDataLoader(
                PretokenizedReplayDataLoader.Config(replay_steps=steps),
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                tokenizer=None,
                seq_len=SEQ_LEN,
                local_batch_size=BATCH,
            )
        )
        rows = []
        iterator = iter(loader)
        for _ in range(steps):
            inputs, labels = next(iterator)
            rows.extend(
                zip(inputs["input"], inputs["positions"], labels, strict=True)
            )
        return rows

    def _megatron_rows(self, *, steps: int, dp_rank: int, dp_world_size: int):
        """The (input, positions, label) rows of the Megatron driver's iterator, one sample to a microbatch."""
        from benchmarks.e2e.engines.megatron_stock.driver import data

        iterator = self._skip_on_a_read_only_cache(
            lambda: data.build_iterator(
                _spec(steps),
                rows_per_sample=1,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
            )
        )
        return [
            (
                microbatch["tokens"][0],
                microbatch["position_ids"][0],
                microbatch["labels"][0],
            )
            for microbatch in (
                next(iterator) for _ in range(iterator.microbatch_count)
            )
        ]

    def _assert_rows_equal(self, rows, reference) -> None:
        self.assertEqual(len(rows), len(reference))
        for (tokens, positions, labels), (inputs, label) in zip(rows, reference):
            self.assertTrue(torch.equal(tokens, inputs["input"]))
            self.assertTrue(torch.equal(positions, inputs["positions"]))
            self.assertTrue(torch.equal(labels, label))

    def test_each_engine_reads_the_reference_stream(self) -> None:
        steps = 3
        for dp_rank, dp_world_size in ((0, 1), (0, 2), (1, 2)):
            with self.subTest(dp_rank=dp_rank, dp_world_size=dp_world_size):
                reference = self._skip_on_a_read_only_cache(
                    lambda: self._reference(
                        count=steps * BATCH,
                        dp_rank=dp_rank,
                        dp_world_size=dp_world_size,
                    )
                )
                for engine, rows in (
                    (
                        "torchtitan",
                        self._titan_rows(
                            steps=steps, dp_rank=dp_rank, dp_world_size=dp_world_size
                        ),
                    ),
                    (
                        "megatron_stock",
                        self._megatron_rows(
                            steps=steps, dp_rank=dp_rank, dp_world_size=dp_world_size
                        ),
                    ),
                ):
                    with self.subTest(engine=engine):
                        self._assert_rows_equal(rows, reference)

    def test_two_data_parallel_ranks_read_different_tokens(self) -> None:
        """Two ranks that read the same tokens run one step twice and report twice the throughput."""
        first, second = (
            self._skip_on_a_read_only_cache(
                lambda rank=rank: materialize(_spec(2), rank, 2)
            )
            for rank in (0, 1)
        )
        self.assertEqual(len(first), len(second))
        matching = sum(
            1
            for (a_inputs, _), (b_inputs, _) in zip(first, second)
            if torch.equal(a_inputs["input"], b_inputs["input"])
        )
        self.assertEqual(matching, 0)

    def test_another_dataset_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "'c4_test' alone"):
            DataSpec(dataset="c4", seq_len=SEQ_LEN, local_batch_size=1, steps=1)

    def test_a_dataset_path_is_refused(self) -> None:
        from benchmarks.e2e.engines.torchtitan.plugins.replay import (
            PretokenizedReplayDataLoader,
        )

        with self.assertRaisesRegex(ValueError, "remove --dataloader.dataset-path"):
            PretokenizedReplayDataLoader(
                PretokenizedReplayDataLoader.Config(dataset_path="/tmp/c4"),
                dp_world_size=1,
                dp_rank=0,
                tokenizer=None,
                seq_len=SEQ_LEN,
                local_batch_size=BATCH,
            )

    def test_replay_exhaustion_is_loud(self) -> None:
        from benchmarks.e2e.engines.torchtitan.plugins.replay import (
            PretokenizedReplayDataset,
        )

        sample = ({"input": torch.zeros(4)}, torch.zeros(4))
        iterator = iter(PretokenizedReplayDataset([sample, sample]))
        next(iterator)
        next(iterator)
        with self.assertRaisesRegex(RuntimeError, "replay exhausted"):
            next(iterator)


if __name__ == "__main__":
    unittest.main()
