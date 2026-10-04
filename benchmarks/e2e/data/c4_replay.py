"""The pre-tokenized c4_test stream that every engine trains on, materialized at startup."""

from __future__ import annotations

import torch

from benchmarks.e2e.engines.api import REPLAY_DATASET, DataSpec
from benchmarks.execution.paths import TITAN_DIR

C4_TEST_PATH = TITAN_DIR / "tests" / "assets" / "c4_test"
TOKENIZER_PATH = TITAN_DIR / "tests" / "assets" / "tokenizer"

Sample = tuple[dict[str, torch.Tensor], torch.Tensor]
"""One sample of TorchTitan's text dataset: the ``input`` and ``positions`` tensors, and the labels."""


def materialize(spec: DataSpec, dp_rank: int, dp_world_size: int) -> list[Sample]:
    """The ``spec.steps * spec.local_batch_size`` samples of one data-parallel slice, in stream order."""
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

    dataset = HuggingFaceTextDataset(
        dataset_name=REPLAY_DATASET,
        dataset_path=str(C4_TEST_PATH),
        tokenizer=HuggingFaceTokenizer(tokenizer_path=str(TOKENIZER_PATH)),
        seq_len=spec.seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=True,
    )
    iterator = iter(dataset)
    return [next(iterator) for _ in range(spec.steps * spec.local_batch_size)]
