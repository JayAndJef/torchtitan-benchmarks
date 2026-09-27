"""The pre-tokenized c4_test stream that every engine trains on, materialized at startup.

Each engine adapts the samples to its own loader. So both engines read the
same tokens, and no measured step pays a data cost.
"""

from __future__ import annotations

import torch

from benchmarks.e2e.engines.api import DataSpec
from benchmarks.execution.paths import TITAN_DIR


DATASET = "c4_test"
"""The one dataset that the stream holds."""

C4_TEST_PATH = TITAN_DIR / "tests" / "assets" / "c4_test"
TOKENIZER_PATH = TITAN_DIR / "tests" / "assets" / "tokenizer"

Sample = tuple[dict[str, torch.Tensor], torch.Tensor]
"""One sample of TorchTitan's text dataset: the ``input`` and ``positions`` tensors, and the labels."""


def materialize(spec: DataSpec, dp_rank: int, dp_world_size: int) -> list[Sample]:
    """The ``spec.steps * spec.local_batch_size`` samples of one data-parallel slice, in stream order."""
    if spec.dataset != DATASET:
        raise ValueError(
            f"the replay stream holds the dataset {DATASET!r} alone, and the "
            f"run asks for {spec.dataset!r}"
        )
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

    dataset = HuggingFaceTextDataset(
        dataset_name=DATASET,
        dataset_path=str(C4_TEST_PATH),
        tokenizer=HuggingFaceTokenizer(tokenizer_path=str(TOKENIZER_PATH)),
        seq_len=spec.seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=True,
    )
    iterator = iter(dataset)
    return [next(iterator) for _ in range(spec.steps * spec.local_batch_size)]
