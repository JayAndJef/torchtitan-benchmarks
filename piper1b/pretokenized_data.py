"""Pre-tokenized replay dataloader for the engine-comparison scenario.

Wraps the stock torchtitan c4_test pipeline (same dataset class, same
tokenizer), drains the first ``replay_steps x local_batch_size`` samples at
construction, and replays the materialized tensors during training. Measured
steps therefore carry ~zero data-host cost, matching the Megatron driver's
treatment (megatron_baseline/data.py drains the identical class), so the two
engines see bit-identical token streams and identical per-step host work.

Requesting more batches than were materialized is a hard error, not a wrap:
silently reusing data would change the workload relative to the lazy loader.
The scenario using this loader pins --training.steps accordingly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets.text_datasets import (
    HuggingFaceTextDataLoader,
    HuggingFaceTextDataset,
)


class PretokenizedReplayDataset(IterableDataset, Stateful):
    """Materialize the first samples of an inner dataset, then replay them."""

    def __init__(self, inner: IterableDataset, *, num_samples: int) -> None:
        iterator = iter(inner)
        self._samples: list[tuple[dict[str, torch.Tensor], torch.Tensor]] = [
            next(iterator) for _ in range(num_samples)
        ]

    def __iter__(self):
        yield from self._samples
        raise RuntimeError(
            f"pre-tokenized replay exhausted after {len(self._samples)} "
            "samples: --training.steps exceeds the scenario's replay_steps"
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict


class PretokenizedReplayDataLoader(ParallelAwareDataloader):
    """HuggingFaceTextDataLoader twin that pre-materializes the sample stream."""

    @dataclass(kw_only=True, slots=True)
    class Config(HuggingFaceTextDataLoader.Config):
        replay_steps: int = 40
        """Number of training steps' worth of samples to materialize."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs,
    ):
        if dp_world_size != 1:
            raise ValueError(
                "PretokenizedReplayDataLoader is single-GPU only "
                f"(dp_world_size={dp_world_size})"
            )
        inner = HuggingFaceTextDataset(
            dataset_name=config.dataset,
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            infinite=config.infinite,
        )
        dataset = PretokenizedReplayDataset(
            inner, num_samples=config.replay_steps * local_batch_size
        )
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            snapshot_every_n_steps=snapshot_every_n_steps,
            batch_size=local_batch_size,
        )
