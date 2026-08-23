"""Pre-tokenized replay dataloader for the engine-comparison scenario.

Wraps the stock torchtitan c4_test pipeline (same dataset class, same
tokenizer), drains the first ``replay_steps x local_batch_size`` samples at
construction, and replays the materialized tensors during training. Measured
steps therefore carry ~zero data-host cost, matching the Megatron driver's
treatment (benchmarks/e2e/megatron/data.py drains the identical class), so
the two engines see bit-identical token streams and identical per-step host
work.

Requesting more batches than were materialized is a hard error, not a wrap:
silently reusing data would change the workload relative to the lazy loader.
``replay_steps`` therefore tracks the run's step count: the config registry
defaults it to the config's own ``training.steps`` and the benchmark runner
delivers ``--dataloader.replay-steps`` next to ``--training.steps`` for every
arm of a scenario whose workload sets ``replay_dataloader``.

**Under a data-parallel degree each rank replays its own shard.** The
forwarding was always here: ``dp_rank`` and ``dp_world_size`` have reached
the stock dataset class since this loader was written, and its
``split_dataset_by_node`` is the split TorchTitan's own loader uses. What
stood above it was a refusal of ``dp_world_size != 1``, kept while no run
could ask for one, and only that refusal is gone. The megatron driver
drains the same class with the same two values, so the engines stay
bit-identical rank for rank. The materialized count is per rank, and a rank
still hard-fails at exhaustion.
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
        """Training steps' worth of samples to materialize.

        Must be >= --training.steps or the run dies when the stream is
        exhausted. The registry sets it from the config's own step count and
        the runner overrides both together."""

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
        # Each data-parallel rank materializes its OWN shard of the stream.
        # ``HuggingFaceTextDataset`` calls ``split_dataset_by_node(ds,
        # dp_rank, dp_world_size)``, so the two arguments below are what give
        # the ranks different tokens; the megatron driver drains the same
        # class with the same two values, which is what keeps the two engines
        # bit-identical under a data-parallel degree as well. The count is
        # per rank, so a dp 2 run still replays ``replay_steps`` steps.
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
