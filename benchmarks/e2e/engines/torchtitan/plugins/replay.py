"""The TorchTitan dataloader that replays the samples of the shared c4_test stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer

from benchmarks.e2e.data.c4_replay import DATASET, Sample, materialize
from benchmarks.e2e.engines.api import DataSpec


class PretokenizedReplayDataset(IterableDataset, Stateful):
    """The samples of the run, in order; a step past the last sample raises."""

    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples

    def __iter__(self):
        yield from self._samples
        raise RuntimeError(
            f"pre-tokenized replay exhausted after {len(self._samples)} "
            "samples: --training.steps exceeds --dataloader.replay-steps"
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        del state_dict


class PretokenizedReplayDataLoader(ParallelAwareDataloader):
    """The dataloader of one data-parallel rank, over the samples that ``materialize`` gives it."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = DATASET
        replay_steps: int = 40
        """The steps whose samples the loader materializes; it must be at least ``--training.steps``."""

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
    ):
        # materialize reads the same tokenizer files as the trainer.
        del tokenizer
        spec = DataSpec(
            dataset=config.dataset,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
            steps=config.replay_steps,
        )
        super().__init__(
            PretokenizedReplayDataset(materialize(spec, dp_rank, dp_world_size)),
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            snapshot_every_n_steps=snapshot_every_n_steps,
            batch_size=local_batch_size,
        )
