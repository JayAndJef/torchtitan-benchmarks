"""The TorchTitan dataloader that replays the samples of the shared c4_test stream."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset, default_collate

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer

from benchmarks.e2e.data.c4_replay import Sample, materialize
from benchmarks.e2e.engines.api import REPLAY_DATASET, DataSpec
from benchmarks.models.piper_qwen3.components.attention.packed import (
    microbatch_offsets,
)


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


def collate_with_offsets(
    samples: list[Sample], *, rows_per_microbatch: int
) -> Sample:
    """The default batch, plus the offsets of each microbatch under the input key ``attention_masks``."""
    inputs, labels = default_collate(samples)
    inputs["attention_masks"] = microbatch_offsets(
        inputs["positions"], rows_per_microbatch
    )
    return inputs, labels


class PretokenizedReplayDataLoader(ParallelAwareDataloader):
    """The dataloader of one data-parallel rank, over the samples that ``materialize`` gives it."""

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        dataset: str = REPLAY_DATASET
        replay_steps: int = 40
        """The steps whose samples the loader materializes; it must be at least ``--training.steps``."""
        offset_rows: int = 0
        """The rows of one pipeline microbatch, whose document offsets each batch carries; 0 sends no offsets."""

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
        if config.dataset_path is not None:
            raise ValueError(
                f"the replay loader reads {REPLAY_DATASET!r} from the TorchTitan "
                f"checkout, and the config names the path {config.dataset_path!r}; "
                "remove --dataloader.dataset-path"
            )
        if config.offset_rows < 0 or (
            config.offset_rows and local_batch_size % config.offset_rows
        ):
            raise ValueError(
                f"--dataloader.offset-rows {config.offset_rows} must be 0, or "
                f"divide the local batch size {local_batch_size}"
            )
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
            # None is the default collate, so the other arms keep their batches.
            collate_fn=(
                partial(collate_with_offsets, rows_per_microbatch=config.offset_rows)
                if config.offset_rows
                else None
            ),
        )
