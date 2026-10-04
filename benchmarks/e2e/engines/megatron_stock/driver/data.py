"""The shared c4_test stream, packed for Megatron's external dataloader."""

from __future__ import annotations

import torch

from benchmarks.e2e.data.c4_replay import Sample, materialize
from benchmarks.e2e.engines.api import REPLAY_DATASET, DataSpec


MICROBATCH_KEYS: tuple[str, ...] = (
    "tokens",
    "labels",
    "loss_mask",
    "position_ids",
    "cu_seqlens",
    "max_seqlen",
    "attention_mask",
    "cu_seqlens_padded",
)
"""The keys of one microbatch; each key of ``pretrain_gpt``'s ``BATCH_KEYS`` that is absent here reaches ``get_batch`` as ``None``."""


def document_offsets(positions: torch.Tensor, seq_len: int) -> torch.Tensor:
    """The document offsets ``[0, ..., seq_len]`` of one packed sample, in int32 and without padding."""
    if positions.numel() != seq_len:
        raise ValueError(
            f"a sample carries {positions.numel()} positions, not the "
            f"{seq_len} the sequence length declares"
        )
    starts = (positions == 0).nonzero(as_tuple=True)[0].to(torch.int32)
    if starts.numel() == 0 or int(starts[0].item()) != 0:
        raise ValueError(
            "the first position of a sample is not a document start, so the "
            "packed-document boundaries cannot be read from it"
        )
    return torch.cat(
        [starts, torch.tensor([seq_len], dtype=torch.int32)]
    )


def _microbatch(
    rows: list[Sample], *, packed_len: int
) -> dict[str, torch.Tensor | None]:
    """One microbatch on the CPU: one packed row of ``packed_len`` tokens, with ``cu_seqlens`` padded to ``packed_len + 1`` entries as ``GPTDataset`` pads it."""
    tokens = torch.cat([inputs["input"] for inputs, _ in rows]).to(torch.int64)
    positions = torch.cat([inputs["positions"] for inputs, _ in rows]).to(torch.int64)
    labels = torch.cat([label for _, label in rows]).to(torch.int64)
    cu_seqlens = document_offsets(positions, packed_len)
    longest = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
    # Megatron's get_batch strips the trailing copies of packed_len.
    padded = torch.cat(
        [
            cu_seqlens,
            torch.full(
                (packed_len + 1 - cu_seqlens.numel(),),
                packed_len,
                dtype=torch.int32,
            ),
        ]
    )
    return {
        "tokens": tokens.unsqueeze(0),
        "labels": labels.unsqueeze(0),
        # Each token counts toward the loss, as in TorchTitan.
        "loss_mask": torch.ones(
            (1, packed_len), dtype=torch.float32
        ),
        "position_ids": positions.unsqueeze(0),
        "cu_seqlens": padded.unsqueeze(0),
        "max_seqlen": torch.tensor([longest], dtype=torch.int32),
        "attention_mask": None,
        "cu_seqlens_padded": None,
    }


class StockReplayIterator:
    """The microbatches of the whole run, built once and served in order; exhaustion raises."""

    def __init__(
        self,
        samples: list[Sample],
        *,
        rows_per_sample: int,
        seq_len: int,
    ) -> None:
        if rows_per_sample < 1:
            raise ValueError(
                f"rows per sample {rows_per_sample} must be >= 1"
            )
        if len(samples) % rows_per_sample:
            raise ValueError(
                f"{len(samples)} samples do not divide into packs of "
                f"{rows_per_sample} row(s)"
            )
        self._packed_len = rows_per_sample * seq_len
        groups = [
            samples[start : start + rows_per_sample]
            for start in range(0, len(samples), rows_per_sample)
        ]
        self._microbatches = [
            _microbatch(group, packed_len=self._packed_len) for group in groups
        ]
        self._served = 0

    @property
    def packed_len(self) -> int:
        """The tokens in one microbatch."""
        return self._packed_len

    @property
    def microbatch_count(self) -> int:
        return len(self._microbatches)

    def __iter__(self) -> "StockReplayIterator":
        return self

    def __next__(self) -> dict[str, torch.Tensor | None]:
        if self._served >= len(self._microbatches):
            raise RuntimeError(
                f"the stock replay stream is exhausted after "
                f"{len(self._microbatches)} microbatches; raise the sample "
                "count rather than wrapping, because a wrap trains a second "
                "epoch under the first epoch's label"
            )
        microbatch = self._microbatches[self._served]
        self._served += 1
        return microbatch


def build_iterator(
    spec: DataSpec, *, rows_per_sample: int, dp_rank: int, dp_world_size: int
) -> StockReplayIterator:
    """The stream of this rank, packed ``rows_per_sample`` rows to a microbatch."""
    return StockReplayIterator(
        materialize(spec, dp_rank, dp_world_size),
        rows_per_sample=rows_per_sample,
        seq_len=spec.seq_len,
    )


def train_valid_test_datasets_provider(
    train_val_test_num_samples: object,
) -> tuple[StockReplayIterator, None, None]:
    """The dataset provider that the driver gives ``pretrain``; it ignores Megatron's sample counts."""
    from megatron.core import mpu
    from megatron.training import get_args

    args = get_args()
    resolved = mpu.get_data_parallel_world_size()
    if resolved != args.data_parallel_size:
        raise RuntimeError(
            f"megatron built a data-parallel group of {resolved} rank(s) "
            f"where its own arguments say {args.data_parallel_size}; the "
            "token slice and the recorded mesh would disagree"
        )
    spec = DataSpec(
        dataset=REPLAY_DATASET,
        # One TorchTitan row, because --seq-length is the packed sample.
        seq_len=args.bench_seq_len,
        local_batch_size=args.bench_local_batch_size,
        steps=args.train_iters,
    )
    return (
        build_iterator(
            spec,
            rows_per_sample=args.bench_rows_per_sample,
            dp_rank=mpu.get_data_parallel_rank(),
            dp_world_size=mpu.get_data_parallel_world_size(),
        ),
        None,
        None,
    )

