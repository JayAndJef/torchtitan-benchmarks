"""The c4_test stream of TorchTitan's own dataset class, packed for Megatron's external dataloader."""

from __future__ import annotations

import torch

from benchmarks.execution.paths import TITAN_DIR

C4_TEST_PATH = TITAN_DIR / "tests" / "assets" / "c4_test"
TOKENIZER_PATH = TITAN_DIR / "tests" / "assets" / "tokenizer"


def materialize_titan_samples(
    *,
    seq_len: int,
    num_samples: int,
    dp_rank: int = 0,
    dp_world_size: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """The first ``num_samples`` (input, positions, label) triples of one data-parallel slice, as a TorchTitan arm reads them."""
    from torchtitan.components.tokenizer import HuggingFaceTokenizer
    from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataset

    tokenizer = HuggingFaceTokenizer(tokenizer_path=str(TOKENIZER_PATH))
    dataset = HuggingFaceTextDataset(
        dataset_name="c4_test",
        dataset_path=str(C4_TEST_PATH),
        tokenizer=tokenizer,
        seq_len=seq_len,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        infinite=True,
    )
    iterator = iter(dataset)
    samples = []
    for _ in range(num_samples):
        inputs, label = next(iterator)
        samples.append((inputs["input"], inputs["positions"], label))
    return samples


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
    rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    packed_len: int,
    padded_documents: int,
) -> dict[str, torch.Tensor | None]:
    """One microbatch on the CPU: one packed row of ``packed_len`` tokens, with ``cu_seqlens`` padded to ``padded_documents``."""
    tokens = torch.cat([row[0] for row in rows]).to(torch.int64)
    positions = torch.cat([row[1] for row in rows]).to(torch.int64)
    labels = torch.cat([row[2] for row in rows]).to(torch.int64)
    cu_seqlens = document_offsets(positions, packed_len)
    pad = padded_documents - cu_seqlens.numel()
    if pad < 0:
        raise ValueError(
            f"a pack holds {cu_seqlens.numel()} cu_seqlens entries, above "
            f"the run's padded width of {padded_documents}; padding cannot "
            "remove a document"
        )
    longest = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
    padded = torch.cat(
        [cu_seqlens, torch.full((pad,), packed_len, dtype=torch.int32)]
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
        samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
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
        # No collective reads this width, so each rank can have its own.
        self._padded_documents = max(
            document_offsets(
                torch.cat([row[1] for row in group]), self._packed_len
            ).numel()
            for group in groups
        )
        self._microbatches = [
            _microbatch(
                group,
                packed_len=self._packed_len,
                padded_documents=self._padded_documents,
            )
            for group in groups
        ]
        self._served = 0

    @property
    def padded_documents(self) -> int:
        """The ``cu_seqlens`` width of each microbatch of this rank."""
        return self._padded_documents

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
    *,
    seq_len: int,
    steps: int,
    local_batch_size: int,
    rows_per_sample: int,
    dp_rank: int,
    dp_world_size: int,
) -> StockReplayIterator:
    """The stream of this rank: ``steps * local_batch_size`` rows of ``seq_len`` tokens, packed ``rows_per_sample`` to a microbatch."""
    samples = materialize_titan_samples(
        seq_len=seq_len,
        num_samples=steps * local_batch_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )
    return StockReplayIterator(
        samples, rows_per_sample=rows_per_sample, seq_len=seq_len
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
    return (
        build_iterator(
            # One TorchTitan row, because --seq-length is the packed sample.
            seq_len=args.bench_seq_len,
            steps=args.train_iters,
            local_batch_size=args.bench_local_batch_size,
            rows_per_sample=args.bench_rows_per_sample,
            dp_rank=mpu.get_data_parallel_rank(),
            dp_world_size=mpu.get_data_parallel_world_size(),
        ),
        None,
        None,
    )

