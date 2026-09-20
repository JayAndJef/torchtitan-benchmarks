"""The exact TorchTitan c4_test batch stream, materialized and packed as THD.

Parity is by construction, not reimplementation: samples are drained from
torchtitan's own HuggingFaceTextDataset with torchtitan's own tokenizer, so
tokens, labels, and per-document positions are bit-identical to what the
titan arms' pre-tokenized replay loader (benchmarks/e2e/data/piper_qwen3.py)
serves — a property the test suite asserts directly. The only Megatron-
specific step is the batch -> THD conversion: each training batch's rows are
concatenated into one packed sequence whose cu_seqlens mark the document
boundaries (positions == 0), reproducing titan's block-diagonal causal
attention pattern in TE's packed-sequence format.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmarks.e2e.megatron_stock.data import (  # noqa: F401
    C4_TEST_PATH,
    TOKENIZER_PATH,
    materialize_titan_samples,
)


@dataclass
class ThdBatch:
    """One training step's batch in TE packed-sequence (thd) form."""

    tokens: torch.Tensor  # int64 [1, batch * seq_len]
    labels: torch.Tensor  # int64 [1, batch * seq_len], pre-shifted by packing
    positions: torch.Tensor  # int64 [1, batch * seq_len], restart per document
    cu_seqlens: torch.Tensor  # int32 [num_documents + 1]
    max_seqlen: int


def padded_microbatches(
    batches: list[ThdBatch],
    *,
    max_documents: int,
    tokens_per_microbatch: int,
) -> list[dict[str, torch.Tensor]]:
    """Each batch as the driver feeds it, with ``cu_seqlens`` padded to one
    length.

    The document count varies per pack, and a varying shape re-records a
    captured graph every step. Padding with trailing full-offset entries
    gives every microbatch one shape and adds no segment: an entry equal to
    the last real offset describes a document of zero length.

    ``max_documents`` is the caller's, and under a data-parallel degree it
    must be the maximum over **every** rank's packs. Each rank holds
    different documents, so a per-rank maximum would give the ranks
    different static shapes under one label. The caller takes the collective;
    this function only obeys the number it is given, which is what lets a
    test state the property without a device.
    """
    if max_documents < max(
        (batch.cu_seqlens.numel() for batch in batches), default=0
    ):
        raise ValueError(
            f"max_documents {max_documents} is below a pack's own document "
            "count; padding cannot remove a document"
        )
    microbatches = []
    for batch in batches:
        pad = max_documents - batch.cu_seqlens.numel()
        microbatches.append(
            {
                "tokens": batch.tokens,
                "labels": batch.labels,
                "cu_seqlens": torch.cat(
                    [
                        batch.cu_seqlens,
                        torch.full(
                            (pad,), tokens_per_microbatch, dtype=torch.int32
                        ),
                    ]
                ),
            }
        )
    return microbatches


def thd_batches(
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
) -> list[ThdBatch]:
    """Group consecutive samples into batches (titan's sequential batching)
    and pack each batch's rows into one THD sequence."""
    if len(samples) % batch_size:
        raise ValueError(
            f"{len(samples)} samples do not divide into batches of {batch_size}"
        )
    batches = []
    for start in range(0, len(samples), batch_size):
        rows = samples[start : start + batch_size]
        tokens = torch.cat([row[0] for row in rows])
        positions = torch.cat([row[1] for row in rows])
        labels = torch.cat([row[2] for row in rows])
        # Every document (and every row: _normalize_positions re-bases
        # chunk-leading fragments) starts at position 0, so positions == 0
        # enumerates exactly the packed-document boundaries titan's flex
        # attention mask uses.
        starts = (positions == 0).nonzero(as_tuple=True)[0].to(torch.int32)
        if starts[0].item() != 0:
            raise ValueError("first packed position is not a document start")
        cu_seqlens = torch.cat(
            [starts, torch.tensor([positions.numel()], dtype=torch.int32)]
        )
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        batches.append(
            ThdBatch(
                tokens=tokens.unsqueeze(0),
                labels=labels.unsqueeze(0),
                positions=positions.unsqueeze(0),
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        )
    return batches
