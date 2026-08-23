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
    """Drain the first num_samples (input, positions, label) triples from
    torchtitan's c4_test dataset, exactly as the titan arms consume them.

    ``dp_rank`` and ``dp_world_size`` are the data-parallel slice, and they
    default to the whole stream, which is what every megatron number under
    ``out/`` was measured on. They are forwarded to the stock dataset class
    unchanged: it calls ``split_dataset_by_node(ds, dp_rank, dp_world_size)``
    on the raw documents, which is the split torchtitan's own loader gives
    its ranks. So this function reproduces the titan stream rank for rank
    rather than reimplementing a split, which is what the parity between the
    two engines rests on.

    ``num_samples`` is PER RANK. A data-parallel rank reads a batch of its
    own each step, so every rank drains the same count from a different
    shard.
    """
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
