"""The c4_test stream, shaped for stock Megatron's external dataloader.

Both engines read one stream, rank for rank.
``materialize_titan_samples`` below drains TorchTitan's own dataset class
with TorchTitan's own tokenizer, and the test suite asserts that its output
is bit-identical to the TorchTitan replay loader's. The rest of this module
reshapes that output and adds nothing to it.

``--dataloader-type external`` passes the iterator below through unchanged
(``megatron/training/datasets/data_samplers.py``). Megatron's own
``MegatronPretrainingSampler`` would shard a global stream by a different
rule than TorchTitan's ``split_dataset_by_node``, so the two engines would
put different tokens on the same rank.

Three rules govern the iterator, and each answers a way a run can be wrong:

1. **Every rank builds one.** ``--dataloader-inter-document-masking`` makes
   the middle pipeline stages read the batch too, for the ``cu_seqlens`` the
   attention needs (``pretrain_gpt.py``'s ``get_batch``). A middle stage
   with no iterator would fail on its first microbatch.
2. **The key is the data-parallel rank, never the global rank.** The stages
   of one pipeline train one model on one batch, so they must read the same
   tokens in the same order. ``mpu.get_data_parallel_rank()`` returns the
   same value on every stage of one pipeline.
3. **Exhaustion raises.** A wrap would train a second epoch under the first
   epoch's label, and no validation rule would see it.

**One microbatch is one packed sequence, never a batch of rows.** The
iterator concatenates ``rows_per_sample`` titan rows into a single
``(1, rows * seq_len)`` sample. ``benchmarks/e2e/megatron_stock/flags.py``'s
``microbatch_geometry`` gives the reason: Megatron flattens a ``(m, S)``
microbatch to ``(1, m*S)`` and then allocates its pipeline receive buffer
as ``(S, m, H)``, so a batched microbatch reaches the next stage permuted.
At ``m`` 1 there is nothing to flatten and every shape agrees.

The attention is unchanged by the packing. ``cu_seqlens`` marks every
document, and every titan row starts at position 0, so a row boundary is a
document boundary and the mask stays block-diagonal exactly where it was.

The dict keys and dtypes are Megatron's, not ours.
``megatron/core/utils.py``'s ``_merge_cu_seqlens_across_micro_batch`` reads
a ``(micro_batch_size, padded_length)`` ``cu_seqlens`` whose rows start at
0, end at ``seq_length``, and are right-padded with more copies of
``seq_length``. It strips the padding by finding the **first** entry equal
to ``seq_length``, so a row must carry no other value that large. The
offsets rise strictly to ``seq_length``, so the first such entry is always
the real end of the row.
"""

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
    """Drain the first num_samples (input, positions, label) triples from
    torchtitan's c4_test dataset, exactly as the titan arms consume them.

    ``dp_rank`` and ``dp_world_size`` are the data-parallel slice, and they
    default to the whole stream. They are forwarded to the stock dataset
    class unchanged: it calls ``split_dataset_by_node(ds, dp_rank,
    dp_world_size)`` on the raw documents, which is the split torchtitan's
    own loader gives its ranks. So this function reproduces the titan stream
    rank for rank rather than reimplementing a split, which is what the
    parity between the two engines rests on.

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


# The keys one microbatch carries. ``pretrain_gpt.py``'s ``BATCH_KEYS``
# names ten; a key this dict omits reaches ``get_batch`` as None, which is
# what ``attention_mask`` and ``cu_seqlens_padded`` need. They are listed
# here as None so a reader sees the whole contract in one place.
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


def document_offsets(positions: torch.Tensor, seq_len: int) -> torch.Tensor:
    """One packed sample's document boundaries, as Megatron reads them.

    ``positions`` restarts at 0 at every document, and TorchTitan re-bases a
    chunk-leading fragment the same way, so ``positions == 0`` enumerates
    exactly the packed-document starts its flex-attention mask uses. A titan
    row boundary is one of them, because every row starts at position 0.

    The result is ``[0, d1, ..., seq_len]`` in int32, with no padding. It
    always starts at 0 and always ends at ``seq_len``, where ``seq_len`` is
    the **packed** length.
    """
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
    """One microbatch dict, on the CPU: one packed row of ``packed_len``.

    ``get_batch`` moves every tensor to the device itself, so nothing here
    touches CUDA. That is what lets a test read a microbatch on a host with
    no GPU.
    """
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
        # Every token of the c4_test stream is a real token, so every one of
        # them counts toward the loss. TorchTitan does the same.
        "loss_mask": torch.ones(
            (1, packed_len), dtype=torch.float32
        ),
        "position_ids": positions.unsqueeze(0),
        "cu_seqlens": padded.unsqueeze(0),
        "max_seqlen": torch.tensor([longest], dtype=torch.int32),
        # Megatron builds no mask tensor under
        # --no-create-attention-mask-in-dataloader, and it needs no padded
        # cu_seqlens without context parallelism.
        "attention_mask": None,
        "cu_seqlens_padded": None,
    }


class StockReplayIterator:
    """The whole run's microbatches, materialized once and served in order.

    Materializing at construction matches the TorchTitan replay loader the
    other arm of this scenario runs, so neither engine pays a per-step data
    cost the other does not.

    It is an iterator and not an iterable: Megatron wraps the object in a
    ``RerunDataIterator`` and calls ``next()`` on it directly.
    """

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
        # One padded width for the whole run, taken over this rank's own
        # packs. Every rank holds different documents, so this number is a
        # per-rank number -- and it may be, because Megatron strips the
        # padding inside each rank and no collective reads the width. The
        # tuned driver takes a global maximum instead, because a captured
        # CUDA graph needs one static shape across the mesh.
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
        """The ``cu_seqlens`` width every microbatch of this rank carries."""
        return self._padded_documents

    @property
    def packed_len(self) -> int:
        """Tokens in one microbatch: ``rows_per_sample * seq_len``."""
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
    """This rank's whole stream.

    ``seq_len`` is one titan row, not the packed sample.
    ``local_batch_size`` is one data-parallel rank's own batch, so the
    sample count is ``steps * local_batch_size`` per rank. Each step then
    serves ``local_batch_size // rows_per_sample`` microbatches, and every
    stage of one pipeline reads all of them.
    """
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
    """The one argument this driver substitutes for ``pretrain_gpt``'s.

    Megatron passes its own target sample counts. This provider ignores
    them: the run's sample count is ``--train-iters`` times the harness
    batch size, and the harness owns both.

    The valid and test slots are None. ``--eval-iters 0`` means Megatron
    builds no loader for either.
    """
    from megatron.core import mpu
    from megatron.training import get_args

    args = get_args()
    # Megatron resolves the data-parallel degree twice: the parser derives
    # ``args.data_parallel_size`` from ``WORLD_SIZE`` and the other degrees,
    # and ``initialize_model_parallel`` builds the group this reads. The
    # marker line arm rule 12 matches states the first. The token slice
    # below uses the second. A disagreement would put the wrong shard on
    # this rank while the log named the right mesh, and every rule would
    # pass. This is where the stock driver makes the check, because
    # ``pretrain()`` owns ``initialize_model_parallel``.
    # It is not the first of our code to run after that call -- the model
    # provider and install_data_parallel_marker's wrapper run first -- but
    # it is before any token is read, which is the property that matters.
    resolved = mpu.get_data_parallel_world_size()
    if resolved != args.data_parallel_size:
        raise RuntimeError(
            f"megatron built a data-parallel group of {resolved} rank(s) "
            f"where its own arguments say {args.data_parallel_size}; the "
            "token slice and the recorded mesh would disagree"
        )
    return (
        build_iterator(
            # The titan row length, not --seq-length: that one is the
            # packed sample, which is rows_per_sample rows long.
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


# Megatron reads this flag to decide whether every rank builds the data, or
# only tensor-parallel rank 0 does (``training.py``'s
# ``build_train_valid_test_data_loaders``). Every rank must build one here:
# see rule 1 in this module's docstring.
train_valid_test_datasets_provider.is_distributed = True
