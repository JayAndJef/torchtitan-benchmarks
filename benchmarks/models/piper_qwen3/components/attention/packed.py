"""Grouped-query attention that builds its varlen offsets from each microbatch's ``positions``.

The trainer builds no mask for an inner attention config that it does not
know, and the pipeline splitter cuts ``positions`` per microbatch. So each
layer builds offsets that are right for the rows it holds, with no host sync.
"""

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import (
    AttentionMasksType,
    GQAttention,
    VarlenMetadata,
)

MAX_DOCUMENTS = 32
"""The document cap of one microbatch; the c4_test replay stream holds at most 23 at batch 4 and sequence length 4096."""


def cu_seqlens_from_positions(positions: torch.Tensor, max_docs: int) -> torch.Tensor:
    """The ``max_docs + 1`` int32 offsets of the documents in ``positions``, padded with empty documents."""
    torch._assert_async(
        (positions[:, 0] == 0).all(), "each row must start a document"
    )
    flat = positions.reshape(-1)
    is_start = flat == 0
    torch._assert_async(
        is_start.sum() <= max_docs, "a microbatch holds more documents than max_documents"
    )
    starts = torch.nonzero_static(
        is_start, size=max_docs, fill_value=flat.numel()
    ).flatten()
    end = torch.full((1,), flat.numel(), dtype=starts.dtype, device=starts.device)
    return torch.cat([starts, end]).to(torch.int32)


class PackedGQAttention(GQAttention):
    """GQAttention whose inner varlen kernel reads offsets built from this microbatch's ``positions``."""

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        max_documents: int

    def __init__(self, config: "PackedGQAttention.Config"):
        super().__init__(config)
        self.max_documents = config.max_documents

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise TypeError(
                "PackedGQAttention builds its own offsets, but the trainer "
                f"passed a {type(attention_masks).__name__}"
            )
        if positions is None:
            raise ValueError("PackedGQAttention needs the per-token positions")
        # max_q = L holds because every row starts a document.
        L = x_BLD.shape[1]
        cu = cu_seqlens_from_positions(positions, self.max_documents)
        return super().forward(x_BLD, VarlenMetadata(cu, cu, L, L), positions)
