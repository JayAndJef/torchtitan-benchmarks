"""Grouped-query attention that reads the varlen offsets of its microbatch from the batch.

The replay loader computes the exact offsets of each pipeline microbatch on
the CPU and sends them under the batch key ``attention_masks``. The trainer
builds no mask for an inner attention config that it does not know, so the
key reaches each layer untouched, and the pipeline splitter gives each
microbatch its own row.
"""

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import GQAttention, VarlenMetadata


def microbatch_offsets(
    positions: torch.Tensor, rows_per_microbatch: int
) -> torch.Tensor:
    """The int32 ``[G, W]`` document offsets of each microbatch of ``rows_per_microbatch`` rows; a shorter row repeats its last offset."""
    if positions.device.type != "cpu":
        raise ValueError(
            f"the offsets are built on the CPU, but positions is on {positions.device}"
        )
    if positions.dim() != 2:
        raise ValueError(
            f"positions must be [rows, seq_len], but has the shape {tuple(positions.shape)}"
        )
    rows = positions.shape[0]
    if rows_per_microbatch < 1 or rows % rows_per_microbatch:
        raise ValueError(
            f"{rows} rows do not divide into microbatches of "
            f"{rows_per_microbatch} row(s)"
        )
    if not bool((positions[:, 0] == 0).all()):
        raise ValueError(
            "a row does not start a document, so a document can cross a row "
            "and max_q = seq_len does not hold"
        )
    groups = []
    for chunk in positions.split(rows_per_microbatch):
        flat = chunk.reshape(-1)
        starts = (flat == 0).nonzero(as_tuple=True)[0].to(torch.int32)
        end = torch.tensor([flat.numel()], dtype=torch.int32)
        groups.append(torch.cat([starts, end]))
    width = max(group.numel() for group in groups)
    return torch.stack(
        [torch.cat([group, group[-1:].expand(width - group.numel())]) for group in groups]
    )


class PackedGQAttention(GQAttention):
    """GQAttention whose inner varlen kernel reads the loader's offsets of this microbatch."""

    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        pass

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: torch.Tensor | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # These checks run at trace time, so the graph holds no assert kernel.
        if attention_masks is None:
            raise ValueError(
                "PackedGQAttention reads its offsets from the batch key "
                "'attention_masks', and the batch has none; set "
                "--dataloader.offset-rows"
            )
        if not isinstance(attention_masks, torch.Tensor):
            raise TypeError(
                "PackedGQAttention needs the loader's offsets tensor, but got "
                f"a {type(attention_masks).__name__}"
            )
        if attention_masks.dtype != torch.int32:
            raise TypeError(
                f"the offsets must be int32, but are {attention_masks.dtype}"
            )
        if attention_masks.dim() != 2 or attention_masks.shape[0] != 1:
            raise ValueError(
                "the offsets of one microbatch must be [1, W], but have the "
                f"shape {tuple(attention_masks.shape)}"
            )
        if positions is None:
            raise ValueError("PackedGQAttention needs the per-token positions")
        # max_q = L holds because every row starts a document.
        L = x_BLD.shape[1]
        cu = attention_masks[0]
        return super().forward(x_BLD, VarlenMetadata(cu, cu, L, L), positions)
