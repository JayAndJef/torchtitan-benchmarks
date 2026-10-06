"""FA3 varlen attention on packed documents, as a TorchTitan override.

Activation:
    --override.imports benchmarks.models.piper_qwen3.components.attention.fa3_override.packed_fa3_attention
"""

from dataclasses import dataclass

import torch
from torch.nn.attention import SDPBackend, current_flash_attention_impl, sdpa_kernel

from torchtitan.config import derive, override
from torchtitan.models.common.attention import GQAttention, VarlenAttention
from torchtitan.protocols.module import Module

from benchmarks.models.piper_qwen3.components.attention.packed import PackedGQAttention

OFFSETS_LENGTH_SOURCE = "L['attention_masks']:1"
"""The ``dynamic_sources`` entry for dim 1 of the block's offsets, which is their length."""


def mark_offsets_length_dynamic() -> None:
    """Add ``OFFSETS_LENGTH_SOURCE`` to ``torch.compiler.config.dynamic_sources`` once; a torch without the field raises."""
    config = torch.compiler.config
    if not hasattr(config, "dynamic_sources"):
        raise RuntimeError(
            "this torch has no torch.compiler.config.dynamic_sources, so each "
            "compiled block recompiles on the second offsets length"
        )
    entries = [entry for entry in config.dynamic_sources.replace(" ", "").split(",") if entry]
    if OFFSETS_LENGTH_SOURCE not in entries:
        config.dynamic_sources = ",".join([*entries, OFFSETS_LENGTH_SOURCE])


# A second offsets length must not recompile the block on a measured step.
mark_offsets_length_dynamic()


class PackedFA3Attention(VarlenAttention):
    """The fork's FA3 varlen attention, under a config type that the trainer builds no metadata for."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: tuple[int, int] = (-1, 0)

    def __init__(self, config: "PackedFA3Attention.Config"):
        super().__init__(config)
        _require_fa3()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """The parent's forward with flash as the only SDPA backend, because ``varlen_attn`` prefers an eligible cuDNN backend."""
        _require_fa3()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return super().forward(*args, **kwargs)


def _require_fa3() -> None:
    """Raise unless FA3 is the active flash attention, which the flash backend of ``varlen_attn`` calls."""
    if current_flash_attention_impl() != "FA3":
        raise RuntimeError(
            "PackedFA3Attention needs FA3, but the active flash attention "
            f"is {current_flash_attention_impl()!r}"
        )


@override(
    target=GQAttention.Config,
    exact=True,
    description="FA3 varlen attention on packed documents, offsets read from the batch.",
)
def packed_fa3_attention(cfg: GQAttention.Config) -> PackedGQAttention.Config:
    return derive(
        cfg,
        PackedGQAttention.Config,
        inner_attention=derive(cfg.inner_attention, PackedFA3Attention.Config),
    )
