"""FA3 varlen attention on packed documents, as a TorchTitan override.

Activation:
    --override.imports benchmarks.models.piper_qwen3.components.attention.fa3_override.packed_fa3_attention
"""

from dataclasses import dataclass

from torch.nn.attention import current_flash_attention_impl

from torchtitan.config import derive, override
from torchtitan.models.common.attention import GQAttention, VarlenAttention
from torchtitan.protocols.module import Module

from benchmarks.models.piper_qwen3.components.attention.packed import PackedGQAttention


class PackedFA3Attention(VarlenAttention):
    """The fork's FA3 varlen attention, under a config type that the trainer builds no metadata for."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: tuple[int, int] = (-1, 0)

    def __init__(self, config: "PackedFA3Attention.Config"):
        super().__init__(config)
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
