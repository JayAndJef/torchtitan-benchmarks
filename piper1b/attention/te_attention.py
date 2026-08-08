"""Out-of-tree torchtitan override: inner attention via TransformerEngine.

Swaps the model's inner attention for
``transformer_engine.pytorch.DotProductAttention`` in THD (packed-token) form,
which is the exact code path the Megatron baseline arm runs. That makes the
attention row of the engine comparison a kernel comparison rather than an
implementation comparison.

    --override.imports piper1b.attention.te_attention.te_attention

``TEAttention`` subclasses ``VarlenAttention`` for one reason: the decoder
hands a ``VarlenMetadata`` (cu_seqlens) to any inner attention whose config is
a ``VarlenAttention.Config``, which is precisely the THD metadata TE needs.
Subclassing gets that for free with no runtime cost.

Two subtleties make the subclassing non-obvious:

- The nested ``Config`` is **mandatory**. ``Configurable.__init_subclass__``
  sets ``_owner`` on a subclass's own nested Config; without one,
  ``TEAttention.Config`` *is* ``VarlenAttention.Config`` and would silently
  build a ``VarlenAttention``.
- ``__init__`` must be overridden. ``VarlenAttention.__init__`` force-activates
  FlashAttention-3, which raises on any box where FA3 is not installed. This
  arm has no business depending on FA3, so it does not inherit that.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torchtitan.config import derive, override
from torchtitan.models.common.attention import VarlenAttention, VarlenMetadata


class TEAttention(VarlenAttention):
    """TransformerEngine fused attention over packed (THD) sequences."""

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        # Deliberately does NOT call VarlenAttention.__init__: that activates
        # FA3, which this arm neither uses nor should require.
        super(VarlenAttention, self).__init__()
        self.window_size = config.window_size
        self._attention = None

    def _build(self, num_heads: int, num_gqa_groups: int, head_dim: int, scale):
        from megatron_baseline.location import configure_te_environment

        configure_te_environment()
        from transformer_engine.pytorch import DotProductAttention

        return DotProductAttention(
            num_attention_heads=num_heads,
            kv_channels=head_dim,
            num_gqa_groups=num_gqa_groups,
            attention_dropout=0.0,
            qkv_format="thd",
            # THD packing requires a padding mask type; plain "causal" is
            # rejected outright by TE for this layout.
            attn_mask_type="padding_causal",
            softmax_scale=scale,
        ).to(device=torch.cuda.current_device())

    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(attention_masks, VarlenMetadata), (
            "TEAttention needs VarlenMetadata (cu_seqlens); got "
            f"{type(attention_masks)}"
        )
        B, L, N, H = q_BLNH.shape
        T = B * L

        if self._attention is None:
            self._attention = self._build(
                num_heads=N,
                num_gqa_groups=k_BLNH.shape[2],
                head_dim=H,
                scale=scale if scale is not None else H**-0.5,
            )

        # TE consumes THD: (total_tokens, heads, head_dim). Match
        # VarlenAttention's dtype handling exactly -- cast in, cast back --
        # so the arms differ in kernel, not in precision.
        out_TNH = self._attention(
            q_BLNH.reshape(T, N, H).to(torch.bfloat16),
            k_BLNH.reshape(T, -1, H).to(torch.bfloat16),
            v_BLNH.reshape(T, -1, H).to(torch.bfloat16),
            cu_seqlens_q=attention_masks.cu_seq_q,
            cu_seqlens_kv=attention_masks.cu_seq_k,
            max_seqlen_q=L,
            max_seqlen_kv=L,
        )
        return out_TNH.view(B, L, N, H).to(q_BLNH.dtype)


@override(
    target=VarlenAttention.Config,
    exact=True,
    description="TransformerEngine fused attention (THD) as the inner attention.",
)
def te_attention(cfg: VarlenAttention.Config) -> TEAttention.Config:
    return derive(cfg, TEAttention.Config)
