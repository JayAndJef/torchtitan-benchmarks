"""The TorchTitan Qwen3 model of one shape: its config, and an in-process build of it for kernel-bench."""

from __future__ import annotations

from typing import Sequence

import torch
from torchtitan.models.common import CosSinRoPE, Embedding, Linear
from torchtitan.models.qwen3 import (
    _build_qwen3_moe_layers,
    _EMBEDDING_INIT,
    _output_linear_init,
    _qwen3_norm,
    Qwen3Model,
)

from benchmarks.models.piper_qwen3.shape import PiperShape


def _piper_1b_model(
    *, fuse_qkv: bool, shape: PiperShape, attn_backend: str = "flex"
) -> Qwen3Model.Config:
    """The TorchTitan config of the Qwen3 MoE model of ``shape``."""
    dim = shape.dim
    head_dim = shape.head_dim
    n_layers = shape.n_layers
    vocab_size = shape.vocab_size
    layers = _build_qwen3_moe_layers(
        fuse_qkv=fuse_qkv,
        n_layers=n_layers,
        dim=dim,
        n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads,
        head_dim=head_dim,
        moe_hidden_dim=shape.moe_hidden_dim,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        attn_backend=attn_backend,
        moe_comm_backend="standard",
        rope=CosSinRoPE.Config(
            dim=head_dim,
            max_seq_len=shape.max_seq_len,
            theta=shape.rope_theta,
        ),
    )
    # Piper trains without aux-free load balancing.
    for layer in layers:
        layer.moe.load_balance_coeff = None
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            # Piper ties no weights, so the embedding gets its own init.
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def apply_config_overrides(
    config,
    targets: Sequence[str],
    *,
    expected: int,
) -> list[str]:
    """Apply ``targets`` to ``config``, and raise unless exactly ``expected`` nodes change.

    Returns the ``[Override]`` lines that the trainer also logs.
    """
    from torchtitan.config.override import OverrideConfig, apply_overrides

    if not targets and expected == 0:
        return []
    lines = apply_overrides(OverrideConfig(imports=list(targets)), config)
    if len(lines) != expected:
        raise RuntimeError(
            f"expected {expected} override replacement(s) from "
            f"{list(targets)}, got {len(lines)}: "
            + ("; ".join(lines) if lines else "nothing matched")
            + ". An override that matches nothing leaves the arm measuring "
            "the baseline under its own name, and the arms whose kernels have "
            "no distinctive name have no other guard."
        )
    return lines


def build_titan_model(
    *,
    shape: PiperShape,
    fuse_qkv: bool = True,
    attn_backend: str = "flex",
    overrides: Sequence[str] = (),
    overrides_per_block: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    eval_mode: bool = False,
):
    """The Qwen3 model of ``shape``, built on ``device`` with ``dtype`` as the default dtype, as the trainer builds it.

    The function restores the default dtype, and it seeds before
    ``init_states``, so two processes that build one arm hold the same
    parameters.
    """
    config = _piper_1b_model(
        fuse_qkv=fuse_qkv, shape=shape, attn_backend=attn_backend
    )
    apply_config_overrides(
        config, overrides, expected=overrides_per_block * shape.n_layers
    )
    device = torch.device(device)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with device:
            model = config.build()
    finally:
        torch.set_default_dtype(previous_dtype)
    torch.manual_seed(seed)
    model.init_states(buffer_device=device)
    if eval_mode:
        model.eval()
    return model
