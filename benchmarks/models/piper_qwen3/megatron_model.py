"""Qwen3-1B (piper flavor) as a bare megatron-core GPTModel with the TE spec.

The builder takes its two halves from data and adds nothing of its own.
Geometry comes from ``benchmarks.models.piper_qwen3.shape.PiperShape`` -- the
same object ``benchmarks/models/piper_qwen3/config_registry.py`` builds the
TorchTitan twin from -- so ``--model-size`` moves both engines together and
neither can drift. At the default ``normal`` shape that is dim 1024, 16
layers, 16 q / 8 kv heads, head_dim 64, expert ffn 3584, vocab 151936.

Behaviour comes from an ``McoreProfile``
(``benchmarks/models/piper_qwen3/mcore_profiles.py``). Every flag this module
used to write inline lives there, so a variant -- megatron with the native
cross-entropy, with a fusion off, with the per-expert GEMMs -- is a named
profile rather than an edit here. Read that module for why a profile is a
``(config_overrides, spec_kwargs)`` pair and not a flat dict.

``BASE`` mirrors the qwen3_piper_1b TorchTitan config (the parity contract in
the scenario docs): RMSNorm eps 1e-6, no biases, SwiGLU; MoE on every layer
with 4 experts, top-2, softmax-then-topk with top-k renormalization (Qwen3
norm_topk_prob == megatron's post-softmax default), fp32 router math, no aux
loss and no expert bias; per-head qk RMSNorm before RoPE; NeoX rotate-half
RoPE with theta 1e6; vocab exactly 151936 (128 x 1187, so no padding change);
untied embeddings.

Precision is plain bf16: params_dtype bf16 plus a blanket .bfloat16() after
construction (torch-norm/TE-norm params otherwise materialize fp32), no
autocast, no fp32 masters.

This module resolves the profile's encoded torch values (``"silu"``,
``"bfloat16"``) and constructs. It is worker-side: it may import torch and
megatron, and the registry it reads may not.
"""

from __future__ import annotations

from typing import Any

from benchmarks.models.piper_qwen3.mcore_profiles import (
    ACTIVATION_FUNC_FIELDS,
    ACTIVATION_FUNCS,
    DTYPE_FIELDS,
    DTYPES,
    McoreProfile,
    layer_spec_kwargs,
    transformer_config_kwargs,
)
from benchmarks.models.piper_qwen3.shape import PiperShape


def _resolve(table: dict[str, Any], name: str, value: Any) -> Any:
    """Turn one encoded profile value into the torch object it names."""
    try:
        return table[value]
    except KeyError as error:
        raise ValueError(
            f"{name}={value!r} names no torch value this builder can "
            f"resolve. Available: {', '.join(sorted(table))}"
        ) from error


def build_model(
    *,
    seq_len: int,
    # No default, matching the titan builders this is the twin of: a default
    # could only be reached by a future omission, and would then build the
    # normal geometry silently under whatever --model-size the run asked for.
    # Both call sites already pass it.
    shape: PiperShape,
    # No default, for the same reason one step further out. An omitted profile
    # would build the base behaviour under whatever arm label asked for a
    # variant -- a wrong number rather than a missing one, and one no
    # correctness gate can see, because every profile here is numerically
    # valid. Pass mcore_profiles.BASE explicitly to get the historical model.
    profile: McoreProfile,
    cuda_graph_impl: str | None = None,
    cuda_graph_modules: tuple[str, ...] = (),
    use_cpu_initialization: bool = False,
):
    """Build the bare GPTModel in bf16 (on the current CUDA device unless
    use_cpu_initialization, which tests and the parity tool use)."""
    import torch
    import torch.nn.functional as F
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.transformer_config import TransformerConfig

    activations = {"silu": F.silu, "gelu": F.gelu}
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    # The registry declares the names; this module owns the objects. A name
    # added to one and not the other is a declaration nothing can build, so
    # fail here rather than at the TransformerConfig call with a torch error.
    assert set(activations) == set(ACTIVATION_FUNCS), (
        "mcore_profiles.ACTIVATION_FUNCS and this resolver disagree"
    )
    assert set(dtypes) == set(DTYPES), (
        "mcore_profiles.DTYPES and this resolver disagree"
    )

    kwargs = transformer_config_kwargs(
        shape=shape,
        profile=profile,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=cuda_graph_modules,
        use_cpu_initialization=use_cpu_initialization,
    )
    for name in ACTIVATION_FUNC_FIELDS:
        if name in kwargs:
            kwargs[name] = _resolve(activations, name, kwargs[name])
    for name in DTYPE_FIELDS:
        if name in kwargs:
            kwargs[name] = _resolve(dtypes, name, kwargs[name])

    config = TransformerConfig(**kwargs)
    spec = get_gpt_layer_with_transformer_engine_spec(
        **layer_spec_kwargs(shape=shape, profile=profile)
    )
    if cuda_graph_impl == "local":
        # The stock GPT specs build plain TransformerLayer, whose local-impl
        # manager can only capture the WHOLE layer forward -- impossible for
        # dynamic MoE (the dispatcher D2H-copies tokens_per_expert). Partial
        # capture (router + preprocess graphed, expert dispatch and attention
        # eager at this rev) lives in MoETransformerLayer, selected the same
        # way megatron's own modelopt/hybrid specs do.
        from megatron.core.transformer.transformer_layer import MoETransformerLayer

        spec.module = MoETransformerLayer
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=shape.vocab_size,
        max_sequence_length=seq_len,
        pre_process=True,
        post_process=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=int(shape.rope_theta),
    )
    if not use_cpu_initialization:
        model.cuda()
    model.bfloat16()
    return model
