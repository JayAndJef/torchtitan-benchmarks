"""Qwen3 (piper flavor) as a bare megatron-core GPTModel with the TE spec.

The builder takes its two halves from data and adds nothing of its own.
Geometry comes from ``benchmarks.models.piper_qwen3.shape.PiperShape``, the
same object the TorchTitan twin is built from, so ``--model-size`` moves
both engines together and neither can drift. Behaviour comes from an
``McoreProfile`` (``benchmarks/models/piper_qwen3/mcore_profiles.py``), so
a variant is a named profile rather than an edit here.

**The layer spec is derived, not written.** ``get_gpt_decoder_block_spec``
reads the built config and turns ``num_moe_experts``, ``moe_grouped_gemm``
and ``qk_layernorm`` into module-class choices, which is the path
megatron's own entrypoint takes. This module used to call the inner factory
``get_gpt_layer_with_transformer_engine_spec``, which receives no config,
so those three settings had to be written a second time and kept in
agreement by hand. Megatron checks the ``moe_grouped_gemm`` agreement
nowhere, so a disagreement built one implementation and labelled it the
other. Do not go back to the inner factory.

``blank_parts`` is the one edit this module makes to that derived spec, and
it only ever **removes** a part. The config is not touched, so the three
single-valued fields above keep their values and the disagreement hazard is
not reached. ``_blank_layer_parts`` carries the rules.

``BASE`` mirrors the qwen3_piper_1b TorchTitan config: RMSNorm eps 1e-6, no
biases, SwiGLU; MoE on every layer with 4 experts, top-2,
softmax-then-topk with top-k renormalization; fp32 router math, no aux loss
and no expert bias; per-head qk RMSNorm before RoPE; NeoX rotate-half RoPE
with theta 1e6; vocab exactly 151936 (128 x 1187, so no padding change);
untied embeddings.

Precision is plain bf16: ``params_dtype`` bf16 plus a blanket
``.bfloat16()`` after construction, because torch-norm and TE-norm params
otherwise materialize fp32. There is no autocast and there are no fp32
masters.

This module resolves the profile's encoded values and constructs. It is
worker-side: it may import torch and megatron, and the registry it reads
may not. ``attention_backend`` is the encoded field that is not a torch
value. It names a member of megatron's ``AttnBackend`` enum, which is not
JSON-safe, so the parent-side registry carries the name and this module
holds the enum.
"""

from __future__ import annotations

from typing import Any

from benchmarks.models.piper_qwen3.mcore_profiles import (
    ACTIVATION_FUNC_FIELDS,
    ACTIVATION_FUNCS,
    ATTENTION_BACKEND_FIELDS,
    ATTENTION_BACKENDS,
    DTYPE_FIELDS,
    DTYPES,
    McoreProfile,
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


def _blank_layer_parts(spec: Any, parts: tuple[str, ...]) -> Any:
    """The same derived block spec, with the named layer parts made inert.

    A kernel scenario times one cut of the model, and the rest of the layer
    is almost all of the build: the mlp part alone is 93% of a layer at
    ``1b`` and 97% at ``48b``. This function removes a part before megatron
    allocates it. It replaces one field of the derived spec and writes no
    spec, no module class and no submodule tree of its own, so megatron
    still builds the layer through its own ``TransformerLayer.__init__``
    and the config keeps every value the profile and the shape gave it.

    **The replacement value is megatron's own dataclass default**, read off
    ``dataclasses.fields`` rather than written here. That keeps the two
    identity classes apart: the six module slots default to ``IdentityOp``
    and the three bias-dropout-add slots default to ``IdentityFuncOp``,
    which returns a function rather than a tensor. A field is blankable
    only when its declared default is an ``IdentityOp`` subclass, so a
    megatron bump that defaults a slot to a real module fails loudly rather
    than putting a different module in the layer.

    **Blank only the parts the scenario never reads.** Two failure modes
    follow, and only one is loud. A caller that navigates into a blanked
    part gets an ``AttributeError``, because an ``IdentityOp`` holds no
    child. A caller that calls one gets its own input back and reads as a
    large win. ``benchmarks/kernel/operations/common.py``'s
    ``MCORE_BLANK_MLP`` names which builders may pass it, and
    ``tests/test_megatron_model.py`` pins that roster, because no run-time
    check can separate an identity from a fast kernel.

    Megatron constructs the nine parts in the order the dataclass declares
    them, so every part built before the first blanked one keeps its exact
    weights. That holds for the GPU initialization path; see
    ``build_model`` for why the host path is refused.
    """
    import dataclasses

    from megatron.core.transformer.identity_op import IdentityOp

    layer_specs = list(spec.layer_specs)
    if not layer_specs:
        raise ValueError("the derived block spec holds no layer")
    # The submodules dataclass of the spec in hand, never a class named here.
    blankable = {
        field.name: field.default
        for field in dataclasses.fields(type(layer_specs[0].submodules))
        if isinstance(field.default, type) and issubclass(field.default, IdentityOp)
    }
    unknown = sorted(set(parts) - set(blankable))
    if unknown:
        raise ValueError(
            f"blank_parts names {unknown}, which this layer spec cannot "
            f"blank. The parts it can blank are: "
            f"{', '.join(sorted(blankable))}"
        )
    blanked = {name: blankable[name] for name in parts}
    # Every layer entry can be ONE shared object, so rebuild, never mutate.
    return dataclasses.replace(
        spec,
        layer_specs=[
            dataclasses.replace(
                layer_spec,
                submodules=dataclasses.replace(layer_spec.submodules, **blanked),
            )
            for layer_spec in layer_specs
        ],
    )


def _assert_parts_are_blank(model: Any, parts: tuple[str, ...]) -> None:
    """Refuse to hand back a model whose blanking did not take.

    Every other geometry axis has an arm that checks it against the shape.
    A blanked part has no such reader, because its whole point is that
    nobody reads it, so the check lives here where it always runs rather
    than in each caller where it can be forgotten.
    """
    from megatron.core.transformer.identity_op import IdentityOp

    layer = model.decoder.layers[0]
    for name in parts:
        part = getattr(layer, name)
        if not isinstance(part, IdentityOp):
            raise RuntimeError(
                f"blank_parts asked for {name!r} to be blank, and decoder "
                f"layer 0 holds a {type(part).__name__}. The spec edit "
                f"did not reach the built layer."
            )


def build_model(
    *,
    seq_len: int,
    # No default: an omission would silently build the wrong geometry.
    shape: PiperShape,
    # No default: an omission would build base under a variant's label.
    profile: McoreProfile,
    cuda_graph_impl: str | None = None,
    cuda_graph_modules: tuple[str, ...] = (),
    use_cpu_initialization: bool = False,
    blank_parts: tuple[str, ...] = (),
    # The pipeline split: all three ask for the whole model on one rank.
    pipeline_model_parallel_size: int = 1,
    pre_process: bool = True,
    post_process: bool = True,
    # The treatment every published megatron number was measured under.
    batch_p2p_sync: bool = True,
):
    """Build the bare GPTModel in bf16 on the current CUDA device.

    ``use_cpu_initialization`` builds on the host instead, which the tests
    and the parity tool use.

    ``pipeline_model_parallel_size`` reaches ``TransformerConfig``, where
    megatron's ``get_num_layers_to_build`` divides ``config.num_layers`` by
    it, so the derived block spec already holds this rank's layers alone.
    ``pre_process`` and ``post_process`` decide the two ends: the embedding
    table on the first stage, the final norm and the output head on the
    last. The caller resolves all three, because this builder consults no
    global state. **The caller must pass a degree that agrees with
    ``initialize_model_parallel``**: megatron reads the degree off the
    config here and off ``parallel_state`` in the schedule, and it checks
    the two against each other nowhere, so a disagreement builds one
    partition and communicates another.

    ``batch_p2p_sync`` reaches ``TransformerConfig`` too. A True value puts
    one ``torch.cuda.synchronize()`` behind every batched pipeline message.

    ``blank_parts`` names transformer-layer parts this caller does not use.
    Each named part becomes megatron's own ``IdentityOp``, which allocates
    nothing and returns its first argument. See ``_blank_layer_parts`` for
    the rules a caller must obey.

    **``blank_parts`` and ``use_cpu_initialization`` are refused
    together.** A blanking caller relies on every part built before the
    first blanked one keeping its exact weights. The CUDA path holds that:
    the expert weights fork the expert-parallel RNG state and the router
    gate takes the host generator, so removing the mlp part moves nothing
    on the model-parallel state the other parts draw from. The host path
    sends every weight through ``_initialize_affine_weight_cpu``, which
    draws from one global host generator with no tracker fork, so a removed
    part shifts every draw after it. No caller asks for both today. This
    raises rather than trusting that, because the shift is silent and every
    affected value stays numerically valid.
    """
    # Checked before the first import, so a CPU test reaches it.
    if blank_parts and use_cpu_initialization:
        raise ValueError(
            "blank_parts and use_cpu_initialization cannot be combined: the "
            "host initialization path draws every weight from one global "
            "generator, so a blanked part shifts the values of every part "
            "built after it"
        )

    import torch
    import torch.nn.functional as F
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    activations = {"silu": F.silu, "gelu": F.gelu}
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    # Megatron's own enum; unset, TE picks and the profile names no kernel.
    attention_backends = {member.name: member for member in AttnBackend}
    # The registry declares the names; this module owns the objects.
    assert set(activations) == set(ACTIVATION_FUNCS), (
        "mcore_profiles.ACTIVATION_FUNCS and this resolver disagree"
    )
    assert set(dtypes) == set(DTYPES), (
        "mcore_profiles.DTYPES and this resolver disagree"
    )
    assert set(attention_backends) == set(ATTENTION_BACKENDS), (
        "mcore_profiles.ATTENTION_BACKENDS and megatron's AttnBackend enum "
        "disagree; a submodule bump changed the roster"
    )

    kwargs = transformer_config_kwargs(
        shape=shape,
        profile=profile,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_modules=cuda_graph_modules,
        use_cpu_initialization=use_cpu_initialization,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        batch_p2p_sync=batch_p2p_sync,
    )
    for name in ACTIVATION_FUNC_FIELDS:
        if name in kwargs:
            kwargs[name] = _resolve(activations, name, kwargs[name])
    for name in DTYPE_FIELDS:
        if name in kwargs:
            kwargs[name] = _resolve(dtypes, name, kwargs[name])
    for name in ATTENTION_BACKEND_FIELDS:
        if name in kwargs:
            kwargs[name] = _resolve(attention_backends, name, kwargs[name])

    config = TransformerConfig(**kwargs)
    # Deriving from the config is what keeps the three fields single-valued.
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
    if blank_parts:
        spec = _blank_layer_parts(spec, blank_parts)
    if cuda_graph_impl == "local":
        # The stock layer graphs the whole forward, which dynamic MoE refuses.
        from megatron.core.transformer.transformer_layer import MoETransformerLayer

        for layer_spec in spec.layer_specs:
            layer_spec.module = MoETransformerLayer
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=shape.vocab_size,
        max_sequence_length=seq_len,
        pre_process=pre_process,
        post_process=post_process,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=int(shape.rope_theta),
    )
    if not use_cpu_initialization:
        model.cuda()
    model.bfloat16()
    if blank_parts:
        _assert_parts_are_blank(model, blank_parts)
    return model
