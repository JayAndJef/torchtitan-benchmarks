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
profile rather than an edit here.

**The layer spec is derived, not written.** ``get_gpt_decoder_block_spec``
reads the built config and turns ``num_moe_experts``, ``moe_grouped_gemm`` and
``qk_layernorm`` into module-class choices (``gpt_layer_specs.py:592-594``),
which is the path megatron's own entrypoint takes. This module used to call the
inner factory ``get_gpt_layer_with_transformer_engine_spec`` instead. That
factory receives no config, so those three settings had to be written a second
time and kept in agreement by hand -- and for ``moe_grouped_gemm`` megatron
checks the agreement nowhere, so a disagreement built one implementation and
labelled it the other. Do not go back to the inner factory.

``blank_parts`` is the one edit this module makes to that derived spec, and
it only ever **removes** a part. A kernel scenario times one cut, so it needs
one part of the layer; the rest is allocation it pays for and never reads.
Each named part becomes the ``IdentityOp`` megatron's own dataclass declares
as that slot's default, and megatron then builds the layer through its own
constructor. The config is not touched, so the three single-valued fields
above keep their values and the disagreement hazard is not reached.
``_blank_layer_parts`` carries the rules.

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

This module resolves the profile's encoded values (``"silu"``,
``"bfloat16"``, ``"fused"``) and constructs. It is worker-side: it may import
torch and megatron, and the registry it reads may not.

``attention_backend`` is the third encoded field and the one that is not a
torch value. It names a member of megatron's ``AttnBackend`` enum, which is
not JSON-safe, so the parent-side registry carries the name and this module
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

    A kernel scenario times one cut of the model. It does not need the rest
    of the layer to exist, and at the wide shapes the rest of the layer is
    almost all of the build: the mlp part alone is 93% of a layer at ``1b``
    and 97% at ``48b``. This function removes a part before megatron
    allocates it.

    **This edits the derived spec. It writes no spec.**
    ``get_gpt_decoder_block_spec`` derives the spec exactly as it always did,
    and this replaces one field of the derived result. Megatron then builds
    the layer through its own ``TransformerLayer.__init__``. Nothing here
    writes a module class, a constructor argument or a submodule tree.

    **The replacement value is megatron's own dataclass default**, read off
    ``dataclasses.fields`` rather than written here. Megatron blanks a part
    the same way throughout ``gpt_layer_specs.py`` -- ``q_layernorm``,
    ``pre_mlp_layernorm`` and others -- so the mechanism is the one the file
    already uses.

    Reading the default also keeps the two identity classes apart. The six
    module slots default to ``IdentityOp``. The three bias-dropout-add slots
    default to ``IdentityFuncOp``, which returns a function rather than a
    tensor. A caller that wrote the wrong one of the two would get a
    confusing failure at call time.

    A field is blankable only when its declared default is an ``IdentityOp``
    subclass. That excludes ``sharded_state_dict_keys_map``, which is a dict,
    and it fails loudly if a megatron bump ever defaults a slot to a real
    module -- blanking such a slot would put a different module in the layer
    rather than remove one.

    **The config is untouched.** ``num_moe_experts``, ``moe_grouped_gemm``
    and ``qk_layernorm`` keep the values the profile and the shape gave them,
    so the hazard this module's docstring warns about -- a config that says
    one thing and a spec that says another, which megatron checks nowhere --
    is not reached.

    **What a caller must obey. Blank only the parts the scenario never
    reads.** Two failure modes follow, and only one of them is loud. A caller
    that **navigates** into a blanked part -- ``layer.mlp.experts`` -- gets an
    ``AttributeError``, because an ``IdentityOp`` holds no child. A caller
    that **calls** a blanked part gets its own input back, times an identity,
    and reads as a large win. ``benchmarks/kernel/operations/common.py``'s
    ``MCORE_BLANK_MLP`` names which builders may pass it, and
    ``tests/test_megatron_model.py`` pins that roster, because no run-time
    check can separate an identity from a fast kernel.

    **What a caller may rely on.** Megatron constructs the nine parts in the
    order this dataclass declares them, so every part built before the first
    blanked one keeps its exact weights. That statement holds for the GPU
    initialization path; see ``build_model`` for why the host path is refused.
    """
    import dataclasses

    from megatron.core.transformer.identity_op import IdentityOp

    layer_specs = list(spec.layer_specs)
    if not layer_specs:
        raise ValueError("the derived block spec holds no layer")
    # The submodules dataclass of the spec in hand, not a class named here:
    # megatron picks the dense or the MoE layer submodules from the config,
    # and a future variant may declare its own.
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
    # A block spec holds one entry per layer and, at moe_layer_freq=1, every
    # entry is the SAME object. Rebuilding the list gives each layer its own
    # blanked copy, so no layer keeps a part another layer dropped, and the
    # derived original is left unmutated for any other reader.
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

    Every other geometry axis is checked against the shape by the arm that
    reads it -- ``num_moe_experts`` in the router builder, ``kv_channels`` in
    the rope builder, the weight shapes in the three projection builders.
    The blanked parts have no such reader, because their whole point is that
    nobody reads them. So the check lives here, where it always runs, rather
    than in each caller, where it can be forgotten.
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
    blank_parts: tuple[str, ...] = (),
):
    """Build the bare GPTModel in bf16 (on the current CUDA device unless
    use_cpu_initialization, which tests and the parity tool use).

    ``blank_parts`` names transformer-layer parts this caller does not use.
    Each named part becomes megatron's own ``IdentityOp``, which allocates
    nothing and returns its first argument. The default builds every part, so
    an omitted argument keeps the historical model. See ``_blank_layer_parts``
    for the rules a caller must obey.

    **``blank_parts`` and ``use_cpu_initialization`` are refused together.**
    The promise a blanking caller relies on is that every part built before
    the first blanked one keeps its exact weights. On the CUDA path that
    holds: the expert weights fork the expert-parallel RNG state and the
    router gate takes the host generator, so removing the mlp part moves
    nothing on the model-parallel state that the other parts and the output
    layer draw from. On the host path it does not hold.
    ``use_cpu_initialization`` sends every weight through
    ``_initialize_affine_weight_cpu``
    (``tensor_parallel/layers.py:184-225``), which draws from the one global
    host generator with no tracker fork, so a removed part shifts every draw
    after it. No caller asks for both today. This raises rather than trusting
    that, because the shift is silent and every affected value stays
    numerically valid.
    """
    # Checked before the first import, so a CPU test reaches it without
    # megatron on sys.path and without a device.
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
    # Megatron's own enum, keyed by its member names. A profile names one and
    # this resolves it, the way it resolves "silu" and "bfloat16". Left unset,
    # TransformerConfig defaults to AttnBackend.auto and TransformerEngine
    # picks a backend at run time, so the profile would not say which kernel
    # the arm measured.
    attention_backends = {member.name: member for member in AttnBackend}
    # The registry declares the names; this module owns the objects. A name
    # added to one and not the other is a declaration nothing can build, so
    # fail here rather than at the TransformerConfig call with a torch error.
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
    # Megatron derives the layer spec from the config it was just handed. That
    # derivation is what keeps num_moe_experts, moe_grouped_gemm and
    # qk_layernorm single-valued: the module-class choice and the run-time
    # reads both come off one field, so they cannot contradict each other.
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
    if blank_parts:
        spec = _blank_layer_parts(spec, blank_parts)
    if cuda_graph_impl == "local":
        # The stock GPT specs build plain TransformerLayer, whose local-impl
        # manager can only capture the WHOLE layer forward -- impossible for
        # dynamic MoE (the dispatcher D2H-copies tokens_per_expert). Partial
        # capture (router + preprocess graphed, expert dispatch and attention
        # eager at this rev) lives in MoETransformerLayer, selected the same
        # way megatron's own modelopt/hybrid specs do.
        #
        # A block spec holds one entry per layer. At moe_layer_freq=1 every
        # entry is the SAME object (gpt_layer_specs.py:668 appends
        # moe_layer_spec itself), so this assigns to one spec repeatedly --
        # harmless, and it stays correct if a shape ever mixes dense and MoE
        # layers, which a single assignment would not.
        from megatron.core.transformer.transformer_layer import MoETransformerLayer

        for layer_spec in spec.layer_specs:
            layer_spec.module = MoETransformerLayer
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
    if blank_parts:
        _assert_parts_are_blank(model, blank_parts)
    return model
