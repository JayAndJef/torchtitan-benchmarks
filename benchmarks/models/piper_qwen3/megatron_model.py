"""The Qwen3 model of Piper as a bare megatron-core ``GPTModel``, built from a ``PiperShape`` and an ``McoreProfile``."""

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
    """The derived block spec, with each part in ``parts`` replaced by the ``IdentityOp`` default of its field.

    Blank only the parts that the caller never reads: a blanked part returns
    its input, and a timed call of it reads as a fast kernel.
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
    """Raise when decoder layer 0 of ``model`` holds a real module in a part that ``parts`` names."""
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
    shape: PiperShape,
    profile: McoreProfile,
    cuda_graph_impl: str | None = None,
    cuda_graph_modules: tuple[str, ...] = (),
    use_cpu_initialization: bool = False,
    blank_parts: tuple[str, ...] = (),
    pipeline_model_parallel_size: int = 1,
    pre_process: bool = True,
    post_process: bool = True,
    batch_p2p_sync: bool = True,
):
    """The bare ``GPTModel`` in bf16, on the current CUDA device or, under ``use_cpu_initialization``, on the host.

    The caller passes a ``pipeline_model_parallel_size`` that agrees with
    ``initialize_model_parallel``, because Megatron never compares the two.
    """
    # Checked before the first import, so that a CPU test reaches it.
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
    attention_backends = {member.name: member for member in AttnBackend}
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
    # The spec comes from the config, so the config alone sets the MoE and qk-norm choices.
    spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
    if blank_parts:
        spec = _blank_layer_parts(spec, blank_parts)
    if cuda_graph_impl == "local":
        # The stock layer graphs the whole forward, and dynamic MoE refuses that.
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
