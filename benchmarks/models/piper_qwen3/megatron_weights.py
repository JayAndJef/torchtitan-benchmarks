"""The per-parameter map from TorchTitan's layout into megatron-core's.

One implementation, and -- once cross-engine kernel scenarios exist -- the
arm builders that must hand both engines the same weights before they compare
outputs. A second implementation would be a second thing to prove, and the
QKV grouped-interleave is already proved bitwise by ``assert_qkv_roundtrip``.

**Every transfer carries a component tag**, so a caller can take a slice.
A scenario that measures only the expert MLP needs only the ``experts``
weights; loading the rest would cost a full model's memory to compare one
GEMM. The tags name the components the cross-engine partition cuts on, which
is why they are finer than a whole-model transfer needs.

**This is a correctness mechanism, not a timing one.** A GEMM takes the same
time whatever its values are. The one place values reach timing is MoE
routing, and that is handled by giving both engines the same synthetic
``tokens_per_expert`` rather than by matching router weights.

Worker-side: it manipulates tensors, so it imports torch at module scope.
"""

from __future__ import annotations

from typing import Iterator

import torch

from benchmarks.models.piper_qwen3.shape import PiperShape

# The component each transfer belongs to, named after the cross-engine
# cuts rather than after megatron's module tree.
COMPONENTS = (
    "embedding",
    "qkv",
    "qk_norm",
    "attn_out",
    "ffn_norm",
    "router",
    "experts",
    "final_norm",
    "lm_head",
)


def assert_qkv_roundtrip(
    grouped: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    shape: PiperShape,
) -> None:
    """Reconstruct wq/wk/wv from the interleave and require bitwise equality.

    A wrong interleave produces rel_l2 of order 1 in the full forward pass,
    which is a slow and ambiguous way to find a five-literal reshape bug.
    This isolates the reshape itself.
    """
    view = grouped.view(
        shape.n_kv_heads, shape.heads_per_group + 2, shape.head_dim, shape.dim
    )
    back_q = view[:, : shape.heads_per_group].reshape(wq.shape)
    back_k = view[:, shape.heads_per_group].reshape(wk.shape)
    back_v = view[:, shape.heads_per_group + 1].reshape(wv.shape)
    for name, got, want in (
        ("wq", back_q, wq),
        ("wk", back_k, wk),
        ("wv", back_v, wv),
    ):
        if not torch.equal(got, want):
            raise AssertionError(
                f"qkv interleave is not invertible for {name} at shape "
                f"{shape.name} (dim {shape.dim}, n_kv_heads {shape.n_kv_heads})"
            )


def grouped_qkv(
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    shape: PiperShape,
) -> torch.Tensor:
    """Titan's separate wq/wk/wv as megatron's grouped interleave.

    Titan's fused module exposes unfused-style wq/wk/wv in its state dict (the
    merge hook re-packs on load). Megatron wants
    ``(n_kv_heads, [q_per_group..., k, v], head_dim, dim)``, which is the same
    concatenation titan's own fused init uses.

    ``wq`` is ``[n_heads*head_dim, dim]``; ``wk``/``wv`` are
    ``[n_kv_heads*head_dim, dim]``. The concat over dim 1 gives
    ``(n_kv_heads, heads_per_group + 2, head_dim, dim)``, and
    ``qkv_out_features`` is ``(n_heads + 2*n_kv_heads)*head_dim``.
    """
    grouped = torch.cat(
        [
            wq.view(
                shape.n_kv_heads,
                shape.heads_per_group,
                shape.head_dim,
                shape.dim,
            ),
            wk.view(shape.n_kv_heads, 1, shape.head_dim, shape.dim),
            wv.view(shape.n_kv_heads, 1, shape.head_dim, shape.dim),
        ],
        dim=1,
    ).reshape(shape.qkv_out_features, shape.dim)
    assert_qkv_roundtrip(grouped, wq, wk, wv, shape)
    return grouped


def weight_transfers(
    state: dict[str, torch.Tensor], shape: PiperShape
) -> Iterator[tuple[str, str, torch.Tensor]]:
    """``(component, megatron parameter name, tensor)`` for every weight.

    A generator, so a caller taking one component's slice never materializes
    the grouped QKV tensors of the layers it does not want.
    """
    yield (
        "embedding",
        "embedding.word_embeddings.weight",
        state["tok_embeddings.weight"],
    )
    yield "lm_head", "output_layer.weight", state["lm_head.weight"]
    yield "final_norm", "decoder.final_layernorm.weight", state["norm.weight"]
    for layer in range(shape.n_layers):
        titan = f"layers.{layer}"
        mega = f"decoder.layers.{layer}"
        yield (
            "qkv",
            f"{mega}.self_attention.linear_qkv.weight",
            grouped_qkv(
                state[f"{titan}.attention.qkv_linear.wq.weight"],
                state[f"{titan}.attention.qkv_linear.wk.weight"],
                state[f"{titan}.attention.qkv_linear.wv.weight"],
                shape,
            ),
        )
        # Tagged qkv because megatron fuses this norm into linear_qkv.
        yield (
            "qkv",
            f"{mega}.self_attention.linear_qkv.layer_norm_weight",
            state[f"{titan}.attention_norm.weight"],
        )
        yield (
            "attn_out",
            f"{mega}.self_attention.linear_proj.weight",
            state[f"{titan}.attention.wo.weight"],
        )
        yield (
            "qk_norm",
            f"{mega}.self_attention.q_layernorm.weight",
            state[f"{titan}.attention.q_norm.weight"],
        )
        yield (
            "qk_norm",
            f"{mega}.self_attention.k_layernorm.weight",
            state[f"{titan}.attention.k_norm.weight"],
        )
        yield (
            "ffn_norm",
            f"{mega}.pre_mlp_layernorm.weight",
            state[f"{titan}.ffn_norm.weight"],
        )
        yield (
            "router",
            f"{mega}.mlp.router.weight",
            state[f"{titan}.moe.router.gate.weight"],
        )
        w1 = state[f"{titan}.moe.routed_experts.inner_experts.w1_EFD"]
        w2 = state[f"{titan}.moe.routed_experts.inner_experts.w2_EDF"]
        w3 = state[f"{titan}.moe.routed_experts.inner_experts.w3_EFD"]
        for expert in range(shape.num_experts):
            # megatron gated fc1 rows: [gate (titan w1); up (titan w3)].
            yield (
                "experts",
                f"{mega}.mlp.experts.linear_fc1.weight{expert}",
                torch.cat([w1[expert], w3[expert]], dim=0),
            )
            yield (
                "experts",
                f"{mega}.mlp.experts.linear_fc2.weight{expert}",
                w2[expert],
            )


def transfer_weights(
    titan,
    megatron,
    shape: PiperShape,
    components: tuple[str, ...] | None = None,
) -> int:
    """Copy titan parameters into the megatron layout, in place.

    ``components`` restricts the copy to those tags; ``None`` means all of
    them. Returns the number of parameters written, so a caller that expected
    a slice can check it got one rather than silently copying nothing.
    """
    if components is not None:
        unknown = sorted(set(components) - set(COMPONENTS))
        if unknown:
            raise ValueError(
                f"unknown weight component(s) {', '.join(unknown)}. "
                f"Available: {', '.join(COMPONENTS)}"
            )
    wanted = set(components) if components is not None else None
    state = dict(titan.state_dict())
    targets = dict(megatron.named_parameters())
    written = 0
    for component, name, tensor in weight_transfers(state, shape):
        if wanted is not None and component not in wanted:
            continue
        try:
            target = targets[name]
        except KeyError as error:
            raise KeyError(
                f"megatron has no parameter {name!r}; the layouts have "
                "diverged, so the transfer would silently leave it at its "
                "initialization"
            ) from error
        if target.shape != tensor.shape:
            raise ValueError(
                f"{name}: shape {tuple(target.shape)} != "
                f"source {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            target.copy_(tensor)
        written += 1
    if written == 0:
        raise ValueError(
            f"transferred nothing for components {components!r} at shape "
            f"{shape.name!r}"
        )
    return written
