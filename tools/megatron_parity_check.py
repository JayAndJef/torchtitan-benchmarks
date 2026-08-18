"""One-off GPU parity check: TorchTitan Qwen3-1B vs the Megatron twin.

Builds both models, transfers the titan weights into the megatron layout,
runs one real pre-tokenized c4_test batch through both (titan: flex
attention with the document block mask; megatron: TE fused attention in THD
form with the same cu_seqlens), and reports logit and loss agreement. Also
asserts the exact parameter-count match and prints a per-tensor shape
census diff.

Run manually during bring-up (not part of CI):

    source ./cuda_compat.sh
    CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python tools/megatron_parity_check.py
    CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python tools/megatron_parity_check.py \
        --model-size huge

Expected agreement is bf16-level (rel_l2 around 2e-3), not bitwise: the
engines use different kernels and different reduction orders.

The ceiling is ``PiperShape.parity_gate``, per shape (normal 2e-2, huge
5e-2), and it is bf16-accumulation-scaled rather than arbitrary: rel_l2 grows
roughly with the square root of the reduction length, so the 12x dim of the
huge shape moves the normal shape's measured 5.5e-3 to 2.0e-2 (sqrt(12) =
3.46; predicted 1.9e-2). MEASURED with --fp32-reference on 20260809: running
the same weights in fp32 puts titan's own bf16 output 3.251e-2 from the
reference and megatron's 3.286e-2, a ratio of 1.011 -- and the
engine-to-engine distance (2.027e-2) is SMALLER than either engine's distance
to fp32. The two engines are therefore equally correct and the residual is
reduction order, not layout. The interleave itself is proved separately and
bitwise by _assert_qkv_roundtrip. Do not widen a gate without that evidence.

Memory: two full models plus fp32 logit copies live on one GPU, so the huge
shape needs an otherwise idle card (~55-60 GiB at dim 12288).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double().flatten()
    b = b.double().flatten()
    return (a - b).norm().item() / b.norm().item()


def build_titan_model(shape, dtype=torch.bfloat16):
    from piper1b.config_registry import _piper_1b_model

    config = _piper_1b_model(fuse_qkv=True, shape=shape)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("cuda"):
            model = config.build()
    finally:
        torch.set_default_dtype(previous_dtype)
    torch.manual_seed(42)
    model.init_states(buffer_device=torch.device("cuda"))
    model.eval()
    return model


def build_megatron_model(shape):
    from megatron_baseline.model import build_model

    model = build_model(seq_len=1024, shape=shape)
    model.eval()
    return model


def transfer_weights(titan, megatron, shape) -> None:
    """Copy titan parameters into the megatron layout, in place."""
    state = dict(titan.state_dict())

    def put(megatron_name: str, tensor: torch.Tensor) -> None:
        target = dict(megatron.named_parameters())[megatron_name]
        if target.shape != tensor.shape:
            raise ValueError(
                f"{megatron_name}: shape {tuple(target.shape)} != "
                f"source {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            target.copy_(tensor)

    put("embedding.word_embeddings.weight", state["tok_embeddings.weight"])
    put("output_layer.weight", state["lm_head.weight"])
    put("decoder.final_layernorm.weight", state["norm.weight"])
    for layer in range(shape.n_layers):
        titan_prefix = f"layers.{layer}"
        mega_prefix = f"decoder.layers.{layer}"
        # Titan's fused module exposes unfused-style wq/wk/wv in its state
        # dict (the merge hook re-packs on load). Assemble megatron's grouped
        # interleave (n_kv_heads, [q_per_group..., k, v], head_dim, dim)
        # explicitly -- the same concatenation titan's fused init uses.
        wq = state[f"{titan_prefix}.attention.qkv_linear.wq.weight"]
        wk = state[f"{titan_prefix}.attention.qkv_linear.wk.weight"]
        wv = state[f"{titan_prefix}.attention.qkv_linear.wv.weight"]
        # wq is [n_heads*head_dim, dim] = [dim, dim]; wk/wv are
        # [n_kv_heads*head_dim, dim] = [dim/2, dim]. The concat over dim=1
        # gives (n_kv_heads, heads_per_group + 2, head_dim, dim) and
        # qkv_out_features = (n_heads + 2*n_kv_heads)*head_dim = 2*dim.
        grouped = torch.cat(
            [
                wq.view(shape.n_kv_heads, shape.heads_per_group, shape.head_dim, shape.dim),
                wk.view(shape.n_kv_heads, 1, shape.head_dim, shape.dim),
                wv.view(shape.n_kv_heads, 1, shape.head_dim, shape.dim),
            ],
            dim=1,
        ).reshape(shape.qkv_out_features, shape.dim)
        _assert_qkv_roundtrip(grouped, wq, wk, wv, shape)
        put(f"{mega_prefix}.self_attention.linear_qkv.weight", grouped)
        put(
            f"{mega_prefix}.self_attention.linear_qkv.layer_norm_weight",
            state[f"{titan_prefix}.attention_norm.weight"],
        )
        put(
            f"{mega_prefix}.self_attention.linear_proj.weight",
            state[f"{titan_prefix}.attention.wo.weight"],
        )
        put(
            f"{mega_prefix}.self_attention.q_layernorm.weight",
            state[f"{titan_prefix}.attention.q_norm.weight"],
        )
        put(
            f"{mega_prefix}.self_attention.k_layernorm.weight",
            state[f"{titan_prefix}.attention.k_norm.weight"],
        )
        put(
            f"{mega_prefix}.pre_mlp_layernorm.weight",
            state[f"{titan_prefix}.ffn_norm.weight"],
        )
        put(
            f"{mega_prefix}.mlp.router.weight",
            state[f"{titan_prefix}.moe.router.gate.weight"],
        )
        w1 = state[f"{titan_prefix}.moe.routed_experts.inner_experts.w1_EFD"]
        w2 = state[f"{titan_prefix}.moe.routed_experts.inner_experts.w2_EDF"]
        w3 = state[f"{titan_prefix}.moe.routed_experts.inner_experts.w3_EFD"]
        for expert in range(shape.num_experts):
            # megatron gated fc1 rows: [gate (titan w1); up (titan w3)].
            put(
                f"{mega_prefix}.mlp.experts.linear_fc1.weight{expert}",
                torch.cat([w1[expert], w3[expert]], dim=0),
            )
            put(
                f"{mega_prefix}.mlp.experts.linear_fc2.weight{expert}",
                w2[expert],
            )


def _assert_qkv_roundtrip(grouped, wq, wk, wv, shape) -> None:
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
    for name, got, want in (("wq", back_q, wq), ("wk", back_k, wk), ("wv", back_v, wv)):
        if not torch.equal(got, want):
            raise AssertionError(
                f"qkv interleave is not invertible for {name} at shape "
                f"{shape.name} (dim {shape.dim}, n_kv_heads {shape.n_kv_heads})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="TorchTitan vs Megatron parity")
    parser.add_argument("--model-size", default="normal")
    parser.add_argument(
        "--gate",
        type=float,
        default=None,
        help=(
            "logit rel_l2 ceiling; defaults to the shape's own parity_gate "
            "in piper1b/model_shape.py (normal 2e-2, huge 5e-2)"
        ),
    )
    parser.add_argument(
        "--fp32-reference",
        action="store_true",
        help=(
            "Also run the same weights in fp32 and report how far each "
            "engine's bf16 output sits from it -- the evidence that a "
            "rel_l2 above the gate is accumulation, not a layout bug."
        ),
    )
    args = parser.parse_args()

    from piper1b.model_shape import shape_by_name

    shape = shape_by_name(args.model_size)
    if args.gate is None:
        args.gate = shape.parity_gate
    print(
        f"model size: {shape.name} (dim {shape.dim}, {shape.n_layers} layers)"
        f"  gate: {args.gate:.0e}"
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29713")
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from megatron_baseline.location import (
        add_megatron_to_path,
        configure_te_environment,
    )

    add_megatron_to_path()
    configure_te_environment()
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(42)

    titan = build_titan_model(shape)
    megatron = build_megatron_model(shape)

    titan_count = sum(p.numel() for p in titan.parameters())
    megatron_count = sum(p.numel() for p in megatron.parameters())
    print(f"param count: titan {titan_count:,}  megatron {megatron_count:,}")
    assert titan_count == megatron_count == shape.param_count, (
        f"expected {shape.param_count:,} parameters for shape {shape.name!r}"
    )

    transfer_weights(titan, megatron, shape)
    print("weight transfer complete (every tensor shape matched)")

    from megatron_baseline.data import materialize_titan_samples, thd_batches

    samples = materialize_titan_samples(seq_len=1024, num_samples=4)
    batch = thd_batches(samples, batch_size=4)[0]

    with torch.no_grad():
        # Titan: batched rows + flex document mask.
        tokens_bl = batch.tokens.view(4, 1024).cuda()
        positions_bl = batch.positions.view(4, 1024).cuda()
        masks = titan.get_attention_masks(positions=positions_bl)
        titan_logits = titan(tokens_bl, positions=positions_bl, attention_masks=masks)

        # Megatron: the same tokens in THD form with the same boundaries.
        cu = batch.cu_seqlens.cuda()
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=1024,
            max_seqlen_kv=1024,
        )
        megatron_logits = megatron(
            batch.tokens.cuda(),
            position_ids=None,
            attention_mask=None,
            packed_seq_params=packed,
        )

    titan_flat = titan_logits.reshape(-1, titan_logits.shape[-1]).float()
    megatron_flat = megatron_logits.reshape(-1, megatron_logits.shape[-1]).float()
    print(f"logit shapes: titan {tuple(titan_flat.shape)} megatron {tuple(megatron_flat.shape)}")
    error = rel_l2(titan_flat, megatron_flat)

    labels = batch.labels.view(-1).cuda()
    titan_loss = torch.nn.functional.cross_entropy(titan_flat, labels)
    megatron_loss = torch.nn.functional.cross_entropy(megatron_flat, labels)
    print(f"logits rel_l2: {error:.3e}")
    print(f"loss: titan {titan_loss.item():.6f}  megatron {megatron_loss.item():.6f}")

    if args.fp32_reference:
        # Is a rel_l2 above the bf16 gate a layout bug or just accumulation?
        # The discriminator is an fp32 run of the SAME weights: if megatron's
        # bf16 output is no further from it than titan's own bf16 output is,
        # both engines are equally correct and the difference is the
        # reduction order, not the model. (rel_l2 grows roughly with the
        # square root of the reduction length, so a 12x dim legitimately
        # pushes ~5e-3 toward 2e-2.)
        del megatron
        torch.cuda.empty_cache()
        reference_model = build_titan_model(shape, dtype=torch.float32)
        reference_model.load_state_dict(
            {name: tensor.float() for name, tensor in titan.state_dict().items()}
        )
        with torch.no_grad():
            reference_logits = reference_model(
                tokens_bl, positions=positions_bl, attention_masks=masks
            )
        reference_flat = reference_logits.reshape(
            -1, reference_logits.shape[-1]
        ).float()
        titan_vs_fp32 = rel_l2(titan_flat, reference_flat)
        megatron_vs_fp32 = rel_l2(megatron_flat, reference_flat)
        reference_loss = torch.nn.functional.cross_entropy(reference_flat, labels)
        print(f"fp32 reference loss: {reference_loss.item():.6f}")
        print(f"titan bf16    vs fp32 reference rel_l2: {titan_vs_fp32:.3e}")
        print(f"megatron bf16 vs fp32 reference rel_l2: {megatron_vs_fp32:.3e}")
        print(
            "ratio (megatron/titan distance to fp32): "
            f"{megatron_vs_fp32 / titan_vs_fp32:.3f}"
        )

    assert error < args.gate, (
        f"logit rel_l2 {error} exceeds the bf16 gate {args.gate}"
    )
    print("PARITY CHECK PASSED")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
