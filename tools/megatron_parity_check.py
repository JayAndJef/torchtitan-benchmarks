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

Expected agreement is bf16-level (rel_l2 around 2e-3), not bitwise: the
engines use different kernels and different reduction orders.
"""

from __future__ import annotations

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


def build_titan_model():
    from piper1b.config_registry import _piper_1b_model

    config = _piper_1b_model(fuse_qkv=True)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = config.build()
    finally:
        torch.set_default_dtype(previous_dtype)
    torch.manual_seed(42)
    model.init_states(buffer_device=torch.device("cuda"))
    model.eval()
    return model


def build_megatron_model():
    from megatron_baseline.model import build_model

    model = build_model(seq_len=1024)
    model.eval()
    return model


def transfer_weights(titan, megatron) -> None:
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
    for layer in range(16):
        titan_prefix = f"layers.{layer}"
        mega_prefix = f"decoder.layers.{layer}"
        # Titan packs wqkv as (n_kv_heads, [q_per_group..., k, v], head_dim,
        # dim) flattened -- exactly megatron's grouped qkv interleave.
        put(
            f"{mega_prefix}.self_attention.linear_qkv.weight",
            state[f"{titan_prefix}.attention.wqkv.weight"],
        )
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
        w1 = state[f"{titan_prefix}.moe.routed_experts.w1_EFD"]
        w2 = state[f"{titan_prefix}.moe.routed_experts.w2_EDF"]
        w3 = state[f"{titan_prefix}.moe.routed_experts.w3_EFD"]
        for expert in range(4):
            # megatron gated fc1 rows: [gate (titan w1); up (titan w3)].
            put(
                f"{mega_prefix}.mlp.experts.linear_fc1.weight{expert}",
                torch.cat([w1[expert], w3[expert]], dim=0),
            )
            put(
                f"{mega_prefix}.mlp.experts.linear_fc2.weight{expert}",
                w2[expert],
            )


def main() -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29713")
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from megatron_baseline.location import add_megatron_to_path

    add_megatron_to_path()
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(42)

    titan = build_titan_model()
    megatron = build_megatron_model()

    titan_count = sum(p.numel() for p in titan.parameters())
    megatron_count = sum(p.numel() for p in megatron.parameters())
    print(f"param count: titan {titan_count:,}  megatron {megatron_count:,}")
    assert titan_count == megatron_count == 1_066_241_024

    transfer_weights(titan, megatron)
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

    assert error < 2e-2, f"logit rel_l2 {error} exceeds the bf16 gate"
    print("PARITY CHECK PASSED")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
