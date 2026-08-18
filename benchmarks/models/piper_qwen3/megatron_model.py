"""Qwen3-1B (piper flavor) as a bare megatron-core GPTModel with the TE spec.

Every geometry knob is taken from
``benchmarks.models.piper_qwen3.shape.PiperShape`` -- the same object
``benchmarks/models/piper_qwen3/config_registry.py`` builds the TorchTitan
twin from -- so ``--model-size`` moves both engines together and neither can
drift. At the
default ``normal`` shape that is dim 1024, 16 layers, 16 q / 8 kv heads,
head_dim 64, expert ffn 3584, vocab 151936.

Everything else mirrors the qwen3_piper_1b TorchTitan config (the parity
contract in the scenario docs): RMSNorm eps 1e-6, no biases, SwiGLU; MoE on
every layer with 4 experts, top-2, softmax-then-topk with top-k
renormalization (Qwen3 norm_topk_prob == megatron's post-softmax default),
fp32 router math, no aux loss and no expert bias; per-head qk RMSNorm before
RoPE; NeoX rotate-half RoPE with theta 1e6; vocab exactly 151936 (128 x 1187,
so no padding change); untied embeddings.

Precision is plain bf16: params_dtype bf16 plus a blanket .bfloat16() after
construction (torch-norm/TE-norm params otherwise materialize fp32), no
autocast, no fp32 masters. gradient_accumulation_fusion stays off so a bare
backward populates normal .grad (the fused path requires apex main_grad
buffers) -- the one performance default we knowingly decline.

Every other fusion megatron's own training entrypoint would enable is
enabled here explicitly. Building TransformerConfig directly bypasses
megatron/training/arguments.py, where those defaults actually live, so the
dataclass defaults are the wrong baseline; see the fusion block below.
"""

from __future__ import annotations

from benchmarks.models.piper_qwen3.shape import NORMAL, PiperShape


def build_model(
    *,
    seq_len: int,
    shape: PiperShape = NORMAL,
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

    extra = {}
    if use_cpu_initialization:
        extra["use_cpu_initialization"] = True
    if cuda_graph_impl is not None:
        extra["cuda_graph_impl"] = cuda_graph_impl
        if cuda_graph_modules:
            extra["cuda_graph_modules"] = list(cuda_graph_modules)
        # TE's attention asserts on the RNG tracker type inside captured
        # graphs; its own tracker is the supported one.
        extra["use_te_rng_tracker"] = True

    config = TransformerConfig(
        num_layers=shape.n_layers,
        hidden_size=shape.dim,
        num_attention_heads=shape.n_heads,
        num_query_groups=shape.n_kv_heads,
        kv_channels=shape.head_dim,
        # Required field; unused (every layer is MoE) but kept at the expert
        # width so nothing dense-sized is ever allocated from it.
        ffn_hidden_size=shape.moe_hidden_dim,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        add_bias_linear=False,
        add_qkv_bias=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        num_moe_experts=shape.num_experts,
        moe_layer_freq=1,
        moe_router_topk=shape.top_k,
        moe_ffn_hidden_size=shape.moe_hidden_dim,
        moe_router_score_function="softmax",
        # Post-softmax top-k: top-k on logits then softmax over the selected
        # k, algebraically identical to Qwen3's softmax-then-topk-then-renorm.
        moe_router_pre_softmax=False,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_enable_expert_bias=False,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="allgather",
        moe_grouped_gemm=True,
        apply_rope_fusion=True,
        # TransformerConfig's dataclass defaults are NOT megatron's defaults:
        # the ones a real run gets are set in the argparse layer we bypass by
        # constructing the config directly. --no-bias-swiglu-fusion and
        # --no-bias-dropout-fusion are both action="store_false" (so argparse
        # defaults them True) and argument_utils.py forwards the swiglu one as
        # bias_activation_fusion. Leaving them at the dataclass False ran the
        # unfused chunk/silu/mul/copy path in moe/experts.py and cost 11.9 GPU
        # ms/step -- an accidental handicap, not a property of the engine.
        bias_activation_fusion=True,
        bias_dropout_fusion=True,
        # Opt-in in megatron, but shipped in NVIDIA's own Qwen3 example
        # configs, and both are parity-safe pure kernel fusions (measured:
        # loss trajectories match the unfused run to ~1e-3).
        cross_entropy_loss_fusion=True,
        # 'te' routes the loss through transformer_engine.pytorch.
        # parallel_cross_entropy -- the SAME implementation our te_fused_ce
        # arm wraps (vendored at benchmarks/models/piper_qwen3/components/
        # lm_head/te_cross_entropy.py), and
        # the one piper_optimized_te_ce further optimizes. Choosing it makes
        # the loss path an apples-to-apples kernel comparison instead of an
        # algorithm comparison.
        #
        # 'native' is megatron's own jit_fuser CE: it upcasts the whole
        # [tokens, 151936] logits tensor to fp32 and makes ~6 full-tensor
        # traversals (max, sub, exp, sum, div) whose fp32 softmax it also
        # keeps resident for backward -- 88 GPU ms/step at batch 48 versus
        # 14.9 for the online-softmax TE-family kernel, which streams the
        # row in two bf16 passes and writes the gradient in place.
        #
        # NOTE: megatron's *training entrypoint* refuses this combination
        # (arguments.py ~1630) citing known stability issues. The core config
        # only warns (model_parallel_config.py ~536), and we construct the
        # config directly, so we get it. That is a deliberate choice to
        # measure megatron's fastest available loss path; it is NOT the
        # configuration a stock `pretrain_gpt.py` user runs. Report numbers
        # from this setting as "megatron with its fastest available CE", and
        # flip back to "native" for "megatron as NVIDIA ships it".
        cross_entropy_fusion_impl="te",
        moe_permute_fusion=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        gradient_accumulation_fusion=False,
        **extra,
    )
    # num_experts here is a *separate* argument from the config field above
    # (it selects the MoE submodule spec); both must move with the shape.
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=shape.num_experts,
        moe_grouped_gemm=True,
        qk_layernorm=True,
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
