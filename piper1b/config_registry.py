"""Out-of-tree torchtitan configs for the piper Qwen3-1B model.

Ports /data/zejiaqi/piper/examples/models/qwen3.py case '1B' onto current
torchtitan (dim=1024, n_layers=16, n_heads=16, n_kv_heads=8, head_dim=64,
MoE with 4 experts / top_k=2 / inter_dim=3584, qk_norm, rope theta 1e6,
max_seq_len 2048, no weight tying, vocab 151936, load_balance_coeff=None).

Every geometry knob comes from ``piper1b.model_shape.PiperShape`` -- the same
object ``megatron_baseline/model.py`` builds its twin from -- so the two
engines cannot drift. Each public config has a ``_huge`` twin built from the
``huge`` shape; the naming convention ``<config>_<size>`` is what
``benchmarks/runtime.py`` resolves ``--model-size`` through, and
``tests/test_model_shape.py`` asserts the closure over every (scenario, arm,
size) triple.

Known deltas vs piper (identical across all benchmark arms, so they do not
affect the RoPE comparison):
- torchtitan's MoE layer builder hardcodes route_norm=True (piper: False);
- experts run as torchtitan GroupedExperts (torch._grouped_mm) rather than
  piper's BmmExperts (same post-expert score placement, different kernels);
- the c4_test tokenizer (vocab 2020) is used against the full 151936-row
  embedding, so losses are not comparable to real Qwen3 training.

Usage:
    PYTHONPATH=/data/zejiaqi/torchtitan-benchmarks torchtitan_train \
        --module piper1b --config qwen3_piper_1b ...
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common import CosSinRoPE, Embedding, Linear
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import (
    _build_qwen3_moe_layers,
    _EMBEDDING_INIT,
    _output_linear_init,
    _qwen3_norm,
    Qwen3Model,
)
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from piper1b.lm_head.losses import (
    FusedLinearCrossEntropyLoss,
    PiperOptimizedCrossEntropyLoss,
    TECrossEntropyLoss,
)
from piper1b.model_shape import HUGE, NORMAL, PiperShape
from piper1b.parallelize import parallelize_piper1b
from piper1b.pretokenized_data import PretokenizedReplayDataLoader


def _piper_1b_model(
    *, fuse_qkv: bool, attn_backend: str = "flex", shape: PiperShape = NORMAL
) -> Qwen3Model.Config:
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
    # piper sets load_balance_coeff=None (no aux-free load balancing); the
    # torchtitan builder defaults to 1e-3, so match piper explicitly.
    for layer in layers:
        layer.moe.load_balance_coeff = None
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            # Real init: weight tying is OFF (as in piper), so the embedding is
            # a free parameter. (_EMBEDDING_SKIP_INIT is only valid with tying.)
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def qwen3_piper_1b(*, shape: PiperShape = NORMAL) -> Trainer.Config:
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="full_logits",
        shape=shape,
    )


def qwen3_piper_1b_varlen(*, shape: PiperShape = NORMAL) -> Trainer.Config:
    """Piper-1B with FlashAttention-3 varlen instead of FlexAttention.

    ``attn_backend="varlen"`` selects VarlenAttention, whose constructor
    activates FA3 -- so this config requires the flash3 dependency group to be
    installed and will raise ModuleNotFoundError without it. The masking is
    equivalent, not merely similar: both backends receive the same packed
    document boundaries, and their outputs agree with an fp64 reference to
    ~2e-3 rel_l2 (measured).
    """
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="full_logits",
        attn_backend="varlen",
        shape=shape,
    )


def qwen3_piper_1b_flex_flash(*, shape: PiperShape = NORMAL) -> Trainer.Config:
    """Piper-1B with FlexAttention lowered to FlashAttention-4 kernels.

    ``attn_backend="flex_flash"`` keeps FlexAttention and its BlockMask -- only
    the lowering changes, from an Inductor Triton template to FA4 CuTe DSL
    kernels. That makes this the arm that isolates the kernel family, since
    the varlen arm changes the masking mechanism as well.

    Requires the fa4 dependency group. Unlike varlen there is no silent
    degradation to guard against: BACKEND="FLASH" raises when flash_attn.cute
    is missing rather than falling back to Triton.
    """
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="full_logits",
        attn_backend="flex_flash",
        shape=shape,
    )


def qwen3_piper_1b_unfused_qkv(*, shape: PiperShape = NORMAL) -> Trainer.Config:
    return _piper_1b_trainer(
        fuse_qkv=False,
        loss_kind="full_logits",
        shape=shape,
    )


def qwen3_piper_1b_full_logits(*, shape: PiperShape = NORMAL) -> Trainer.Config:
    """Piper's vanilla full lm_head followed by cross entropy."""
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="full_logits",
        shape=shape,
    )


def qwen3_piper_1b_fused_linear_ce(
    *, shape: PiperShape = NORMAL
) -> Trainer.Config:
    """Full-token PyTorch-native fused linear plus cross entropy."""
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="fused_linear_ce",
        shape=shape,
    )


def qwen3_piper_1b_te_fused_ce(*, shape: PiperShape = NORMAL) -> Trainer.Config:
    """Full-token lm_head followed by TransformerEngine fused CE."""
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="te_fused_ce",
        shape=shape,
    )


def qwen3_piper_1b_piper_optimized_te_ce(
    *, attn_backend: str = "flex", shape: PiperShape = NORMAL
) -> Trainer.Config:
    """Full-token lm_head followed by Piper-optimized TE-derived CE."""
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="piper_optimized_te_ce",
        attn_backend=attn_backend,
        shape=shape,
    )


def qwen3_piper_1b_pretokenized(*, shape: PiperShape = NORMAL) -> Trainer.Config:
    """Stock model on the pre-tokenized replay stream (piper1b_megatron)."""
    return _with_pretokenized_replay(qwen3_piper_1b(shape=shape))


def qwen3_piper_1b_piper_optimized_te_ce_pretokenized(
    *, shape: PiperShape = NORMAL
) -> Trainer.Config:
    """Piper-optimized TE CE loss on the pre-tokenized replay stream."""
    return _with_pretokenized_replay(
        qwen3_piper_1b_piper_optimized_te_ce(shape=shape)
    )


# --- huge-shape twins -------------------------------------------------------
#
# One explicit def per public config, not a globals() loop: torchtitan
# resolves a config with getattr(module, name), which would work either way,
# but explicit names stay greppable, importable by tests, and carry a
# __name__ (tests/test_runner.py's ParallelizeTests subTest relies on it).


def qwen3_piper_1b_huge() -> Trainer.Config:
    """qwen3_piper_1b at the huge shape (piper1b/model_shape.py)."""
    return qwen3_piper_1b(shape=HUGE)


def qwen3_piper_1b_varlen_huge() -> Trainer.Config:
    """qwen3_piper_1b_varlen at the huge shape."""
    return qwen3_piper_1b_varlen(shape=HUGE)


def qwen3_piper_1b_flex_flash_huge() -> Trainer.Config:
    """qwen3_piper_1b_flex_flash at the huge shape."""
    return qwen3_piper_1b_flex_flash(shape=HUGE)


def qwen3_piper_1b_unfused_qkv_huge() -> Trainer.Config:
    """qwen3_piper_1b_unfused_qkv at the huge shape."""
    return qwen3_piper_1b_unfused_qkv(shape=HUGE)


def qwen3_piper_1b_full_logits_huge() -> Trainer.Config:
    """qwen3_piper_1b_full_logits at the huge shape."""
    return qwen3_piper_1b_full_logits(shape=HUGE)


def qwen3_piper_1b_fused_linear_ce_huge() -> Trainer.Config:
    """qwen3_piper_1b_fused_linear_ce at the huge shape."""
    return qwen3_piper_1b_fused_linear_ce(shape=HUGE)


def qwen3_piper_1b_te_fused_ce_huge() -> Trainer.Config:
    """qwen3_piper_1b_te_fused_ce at the huge shape."""
    return qwen3_piper_1b_te_fused_ce(shape=HUGE)


def qwen3_piper_1b_piper_optimized_te_ce_huge() -> Trainer.Config:
    """qwen3_piper_1b_piper_optimized_te_ce at the huge shape."""
    return qwen3_piper_1b_piper_optimized_te_ce(shape=HUGE)


def qwen3_piper_1b_pretokenized_huge() -> Trainer.Config:
    """qwen3_piper_1b_pretokenized at the huge shape."""
    return qwen3_piper_1b_pretokenized(shape=HUGE)


def qwen3_piper_1b_piper_optimized_te_ce_pretokenized_huge() -> Trainer.Config:
    """qwen3_piper_1b_piper_optimized_te_ce_pretokenized at the huge shape."""
    return qwen3_piper_1b_piper_optimized_te_ce_pretokenized(shape=HUGE)


def _with_pretokenized_replay(config: Trainer.Config) -> Trainer.Config:
    """Swap the dataloader for the replay variant used by the megatron
    comparison: same c4_test pipeline, materialized at startup so measured
    steps carry ~zero data-host cost (matching the Megatron driver).
    replay_steps tracks the config's own step count; running with more steps
    fails loudly instead of silently reusing data, so the runner delivers
    --dataloader.replay-steps alongside --training.steps (the workload
    declares the loader via Workload.replay_dataloader)."""
    config.dataloader = PretokenizedReplayDataLoader.Config(
        dataset="c4_test",
        replay_steps=config.training.steps,
    )
    return config


def _piper_1b_trainer(
    *,
    fuse_qkv: bool,
    loss_kind: str,
    attn_backend: str = "flex",
    shape: PiperShape = NORMAL,
) -> Trainer.Config:
    model_spec = ModelSpec(
        name="qwen3",
        flavor="piper_1B",
        model=_piper_1b_model(
            fuse_qkv=fuse_qkv, attn_backend=attn_backend, shape=shape
        ),
        parallelize_fn=parallelize_piper1b,
        pipelining_fn=pipeline_llm,
        # No register_moe_load_balancing_hook: load_balance_coeff is None
        # (matching piper), so there is no expert-bias state to update.
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
    cross_entropy = CrossEntropyLoss.Config(
        global_vocab_size=decoder_vocab_size(model_spec),
    )
    if loss_kind == "full_logits":
        loss = cross_entropy
    elif loss_kind == "fused_linear_ce":
        loss = FusedLinearCrossEntropyLoss.Config(
            batch_chunk_size=None,
            chunking_method=None,
        )
    elif loss_kind == "te_fused_ce":
        loss = TECrossEntropyLoss.Config()
    elif loss_kind == "piper_optimized_te_ce":
        loss = PiperOptimizedCrossEntropyLoss.Config()
    else:
        raise ValueError(f"Unknown piper-1B loss kind: {loss_kind}")

    return Trainer.Config(
        loss=loss,
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            # piper harness defaults: batch 4, seq 1024
            local_batch_size=4,
            seq_len=1024,
            steps=40,
            # Plain bf16 params/grads/optimizer states, matching piper. No
            # FSDP means no mixed-precision engine, so this is the only bf16
            # mechanism; parallelize_piper1b enforces it.
            dtype="bfloat16",
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
