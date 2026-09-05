"""Out-of-tree torchtitan configs for the piper Qwen3-1B model.

Ports /data/zejiaqi/piper/examples/models/qwen3.py case '1B' onto current
torchtitan (dim=1024, n_layers=16, n_heads=16, n_kv_heads=8, head_dim=64,
MoE with 4 experts / top_k=2 / inter_dim=3584, qk_norm, rope theta 1e6,
max_seq_len 4096, no weight tying, vocab 151936, load_balance_coeff=None).

Every geometry knob comes from
``benchmarks.models.piper_qwen3.shape.PiperShape`` -- the same object
``benchmarks/models/piper_qwen3/megatron_model.py`` builds its twin
from -- so the two engines cannot drift. Each public config takes one
``size`` keyword naming a ``PIPER_SHAPES`` entry, so a shape costs one
registry entry and no config edit at all. The runner delivers it as
``--config-arg size=<name>``, which the fork's ``ConfigManager`` forwards
as a keyword argument; the value arrives as a string and ``shape_by_name``
raises on an unknown one. ``tests/test_model_shape.py`` asserts every
(scenario, arm) config accepts every registered size and builds the shape
it names.

**Reverse edge, pending resolution.** The ``_pretokenized`` configs import
``PretokenizedReplayDataLoader`` from ``benchmarks.e2e.data.piper_qwen3``,
so ``models/`` depends on ``e2e/``. This is the one import direction the
package layering does not want, and it is *new*: before the restructure the
configs and the replay loader were siblings in a single model package, so the
reference was intra-package and legal. Splitting that package by ownership --
geometry and configs to ``models/``, the run-shaped replay loader to
``e2e/`` -- is what exposed it.
It is a real edge, not an artifact of annotations: the loader is constructed
at runtime by the dataloader factory. It forms no cycle (nothing under
``e2e/`` imports this module; torchtitan reaches it by ``--module`` through
``ConfigManager``, not by import), and this module's sole non-test consumer
is that ``getattr`` lookup. The candidate resolution is to move the replay
loader under ``models/piper_qwen3/`` alongside the configs that are its only
caller, which deletes the edge outright; it is deferred rather than done
here because the flag-day commit moved code without redesigning it.

Known deltas vs piper (identical across all benchmark arms, so they do not
affect the RoPE comparison):
- torchtitan's MoE layer builder hardcodes route_norm=True (piper: False);
- experts run as torchtitan GroupedExperts (torch._grouped_mm) rather than
  piper's BmmExperts (same post-expert score placement, different kernels);
- the c4_test tokenizer (vocab 2020) is used against the full 151936-row
  embedding, so losses are not comparable to real Qwen3 training.

Usage:
    PYTHONPATH=/data/zejiaqi/torchtitan-benchmarks torchtitan_train \
        --module benchmarks.models.piper_qwen3 --config qwen3_piper_1b ...
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

# See "Reverse edge, pending resolution" above: models/ -> e2e/.
from benchmarks.e2e.data.piper_qwen3 import PretokenizedReplayDataLoader
from benchmarks.models.piper_qwen3.components.lm_head.losses import (
    FusedLinearCrossEntropyLoss,
    PiperOptimizedCrossEntropyLoss,
    TECrossEntropyLoss,
)
from benchmarks.models.piper_qwen3.parallelize import parallelize_piper1b
from benchmarks.models.piper_qwen3.shape import PiperShape, shape_by_name


def _piper_1b_model(
    *, fuse_qkv: bool, shape: PiperShape, attn_backend: str = "flex"
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


# **The ``1b`` in every public name below is the config family, not the
# geometry.** ``--config qwen3_piper_1b --config-arg size=48b`` builds a 48B
# model, and that is correct: the size is one argument and the config is
# another. The names are not renamed to match, because they are a contract --
# the fork's ``ConfigManager`` resolves ``--config`` by ``getattr`` on this
# module, and every manifest on disk records the name it used in ``commands``.
def qwen3_piper_1b(*, size: str = "1b") -> Trainer.Config:
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="full_logits",
        shape=shape_by_name(size),
    )


def qwen3_piper_1b_varlen(*, size: str = "1b") -> Trainer.Config:
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
        shape=shape_by_name(size),
    )


def qwen3_piper_1b_flex_flash(*, size: str = "1b") -> Trainer.Config:
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
        shape=shape_by_name(size),
    )


def qwen3_piper_1b_unfused_qkv(*, size: str = "1b") -> Trainer.Config:
    return _piper_1b_trainer(
        fuse_qkv=False,
        loss_kind="full_logits",
        shape=shape_by_name(size),
    )


def qwen3_piper_1b_full_logits(*, size: str = "1b") -> Trainer.Config:
    """Piper's vanilla full lm_head followed by cross entropy."""
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="full_logits",
        shape=shape_by_name(size),
    )


def qwen3_piper_1b_fused_linear_ce(*, size: str = "1b") -> Trainer.Config:
    """Full-token PyTorch-native fused linear plus cross entropy."""
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="fused_linear_ce",
        shape=shape_by_name(size),
    )


def qwen3_piper_1b_te_fused_ce(*, size: str = "1b") -> Trainer.Config:
    """Full-token lm_head followed by TransformerEngine fused CE."""
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="te_fused_ce",
        shape=shape_by_name(size),
    )


def qwen3_piper_1b_piper_optimized_te_ce(
    *, attn_backend: str = "flex", size: str = "1b"
) -> Trainer.Config:
    """Full-token lm_head followed by Piper-optimized TE-derived CE."""
    return _piper_1b_trainer(
        fuse_qkv=True,
        loss_kind="piper_optimized_te_ce",
        attn_backend=attn_backend,
        shape=shape_by_name(size),
    )


def qwen3_piper_1b_pretokenized(*, size: str = "1b") -> Trainer.Config:
    """Stock model on the pre-tokenized replay stream (piper1b_megatron)."""
    # Pass the size on rather than a resolved shape: the delegate resolves it
    # itself, and resolving here as well would be two places to keep in step.
    return _with_pretokenized_replay(qwen3_piper_1b(size=size))


def qwen3_piper_1b_piper_optimized_te_ce_pretokenized(
    *, size: str = "1b"
) -> Trainer.Config:
    """Piper-optimized TE CE loss on the pre-tokenized replay stream."""
    return _with_pretokenized_replay(
        qwen3_piper_1b_piper_optimized_te_ce(size=size)
    )


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
    shape: PiperShape,
    attn_backend: str = "flex",
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
