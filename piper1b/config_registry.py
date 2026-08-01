"""Out-of-tree torchtitan config: the piper Qwen3-1B model, for RoPE benchmarking.

Ports /data/zejiaqi/piper/examples/models/qwen3.py case '1B' onto current
torchtitan (dim=1024, n_layers=16, n_heads=16, n_kv_heads=8, head_dim=64,
MoE with 4 experts / top_k=2 / inter_dim=3584, qk_norm, rope theta 1e6,
max_seq_len 2048, no weight tying, vocab 151936, load_balance_coeff=None).

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
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
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
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


def _piper_1b_model() -> Qwen3Model.Config:
    dim = 1024
    head_dim = 64
    n_layers = 16
    vocab_size = 151936
    layers = _build_qwen3_moe_layers(
        fuse_qkv=True,
        n_layers=n_layers,
        dim=dim,
        n_heads=16,
        n_kv_heads=8,
        head_dim=head_dim,
        moe_hidden_dim=3584,
        num_experts=4,
        top_k=2,
        attn_backend="flex",
        moe_comm_backend="standard",
        rope=CosSinRoPE.Config(
            dim=head_dim,
            max_seq_len=2048,
            theta=1000000.0,
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


def qwen3_piper_1b() -> Trainer.Config:
    model_spec = ModelSpec(
        name="qwen3",
        flavor="piper_1B",
        model=_piper_1b_model(),
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_llm,
        # No register_moe_load_balancing_hook: load_balance_coeff is None
        # (matching piper), so there is no expert-bias state to update.
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
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
        ),
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
