"""The TorchTitan trainer config that every TorchTitan arm names with ``--config``."""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from benchmarks.e2e.engines.torchtitan.plugins.parallelize import parallelize_piper1b
from benchmarks.e2e.engines.torchtitan.plugins.replay import PretokenizedReplayDataLoader
from benchmarks.models.piper_qwen3.shape import shape_by_name
from benchmarks.models.piper_qwen3.titan_model import _piper_1b_model


def qwen3_piper_1b_pretokenized(*, size: str = "1b") -> Trainer.Config:
    """The Qwen3 MoE model of shape ``size`` on the pre-tokenized c4_test replay stream.

    The fork gives ``--config-arg size=<name>`` to the ``size`` keyword. A
    manifest records the config name, so do not rename the function.
    """
    model_spec = ModelSpec(
        name="qwen3",
        flavor="piper_1B",
        model=_piper_1b_model(fuse_qkv=True, shape=shape_by_name(size)),
        parallelize_fn=parallelize_piper1b,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3StateDictAdapter,
    )
    training = TrainingConfig(
        local_batch_size=4,
        seq_len=1024,
        steps=40,
        dtype="bfloat16",
    )
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_spec),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=PretokenizedReplayDataLoader.Config(replay_steps=training.steps),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=training,
        checkpoint=CheckpointManager.Config(
            interval=500,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
    )
