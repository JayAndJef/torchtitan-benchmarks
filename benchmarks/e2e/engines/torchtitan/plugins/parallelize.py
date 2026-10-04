"""The TorchTitan parallelize function: activation checkpointing, per-block compile, and FSDP above one data-parallel rank."""

from torch.distributed.fsdp import FSDPModule
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
)
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.tools.logging import logger


DATA_PARALLEL_LINE = (
    "piper1b data parallel: fully_shard applied "
    "(dp_replicate={replicate}, dp_shard={shard})"
)
"""The line this module prints after it counts the FSDP units; the TorchTitan validation reads it."""


def skip_data_parallel(parallel_dims: ParallelDims) -> bool:
    """Whether this rank needs none of TorchTitan's data-parallel machinery."""
    return parallel_dims.dp_replicate * parallel_dims.dp_shard == 1


def parallelize_piper1b(
    model: Qwen3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> Qwen3Model:
    """Apply ``parallelize_qwen3``, and refuse a mesh that the manifest cannot name."""
    if parallel_dims.tp != 1:
        raise RuntimeError(
            "piper1b benchmark configs do not support tensor parallelism: "
            "the harness cannot express a tensor-parallel degree, so the "
            f"manifest could not record this run (got tp={parallel_dims.tp})"
        )
    if parallel_dims.cp != 1:
        raise RuntimeError(
            "piper1b benchmark configs do not support context parallelism: "
            "the harness cannot express a context-parallel degree, so the "
            f"manifest could not record this run (got cp={parallel_dims.cp})"
        )
    skip_dp = skip_data_parallel(parallel_dims)
    if not skip_dp:
        # The configured value, because the resolved mesh cannot show an omitted flag.
        if parallelism.data_parallel_shard_degree < 0:
            raise RuntimeError(
                "piper1b benchmark configs need an explicit "
                "--parallelism.data-parallel-shard-degree: TorchTitan reads "
                "an omitted one as every remaining rank, so this run shards "
                "the parameters while the manifest records the "
                "ZeRO level the harness asked for (got "
                "data_parallel_shard_degree="
                f"{parallelism.data_parallel_shard_degree})"
            )
        if parallel_dims.dp_replicate > 1 and parallel_dims.dp_shard > 1:
            raise RuntimeError(
                "piper1b benchmark configs run one data-parallel treatment "
                "at a time, replication or sharding; this mesh does both "
                "and the manifest carries no ZeRO level that "
                f"names it (got dp_replicate={parallel_dims.dp_replicate}, "
                f"dp_shard={parallel_dims.dp_shard})"
            )
    if training.dtype != "bfloat16":
        raise ValueError(
            "piper1b benchmark configs require training.dtype='bfloat16': "
            "with FSDP skipped there is no mixed-precision engine, and the "
            "TE RoPE arm hard-requires bf16 activations "
            f"(got training.dtype={training.dtype!r})"
        )
    model = parallelize_qwen3(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
        skip_dp=skip_dp,
    )
    if not skip_dp:
        units = sum(
            1 for module in model.modules() if isinstance(module, FSDPModule)
        )
        if not units:
            raise RuntimeError(
                "piper1b benchmark configs asked TorchTitan for its "
                f"data-parallel path at dp_replicate="
                f"{parallel_dims.dp_replicate}, and no module came back "
                "wrapped; these ranks would never reduce their gradients"
            )
        logger.info(
            DATA_PARALLEL_LINE.format(
                replicate=parallel_dims.dp_replicate,
                shard=parallel_dims.dp_shard,
            )
            + f"; {units} FSDP units"
        )
    return model
