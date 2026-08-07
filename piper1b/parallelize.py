"""Single-GPU parallelize_fn: AC + per-block compile, plain bf16, no FSDP.

Piper trains on plain bf16 parameters with no FSDP; torchtitan's
parallelize_qwen3 wraps the model in FSDP2 even at world_size 1, where it
acts purely as a mixed-precision engine (fp32 masters plus per-step bf16
unshard cast-copies). That per-step free/realloc of the unsharded params
also ping-pongs their addresses, which violates the CUDA-graph-trees
static-address contract and forces continuous graph re-capture under the
cudagraph compile modes.

Delegating with skip_dp=True applies exactly the AC -> per-block
torch.compile pipeline (including the "Compiling each TransformerBlock"
log line that validate_arm rule 8 matches) and returns before any mesh
resolution or fully_shard call. With FSDP gone, training.dtype="bfloat16"
is the ONLY bf16 mechanism, and the TE RoPE arm hard-requires bf16
activations (its fallback line trips validate_arm rule 4), so the dtype
is enforced here rather than trusted.
"""

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
)
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.parallelize import parallelize_qwen3


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
    if parallel_dims.world_size != 1:
        raise RuntimeError(
            "piper1b benchmark configs are single-GPU only: piper trains on "
            "plain bf16 parameters with no FSDP, and this wrapper skips FSDP "
            f"entirely (got world_size={parallel_dims.world_size})"
        )
    if training.dtype != "bfloat16":
        raise ValueError(
            "piper1b benchmark configs require training.dtype='bfloat16': "
            "with FSDP skipped there is no mixed-precision engine, and the "
            "TE RoPE arm hard-requires bf16 activations "
            f"(got training.dtype={training.dtype!r})"
        )
    return parallelize_qwen3(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ac_config,
        dump_folder=dump_folder,
        skip_dp=True,
    )
