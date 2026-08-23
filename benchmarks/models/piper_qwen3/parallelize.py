"""The parallelize_fn: AC + per-block compile, plain bf16, no FSDP.

Piper trains on plain bf16 parameters with no FSDP; torchtitan's
parallelize_qwen3 wraps the model in FSDP2 even at world_size 1, where it
acts purely as a mixed-precision engine (fp32 masters plus per-step bf16
unshard cast-copies). That per-step free/realloc of the unsharded params
also ping-pongs their addresses, which violates the CUDA-graph-trees
static-address contract and forces continuous graph re-capture under the
cudagraph compile modes.

Delegating with skip_dp=True applies exactly the AC -> per-block
torch.compile pipeline (including the "Compiling each TransformerBlock"
log line that validate_arm arm rule 8 matches) and returns before any mesh
resolution or fully_shard call. With FSDP gone, training.dtype="bfloat16"
is the ONLY bf16 mechanism, and the TE RoPE arm hard-requires bf16
activations (its fallback line trips validate_arm arm rule 4), so the dtype
is enforced here rather than trusted.

**The refusals are per axis, and each names its own reason.** One
``world_size != 1`` check stood here before pipeline parallelism, and it
refused every axis for one axis's reason. A pipeline rank holds a slice of
the layers and needs no gradient synchronization, so it keeps exactly the
plain-bf16 model above; a data-parallel rank does not, and the ``skip_dp``
below is what would silently drop its gradient reduction.

This module reads the ``ParallelDims`` TorchTitan builds from the command
line, never ``benchmarks/e2e/parallelism.py``: it executes inside the
training subprocess, where the harness's own spec is neither present nor
needed. The two agree because the harness builds that command line from the
spec.
"""

from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
)
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.parallelize import parallelize_qwen3


# This wrapper always skips TorchTitan's data-parallel path, so a data-parallel
# degree above 1 would run without one. Named here because the refusal below
# and the delegation at the bottom must move together: the day a data-parallel
# run is supported, both change in one edit.
SKIP_DP = True


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
    # Tensor and context parallelism are out of scope for this repo's
    # parallelism axis: ParallelismSpec cannot express either, so a run that
    # reached here with one came from a passthrough flag the harness refuses,
    # and the manifest would name a mesh the run did not have.
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
    # TorchTitan ships no DDP class; fully_shard is its only data-parallel
    # path, and SKIP_DP returns before it. A data-parallel run that got past
    # this line would train two ranks on different data, never reduce the
    # gradients, and report roughly twice the true throughput -- a wrong
    # number that every other check passes.
    data_parallel = parallel_dims.dp_replicate * parallel_dims.dp_shard
    if SKIP_DP and data_parallel != 1:
        raise RuntimeError(
            "piper1b benchmark configs skip TorchTitan's data-parallel path, "
            "so a data-parallel degree above 1 would run with no gradient "
            f"reduction (got dp_replicate={parallel_dims.dp_replicate}, "
            f"dp_shard={parallel_dims.dp_shard})"
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
        skip_dp=SKIP_DP,
    )
