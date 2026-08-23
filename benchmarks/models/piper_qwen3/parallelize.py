"""The parallelize_fn: AC + per-block compile, plain bf16, FSDP only under DP.

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

**Whether to skip is now a property of the run, not a constant.**
``skip_data_parallel`` reads the delivered mesh, and it is the same
predicate ``benchmarks/e2e/parallelism.py``'s ``skip_dp`` applies to the
spec: both say "no data-parallel machinery is needed" exactly when the
data-parallel degree is 1. So a single-GPU run and a pipeline-only run keep
the plain-bf16 model above, and a data-parallel run gets TorchTitan's own
DP path -- which is `fully_shard`, because TorchTitan ships no DDP class.

**The refusals are per axis, and each names its own reason.** One
``world_size != 1`` check stood here before pipeline parallelism, and it
refused every axis for one axis's reason. A pipeline rank holds a slice of
the layers and needs no gradient synchronization, so it keeps exactly the
plain-bf16 model above.

**A shard degree above 1 is refused, and that is the DP refusal that
survives.** ``data_parallel_shard_degree`` defaults to **-1** in TorchTitan
(``config/configs.py``), which resolves to "every remaining rank". A dp 2
run whose command line omitted the flag therefore arrives here as
``dp_replicate=1, dp_shard=2`` -- ZeRO-3, not the replication the manifest
records -- and every other check passes. ``titan_mesh`` asks for
``dp_shard=1`` on every spec this repo can run today, so the refusal costs
nothing and catches that substitution. **The expert-parallel stage must lift
it**, because ``titan_mesh`` returns ``dp_shard=ep`` at ``ep > 1``.

This module reads the ``ParallelDims`` TorchTitan builds from the command
line, never ``benchmarks/e2e/parallelism.py``: it executes inside the
training subprocess, where the harness's own spec is neither present nor
needed. The two agree because the harness builds that command line from the
spec.
"""

from torch.distributed.fsdp import FSDPModule
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import (
    ActivationCheckpointingConfig,
)
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.models.qwen3.parallelize import parallelize_qwen3
from torchtitan.tools.logging import logger


# What arm rule 12 matches on a data-parallel titan arm. It is printed only
# after this module has counted the FSDP units the delegate really built, so
# the line states a fact about the model rather than a request that was
# made. Keep in sync with benchmarks/e2e/validation.py's
# _titan_parallelism_markers, which composes the same string from the spec.
DATA_PARALLEL_LINE = (
    "piper1b data parallel: fully_shard applied "
    "(dp_replicate={replicate}, dp_shard={shard})"
)


def skip_data_parallel(parallel_dims: ParallelDims) -> bool:
    """Whether this rank needs none of TorchTitan's data-parallel machinery.

    True exactly when the data-parallel degree is 1, which is the single-GPU
    run and the pipeline-only run. It is the subprocess-side twin of
    ``benchmarks/e2e/parallelism.py``'s ``skip_dp``, read off the mesh
    TorchTitan built instead of off the spec the harness holds.
    """
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
    skip_dp = skip_data_parallel(parallel_dims)
    # The shard degree is the one data-parallel value that can arrive wrong
    # and still train. TorchTitan resolves an omitted
    # --parallelism.data-parallel-shard-degree to every remaining rank, so a
    # dp 2 run would shard the parameters and publish ZeRO-3 under a
    # replication label. Every spec this repo can run asks for shard 1.
    if not skip_dp and parallel_dims.dp_shard != 1:
        raise RuntimeError(
            "piper1b benchmark configs run TorchTitan's data-parallel path "
            "at shard degree 1, which its own config calls DDP; a shard "
            "degree above 1 shards the parameters instead and the manifest "
            f"would record replication (got dp_replicate="
            f"{parallel_dims.dp_replicate}, dp_shard={parallel_dims.dp_shard})"
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
        # The delegate takes skip_dp as a request. This counts what it built.
        # Without the count, a future edit that lost the argument would train
        # two ranks on different data, reduce no gradient, and report roughly
        # twice the true throughput -- a wrong number every other check
        # passes. The log line follows the count, so arm rule 12 reads a
        # measured fact.
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
