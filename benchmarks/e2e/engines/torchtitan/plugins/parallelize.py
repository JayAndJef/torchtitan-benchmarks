"""The parallelize_fn: AC and per-block compile, plain bf16, FSDP under DP.

Piper trains on plain bf16 parameters with no FSDP. ``parallelize_qwen3``
wraps the model in FSDP2 even at world size 1, where it acts purely as a
mixed-precision engine: fp32 masters plus per-step bf16 unshard
cast-copies. That per-step free and realloc also moves the addresses of the
unsharded parameters, which violates the CUDA-graph-trees static-address
contract.

Delegating with ``skip_dp=True`` applies exactly the AC and per-block
``torch.compile`` pipeline, including the log line arm rule 8 matches, and
returns before any mesh resolution or ``fully_shard`` call. With FSDP gone,
``training.dtype="bfloat16"`` is the ONLY bf16 mechanism, and the TE RoPE
arm hard-requires bf16 activations, so this module enforces the dtype
rather than trusting it.

**Whether to skip is a property of the run, not a constant.**
``skip_data_parallel`` reads the delivered mesh, and it is the same
predicate ``benchmarks/e2e/parallelism.py``'s ``skip_dp`` applies to the
spec: no data-parallel machinery is needed exactly when the data-parallel
degree is 1. So a single-GPU run and a pipeline-only run keep the
plain-bf16 model, and a data-parallel run gets ``fully_shard``, because
TorchTitan ships no DDP class.

**The refusals are per axis, and each names its own reason.** A pipeline
rank holds a slice of the layers and needs no gradient synchronization, so
it keeps the plain-bf16 model.

**The dropped shard-degree flag is refused on the raw configured value.**
``data_parallel_shard_degree`` is **-1** in TorchTitan when nobody sends
the flag, which means "take every remaining rank", so a dp 2 run that
omitted it shards the parameters while the manifest records the ZeRO level
the harness asked for. The resolved mesh cannot show that, because an
honest sharded run and a dropped flag reach the same degree. Four facts
make ``-1`` unambiguous on this object:
``ParallelismConfig.__post_init__`` does not touch the field,
``ParallelDims.__post_init__`` resolves ``-1`` onto itself and never onto
the ``ParallelismConfig``, TorchTitan's trainer hands the raw object over
on both the pipeline path and the plain path, and
``benchmarks/models/piper_qwen3/config_registry.py`` sets no parallelism
default.

**A mesh that replicates AND shards is refused too.** The harness asks for
one treatment at a time, so no spec asks for HSDP. A passthrough flag can
build one, and the manifest carries no ZeRO level that names it.

**This module cannot check ``titan_mesh``.** It runs in the training
subprocess and reads only the ``ParallelDims`` TorchTitan built from the
command line, never the harness's own spec. A ``titan_mesh`` that emitted
the wrong pair would produce a command line both guards below accept, and
arm rule 12 composes its own marker from the same function. The
parent-side pin of that table is what catches it.
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


DATA_PARALLEL_LINE = (
    "piper1b data parallel: fully_shard applied "
    "(dp_replicate={replicate}, dp_shard={shard})"
)
"""What arm rule 12 matches on a data-parallel titan arm.

This module prints it only after it has counted the FSDP units the
delegate really built, so the line states a fact about the model rather
than a request that was made. Keep it in sync with
``benchmarks/e2e/validation.py``'s ``_titan_parallelism_markers``, which
composes the same string from the spec.
"""


def skip_data_parallel(parallel_dims: ParallelDims) -> bool:
    """Whether this rank needs none of TorchTitan's data-parallel machinery.

    True exactly when the data-parallel degree is 1, which is the
    single-GPU run and the pipeline-only run. It is the subprocess-side
    twin of ``benchmarks/e2e/parallelism.py``'s ``skip_dp``, read off the
    mesh TorchTitan built.
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
    # The harness expresses neither, so the manifest would name a wrong mesh.
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
        # The RAW configured value: the mesh cannot show a dropped flag.
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
        # One treatment at a time. HSDP does both, and no spec asks for it.
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
        # A lost skip_dp argument would publish twice the true throughput.
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
