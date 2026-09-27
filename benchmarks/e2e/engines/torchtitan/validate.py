"""The TorchTitan validation profile and the mesh lines TorchTitan logs."""

from __future__ import annotations

import re

from benchmarks.e2e.engines.api import DataSpec
from benchmarks.e2e.parallelism import (
    PP_SCHEDULES,
    ParallelismSpec,
    n_microbatches,
    titan_mesh,
)
from benchmarks.e2e.validation import ValidationProfile


def _titan_parallelism_markers(
    spec: ParallelismSpec,
    data: DataSpec,
    megatron_precision: str | None,
    megatron_args: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """What TorchTitan logs about the mesh it really built.

    ``megatron_precision`` reaches every profile, because one call site
    serves all three. This one reads it and states nothing for it: no
    Megatron optimizer holds a TorchTitan arm's state.

    The first line comes from ``ParallelDims``, which TorchTitan builds from
    the command line it was given, so it states the degrees that took effect
    rather than the ones the harness asked for -- and it names ``cp`` and
    ``tp``, which ``ParallelismSpec`` cannot express and which must therefore
    both read 1.

    The second is the pipeline schedule and the microbatch count, and it is
    the one that catches the hazard this axis carries: two engines that agree
    on the split but disagree on how many microbatches they move through it
    would publish two different schedules under one label.

    **The third is ours, and it is the only one that proves a reduction.**
    The mesh line above says a mesh was built, not that anything was wrapped
    in it: TorchTitan logs it from ``ParallelDims`` before ``parallelize_fn``
    runs, so a run whose ``parallelize_piper1b`` skipped the data-parallel
    path prints it and reduces nothing. ``parallelize_piper1b`` therefore
    counts the FSDP units the delegate really built and prints
    ``DATA_PARALLEL_LINE`` after the count. **This module cannot import that
    constant**: ``parallelize.py`` imports torch and torchtitan, and this
    module is parent-side. The string is stated twice and
    ``tests/test_parallel_validation.py`` pins the two against each other,
    exactly as it does for the megatron driver's own line.

    A titan *loss* all-reduce is not evidence of a gradient all-reduce, which
    is why this rule reads a log line rather than only the NCCL trace marker
    arm rule 13 adds. TorchTitan reduces the loss over its ``loss`` mesh on
    every logged step whenever ``dp_cp_enabled`` (``trainer.py``), so an
    ``ncclDevKernel_AllReduce`` appears under dp 2 even with the gradients
    never reduced.
    """
    replicate, shard = titan_mesh(spec)
    markers = [
        f"Building device mesh with parallelism: pp={spec.pp}, "
        f"dp_replicate={replicate}, dp_shard={shard}, cp=1, tp=1, "
        f"ep={spec.ep}"
    ]
    if replicate * shard > 1:
        markers.append(
            "piper1b data parallel: fully_shard applied "
            f"(dp_replicate={replicate}, dp_shard={shard})"
        )
    if spec.pp > 1:
        schedule = PP_SCHEDULES[spec.pp_schedule]
        microbatches = n_microbatches(
            spec, local_batch_size=data.local_batch_size
        )
        markers.append(
            f"Using pipeline schedule {schedule.titan_name} with "
            f"{microbatches} microbatches and "
            f"{spec.pp * schedule.stages_per_rank} stages"
        )
    return tuple(markers)


def _no_p2p_markers(
    spec: ParallelismSpec, megatron_p2p_sync: str | None
) -> tuple[str, ...]:
    """TorchTitan has no p2p sync to prove.

    ``--megatron-p2p-sync`` reaches the megatron command alone, and a
    TorchTitan argv is the same under either value. So no line is asked of
    a titan rank, and its absence is not a failure.
    """
    return ()


def _no_nan_guard_markers(megatron_nan_guard: str | None) -> tuple[str, ...]:
    """TorchTitan has no Megatron NaN guard to prove.

    The value reaches the stock megatron command alone, and a TorchTitan
    argv is the same under either value.
    """
    return ()


def _no_precision_markers(megatron_precision: str | None) -> tuple[str, ...]:
    """TorchTitan has no Megatron optimizer precision to prove.

    The value reaches the stock megatron command alone, and a TorchTitan
    argv is the same under either value.
    """
    return ()


TORCHTITAN_PROFILE = ValidationProfile(
    completion_marker="Training completed",
    # Both of TorchTitan's compile lines carry it, in both directions.
    compile_marker="with torch.compile",
    failure_markers=("falling back to the PyTorch",),
    check_ac_line=True,
    parallelism_markers=_titan_parallelism_markers,
    # Logged only above one pipeline rank.
    pipelined_pattern=re.compile(r"Using pipeline schedule"),
    # Two witnesses, because the resolved mesh line is not ours.
    data_parallel_pattern=re.compile(
        r"dp_replicate=(?!1\b)\d+"
        r"|dp_shard=(?!1\b)\d+"
        r"|piper1b data parallel:"
    ),
    # The option never reaches a TorchTitan arm.
    p2p_markers=_no_p2p_markers,
    # Nor does this one.
    nan_guard_markers=_no_nan_guard_markers,
    precision_markers=_no_precision_markers,
)
"""The TorchTitan profile."""
