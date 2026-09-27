"""The log lines that prove a TorchTitan arm ran the treatment it declares."""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.e2e.engines.api import Arm, DataSpec, RunSpec
from benchmarks.e2e.engines.torchtitan.mesh import SCHEDULES, titan_mesh
from benchmarks.e2e.parallelism import (
    PP_SCHEDULES,
    ParallelismSpec,
    n_microbatches,
)
from benchmarks.e2e.validation import ValidationProfile, validate_against_profile


TORCHTITAN_PROFILE = ValidationProfile(
    completion_marker="Training completed",
    compile_marker="with torch.compile",
    failure_markers=("falling back to the PyTorch",),
    ac_line="Applied SelectiveAC activation checkpointing",
    pipelined_pattern=re.compile(r"Using pipeline schedule"),
    data_parallel_pattern=re.compile(
        r"dp_replicate=(?!1\b)\d+"
        r"|dp_shard=(?!1\b)\d+"
        r"|piper1b data parallel:"
    ),
)
"""The TorchTitan log lines that the shared rules read."""


def mesh_markers(spec: ParallelismSpec, data: DataSpec) -> tuple[str, ...]:
    """The lines that TorchTitan and the parallelize plug-in log about the mesh of ``spec``."""
    replicate, shard = titan_mesh(spec)
    markers = [
        f"Building device mesh with parallelism: pp={spec.pp}, "
        f"dp_replicate={replicate}, dp_shard={shard}, cp=1, tp=1, "
        f"ep={spec.ep}"
    ]
    if replicate * shard > 1:
        # plugins/parallelize.py prints this line, and a test pins the two.
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
            f"Using pipeline schedule {SCHEDULES[spec.pp_schedule].titan_name} "
            f"with {microbatches} microbatches and "
            f"{spec.pp * schedule.stages_per_rank} stages"
        )
    return tuple(markers)


def validate_outputs(run: RunSpec, arm: Arm, arm_dir: Path, log_path: Path) -> None:
    """Raise ``RuntimeError`` when the log or the traces of a TorchTitan arm refuse its numbers."""
    config = arm.config
    required_lines = {}
    if run.parallelism.world_size > 1:
        required_lines["parallelism"] = mesh_markers(run.parallelism, run.data)
    validate_against_profile(
        run,
        arm.name,
        arm_dir,
        log_path,
        profile=TORCHTITAN_PROFILE,
        required_lines=required_lines,
        compile=config.compile,
        overrides_per_block=config.overrides_per_block,
        override_imports=config.override_imports,
        trace_kernel_markers=config.trace_kernel_markers,
    )
