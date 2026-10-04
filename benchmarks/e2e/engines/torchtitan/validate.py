"""The TorchTitan rules that an arm's logs and traces must pass: the treatments it declares, and its profiler traces."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from benchmarks.e2e.engines.api import Arm, CompileMode, DataSpec, RunSpec
from benchmarks.e2e.engines.torchtitan.mesh import SCHEDULES, titan_mesh
from benchmarks.e2e.parallelism import PP_SCHEDULES, ParallelismSpec, n_microbatches
from benchmarks.e2e.validation import trace_refusals


COMPILE_LINE = "with torch.compile"
"""The line that proves whole-block ``torch.compile``."""

SAC_LINE = "Applied SelectiveAC activation checkpointing"
"""The line that proves selective activation checkpointing."""

FALLBACK_LINES = ("falling back to the PyTorch",)
"""The phrases that mean a silent fallback to a slower path."""

OVERRIDE_LINE = re.compile(r"\[Override\]")


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
        microbatches = n_microbatches(spec, local_batch_size=data.local_batch_size)
        markers.append(
            f"Using pipeline schedule {SCHEDULES[spec.pp_schedule].titan_name} "
            f"with {microbatches} microbatches and "
            f"{spec.pp * schedule.stages_per_rank} stages"
        )
    return tuple(markers)


def required_lines(run: RunSpec) -> dict[str, tuple[str, ...]]:
    """The lines that every rank must print, by the treatment that each one proves."""
    if run.parallelism.world_size == 1:
        return {}
    return {"parallelism": mesh_markers(run.parallelism, run.data)}


def _log_refusals(run: RunSpec, arm: Arm, rank: int, log: str) -> list[str]:
    """The rules that one rank's log breaks."""
    config = arm.config
    where = f"on rank {rank}"
    refusals = []
    compiled = COMPILE_LINE in log
    if config.compile is CompileMode.TORCH and not compiled:
        refusals.append(
            f"the arm asks for torch.compile and the engine did not apply it {where}"
        )
    if config.compile is CompileMode.NONE and compiled:
        refusals.append(f"the arm runs eager and the engine compiled the model {where}")
    sac_applied = SAC_LINE in log
    if run.ac_mode == "sac" and not sac_applied:
        refusals.append(
            f"ac mode 'sac' requested but SelectiveAC was not applied {where}"
        )
    if run.ac_mode == "none" and sac_applied:
        refusals.append(f"ac mode 'none' requested but SelectiveAC was applied {where}")
    if config.overrides_per_block:
        expected = config.overrides_per_block * run.shape.n_layers
        found = len(OVERRIDE_LINE.findall(log))
        if found != expected:
            refusals.append(
                f"expected {expected} override applications, found {found} {where}"
            )
        refusals.extend(
            f"override {target!r} did not apply {where}"
            for target in config.override_imports
            if f"[Override] {target}:" not in log
        )
    refusals.extend(
        f"silent fallback marker {marker!r} found in the log {where}"
        for marker in FALLBACK_LINES
        if marker in log
    )
    refusals.extend(
        f"the requested {treatment} did not apply; the engine never logged "
        f"{marker!r} {where}"
        for treatment, markers in required_lines(run).items()
        for marker in markers
        if marker not in log
    )
    return refusals


def validate_outputs(
    run: RunSpec, arm: Arm, arm_dir: Path, rank_logs: Mapping[int, str]
) -> list[str]:
    """Every TorchTitan rule that the arm's logs and traces break."""
    refusals = [
        refusal
        for rank, log in sorted(rank_logs.items())
        for refusal in _log_refusals(run, arm, rank, log)
    ]
    if run.profile:
        refusals.extend(
            trace_refusals(run, arm_dir, arm.config.trace_kernel_markers)
        )
    return refusals
