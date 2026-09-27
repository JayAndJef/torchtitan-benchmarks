"""The harness facts that every arm must prove before the harness publishes its numbers."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from benchmarks.e2e.engines.api import MeshObserved, RankEvidence, RunSpec


LOSS_METRIC = re.compile(r"step:\s*(\d+).*?loss:\s*(nan|-?inf|[0-9.eE+-]+)")
"""The loss pattern; the words come before the digits, because the digit class alone reads ``-inf`` as ``-``."""

GRAD_NORM_METRIC = re.compile(
    r"step:\s*(\d+).*?grad_norm:\s*(nan|-?inf|[0-9.eE+-]+)"
)

TRAJECTORY_METRICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("loss", LOSS_METRIC),
    ("grad_norm", GRAD_NORM_METRIC),
)
"""The two trajectories that a step line carries, by the name that a failure prints."""


def trajectory(text: str, pattern: re.Pattern[str]) -> list[tuple[int, float]]:
    """The (step, value) pairs that ``pattern`` finds in ``text``, in log order."""
    values = []
    for line in text.splitlines():
        match = pattern.search(line)
        if match:
            values.append((int(match.group(1)), float(match.group(2))))
    return values


def non_finite_refusals(rank_logs: Mapping[int, str]) -> list[str]:
    """The first ``nan`` or ``inf`` of each trajectory on each rank."""
    refusals = []
    for rank, text in sorted(rank_logs.items()):
        for metric, pattern in TRAJECTORY_METRICS:
            for step, value in trajectory(text, pattern):
                if not math.isfinite(value):
                    refusals.append(
                        f"rank {rank} logged a non-finite {metric} at step "
                        f"{step} ({value})"
                    )
                    break
    return refusals


def _mesh_refusals(run: RunSpec, rank: int, mesh: MeshObserved) -> list[str]:
    """How the mesh that one rank states differs from the run's parallelism spec."""
    spec = run.parallelism
    refusals = []
    for name, declared, observed, none_text in (
        ("pp", spec.pp, mesh.pp, "no pipeline"),
        ("dp", spec.dp, mesh.dp, "no data parallelism"),
        ("ep", spec.ep, mesh.ep, "no expert parallelism"),
    ):
        if observed == declared:
            continue
        declares = none_text if declared == 1 else f"{name}={declared}"
        refusals.append(
            f"the run declares {declares}, and rank {rank} records "
            f"{name}={observed}"
        )
    # The level shards nothing at one data-parallel rank, so the log cannot state it there.
    if spec.dp > 1:
        if mesh.zero is None:
            refusals.append(
                f"rank {rank} does not state its ZeRO level at dp {spec.dp}"
            )
        elif mesh.zero != spec.zero:
            refusals.append(
                f"the run declares zero {spec.zero}, and rank {rank} records "
                f"zero {mesh.zero}"
            )
    return refusals


def check_evidence(run: RunSpec, evidence: Mapping[int, RankEvidence]) -> list[str]:
    """Every harness fact that the ranks' evidence breaks: completion, model size and mesh."""
    world_size = run.parallelism.world_size
    shape = run.shape
    refusals = [
        f"rank {rank} wrote nothing"
        for rank in range(world_size)
        if rank not in evidence
    ]
    refusals.extend(
        f"rank {rank} wrote output, and the run declares {world_size} rank(s)"
        for rank in sorted(evidence)
        if rank >= world_size
    )
    for rank, facts in sorted(evidence.items()):
        if not facts.completed:
            refusals.append(f"training did not complete on rank {rank}")
    model = f"model size {shape.name!r} ({shape.param_count:,} parameters)"
    counts = {
        rank: facts.param_count
        for rank, facts in sorted(evidence.items())
        if facts.param_count is not None
    }
    if evidence and not counts:
        refusals.append(f"{model} did not apply: no rank states a parameter count")
    refusals.extend(
        f"{model} did not apply on rank {rank}, which states {count:,}"
        for rank, count in counts.items()
        if count != shape.param_count
    )
    for rank, facts in sorted(evidence.items()):
        if facts.mesh is None:
            refusals.append(f"rank {rank} states no mesh")
        else:
            refusals.extend(_mesh_refusals(run, rank, facts.mesh))
    return refusals
