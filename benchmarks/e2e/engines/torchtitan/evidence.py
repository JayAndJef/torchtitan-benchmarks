"""The evidence that one TorchTitan rank's log states: completion, parameter count and mesh."""

from __future__ import annotations

import re

from benchmarks.e2e.engines.api import MeshObserved, RankEvidence


COMPLETION_LINE = "Training completed"
"""The line that every rank of a finished run prints."""

SIZE_LINE = re.compile(r"size: ([0-9,]+) total parameters")
"""The trainer's whole-model parameter count."""

MESH_LINE = re.compile(
    r"Building device mesh with parallelism: pp=(\d+), dp_replicate=(\d+), "
    r"dp_shard=(\d+), cp=\d+, tp=\d+, ep=(\d+)"
)
"""The mesh that TorchTitan builds."""

DATA_PARALLEL_LINE = re.compile(
    r"piper1b data parallel: fully_shard applied "
    r"\(dp_replicate=(\d+), dp_shard=(\d+)\)"
)
"""The data-parallel mesh that the parallelize plug-in wraps."""


def _one(rank: int, fact: str, values: set) -> object | None:
    """The one value of ``fact`` that a rank states, or ``None``; two values raise."""
    if len(values) > 1:
        raise RuntimeError(
            f"rank {rank} states two values of its {fact}: "
            + ", ".join(sorted(str(value) for value in values))
        )
    return next(iter(values), None)


def read_evidence(rank: int, text: str) -> RankEvidence:
    """The evidence in one rank's log; a rank that states no mesh ran on one device."""
    if not text.strip():
        return RankEvidence(rank=rank, completed=False, param_count=None, mesh=None)
    counts = {int(match.group(1).replace(",", "")) for match in SIZE_LINE.finditer(text)}
    meshes = {
        tuple(int(group) for group in match.groups())
        for match in MESH_LINE.finditer(text)
    }
    wrapped = {
        tuple(int(group) for group in match.groups())
        for match in DATA_PARALLEL_LINE.finditer(text)
    }
    stated = _one(rank, "mesh", meshes)
    data_parallel = _one(rank, "data-parallel mesh", wrapped)
    pp, replicate, shard, ep = stated or (1, *(data_parallel or (1, 1)), 1)
    if data_parallel is not None and data_parallel != (replicate, shard):
        raise RuntimeError(
            f"rank {rank} builds dp_replicate={replicate}, dp_shard={shard} "
            f"and wraps dp_replicate={data_parallel[0]}, "
            f"dp_shard={data_parallel[1]}"
        )
    dp = replicate * shard
    return RankEvidence(
        rank=rank,
        completed=COMPLETION_LINE in text,
        param_count=_one(rank, "parameter count", counts),
        mesh=MeshObserved(
            dp=dp,
            pp=pp,
            ep=ep,
            zero=None if dp == 1 else int(shard > 1),
        ),
    )
