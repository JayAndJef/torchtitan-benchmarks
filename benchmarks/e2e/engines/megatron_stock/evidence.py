"""The evidence that one stock Megatron rank's log states: completion, parameter count and mesh."""

from __future__ import annotations

import re

from benchmarks.e2e.engines.api import MeshObserved, RankEvidence
from benchmarks.e2e.engines.megatron_stock.flags import data_parallel_optimizer
from benchmarks.e2e.parallelism import ZERO_MODES


COMPLETION_LINE = "Training completed"
"""The line that the driver prints on every rank of a finished run."""

SIZE_LINE = re.compile(r"size: ([0-9,]+) total parameters")
"""The model builder's whole-model parameter count."""

PARALLELISM_LINE = re.compile(
    r"Megatron-LM stock parallelism: dp=(\d+) pp=(\d+) ep=(\d+)"
)
"""The mesh that Megatron resolved, which the driver prints above one rank."""

DATA_PARALLEL_LINE = re.compile(
    r"Megatron-LM stock data parallel: \S+ over (\d+) ranks .*"
    r"expert_parallel=(\d+), optimizer=(\S+)\)"
)
"""The wrapper and the optimizer that Megatron built, which the driver prints above one data-parallel rank."""

ZERO_BY_OPTIMIZER = {data_parallel_optimizer(zero): zero for zero in ZERO_MODES}
"""The ZeRO level that each optimizer name of the data-parallel line states."""


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
        for match in PARALLELISM_LINE.finditer(text)
    }
    wrapped = {match.groups() for match in DATA_PARALLEL_LINE.finditer(text)}
    stated = _one(rank, "mesh", meshes)
    data_parallel = _one(rank, "data-parallel line", wrapped)
    if data_parallel is None:
        dp, pp, ep = stated or (1, 1, 1)
        zero = None
    else:
        wrapped_dp, wrapped_ep = int(data_parallel[0]), int(data_parallel[1])
        dp, pp, ep = stated or (wrapped_dp, 1, wrapped_ep)
        if (wrapped_dp, wrapped_ep) != (dp, ep):
            raise RuntimeError(
                f"rank {rank} resolves dp={dp} ep={ep} and wraps "
                f"dp={wrapped_dp} ep={wrapped_ep}"
            )
        zero = ZERO_BY_OPTIMIZER.get(data_parallel[2])
    return RankEvidence(
        rank=rank,
        completed=COMPLETION_LINE in text,
        param_count=_one(rank, "parameter count", counts),
        mesh=MeshObserved(dp=dp, pp=pp, ep=ep, zero=zero),
    )
