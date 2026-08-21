"""The contract between an arm builder and the measurement engine.

One dataclass, deliberately alone. ``BuiltArm`` is what every builder under
``benchmarks.kernel.operations`` returns and what every consumer in
``benchmarks.kernel.engine`` accepts -- every family module constructs it,
``measurement`` times its closures, ``correctness`` calls its validity hook,
and ``run`` assembles the result from both. Eight modules across two packages
touch it and nothing else about either.

Its own module because of the direction of that traffic. The builders are
resolved by dotted path inside the worker, so an operations module imports
the engine, never the other way round; leaving ``BuiltArm`` in ``run.py``
would make every family module drag the orchestrator, the statistics, the
result schema and ``benchmarks.artifacts.summaries`` in behind a twenty-line
dataclass, and would leave ``measurement`` and ``correctness`` importing the
module that imports them. Here, the graph is a fan-in with no cycle: this
module imports nothing first-party at all.

``import torch`` is real rather than decorative -- the closure signatures
name ``torch.Tensor`` -- which is also why this module sits on the worker
side of ``tests/test_import_boundaries.py``'s split despite defining no
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from benchmarks.kernel.schema import Mode


@dataclass
class BuiltArm:
    """A constructed arm: per-mode timed closures plus its validity hook.

    Each ``calls`` closure performs exactly one timed operation over
    prebuilt tensors; ``correctness_outputs`` runs the arm once on the
    shared seeded inputs and returns the named tensors its declared
    correctness checks compare.

    It says nothing about *which* modes the arm was supposed to expose, or
    whether the arm is a bandwidth floor. ``KernelArm`` declares both, and
    ``engine.run`` raises when ``calls`` disagrees with the declaration --
    so this type carries what only the build can know, and the registry
    carries what a reader must be able to check without a GPU.
    """

    name: str
    calls: dict[Mode, Callable[[], object]]
    correctness_outputs: Callable[[], dict[str, torch.Tensor]]
    bytes_moved: int | None = None
    notes: dict[str, Any] = field(default_factory=dict)
