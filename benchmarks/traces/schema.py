"""Declarative schema for the compiled regions read out of profiler traces.

``Region`` is the contract between a scenario (which declares the regions it
expects) and the extraction code (which finds them in a trace). It is
deliberately free of first-party imports so that both the end-to-end and the
kernel packages can depend on it without depending on each other.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    """One reported compiled region, identified by direction and call count.

    Graph hashes differ between arms, so a region is matched structurally:
    ``phase`` ("forward" or "backward") comes from whether the graph's CPU
    annotations nest inside ``CompiledFunctionBackward`` autograd frames, and
    ``invocations_per_window`` picks the graph among same-phase partitions
    while pinning the expected sample count. A count mismatch means the
    compiler partitioned the model differently and the graph-to-region
    mapping is no longer valid.
    """

    name: str
    phase: str
    invocations_per_window: int
