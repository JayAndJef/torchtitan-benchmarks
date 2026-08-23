"""Declarative schema for the compiled regions read out of profiler traces.

``Region`` is the contract between a scenario (which declares the regions it
expects) and the extraction code (which finds them in a trace). It is
deliberately free of first-party imports: it sits below every package that
names a region -- ``benchmarks.e2e.registry`` declares them,
``benchmarks.traces.extraction`` finds them, ``benchmarks.e2e.results`` and
``benchmarks.e2e.validation`` consume them, and
``benchmarks.artifacts.manifests`` serializes them -- so none of those has to
import another to spell the type. Nothing under ``benchmarks.kernel``
references regions at all; kernel-isolation runs write no traces.

The trace *file name* grammar lives here for the same reason. Both engines
write one trace per rank per profiler window, named ``rank<n>_trace.json.gz``.
``benchmarks.artifacts.layout`` globs that name and groups by the rank it
carries; ``benchmarks.traces.extraction`` reads the same token to refuse a
pooling call that mixes two ranks. Spelling it once keeps the two from
drifting apart, and neither has to import the other.
"""

import re
from dataclasses import dataclass
from pathlib import Path


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


TRACE_FILE_GLOB = "rank*_trace.json.gz"
_TRACE_FILE_NAME = re.compile(r"\Arank(\d+)_trace\.json\.gz\Z")


def rank_of_trace(path: Path) -> int | None:
    """The rank that wrote this trace, or ``None`` when the name does not say.

    ``None`` is a real answer, not an error: the synthetic traces the tests
    write carry whatever name the test chose, and a caller that only wants to
    know whether two paths came from *different* ranks must be able to say
    "this one does not claim a rank" without failing. A caller that needs the
    rank -- ``trace_files_by_rank`` -- raises on ``None`` itself, because
    there the name matched ``TRACE_FILE_GLOB`` and must therefore parse.
    """
    match = _TRACE_FILE_NAME.match(Path(path).name)
    return int(match.group(1)) if match else None
