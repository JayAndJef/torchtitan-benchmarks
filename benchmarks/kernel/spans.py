"""Which kernel spans exist, and what each one replaces.

A **span** is an implementation that fuses across a scenario cut. It belongs
to no single scenario, so it is declared over an ordered scenario range and
its claim is the span against the **sum of the scenarios it replaces**. The
declaration types live in ``benchmarks.kernel.schema``; this module is
nothing but instances of them, exactly as ``benchmarks.kernel.registry`` is
for scenarios.

**Why spans live here and not in ``registry.py``.** A span names arms inside
other scenarios, so this module has to import the scenario registry to check
that each named arm exists. The dependency runs one way -- a span knows its
scenarios, a scenario knows nothing of spans -- and a separate module is what
makes that visible. Putting the cross-check inline in the scenario registry
would bury module-level validation in the middle of a data file and would
invite a scenario to reference a span back.

**A span is not a scenario, and the two rosters are disjoint.**
``KernelSpan`` composes a ``KernelScenario`` (its own head-to-head) rather
than subclassing one, so a span cannot be handed to anything that expects a
scenario without someone writing ``.measurement``. A span that reached
``KERNEL_SCENARIOS`` would run as a bare scenario and publish one of its two
totals under a name that promises both.

**Every span row carries one systematic bias, and it favours the span.** The
parts total pays one host dispatch chain per enclosed scenario; the span pays
one. CLAUDE.md records that roughly 85% of a kernel number in this repository
is host dispatch rather than device time, so a parts total over N scenarios
holds N-1 extra chains that no fusion removed -- the harness stopped paying
them because it timed one closure instead of N. The published ratio is
therefore smaller than fusion alone would make it, and the effect grows with
the length of the range.

The bias is a property of the **range length**, not of what a span fuses, so
the engine states it and no declaration has to remember to. It is printed
under every span table (``benchmarks.kernel.results.reporting``) and recorded
in every span results file (``KERNEL_SPAN_METHODOLOGY``). Nothing corrects
for it: separating the two would need profiler-summed device time, and
nothing in this repository measures that. A span's own ``description`` should
still say what the fusion is, so a reader knows what the remainder of the
ratio is supposed to be.

Torch-free and parent-side, like the schema and the scenario registry: arm
builders are dotted strings resolved inside the GPU worker, so declaring a
span costs no import.

Never present these numbers as end-to-end results, and never present a span
total as a scenario total: a span answers "what does fusing across these cuts
buy", which no single scenario asks.
"""

from __future__ import annotations

from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.schema import KernelSpan, validate_span_parts


# No span is declared at this rev. The mechanism landed first, deliberately:
# a span declaration needs a runner that can launch one and a merge that can
# hold two totals, and neither existed. Append a ``KernelSpan`` here to
# declare one.
KERNEL_SPANS: dict[str, KernelSpan] = {
    span.name: span for span in ()
}


for _span in KERNEL_SPANS.values():
    # At import, so a part arm that does not exist fails when this module
    # loads rather than as an absent row after a GPU has measured every arm
    # of the span and of every scenario it encloses.
    validate_span_parts(_span, KERNEL_SCENARIOS)


def kernel_span_by_name(name: str) -> KernelSpan:
    try:
        return KERNEL_SPANS[name]
    except KeyError:
        raise ValueError(
            f"Unknown kernel span {name!r}. "
            f"Available: {', '.join(KERNEL_SPANS) or '(none declared)'}"
        ) from None
