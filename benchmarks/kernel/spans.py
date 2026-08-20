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
