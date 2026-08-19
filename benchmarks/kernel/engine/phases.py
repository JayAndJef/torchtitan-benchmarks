"""Wall-clock attribution for the phases of one kernel worker process.

A kernel scenario is one correctness worker plus ``replicates x arms`` timing
workers, spawned sequentially, and each worker is a fresh interpreter. The
timed measurement itself is milliseconds; almost all of a worker's wall clock
is the work that gets it to the point of measuring. Nothing recorded that
before this module, so "the worker takes about twenty seconds" was the whole
of what anyone could say about it.

This records a named span per phase and puts the list in the fragment the
worker writes. Two consequences worth stating:

* **It never touches the timed region.** ``burst_samples`` records the warmup
  and the sample loop as two separate phases, from outside its own event
  timing, and adds no synchronize the measurement did not already make. The
  published numbers come from CUDA events between the same two points they
  always did.
* **A phase boundary in the setup path does synchronize**, through the
  ``on_exit`` hook the caller passes. Kernel launches are asynchronous, so
  without it the device work an input builder queued would be charged to
  whichever phase happened to sync next -- usually the arm build. The setup
  path is not measured, so the cost of the extra synchronize is not a
  measurement error; the mis-attribution it removes would be.

Stdlib only, and deliberately so: the worker imports it before it imports
torch, because the interval from process exec to the first torch import is
one of the phases.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Callable, Iterator


_phases: list[tuple[str, float]] = []


def reset() -> None:
    _phases.clear()


def record(name: str, seconds: float) -> None:
    """Add a span this module did not time itself."""
    _phases.append((name, seconds))


@contextlib.contextmanager
def phase(name: str, on_exit: Callable[[], None] | None = None) -> Iterator[None]:
    """Time the block and record it under ``name``.

    ``on_exit`` runs inside the span, immediately before it closes. Setup-path
    callers pass ``torch.cuda.synchronize`` so that asynchronous device work a
    phase queued is charged to that phase. No timing caller passes one.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        if on_exit is not None:
            on_exit()
        _phases.append((name, time.perf_counter() - started))


def phases() -> list[dict[str, float | str]]:
    """The recorded spans, in the order they closed."""
    return [{"phase": name, "seconds": seconds} for name, seconds in _phases]


def process_start_offset() -> float | None:
    """Seconds from this process's exec to now, or None off Linux.

    Read from ``/proc`` rather than timed, because the interval being measured
    starts before any Python of ours runs: the interpreter's own startup and
    the stdlib imports at the worker's module scope are inside it.
    """
    try:
        with open("/proc/self/stat", "rb") as handle:
            fields = handle.read().rpartition(b")")[2].split()
        # Field 22 of proc(5), which is index 19 after the comm field: the
        # process start time in clock ticks since boot.
        started_ticks = int(fields[19])
        with open("/proc/uptime", "rb") as handle:
            uptime = float(handle.read().split()[0])
    except (OSError, IndexError, ValueError):
        return None
    return uptime - started_ticks / os.sysconf("SC_CLK_TCK")
