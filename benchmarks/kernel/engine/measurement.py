"""How a built arm is timed on the device, and nothing about what it is.

Three passes over ``BuiltArm.calls``, each answering a different question and
each with its own reason to change:

* ``interleaved_samples`` is the primary measurement -- one cycle runs every
  arm that supports the mode once, between adjacent entries of a preallocated
  CUDA event matrix, with a single synchronize after all cycles. Drift hits
  every arm equally, so the per-cycle deltas are *paired* and the
  Welch/MWU/Wilcoxon/Cohen's d numbers computed from them are inferential for
  that run, unlike the e2e span diagnostics. Interleaved and isolated timings
  of the same kernels agree within 0.7%.
* ``memory_pass`` is the secondary metric, and is deliberately *not*
  interleaved: peak allocation is a property of one arm's call in isolation,
  so it runs each arm alone with the peak counter reset per iteration.
* ``burst_pass`` is the ``--burst`` diagnostic, which separates kernel cost
  from dispatch cost. It exists because the event-timed medians above are
  wall time: on this host-bound workload the rope arms are 90%+ dispatch
  (device work ~11-13 us inside 131-286 us walls), so a kernel-speed claim
  needs this or profiler-summed device time, never the wall median.

Split from ``run.py`` because these are the methodology, and the methodology
is what a reviewer questions: the GC pause, the discarded first cycle, the
absence of an L2 flush and the event-matrix layout are the properties that
make the numbers mean anything, and they are hard to see inside a
150-line orchestrator. ``memory_pass`` stays here rather than in a module of
its own -- it is ten lines, has one caller, and is the same kind of thing as
``burst_pass``: a device pass over a ``BuiltArm`` closure.
"""

from __future__ import annotations

import contextlib
import gc
from statistics import median

import torch

from benchmarks.kernel.engine.arm import BuiltArm


@contextlib.contextmanager
def _gc_paused():
    """Keep collection cycles out of the timed region.

    The measured arms only stay ahead of the GPU if the host keeps enqueuing;
    a collection pause starves the stream and lands as idle time inside
    whichever arm's interval was open. Measured on the swiglu modules: sd
    fell from ~63 us to ~1.4 us and every 2x outlier disappeared, with the
    median unchanged. ``timeit`` disables the collector for the same reason.
    """
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def interleaved_samples(
    arms: list[BuiltArm], mode: str, n: int, warmup: int
) -> dict[str, list[float]]:
    """Time all arms that support ``mode``, one cycle at a time."""
    active = [arm for arm in arms if mode in arm.calls]
    if not active:
        return {}
    for _ in range(warmup):
        for arm in active:
            arm.calls[mode]()
    torch.cuda.synchronize()
    # One extra cycle absorbs the post-sync cold start: the queue is empty
    # after the warmup synchronize, so the first cycle cannot overlap host
    # dispatch with device work and reads systematically high.
    events = [
        [torch.cuda.Event(enable_timing=True) for _ in range(len(active) + 1)]
        for _ in range(n + 1)
    ]
    with _gc_paused():
        for row in events:
            row[0].record()
            for slot, arm in enumerate(active):
                arm.calls[mode]()
                row[slot + 1].record()
        torch.cuda.synchronize()
    return {
        arm.name: [
            row[slot].elapsed_time(row[slot + 1]) * 1e3 for row in events[1:]
        ]
        for slot, arm in enumerate(active)
    }


def memory_pass(arm: BuiltArm, mode: str, iters: int) -> float:
    """Peak allocated GiB across ``iters`` isolated calls of ``mode``."""
    peak = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        arm.calls[mode]()
        torch.cuda.synchronize()
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
    return peak


def burst_pass(
    arm: BuiltArm, bursts: tuple[int, ...], iters: int
) -> dict[str, float]:
    """Median us per call at increasing back-to-back burst sizes.

    Distinguishes kernel cost from dispatch cost: if per-call time collapses
    as the burst grows, single-call timing was host-dispatch-bound.
    """
    call = arm.calls["forward"]
    result = {}
    with _gc_paused():
        for burst in bursts:
            for _ in range(20):
                call()
            torch.cuda.synchronize()
            samples = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(burst):
                    call()
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end) * 1e3 / burst)
            result[str(burst)] = median(samples)
    return result
