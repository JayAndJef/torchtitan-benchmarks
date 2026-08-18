"""How a built arm is timed on the device, and nothing about what it is.

Three passes over ``BuiltArm.calls``, each answering a different question and
each with its own reason to change:

* ``burst_samples`` is the primary measurement. It times **one** arm in one
  mode: synchronize, record a start event, launch ``k`` calls back to back,
  record an end event, synchronize, divide by ``k``. Each sample is therefore
  a burst-amortized per-call time, and the pass returns ``n`` of them.
* ``memory_pass`` is the secondary metric. Peak allocation is a property of
  one arm's call in isolation, so it runs the arm alone with the peak counter
  reset per iteration.
* ``burst_ladder`` is the ``--burst`` diagnostic. Where ``burst_samples``
  fixes one ``k``, this sweeps 1/4/16/64 in one mode, so a reader can see how
  much dispatch cost the chosen ``k`` amortized away.

**Why the measurand is burst-amortized device time.** The previous primary
pass timed one call per arm per cycle and reported the wall median, which on
this host-bound workload is mostly host dispatch: the rope arms are 90%+
dispatch, with device work of ~11-13 us inside 131-286 us walls. Dispatch
cost is real end-to-end, but in an *isolated* benchmark it is the harness,
not the kernel, and it swamped the quantity these scenarios exist to compare.
A burst of ``k`` back-to-back calls is also what a component looks like
inside a steady-state training step, where the same kernel runs on every
layer of every iteration.

**Why one ``k`` for every arm in a scenario.** A ``k`` chosen per arm makes
arms incomparable -- a k=64 arm overlaps 64 launches with device work and a
k=4 arm overlaps 4 -- and the residual bias runs in the same direction as the
effect under test. ``k`` is scenario-wide, and the schema records it.

**Known bias, stated rather than equalized.** The previous pass interleaved
arms and claimed cache state was equalized across them. Timing one arm at a
time makes that claim false, so it is withdrawn rather than carried: an arm
whose working set fits in L2 benefits from bursting more than one whose does
not. There is still no L2 flush.

Split from ``run.py`` because these are the methodology, and the methodology
is what a reviewer questions: the GC pause, the discarded first burst and the
absence of an L2 flush are the properties that make the numbers mean
anything, and they are hard to see inside an orchestrator.
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


def burst_samples(
    arm: BuiltArm, mode: str, k: int, n: int, warmup_calls: int
) -> list[float]:
    """``n`` burst-amortized per-call microsecond samples for one arm+mode.

    Returns an empty list when the arm does not declare ``mode``.
    """
    if mode not in arm.calls:
        return []
    if k < 1:
        raise ValueError(f"burst k must be >= 1, got {k}")
    call = arm.calls[mode]
    for _ in range(warmup_calls):
        call()
    torch.cuda.synchronize()
    # One extra burst absorbs the post-sync cold start: the queue is empty
    # after the warmup synchronize, so the first burst cannot overlap host
    # dispatch with device work and reads systematically high.
    samples: list[float] = []
    with _gc_paused():
        for _ in range(n + 1):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(k):
                call()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) * 1e3 / k)
    return samples[1:]


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


def burst_ladder(
    arm: BuiltArm, mode: str, bursts: tuple[int, ...], iters: int
) -> dict[str, float]:
    """Median us per call at increasing back-to-back burst sizes.

    Shows how much dispatch cost the primary pass's ``k`` amortized away: if
    per-call time is still falling at the top of the ladder, ``k`` is too
    small for this arm.

    ``mode`` is a parameter rather than a hardcoded ``"forward"``. The old
    version read ``arm.calls["forward"]`` directly, which made the diagnostic
    unavailable to ``lm_head`` -- its arms declare ``forward_backward`` only,
    because ``FusedLinearCrossEntropyLoss`` runs its backward inside
    ``__call__``.
    """
    if mode not in arm.calls:
        return {}
    call = arm.calls[mode]
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
