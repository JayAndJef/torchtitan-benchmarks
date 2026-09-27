"""The TorchTitan arguments that make a profiled run write its trace windows."""

from __future__ import annotations

from benchmarks.e2e.engines.api import ProfileWindow


def profiler_args(window: ProfileWindow) -> tuple[str, ...]:
    """The ``--profiler.*`` arguments of ``window``; TorchTitan writes the traces under ``--dump-folder``."""
    return (
        "--profiler.enable_profiling",
        "--profiler.profile_freq",
        str(window.freq),
        "--profiler.profiler_active",
        str(window.active),
        "--profiler.profiler_warmup",
        str(window.warmup),
    )
