"""The Megatron arguments that make a profiled run write its trace windows."""

from __future__ import annotations

from benchmarks.e2e.engines.api import RunSpec


def profiler_args(steps: int) -> tuple[str, ...]:
    """Megatron's profiler flags for a run of ``steps`` steps; the driver's shim sets the schedule and the path."""
    return (
        "--profile",
        "--use-pytorch-profiler",
        "--profile-step-start",
        "1",
        "--profile-step-end",
        str(steps),
    )


def partial_cycle_refusal(arm_name: str, run: RunSpec) -> str | None:
    """Why a profiled run ends inside a profiler cycle, or ``None``."""
    steps = run.data.steps
    freq = run.window.freq
    if not run.profile or steps % freq == 0:
        return None
    return (
        f"{arm_name}: steps ({steps}) must be a whole number of profiler "
        f"cycles of {freq} for the stock megatron arm, because Megatron "
        "steps the profiler after it stops it; choose --steps as a multiple "
        f"of {freq}"
    )
