"""The run request: what the operator asked for, before the runner resolves it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmarks.e2e.overrides import Override
from benchmarks.e2e.parallelism import ParallelismSpec


@dataclass(frozen=True)
class RequestedAxes:
    """The run-wide values that the operator asked for; ``None`` means not asked."""

    ac_mode: str | None = None
    model_size: str | None = None
    parallelism: ParallelismSpec | None = None
    profile: bool | None = None
    warmup_steps: int | None = None


@dataclass(frozen=True)
class RunRequest:
    """The inputs of one run, as the command line gives them."""

    gpu: str
    """The ``<gpu>`` positional as the operator typed it."""
    scenario_name: str | None = None
    """The scenario; only a resume may leave it out, because the manifest names it."""
    arm_names: tuple[str, ...] = ()
    """The ordered arm subset; empty selects every arm."""
    hardware: str = "auto"
    out_dir: Path | None = None
    resume_dir: Path | None = None
    seq_len: int | None = None
    steps: int | None = None
    batch: int | None = None
    overrides: tuple[Override, ...] = ()
    """The ``--set`` overrides, in the order given."""
    timestamp: str | None = None
    occurrence: int = 1
    """Which run of this scenario name this is, from 1; a later run gets its own directory."""
    cache_root: Path | None = None
    compiler_env: Path | None = None
    axes: RequestedAxes = RequestedAxes()
