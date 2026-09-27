"""Test helpers that build a run and reach one arm's engine through the registry."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from benchmarks.e2e.engines.api import Arm, DataSpec, ProfileWindow, RunSpec
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.e2e.parallelism import TRIVIAL_SPEC, ParallelismSpec
from benchmarks.e2e.registry import DEFAULT_WARMUP_STEPS, ENGINES, SEED
from benchmarks.execution.affinity import CpuPinning
from benchmarks.execution.launcher import command_line
from benchmarks.models.piper_qwen3.shape import shape_by_name


def run_spec(
    model_size: str = "1b",
    *,
    data: DataSpec = ENGINES.data,
    window: ProfileWindow = ENGINES.window,
    parallelism: ParallelismSpec = TRIVIAL_SPEC,
    ac_mode: str = "sac",
    profile: bool = True,
    warmup_steps: int | None = None,
    seed: int | None = SEED,
    **data_fields: int,
) -> RunSpec:
    """A run of the ``engines`` scenario; ``data_fields`` replace fields of ``data``."""
    if warmup_steps is None and not profile:
        warmup_steps = DEFAULT_WARMUP_STEPS
    return RunSpec(
        shape=shape_by_name(model_size),
        data=replace(data, **data_fields),
        parallelism=parallelism,
        ac_mode=ac_mode,
        profile=profile,
        window=window,
        warmup_steps=warmup_steps,
        seed=seed,
    )


def configured(arm: Arm, **fields: object) -> Arm:
    """``arm``, with ``fields`` replaced in its config."""
    return replace(arm, config=replace(arm.config, **fields))


UNPINNED = CpuPinning((), "none: test")
"""A host pinning with no prefix."""


def command(run: RunSpec, arm: Arm, arm_dir: Path | str) -> list[str]:
    """The unpinned command line of the arm's launch."""
    launch = engine_for(arm).launch(run, arm, Path(arm_dir))
    return list(
        command_line(
            launch, world_size=run.parallelism.world_size, pinning=UNPINNED
        )
    )


def validate(
    run: RunSpec, arm: Arm, arm_dir: Path | str, log_path: Path | str
) -> None:
    """Validate one arm through its engine."""
    engine_for(arm).validate(run, arm, Path(arm_dir), Path(log_path))
