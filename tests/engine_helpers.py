"""Test helpers that build a run and reach one arm's engine through the registry."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from benchmarks.artifacts.manifests import ArmRecord, write_manifest
from benchmarks.e2e.engines.api import Arm, DataSpec, ProfileWindow, RunSpec
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.e2e.parallelism import TRIVIAL_SPEC, ParallelismSpec
from benchmarks.e2e.registry import DEFAULT_WARMUP_STEPS, ENGINES, SEED
from benchmarks.e2e.validation import validate_arm
from benchmarks.execution.affinity import CpuPinning
from benchmarks.execution.launcher import (
    command_line,
    environment_delta,
    pinning_record,
)
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
    """Validate one arm as the runner does: the harness facts, then the arm's engine."""
    validate_arm(run, arm, engine_for(arm), Path(arm_dir), Path(log_path))


TEST_METADATA = {
    "requested_gpu": "0",
    "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
    "torch_version": "test",
    "torchtitan_git_rev": "titan-rev",
    "benchmarks_git_rev": "bench-rev",
    "megatron_git_rev": "megatron-rev",
    "cpu_pinning": UNPINNED.description,
}
"""A provenance block for a manifest that no host probe wrote."""


def titan_step_line(
    step: int,
    *,
    tps: int = 1000,
    loss: float = 1.0,
    grad_norm: float = 2.0,
    memory: float = 3.0,
) -> str:
    """One step line in the format of the TorchTitan fork, with its color codes."""
    return (
        f"[titan] 2026-09-26 10:00:00,000 - root - INFO - \x1b[31mstep: {step:2}  "
        f"\x1b[32mloss: {loss:8.5f}  "
        f"\x1b[38;2;180;60;0mgrad_norm: {grad_norm:7.4f}  "
        f"\x1b[38;2;54;234;195mmemory: {memory:5.2f}GiB(2.14%)  "
        f"\x1b[34mtps: {tps:,}  \x1b[36mtflops: 12.50  "
        "\x1b[35mmfu: 1.26%\x1b[39m\n"
    )


def write_run_manifest(
    out_dir: Path,
    run: RunSpec,
    arms: tuple[Arm, ...],
    *,
    pinning: CpuPinning = UNPINNED,
) -> None:
    """Write the schema 19 manifest that the runner writes for ``arms`` of the ``engines`` scenario."""
    world_size = run.parallelism.world_size
    records = []
    for arm in arms:
        launch = engine_for(arm).launch(run, arm, out_dir / arm.name)
        records.append(
            ArmRecord(
                arm=arm,
                command=command_line(launch, world_size=world_size, pinning=pinning),
                env_delta=environment_delta(launch, world_size=world_size, gpu="0"),
                cpu_pinning=pinning_record(launch, pinning),
                execution_model=engine_for(arm).execution_model(run, arm),
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        out_dir,
        scenario=ENGINES,
        hardware="test-gpu",
        metadata={**TEST_METADATA, "cpu_pinning": pinning.description},
        run=run,
        arms=tuple(records),
    )
