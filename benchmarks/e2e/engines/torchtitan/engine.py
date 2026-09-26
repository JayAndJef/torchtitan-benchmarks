"""The TorchTitan engine, around the ``run_train.sh`` builder and ``TORCHTITAN_PROFILE``."""

from __future__ import annotations

from pathlib import Path

from benchmarks.e2e.engines.api import Arm, CompileMode, Engine, RunSpec
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig
from benchmarks.e2e.launch import titan_command
from benchmarks.e2e.parallelism import PP_SCHEDULES
from benchmarks.e2e.validation import (
    TORCHTITAN_PROFILE,
    validate_against_profile,
)


class TorchTitanEngine(Engine):
    """Our TorchTitan fork's ``torchtitan.train``, with this repository's model port."""

    name = "torchtitan"
    config_type = TorchTitanConfig

    def check(self, run: RunSpec, arm: Arm) -> list[str]:
        schedule = PP_SCHEDULES.get(run.parallelism.pp_schedule or "")
        if (
            schedule is not None
            and schedule.requires_uncompiled
            and arm.config.compile is CompileMode.TORCH
        ):
            return [
                f"pipeline schedule {schedule.name!r} raises on a compiled "
                f"stage module, and {arm.name} asks for torch.compile; select "
                "the eager arms alone, or choose another --pp-schedule"
            ]
        return []

    def command(self, run: RunSpec, arm: Arm, arm_dir: Path) -> list[str]:
        return titan_command(run, arm, arm_dir)

    def validate(
        self, run: RunSpec, arm: Arm, arm_dir: Path, log_path: Path
    ) -> None:
        config = arm.config
        validate_against_profile(
            run,
            arm.name,
            arm_dir,
            log_path,
            engine_profile=TORCHTITAN_PROFILE,
            compile=config.compile,
            overrides_per_block=config.overrides_per_block,
            override_imports=config.override_imports,
            trace_kernel_markers=config.trace_kernel_markers,
            extra_flags=config.extra_flags,
        )
