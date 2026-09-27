"""The TorchTitan engine: our fork's ``torchtitan.train`` with the plug-ins in ``plugins``."""

from __future__ import annotations

from pathlib import Path

from benchmarks.e2e.engines.api import Arm, CompileMode, Engine, Launch, RunSpec
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig
from benchmarks.e2e.engines.torchtitan.flags import (
    TRAIN_MODULE,
    passthrough_refusals,
    trainer_args,
)
from benchmarks.e2e.engines.torchtitan.validate import validate_outputs
from benchmarks.e2e.parallelism import PP_SCHEDULES


class TorchTitanEngine(Engine):
    """The TorchTitan engine."""

    name = "torchtitan"
    config_type = TorchTitanConfig

    def check(self, run: RunSpec, arm: Arm) -> list[str]:
        refusals = []
        schedule = PP_SCHEDULES.get(run.parallelism.pp_schedule or "")
        if (
            schedule is not None
            and schedule.requires_uncompiled
            and arm.config.compile is CompileMode.TORCH
        ):
            refusals.append(
                f"pipeline schedule {schedule.name!r} raises on a compiled "
                f"stage module, and {arm.name} asks for torch.compile; select "
                "the eager arms alone, or choose another --pp-schedule"
            )
        refusals.extend(passthrough_refusals(arm.name, arm.config.extra_flags))
        return refusals

    def launch(self, run: RunSpec, arm: Arm, arm_dir: Path) -> Launch:
        return Launch(
            target=("-m", TRAIN_MODULE, *trainer_args(run, arm.config, arm_dir)),
            processes="per_rank",
            pin=True,
        )

    def validate(
        self, run: RunSpec, arm: Arm, arm_dir: Path, log_path: Path
    ) -> None:
        validate_outputs(run, arm, arm_dir, log_path)
