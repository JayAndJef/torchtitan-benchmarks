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
from benchmarks.e2e.engines.torchtitan.mesh import SCHEDULES, execution_model
from benchmarks.e2e.engines.torchtitan.validate import validate_outputs


ZERO2_AT_PP1 = (
    "--zero 1 was requested at pp 1. One microbatch puts the "
    "gradient reduce-scatter inside the only backward pass, so the "
    "TorchTitan arms (torchtitan) hold ZeRO-2 "
    "rather than the ZeRO-1 shape the level names. Megatron holds "
    "ZeRO-1 at every mesh. Do not read the two engines of this cell "
    "as one ZeRO level"
)
"""The warning of a TorchTitan arm at ZeRO level 1 and one pipeline rank."""


class TorchTitanEngine(Engine):
    """The TorchTitan engine."""

    name = "torchtitan"
    config_type = TorchTitanConfig

    def check(self, run: RunSpec, arm: Arm) -> list[str]:
        refusals = []
        name = run.parallelism.pp_schedule
        schedule = SCHEDULES.get(name) if name is not None else None
        if name is not None and schedule is None:
            refusals.append(
                f"{arm.name}: TorchTitan runs no pipeline schedule {name!r}; "
                f"choose one of {', '.join(SCHEDULES)}"
            )
        if (
            schedule is not None
            and schedule.requires_uncompiled
            and arm.config.compile is CompileMode.TORCH
        ):
            refusals.append(
                f"pipeline schedule {name!r} raises on a compiled "
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
            host_compiler=arm.config.requires_gcc_toolset,
        )

    def execution_model(self, run: RunSpec, arm: Arm) -> str:
        return execution_model(run.parallelism)

    def warnings(self, run: RunSpec, arm: Arm) -> list[str]:
        spec = run.parallelism
        return [ZERO2_AT_PP1] if spec.zero == 1 and spec.pp == 1 else []

    def validate(
        self, run: RunSpec, arm: Arm, arm_dir: Path, log_path: Path
    ) -> None:
        validate_outputs(run, arm, arm_dir, log_path)
