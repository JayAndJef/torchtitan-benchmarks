"""The stock Megatron-LM engine, around its command builder and ``MEGATRON_STOCK_PROFILE``."""

from __future__ import annotations

from pathlib import Path

from benchmarks.e2e.engines.api import Arm, CompileMode, Engine, RunSpec
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.launch import (
    STOCK_MEGATRON_PP_SCHEDULE,
    megatron_stock_command,
)
from benchmarks.e2e.parallelism import PP_SCHEDULES
from benchmarks.e2e.validation import (
    MEGATRON_STOCK_PROFILE,
    validate_against_profile,
)


class MegatronStockEngine(Engine):
    """Stock ``megatron.training.pretrain``, through this repository's driver."""

    name = "megatron_stock"
    config_type = MegatronStockConfig

    def check(self, run: RunSpec, arm: Arm) -> list[str]:
        schedule = PP_SCHEDULES.get(run.parallelism.pp_schedule or "")
        if schedule is None:
            return []
        if not schedule.megatron_supported:
            return [
                f"pipeline schedule {schedule.name!r} is not implemented by "
                "Megatron-LM, and this run holds a megatron arm; there would be "
                "no cross-engine comparison; choose --pp-schedule "
                f"{STOCK_MEGATRON_PP_SCHEDULE}, or select the TorchTitan arms alone"
            ]
        if schedule.name != STOCK_MEGATRON_PP_SCHEDULE:
            return [
                f"{arm.name}: the stock megatron driver implements "
                f"{STOCK_MEGATRON_PP_SCHEDULE!r} alone, and this run asks for "
                f"{schedule.name!r}; choose --pp-schedule "
                f"{STOCK_MEGATRON_PP_SCHEDULE}, or select the TorchTitan arms alone"
            ]
        return []

    def command(self, run: RunSpec, arm: Arm, arm_dir: Path) -> list[str]:
        return megatron_stock_command(run, arm, arm_dir)

    def validate(
        self, run: RunSpec, arm: Arm, arm_dir: Path, log_path: Path
    ) -> None:
        config = arm.config
        validate_against_profile(
            run,
            arm.name,
            arm_dir,
            log_path,
            engine_profile=MEGATRON_STOCK_PROFILE,
            compile=CompileMode.NONE,
            trace_kernel_markers=config.trace_kernel_markers,
            extra_flags=config.extra_flags,
            megatron_p2p_sync=config.p2p_sync,
            megatron_nan_guard=config.nan_guard,
            megatron_precision=config.precision,
        )
