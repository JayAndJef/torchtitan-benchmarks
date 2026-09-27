"""The stock Megatron-LM engine, around its launch builder and ``MEGATRON_STOCK_PROFILE``."""

from __future__ import annotations

from pathlib import Path

from benchmarks.e2e.engines.api import Arm, Engine, Launch, RunSpec
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.engines.megatron_stock.flags import refuse_megatron_passthrough
from benchmarks.e2e.engines.megatron_stock.validate import (
    MEGATRON_STOCK_PROFILE,
    _megatron_stock_nan_guard_markers,
    _megatron_stock_p2p_markers,
    _megatron_stock_parallelism_markers,
    _megatron_stock_precision_markers,
)
from benchmarks.e2e.parallelism import PP_SCHEDULES
from benchmarks.e2e.validation import validate_against_profile


STOCK_MEGATRON_DRIVER_MODULE = "benchmarks.e2e.engines.megatron_stock.driver.train"
"""What ``python -m`` starts for the stock Megatron arm.

Named once, so a test and the command builder cannot drift apart.
"""

STOCK_MEGATRON_PP_SCHEDULE = "1F1B"
"""The one pipeline schedule the stock driver implements.

Megatron-LM implements more, but this driver builds no model-chunk list, so
it runs ``forward_backward_pipelining_without_interleaving`` alone.
"""


def megatron_stock_launch(run: RunSpec, arm: Arm, arm_dir: Path) -> Launch:
    """The launch of one stock Megatron-LM arm.

    ``benchmarks/e2e/engines/megatron_stock/flags.py`` builds every flag, and this
    function adds the driver module and the passthrough tokens. The
    refusals repeat the run checks, so a caller that builds a launch
    without a run gets a message that names the arm.
    """
    config = arm.config
    spec = run.parallelism
    if run.ac_mode != "none":
        raise ValueError(
            f"{arm.name}: the stock megatron arm runs without recompute; "
            f"ac mode {run.ac_mode!r} has no Megatron parity (use --ac none)"
        )
    if run.seed is None:
        raise ValueError(
            f"{arm.name}: megatron arms require a seeded workload"
        )
    if spec.pp > 1 and spec.pp_schedule != STOCK_MEGATRON_PP_SCHEDULE:
        raise ValueError(
            f"{arm.name}: the stock megatron driver implements "
            f"{STOCK_MEGATRON_PP_SCHEDULE!r} alone, and this run asks for "
            f"{spec.pp_schedule!r}"
        )
    # Below the refusals, so a refused request fails with its own message.
    from benchmarks.e2e.engines.megatron_stock.flags import stock_megatron_flags

    refuse_megatron_passthrough(arm.name, config.extra_flags, spec.zero)
    return Launch(
        target=(
            "-m",
            STOCK_MEGATRON_DRIVER_MODULE,
            *stock_megatron_flags(run, config, arm_dir=str(arm_dir)),
            # Last, because Megatron's parser is last-wins.
            *config.extra_flags,
        ),
        processes="per_rank",
        pin=True,
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

    def launch(self, run: RunSpec, arm: Arm, arm_dir: Path) -> Launch:
        return megatron_stock_launch(run, arm, arm_dir)

    def validate(
        self, run: RunSpec, arm: Arm, arm_dir: Path, log_path: Path
    ) -> None:
        config = arm.config
        spec = run.parallelism
        required_lines = {}
        if spec.world_size > 1:
            required_lines["parallelism"] = _megatron_stock_parallelism_markers(
                spec, run.data, config.precision, config.extra_flags
            ) + _megatron_stock_p2p_markers(spec, config.p2p_sync)
        required_lines["megatron nan guard"] = _megatron_stock_nan_guard_markers(
            config.nan_guard
        )
        required_lines["megatron precision"] = _megatron_stock_precision_markers(
            config.precision
        )
        validate_against_profile(
            run,
            arm.name,
            arm_dir,
            log_path,
            profile=MEGATRON_STOCK_PROFILE,
            required_lines=required_lines,
            trace_kernel_markers=config.trace_kernel_markers,
        )
