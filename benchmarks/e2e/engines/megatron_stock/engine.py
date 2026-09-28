"""The stock Megatron-LM engine: ``megatron.training.pretrain`` through the driver in ``driver``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from benchmarks.e2e.engines.api import (
    Arm,
    Engine,
    Launch,
    RankEvidence,
    RunSpec,
    StepRead,
)
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.engines.megatron_stock.flags import (
    DRIVER_MODULE,
    MEGATRON_LM_SCHEDULES,
    MEGATRON_NAN_GUARD_MODES,
    MEGATRON_P2P_SYNC_MODES,
    MEGATRON_PRECISION_MODES,
    PP_SCHEDULE,
    PRECISION_STATES,
    passthrough_refusals,
    stock_megatron_flags,
)
from benchmarks.e2e.engines.megatron_stock.profiling import partial_cycle_refusal
from benchmarks.e2e.engines.megatron_stock.evidence import read_evidence
from benchmarks.e2e.engines.megatron_stock.steps import read_steps
from benchmarks.e2e.engines.megatron_stock.validate import validate_outputs
from benchmarks.e2e.parallelism import (
    data_parallel_term,
    degree_terms,
    device_term,
)
from benchmarks.execution.paths import TITAN_DIR


class MegatronStockEngine(Engine):
    """The stock Megatron-LM engine."""

    name = "megatron_stock"
    config_type = MegatronStockConfig
    can_profile = True

    def check(self, run: RunSpec, arm: Arm) -> list[str]:
        config = arm.config
        spec = run.parallelism
        refusals = []
        for label, value, choices in (
            ("megatron p2p sync", config.p2p_sync, MEGATRON_P2P_SYNC_MODES),
            ("megatron nan guard", config.nan_guard, MEGATRON_NAN_GUARD_MODES),
            ("megatron precision", config.precision, MEGATRON_PRECISION_MODES),
        ):
            if value not in choices:
                refusals.append(
                    f"{arm.name}: {label} {value!r} is not one of "
                    + ", ".join(repr(choice) for choice in choices)
                )
        schedule = spec.pp_schedule
        if schedule is not None and schedule not in MEGATRON_LM_SCHEDULES:
            refusals.append(
                f"pipeline schedule {schedule!r} is not implemented by "
                "Megatron-LM, and this run holds a megatron arm; there would be "
                "no cross-engine comparison; choose --pp-schedule "
                f"{PP_SCHEDULE}, or select the TorchTitan arms alone"
            )
        elif schedule is not None and schedule != PP_SCHEDULE:
            refusals.append(
                f"{arm.name}: the stock megatron driver implements "
                f"{PP_SCHEDULE!r} alone, and this run asks for "
                f"{schedule!r}; choose --pp-schedule "
                f"{PP_SCHEDULE}, or select the TorchTitan arms alone"
            )
        if config.p2p_sync == "on" and spec.pp == 1:
            refusals.append(
                f"{arm.name}: megatron p2p sync 'on' was requested at pp 1, "
                "where there is no pipeline message to synchronize; add a "
                "pipeline rank, or set the sync to 'off'"
            )
        if config.precision == "lean" and spec.zero == 0:
            refusals.append(
                f"{arm.name}: megatron precision 'lean' needs --zero 1, "
                "because Megatron asserts use_distributed_optimizer under "
                "--use-precision-aware-optimizer"
            )
        if run.ac_mode != "none":
            refusals.append(
                f"{arm.name}: the stock megatron arm runs without recompute; "
                f"ac mode {run.ac_mode!r} has no Megatron parity (use --ac none)"
            )
        if run.seed is None:
            refusals.append(
                f"{arm.name}: the stock megatron arm needs a seeded workload, "
                "because Megatron takes its --seed from the run"
            )
        cycle = partial_cycle_refusal(arm.name, run)
        if cycle is not None:
            refusals.append(cycle)
        refusals.extend(
            passthrough_refusals(arm.name, config.extra_flags, spec.zero)
        )
        return refusals

    def launch(self, run: RunSpec, arm: Arm, arm_dir: Path) -> Launch:
        return Launch(
            target=(
                "-m",
                DRIVER_MODULE,
                *stock_megatron_flags(run, arm.config, arm_dir=str(arm_dir)),
                # Last, because Megatron keeps the last value of a repeated flag.
                *arm.config.extra_flags,
            ),
            processes="per_rank",
            pin=True,
            # The same working directory as the TorchTitan arms, so both engines start alike.
            cwd=TITAN_DIR,
        )

    def execution_model(self, run: RunSpec, arm: Arm) -> str:
        spec = run.parallelism
        return "-".join(
            (
                device_term(spec),
                PRECISION_STATES[arm.config.precision],
                data_parallel_term(spec),
                *degree_terms(spec),
            )
        )

    def read_steps(self, rank: int, text: str) -> StepRead:
        return read_steps(rank, text)

    def read_evidence(self, rank: int, text: str) -> RankEvidence:
        return read_evidence(rank, text)

    def validate(
        self,
        run: RunSpec,
        arm: Arm,
        arm_dir: Path,
        rank_logs: Mapping[int, str],
    ) -> list[str]:
        return validate_outputs(run, arm, arm_dir, rank_logs)
