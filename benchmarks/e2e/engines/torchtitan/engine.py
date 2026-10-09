"""The TorchTitan engine: our fork's ``torchtitan.train`` with the plug-ins in ``plugins``."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from benchmarks.e2e.engines.api import (
    Arm,
    CompileMode,
    Engine,
    Launch,
    RankEvidence,
    RunSpec,
    StepRead,
)
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig
from benchmarks.e2e.engines.torchtitan.flags import (
    TRAIN_MODULE,
    flag_value,
    passthrough_refusals,
    trainer_args,
)
from benchmarks.e2e.engines.torchtitan.mesh import SCHEDULES, execution_model
from benchmarks.e2e.engines.torchtitan.evidence import read_evidence
from benchmarks.e2e.engines.torchtitan.steps import read_steps
from benchmarks.e2e.engines.torchtitan.validate import validate_outputs
from benchmarks.execution.paths import TITAN_DIR


ZERO2_AT_PP1 = (
    "--zero 1 was requested at pp 1. One microbatch puts the "
    "gradient reduce-scatter inside the only backward pass, so the "
    "TorchTitan arms (torchtitan) hold ZeRO-2 "
    "rather than the ZeRO-1 shape the level names. Megatron holds "
    "ZeRO-1 at every mesh. Do not read the two engines of this cell "
    "as one ZeRO level"
)
"""The warning of a TorchTitan arm at ZeRO level 1 and one pipeline rank."""

PACKED_ATTENTION_PACKAGE = "benchmarks.models.piper_qwen3.components.attention."
"""The prefix of each override import whose attention reads the loader's offsets."""

MOE_PACKAGE = "benchmarks.models.piper_qwen3.components.moe."

HOST_COUNT_DISPATCHER_MODULE = f"{MOE_PACKAGE}host_count_dispatcher."
"""The prefix of the override import whose dispatcher returns the rows of each local expert on the host."""

PER_EXPERT_MODULE = f"{MOE_PACKAGE}te_per_expert_experts."
"""The prefix of the override import whose experts read the rows of each expert from the host."""

NO_SPMD_TYPES_MODULES = (
    HOST_COUNT_DISPATCHER_MODULE,
    PER_EXPERT_MODULE,
)
"""The prefixes of the override imports whose custom ops have no SPMD type rule."""

SPMD_BACKEND_FLAG = "--parallelism.spmd-backend"
"""The TorchTitan flag that selects the SPMD backend."""


def _host_count_refusals(run: RunSpec, arm: Arm) -> list[str]:
    """Why the host count dispatcher and the per-expert experts of ``arm`` cannot run: each needs the other, and ep > 1."""
    imports = arm.config.override_imports
    dispatchers = [t for t in imports if t.startswith(HOST_COUNT_DISPATCHER_MODULE)]
    experts = [t for t in imports if t.startswith(PER_EXPERT_MODULE)]
    refusals = []
    if experts and not dispatchers:
        refusals.append(
            f"{arm.name}: the override import {experts[0]} reads the rows of "
            f"each expert from the host, and no override import starts with "
            f"{HOST_COUNT_DISPATCHER_MODULE}; add the host count dispatcher"
        )
    if dispatchers and not experts:
        refusals.append(
            f"{arm.name}: the override import {dispatchers[0]} gives the rows "
            "of each expert on the host, and no override import starts with "
            f"{PER_EXPERT_MODULE}, so the stock experts get host counts; add "
            "the per-expert experts"
        )
    if (dispatchers or experts) and run.parallelism.ep == 1:
        refusals.append(
            f"{arm.name}: the override import {(dispatchers or experts)[0]} "
            "needs an expert-parallel mesh, and the run asks for ep 1; pass "
            "--ep 2 or more, or leave the arm out"
        )
    return refusals


class TorchTitanEngine(Engine):
    """The TorchTitan engine."""

    name = "torchtitan"
    config_type = TorchTitanConfig
    can_profile = True

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
        packed = [
            target
            for target in arm.config.override_imports
            if target.startswith(PACKED_ATTENTION_PACKAGE)
        ]
        if arm.config.packed_offsets and not packed:
            refusals.append(
                f"{arm.name}: packed_offsets is on, and no override import "
                f"starts with {PACKED_ATTENTION_PACKAGE}, so no attention reads "
                "the offsets; turn packed_offsets off or add the override"
            )
        for target in packed if not arm.config.packed_offsets else ():
            refusals.append(
                f"{arm.name}: the override import {target} reads the loader's "
                "offsets, and packed_offsets is off; turn packed_offsets on"
            )
        refusals.extend(_host_count_refusals(run, arm))
        untyped = [
            target
            for target in arm.config.override_imports
            if target.startswith(NO_SPMD_TYPES_MODULES)
        ]
        backend = flag_value(arm.config.extra_flags, SPMD_BACKEND_FLAG)
        if untyped and backend == "spmd_types":
            refusals.append(
                f"{arm.name}: the override import {untyped[0]} has no SPMD type "
                f"rule, and {arm.name}.extra_flags select {SPMD_BACKEND_FLAG} "
                "spmd_types, so each block raises; choose another backend"
            )
        refusals.extend(passthrough_refusals(arm.name, arm.config.extra_flags))
        return refusals

    def launch(self, run: RunSpec, arm: Arm, arm_dir: Path) -> Launch:
        return Launch(
            target=("-m", TRAIN_MODULE, *trainer_args(run, arm.config, arm_dir)),
            processes="per_rank",
            pin=True,
            host_compiler=arm.config.requires_gcc_toolset,
            # The trainer reads its tokenizer from a path relative to the fork checkout.
            cwd=TITAN_DIR,
        )

    def execution_model(self, run: RunSpec, arm: Arm) -> str:
        return execution_model(run.parallelism)

    def warnings(self, run: RunSpec, arm: Arm) -> list[str]:
        spec = run.parallelism
        return [ZERO2_AT_PP1] if spec.zero == 1 and spec.pp == 1 else []

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
