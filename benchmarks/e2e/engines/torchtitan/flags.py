"""The ``torchtitan.train`` command line of one arm, and the flags a passthrough may carry."""

from __future__ import annotations

import re
from pathlib import Path

from benchmarks.e2e.engines.api import CompileMode, RunSpec
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig
from benchmarks.e2e.engines.torchtitan.profiling import profiler_args
from benchmarks.e2e.engines.torchtitan.mesh import (
    SCHEDULES,
    reshard_after_forward,
    titan_mesh,
)
from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.passthrough import matches, ownership


TRAIN_MODULE = "torchtitan.train"
"""The module that ``python -m`` starts for a TorchTitan arm."""


def parallelism_args(spec: ParallelismSpec) -> tuple[str, ...]:
    """The ``--parallelism.*`` arguments of ``spec``; the single-GPU spec has none."""
    args: list[str] = []
    if spec.pp > 1:
        schedule = SCHEDULES[spec.pp_schedule]
        args.extend(
            (
                "--parallelism.pipeline-parallel-degree",
                str(spec.pp),
                "--parallelism.pipeline-parallel-schedule",
                schedule.titan_name,
                "--parallelism.pipeline-parallel-microbatch-size",
                str(spec.pp_microbatch_size),
                # Zero, so the stages split the layers evenly, as Megatron does.
                "--parallelism.pipeline-parallel-first-stage-less-layers",
                "0",
                "--parallelism.pipeline-parallel-last-stage-less-layers",
                "0",
            )
        )
    replicate, shard = titan_mesh(spec)
    if (replicate, shard) != (1, 1):
        # TorchTitan reads an omitted shard degree as every remaining rank.
        args.extend(
            (
                "--parallelism.data-parallel-replicate-degree",
                str(replicate),
                "--parallelism.data-parallel-shard-degree",
                str(shard),
            )
        )
    # ZeRO level 1 keeps whole parameters through the step.
    policy = reshard_after_forward(spec)
    if policy is not None:
        args.extend(("--parallelism.fsdp-reshard-after-forward", policy))
    if spec.ep > 1:
        args.extend(("--parallelism.expert-parallel-degree", str(spec.ep)))
    return tuple(args)


def trainer_args(
    run: RunSpec, config: TorchTitanConfig, arm_dir: Path
) -> tuple[str, ...]:
    """The ``torchtitan.train`` arguments of one arm."""
    args = [
        "--module",
        config.module,
        "--config",
        config.config,
        "--config-arg",
        f"size={run.shape.name}",
        "--training.seq-len",
        str(run.data.seq_len),
        "--training.steps",
        str(run.data.steps),
        "--training.local-batch-size",
        str(run.data.local_batch_size),
    ]
    if config.compile is CompileMode.TORCH:
        args.append("--compile.enable")
    if run.profile:
        args.extend(profiler_args(run.window))
    args.extend(parallelism_args(run.parallelism))
    # The replay loader refuses a step past the samples it holds.
    args.extend(("--dataloader.replay-steps", str(run.data.steps)))
    if run.seed is not None:
        args.extend(("--debug.seed", str(run.seed)))
    if config.override_imports:
        args.extend(("--override.imports", ",".join(config.override_imports)))
    args.extend(config.extra_flags)
    args.extend(("--dump-folder", str(arm_dir)))
    if run.ac_mode == "none":
        # A tyro subcommand token, which comes last.
        args.append("activation-checkpoint:none")
    return tuple(args)


OWNED_FLAGS: dict[str, tuple[str, ...]] = {
    "--seq-len": ("--training.seq-len",),
    "--steps": ("--training.steps",),
    "--batch": ("--training.local-batch-size", "--training.global-batch-size"),
    "--pp-microbatch-size": ("--parallelism.pipeline-parallel-microbatch-size",),
    "--model-size": ("--config", "--config-arg", "--module"),
    "--dp/--pp/--ep": (
        "--parallelism.data-parallel-replicate-degree",
        "--parallelism.data-parallel-shard-degree",
        "--parallelism.tensor-parallel-degree",
        "--parallelism.pipeline-parallel-degree",
        "--parallelism.context-parallel-degree",
        "--parallelism.expert-parallel-degree",
        "--parallelism.module-fqns-per-model-part",
        "--parallelism.pipeline-parallel-first-stage-less-layers",
        "--parallelism.pipeline-parallel-last-stage-less-layers",
        "--parallelism.pipeline-parallel-layers-per-stage",
    ),
    "--pp-schedule": (
        "--parallelism.pipeline-parallel-schedule",
        "--parallelism.pipeline-parallel-schedule-csv",
    ),
    "--zero": ("--parallelism.fsdp-reshard-after-forward",),
    "--ac": ("--activation-checkpoint.*", "activation-checkpoint:*"),
    "--profile": ("--profiler.*",),
    "the arm's compile value": ("--compile.enable",),
    "the arm's override imports": ("--override.*",),
}
"""The TorchTitan flags that each harness option, or arm setting, sets."""

PINNED_FLAGS: dict[str, tuple[str, ...]] = {
    "the precision recipe": (
        "--training.dtype",
        "--training.mixed-precision-param",
        "--training.mixed-precision-reduce",
    ),
    "the optimizer matched across engines": (
        "--optimizer.*",
        "--lr-scheduler.*",
        "--training.max-norm",
    ),
    "the routing matched across engines": ("--debug.moe-force-load-balance",),
    "the shared data stream": (
        "--dataloader.*",
        "--tokenizer.*",
        "--hf-assets-path",
        "--debug.seed",
    ),
    "the step lines the evaluation reads": ("--metrics.*",),
    "the timed steps": ("--checkpoint.*", "--validator.*", "--dump-folder"),
}
"""The TorchTitan flags that no option owns and no passthrough may change, by reason."""

PERF_FLAGS: tuple[str, ...] = (
    "--training.enable-cpu-offload",
    "--training.gc-freq",
    "--training.gc-debug",
    "--parallelism.enable-fsdp-symm-mem",
    "--parallelism.enable-async-tensor-parallel",
    "--parallelism.enable-sequence-parallel",
    "--parallelism.spmd-backend",
    "--parallelism.context-parallel-load-balancer",
    "--parallelism.context-parallel-ptrr-mask-key",
    "--compile.components",
    "--compile.backend",
    "--compile.mode",
    "--debug.spmd-typechecking",
    "--debug.deterministic",
    "--debug.deterministic-warn-only",
    "--debug.detect-anomaly",
    "--debug.batch-invariant",
    "--debug.print-config",
    "--debug.save-config-file",
    "--debug.enable-structured-logging",
    "--comm.*",
    "--loss.*",
)
"""The TorchTitan flags that a passthrough may set; every other unowned flag is refused."""

_SUBCOMMAND = re.compile(r"[a-z][a-z0-9_-]*(\.[a-z0-9_-]+)*:[A-Za-z0-9_-]+")


def flag_name(token: str) -> str | None:
    """The flag name in ``token`` as tyro reads it, with ``_`` read as ``-`` and a ``no-`` field prefix removed, or ``None`` for a value."""
    if not token.startswith("--") and not _SUBCOMMAND.fullmatch(token):
        return None
    name = token.split("=", 1)[0].replace("_", "-")
    section, dot, field = name.rpartition(".")
    if dot and field.startswith("no-"):
        return f"{section}.{field[3:]}"
    return name


def refusal(token: str) -> str | None:
    """Why ``token`` cannot pass through to TorchTitan, or ``None``."""
    name = flag_name(token)
    if name is None:
        return None
    reason = ownership(name, OWNED_FLAGS, PINNED_FLAGS)
    if reason is not None:
        return reason
    if any(matches(name, pattern) for pattern in PERF_FLAGS):
        return None
    return "not classified in benchmarks/e2e/engines/torchtitan/flags.py"


def passthrough_refusals(arm_name: str, tokens: tuple[str, ...]) -> list[str]:
    """One refusal that names every passthrough token that is not a perf flag, or none."""
    offenders = [
        f"{token} ({reason})"
        for token in tokens
        if (reason := refusal(token)) is not None
    ]
    if not offenders:
        return []
    return [
        f"{arm_name}: {', '.join(offenders)} cannot pass through "
        f"{arm_name}.extra_flags; set the owning harness option instead"
    ]
