"""Which engine flags a ``--torchtitan-arg`` or ``--megatron-arg`` may carry.

A perf flag passes; a flag that would change a fact the manifest records is
refused and names its owner. ``benchmarks/e2e/megatron_stock/flags.py``
holds the Megatron tables; this module holds the TorchTitan tables and the
matching both engines share. It imports the standard library alone.
"""

from __future__ import annotations

import re

_SUBCOMMAND = re.compile(r"[a-z][a-z0-9_-]*(\.[a-z0-9_-]+)*:[A-Za-z0-9_-]+")

FlagTable = dict[str, tuple[str, ...]]
"""An owner or a reason, to the flag patterns it covers."""

TITAN_OWNED_FLAGS: FlagTable = {
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
    "the arm's compile value": ("--compile.enable",),
    "the arm's override imports": ("--override.*",),
    "--profile": ("--profiler.*",),
}
"""The TorchTitan flags each harness option owns. ``*`` ends a prefix."""

TITAN_PINNED_FLAGS: FlagTable = {
    "the bf16 recipe": (
        "--training.dtype",
        "--training.mixed-precision-param",
        "--training.mixed-precision-reduce",
    ),
    "the optimizer matched to Megatron": (
        "--optimizer.*",
        "--lr-scheduler.*",
        "--training.max-norm",
    ),
    "the routing matched to Megatron": ("--debug.moe-force-load-balance",),
    "the shared data stream": (
        "--dataloader.*",
        "--tokenizer.*",
        "--hf-assets-path",
        "--debug.seed",
    ),
    "the step lines the evaluation reads": ("--metrics.*",),
    "the timed steps": ("--checkpoint.*", "--validator.*", "--dump-folder"),
}
"""The TorchTitan flags no option owns and no passthrough may change."""

TITAN_PERF_FLAGS: tuple[str, ...] = (
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
"""The TorchTitan flags a passthrough may set.

Every TorchTitan flag must match one of the three tables. A flag that
matches none is refused, so a field a fork bump adds fails loudly until
someone classifies it.
"""


def flag_names(tokens: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """The flag names in ``tokens``, with any ``=value`` removed."""
    return tuple(
        token.split("=", 1)[0] for token in tokens if token.startswith("--")
    )


def matches(name: str, pattern: str) -> bool:
    """Whether ``name`` is ``pattern``, or starts with it before a ``*``."""
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


def table_entry(name: str, table: FlagTable) -> str | None:
    """The key of the first ``table`` row that covers ``name``, or ``None``."""
    for key, patterns in table.items():
        if any(matches(name, pattern) for pattern in patterns):
            return key
    return None


def titan_flag_name(token: str) -> str:
    """The canonical spelling of a tyro flag or subcommand token.

    tyro accepts ``_`` and ``-`` alike and spells a false boolean as
    ``--section.no-field``.
    """
    name = token.split("=", 1)[0].replace("_", "-")
    section, dot, field = name.rpartition(".")
    if dot and field.startswith("no-"):
        return f"{section}.{field[3:]}"
    return name


def titan_refusal(token: str) -> str | None:
    """Why ``token`` cannot pass through to TorchTitan, or ``None``."""
    if not token.startswith("--") and not _SUBCOMMAND.fullmatch(token):
        return None
    name = titan_flag_name(token)
    owner = table_entry(name, TITAN_OWNED_FLAGS)
    if owner is not None:
        return f"owned by {owner}"
    reason = table_entry(name, TITAN_PINNED_FLAGS)
    if reason is not None:
        return f"pinned by {reason}"
    if any(matches(name, pattern) for pattern in TITAN_PERF_FLAGS):
        return None
    return "not classified in benchmarks/e2e/passthrough.py"


def refuse_titan_passthrough(
    arm_name: str, torchtitan_args: tuple[str, ...] | list[str]
) -> None:
    """Raise when a ``--torchtitan-arg`` token is not a perf flag."""
    offenders = [
        f"{token} ({reason})"
        for token in torchtitan_args
        if (reason := titan_refusal(token)) is not None
    ]
    if offenders:
        raise ValueError(
            f"{arm_name}: {', '.join(offenders)} cannot pass through "
            "--torchtitan-arg; set the owning harness option instead"
        )
