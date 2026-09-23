"""Which engine flags a ``--torchtitan-arg`` or ``--megatron-arg`` may carry.

A perf flag passes; a flag that would change a fact the manifest records is
refused and names its owner. Each table row holds both engines' patterns,
so one owner or one reason covers both. A pattern ending in ``*`` is a
prefix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from benchmarks.e2e.megatron_stock.flags import (
    LEAN_PRECISION_FLAGS,
    NO_CHECK_FOR_NAN_FLAG,
    ZERO1_FLAGS,
)
from benchmarks.e2e.parallelism import MEGATRON_ENGINES
from benchmarks.e2e.schema import Arm


@dataclass(frozen=True)
class EngineFlags:
    """One row's flag patterns, per engine."""

    torchtitan: tuple[str, ...] = ()
    megatron: tuple[str, ...] = ()


OWNED_FLAGS: dict[str, EngineFlags] = {
    "--seq-len": EngineFlags(
        torchtitan=("--training.seq-len",),
        megatron=("--seq-length", "--max-position-embeddings"),
    ),
    "--steps": EngineFlags(
        torchtitan=("--training.steps",),
        megatron=("--train-iters", "--lr-decay-iters"),
    ),
    "--batch": EngineFlags(
        torchtitan=("--training.local-batch-size", "--training.global-batch-size"),
        megatron=("--micro-batch-size", "--global-batch-size"),
    ),
    "--pp-microbatch-size": EngineFlags(
        torchtitan=("--parallelism.pipeline-parallel-microbatch-size",),
    ),
    "--model-size": EngineFlags(
        torchtitan=("--config", "--config-arg", "--module"),
        megatron=(
            "--num-layers",
            "--hidden-size",
            "--num-attention-heads",
            "--group-query-attention",
            "--num-query-groups",
            "--kv-channels",
            "--ffn-hidden-size",
            "--moe-ffn-hidden-size",
            "--num-experts",
            "--moe-router-topk",
            "--moe-layer-freq",
            "--position-embedding-type",
            "--use-rotary-position-embeddings",
            "--rotary-percent",
            "--rotary-base",
            "--normalization",
            "--norm-epsilon",
            "--swiglu",
            "--disable-bias-linear",
            "--untie-embeddings-and-output-weights",
            "--qk-layernorm",
            "--attention-dropout",
            "--hidden-dropout",
            "--init-method-std",
            "--vocab-size",
            "--padded-vocab-size",
            "--no-pad-vocab-size",
            "--disable-pad-vocab-size",
        ),
    ),
    "--dp/--pp/--ep": EngineFlags(
        torchtitan=(
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
        megatron=(
            "--tensor-model-parallel-size",
            "--pipeline-model-parallel-size",
            "--expert-model-parallel-size",
            "--context-parallel-size",
            "--expert-tensor-parallel-size",
            "--num-layers-per-virtual-pipeline-stage",
            "--num-virtual-stages-per-pipeline-rank",
        ),
    ),
    "--pp-schedule": EngineFlags(
        torchtitan=(
            "--parallelism.pipeline-parallel-schedule",
            "--parallelism.pipeline-parallel-schedule-csv",
        ),
    ),
    "--zero": EngineFlags(
        torchtitan=("--parallelism.fsdp-reshard-after-forward",),
        megatron=ZERO1_FLAGS,
    ),
    "--ac": EngineFlags(
        torchtitan=("--activation-checkpoint.*", "activation-checkpoint:*"),
        megatron=(
            "--recompute-activations",
            "--recompute-granularity",
            "--recompute-method",
            "--recompute-num-layers",
            "--recompute-modules",
        ),
    ),
    "--megatron-precision": EngineFlags(
        megatron=(
            *LEAN_PRECISION_FLAGS,
            "--main-params-dtype",
            "--grad-reduce-in-bf16",
            "--accumulate-allreduce-grads-in-fp32",
            "--bf16",
            "--fp16",
            "--fp8-*",
            "--fp4-*",
            "--no-fp8-wgrad",
            "--disable-fp8-wgrad",
            "--first-last-layers-bf16",
            "--num-layers-at-start-in-bf16",
            "--num-layers-at-end-in-bf16",
        ),
    ),
    "--megatron-nan-guard": EngineFlags(
        megatron=(NO_CHECK_FOR_NAN_FLAG, "--rerun-mode"),
    ),
    "--profile": EngineFlags(
        torchtitan=("--profiler.*",),
        megatron=(
            "--profile",
            "--use-pytorch-profiler",
            "--profile-step-start",
            "--profile-step-end",
            "--profile-ranks",
            "--pytorch-profiler-collect-shapes",
            "--pytorch-profiler-collect-callstack",
            "--pytorch-profiler-collect-chakra",
        ),
    ),
    "the arm's compile value": EngineFlags(torchtitan=("--compile.enable",)),
    "the arm's override imports": EngineFlags(torchtitan=("--override.*",)),
}
"""The flags each harness option, or arm property, sets."""

PINNED_FLAGS: dict[str, EngineFlags] = {
    "the harness driver": EngineFlags(megatron=("--bench-*",)),
    "the precision recipe": EngineFlags(
        torchtitan=(
            "--training.dtype",
            "--training.mixed-precision-param",
            "--training.mixed-precision-reduce",
        ),
    ),
    "the optimizer matched across engines": EngineFlags(
        torchtitan=("--optimizer.*", "--lr-scheduler.*", "--training.max-norm"),
        megatron=(
            "--lr",
            "--lr-decay-style",
            "--lr-warmup-iters",
            "--min-lr",
            "--adam-beta1",
            "--adam-beta2",
            "--adam-eps",
            "--weight-decay",
            "--clip-grad",
        ),
    ),
    "the routing matched across engines": EngineFlags(
        torchtitan=("--debug.moe-force-load-balance",),
        megatron=(
            "--moe-router-load-balancing-type",
            "--moe-aux-loss-coeff",
            "--moe-router-dtype",
        ),
    ),
    "the shared data stream": EngineFlags(
        torchtitan=(
            "--dataloader.*",
            "--tokenizer.*",
            "--hf-assets-path",
            "--debug.seed",
        ),
        megatron=(
            "--seed",
            "--tokenizer-type",
            "--dataloader-type",
            "--data-path",
            "--mock-data",
            "--num-workers",
            "--dataloader-inter-document-masking",
            "--no-create-attention-mask-in-dataloader",
        ),
    ),
    "the step lines the evaluation reads": EngineFlags(
        torchtitan=("--metrics.*",),
        megatron=("--log-interval", "--log-throughput", "--eval-iters", "--eval-interval"),
    ),
    "the timed steps": EngineFlags(
        torchtitan=("--checkpoint.*", "--validator.*", "--dump-folder"),
        megatron=(
            "--save",
            "--load",
            "--save-interval",
            "--persistent-save-interval",
            "--tensorboard-dir",
        ),
    ),
}
"""The flags no option owns and no passthrough may change, by reason."""

PERF_FLAGS = EngineFlags(
    torchtitan=(
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
    ),
    megatron=(
        "--transformer-impl",
        "--moe-token-dispatcher-type",
        "--moe-grouped-gemm",
        "--use-mcore-models",
    ),
)
"""The flags a builder emits, or a config declares, that a passthrough may set.

An unlisted Megatron flag passes, because Megatron offers hundreds of
fusion flags and its parser refuses a misspelling. An unlisted TorchTitan
flag is refused, because the fork's config is small enough to classify
whole, so a field that a bump adds fails loudly.
"""

_SUBCOMMAND = re.compile(r"[a-z][a-z0-9_-]*(\.[a-z0-9_-]+)*:[A-Za-z0-9_-]+")


def engine_side(engine: str) -> str:
    """The ``EngineFlags`` field that holds ``engine``'s patterns."""
    return "megatron" if engine in MEGATRON_ENGINES else "torchtitan"


def flag_name(side: str, token: str) -> str | None:
    """The canonical flag name in ``token``, or ``None`` for a value.

    tyro accepts ``_`` and ``-`` alike and spells a false boolean as
    ``--section.no-field``, so a TorchTitan name is normalized first.
    """
    if side == "megatron":
        return token.split("=", 1)[0] if token.startswith("--") else None
    if not token.startswith("--") and not _SUBCOMMAND.fullmatch(token):
        return None
    name = token.split("=", 1)[0].replace("_", "-")
    section, dot, field = name.rpartition(".")
    if dot and field.startswith("no-"):
        return f"{section}.{field[3:]}"
    return name


def matches(name: str, pattern: str) -> bool:
    """Whether ``name`` is ``pattern``, or starts with it before a ``*``."""
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


def row_for(side: str, name: str, table: dict[str, EngineFlags]) -> str | None:
    """The key of the first ``table`` row that covers ``name``, or ``None``."""
    for key, flags in table.items():
        if any(matches(name, pattern) for pattern in getattr(flags, side)):
            return key
    return None


def refusal(side: str, token: str) -> str | None:
    """Why ``token`` cannot pass through to ``side``, or ``None``."""
    name = flag_name(side, token)
    if name is None:
        return None
    owner = row_for(side, name, OWNED_FLAGS)
    if owner is not None:
        return f"owned by {owner}"
    reason = row_for(side, name, PINNED_FLAGS)
    if reason is not None:
        return f"pinned by {reason}"
    if side == "megatron" or any(
        matches(name, pattern) for pattern in PERF_FLAGS.torchtitan
    ):
        return None
    return "not classified in benchmarks/e2e/passthrough.py"


def refuse_passthrough(
    arm: Arm, tokens: tuple[str, ...] | list[str], zero: int
) -> None:
    """Raise when a passthrough token for ``arm`` is not a perf flag."""
    side = engine_side(arm.engine)
    option = f"--{side}-arg"
    offenders = [
        f"{token} ({reason})"
        for token in tokens
        if (reason := refusal(side, token)) is not None
    ]
    if offenders:
        raise ValueError(
            f"{arm.name}: {', '.join(offenders)} cannot pass through "
            f"{option}; set the owning harness option instead"
        )
    names = {flag_name(side, token) for token in tokens}
    if side == "megatron" and "--overlap-param-gather" in names and zero == 0:
        raise ValueError(
            f"{arm.name}: --overlap-param-gather needs --zero 1, because "
            "Megatron asserts a distributed optimizer for it"
        )


def reach_refusal(
    arms: Iterable[Arm],
    torchtitan_args: tuple[str, ...],
    megatron_args: tuple[str, ...],
) -> str | None:
    """Why a passthrough list reaches no arm of ``arms``, or ``None``."""
    arms = tuple(arms)
    names = ", ".join(arm.name for arm in arms)
    sides = {engine_side(arm.engine) for arm in arms}
    if torchtitan_args and "torchtitan" not in sides:
        return (
            f"--torchtitan-arg reaches no arm of this run: {names} run on "
            "Megatron; select a TorchTitan arm, or omit the option"
        )
    if megatron_args and "megatron" not in sides:
        return (
            f"--megatron-arg reaches no arm of this run: {names} run on "
            "TorchTitan; select the stock megatron arm, or omit the option"
        )
    return None
