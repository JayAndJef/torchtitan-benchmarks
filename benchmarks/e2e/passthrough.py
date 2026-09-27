"""Which engine flags a ``--torchtitan-arg`` or ``--megatron-arg`` may carry.

A perf flag passes; a flag that would change a fact the manifest records is
refused and names its owner. Each engine keeps its own tables. A pattern
ending in ``*`` is a prefix.
"""

from __future__ import annotations

from collections.abc import Mapping

from benchmarks.e2e.megatron_stock.flags import (
    LEAN_PRECISION_FLAGS,
    NO_CHECK_FOR_NAN_FLAG,
    ZERO1_FLAGS,
)


FlagTable = Mapping[str, tuple[str, ...]]
"""One engine's flag patterns, by the owner or the reason of each row."""


def matches(name: str, pattern: str) -> bool:
    """Whether ``name`` is ``pattern``, or starts with it before a ``*``."""
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


def row_for(name: str, table: FlagTable) -> str | None:
    """The key of the first ``table`` row that covers ``name``, or ``None``."""
    for key, patterns in table.items():
        if any(matches(name, pattern) for pattern in patterns):
            return key
    return None


def ownership(name: str, owned: FlagTable, pinned: FlagTable) -> str | None:
    """``owned by <row>`` or ``pinned by <row>`` for ``name``, or ``None`` when no row covers it."""
    owner = row_for(name, owned)
    if owner is not None:
        return f"owned by {owner}"
    reason = row_for(name, pinned)
    if reason is not None:
        return f"pinned by {reason}"
    return None


MEGATRON_OWNED_FLAGS: dict[str, tuple[str, ...]] = {
    "--seq-len": ("--seq-length", "--max-position-embeddings"),
    "--steps": ("--train-iters", "--lr-decay-iters"),
    "--batch": ("--micro-batch-size", "--global-batch-size"),
    "--model-size": (
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
    "--dp/--pp/--ep": (
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--expert-model-parallel-size",
        "--context-parallel-size",
        "--expert-tensor-parallel-size",
        "--num-layers-per-virtual-pipeline-stage",
        "--num-virtual-stages-per-pipeline-rank",
    ),
    "--zero": ZERO1_FLAGS,
    "--ac": (
        "--recompute-activations",
        "--recompute-granularity",
        "--recompute-method",
        "--recompute-num-layers",
        "--recompute-modules",
    ),
    "--megatron-precision": (
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
    "--megatron-nan-guard": (NO_CHECK_FOR_NAN_FLAG, "--rerun-mode"),
    "--profile": (
        "--profile",
        "--use-pytorch-profiler",
        "--profile-step-start",
        "--profile-step-end",
        "--profile-ranks",
        "--pytorch-profiler-collect-shapes",
        "--pytorch-profiler-collect-callstack",
        "--pytorch-profiler-collect-chakra",
    ),
}
"""The Megatron flags that each harness option sets."""

MEGATRON_PINNED_FLAGS: dict[str, tuple[str, ...]] = {
    "the harness driver": ("--bench-*",),
    "the optimizer matched across engines": (
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
    "the routing matched across engines": (
        "--moe-router-load-balancing-type",
        "--moe-aux-loss-coeff",
        "--moe-router-dtype",
    ),
    "the shared data stream": (
        "--seed",
        "--tokenizer-type",
        "--dataloader-type",
        "--data-path",
        "--mock-data",
        "--num-workers",
        "--dataloader-inter-document-masking",
        "--no-create-attention-mask-in-dataloader",
    ),
    "the step lines the evaluation reads": (
        "--log-interval",
        "--log-throughput",
        "--eval-iters",
        "--eval-interval",
    ),
    "the timed steps": (
        "--save",
        "--load",
        "--save-interval",
        "--persistent-save-interval",
        "--tensorboard-dir",
    ),
}
"""The Megatron flags that no option owns and no passthrough may change, by reason."""

MEGATRON_PERF_FLAGS: tuple[str, ...] = (
    "--transformer-impl",
    "--moe-token-dispatcher-type",
    "--moe-grouped-gemm",
    "--use-mcore-models",
)
"""The Megatron flags that the builder emits and that a passthrough may set.

An unlisted Megatron flag passes too, because Megatron's parser refuses a
misspelled flag.
"""


def megatron_flag_name(token: str) -> str | None:
    """The Megatron flag name in ``token``, or ``None`` for a value."""
    return token.split("=", 1)[0] if token.startswith("--") else None


def megatron_refusal(token: str) -> str | None:
    """Why ``token`` cannot pass through to Megatron, or ``None``."""
    name = megatron_flag_name(token)
    if name is None:
        return None
    return ownership(name, MEGATRON_OWNED_FLAGS, MEGATRON_PINNED_FLAGS)


def refuse_megatron_passthrough(
    arm_name: str, tokens: tuple[str, ...], zero: int
) -> None:
    """Raise when a Megatron passthrough token is not a perf flag."""
    offenders = [
        f"{token} ({reason})"
        for token in tokens
        if (reason := megatron_refusal(token)) is not None
    ]
    if offenders:
        raise ValueError(
            f"{arm_name}: {', '.join(offenders)} cannot pass through "
            "--megatron-arg; set the owning harness option instead"
        )
    names = {megatron_flag_name(token) for token in tokens}
    if "--overlap-param-gather" in names and zero == 0:
        raise ValueError(
            f"{arm_name}: --overlap-param-gather needs --zero 1, because "
            "Megatron asserts a distributed optimizer for it"
        )
