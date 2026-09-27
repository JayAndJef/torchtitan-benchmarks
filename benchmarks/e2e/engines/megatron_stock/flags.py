"""The stock Megatron-LM command line of one arm, and the flags that a passthrough may carry.

This module imports no torch, because the harness process builds the command line from it.
"""

from __future__ import annotations

from benchmarks.e2e.engines.api import DataSpec, RunSpec
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.engines.megatron_stock.profiling import profiler_args
from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.passthrough import ownership
from benchmarks.models.piper_qwen3.shape import PiperShape


DRIVER_MODULE = "benchmarks.e2e.engines.megatron_stock.driver.train"
"""The module that ``python -m`` starts for the stock Megatron arm."""

PP_SCHEDULE = "1F1B"
"""The one pipeline schedule that the driver runs."""

MEGATRON_LM_SCHEDULES: tuple[str, ...] = ("1F1B", "Interleaved1F1B")
"""The shared schedule names that Megatron-LM implements."""

MEGATRON_P2P_SYNC_MODES = ("on", "off")
"""The values of ``MegatronStockConfig.p2p_sync``."""

NO_CHECK_FOR_NAN_FLAG = "--no-check-for-nan-in-loss-and-grad"
"""The Megatron flag that turns off ``check_for_nan_in_loss_and_grad``."""

NAN_GUARD_FLAGS: dict[str, tuple[str, ...]] = {
    "on": (),
    "off": (NO_CHECK_FOR_NAN_FLAG,),
}
"""The Megatron flags of each ``MegatronStockConfig.nan_guard`` value."""

MEGATRON_NAN_GUARD_MODES = tuple(NAN_GUARD_FLAGS)
"""The values of ``MegatronStockConfig.nan_guard``."""

LEAN_PRECISION_DTYPE = "bf16"
"""The dtype of each flag that the lean precision sends."""

LEAN_PRECISION_FLAGS: tuple[str, ...] = (
    "--use-precision-aware-optimizer",
    "--main-grads-dtype",
    "--exp-avg-dtype",
    "--exp-avg-sq-dtype",
)
"""The flag names that the lean precision sends."""

PRECISION_FLAGS: dict[str, tuple[str, ...]] = {
    "stock": (),
    "lean": (
        "--use-precision-aware-optimizer",
        "--main-grads-dtype",
        LEAN_PRECISION_DTYPE,
        "--exp-avg-dtype",
        LEAN_PRECISION_DTYPE,
        "--exp-avg-sq-dtype",
        LEAN_PRECISION_DTYPE,
    ),
}
"""The Megatron flags of each ``MegatronStockConfig.precision`` value."""

MEGATRON_PRECISION_MODES = tuple(PRECISION_FLAGS)
"""The values of ``MegatronStockConfig.precision``."""

PRECISION_STATES: dict[str, str] = {
    "stock": "bf16-fp32-master-fp32-grads-fp32-moments",
    "lean": "bf16-fp32-master-bf16-grads-bf16-moments",
}
"""The model state that each precision holds, as one execution-model term."""

MAIN_GRADS_DTYPES: dict[str, str] = {
    "stock": "fp32",
    "lean": LEAN_PRECISION_DTYPE,
}
"""The ``--main-grads-dtype`` that each precision runs; ``stock`` runs Megatron's own default."""

DATA_PARALLEL_WRAPPERS: dict[int, str] = {
    0: "DistributedDataParallel",
    1: "DistributedDataParallel",
}
"""The data-parallel wrapper class that Megatron builds at each ZeRO level."""

NO_SHARD_STRATEGY = "no_shard"
"""The sharding strategy that every ZeRO level of this arm acts on."""

SHARDING_STRATEGIES: dict[int, str] = {
    0: NO_SHARD_STRATEGY,
    1: NO_SHARD_STRATEGY,
}
"""The sharding strategy that each ZeRO level acts on."""

DATA_PARALLEL_OPTIMIZERS: dict[int, str] = {
    0: "Float16OptimizerWithFloat16Params",
    1: "DistributedOptimizer",
}
"""The optimizer class that Megatron builds at each ZeRO level."""

CHAINED_OPTIMIZER = "ChainedOptimizer"
"""The class that holds the optimizers that Megatron returns."""

LEARNING_RATE = "8e-4"
"""TorchTitan's learning rate; the six values below also copy TorchTitan's optimizer."""
LR_WARMUP_ITERS = "2"
ADAM_BETA1 = "0.9"
ADAM_BETA2 = "0.95"
ADAM_EPS = "1e-8"
WEIGHT_DECAY = "0.1"
CLIP_GRAD = "1.0"

NORM_EPSILON = "1e-6"
"""The RMSNorm epsilon of the TorchTitan config."""

INIT_METHOD_STD = "0.01"
"""Piper's initialization width."""

BENCH_ARM_DIR = "--bench-arm-dir"
"""The harness flag that names the arm directory; the driver parses each ``BENCH_*`` flag."""
BENCH_MODEL_SIZE = "--bench-model-size"
BENCH_LOCAL_BATCH_SIZE = "--bench-local-batch-size"
BENCH_PROFILE = "--bench-profile"
"""The harness flag that turns on the profiler shim; the four schedule flags need it."""
BENCH_PROFILE_FREQ = "--bench-profile-freq"
BENCH_PROFILER_WARMUP = "--bench-profiler-warmup"
BENCH_PROFILER_ACTIVE = "--bench-profiler-active"
BENCH_PP_SCHEDULE = "--bench-pp-schedule"
BENCH_SEQ_LEN = "--bench-seq-len"
BENCH_ROWS_PER_SAMPLE = "--bench-rows-per-sample"
BENCH_MIN_TRACE_WINDOWS = "--bench-min-trace-windows"
BENCH_BATCH_P2P_SYNC = "--bench-batch-p2p-sync"
"""The harness flag that carries ``batch_p2p_sync``, because Megatron has no flag for it."""

BENCH_FLAGS: tuple[str, ...] = (
    BENCH_ARM_DIR,
    BENCH_MODEL_SIZE,
    BENCH_LOCAL_BATCH_SIZE,
    BENCH_PROFILE,
    BENCH_PROFILE_FREQ,
    BENCH_PROFILER_WARMUP,
    BENCH_PROFILER_ACTIVE,
    BENCH_PP_SCHEDULE,
    BENCH_SEQ_LEN,
    BENCH_ROWS_PER_SAMPLE,
    BENCH_MIN_TRACE_WINDOWS,
    BENCH_BATCH_P2P_SYNC,
)
"""Every harness flag."""

BENCH_FLAGS_OMITTED_BY_DEFAULT: tuple[str, ...] = (
    BENCH_PP_SCHEDULE,
    BENCH_BATCH_P2P_SYNC,
)
"""The harness flags that a command line at one pipeline rank omits."""

BENCH_PROFILE_SCHEDULE_FLAGS: tuple[str, ...] = (
    BENCH_PROFILE_FREQ,
    BENCH_PROFILER_WARMUP,
    BENCH_PROFILER_ACTIVE,
    BENCH_MIN_TRACE_WINDOWS,
)
"""The harness flags that describe the profiler window; they come with ``BENCH_PROFILE`` alone."""

OVERLAP_GRAD_REDUCE_FLAG = "--overlap-grad-reduce"
"""The Megatron flag that turns on ``overlap_grad_reduce``."""

ALWAYS_OMITTED_FLAGS: tuple[str, ...] = (
    OVERLAP_GRAD_REDUCE_FLAG,
    "--overlap-param-gather",
    "--moe-permute-fusion",
    "--cross-entropy-loss-fusion",
    "--use-flash-attn",
    "--mock-data",
    "--data-path",
    "--tensorboard-dir",
    "--grad-reduce-in-bf16",
    "--profile-ranks",
)
"""The flags that the stock command line omits at every ZeRO level."""

ZERO1_FLAGS: tuple[str, ...] = ("--use-distributed-optimizer",)
"""The flags that ZeRO level 1 sends."""

SHARDING_FLAGS_BY_VALUE: dict[int, tuple[str, ...]] = {
    0: (),
    1: ZERO1_FLAGS,
}
"""The flags that each ZeRO level sends."""


def data_parallel_optimizer(zero: int) -> str:
    """The optimizer name that the data-parallel line states at ``zero``."""
    return f"{CHAINED_OPTIMIZER}[{DATA_PARALLEL_OPTIMIZERS[zero]}]"


def data_parallel_overlap(extra_flags: tuple[str, ...]) -> bool:
    """The ``overlap_grad_reduce`` value that the data-parallel line states."""
    return any(
        token.split("=", 1)[0] == OVERLAP_GRAD_REDUCE_FLAG for token in extra_flags
    )


def grad_reduce_in_fp32(precision: str) -> bool:
    """Whether Megatron reduces the gradients in fp32 at ``precision``."""
    return MAIN_GRADS_DTYPES[precision] == "fp32"


def omitted_flags(zero: int) -> tuple[str, ...]:
    """Every flag that the command line omits at ``zero``."""
    sent = SHARDING_FLAGS_BY_VALUE[zero]
    return ALWAYS_OMITTED_FLAGS + tuple(
        flag for flag in ZERO1_FLAGS if flag not in sent
    )


def microbatch_geometry(
    data: DataSpec, spec: ParallelismSpec
) -> tuple[int, int, int]:
    """``(rows per sample, microbatches per step, Megatron --seq-length)``; one sample packs its rows into one sequence."""
    if spec.pp > 1:
        rows_per_sample = spec.pp_microbatch_size
    else:
        rows_per_sample = data.local_batch_size
    if data.local_batch_size % rows_per_sample:
        raise ValueError(
            f"local batch size {data.local_batch_size} does not divide "
            f"into microbatches of {rows_per_sample} row(s): Megatron needs "
            "an exact global-batch-size to micro-batch-size ratio"
        )
    microbatches = data.local_batch_size // rows_per_sample
    return rows_per_sample, microbatches, rows_per_sample * data.seq_len


def _geometry_flags(shape: PiperShape, *, megatron_seq_length: int) -> list[str]:
    """The model geometry of ``shape``."""
    return [
        "--num-layers",
        str(shape.n_layers),
        "--hidden-size",
        str(shape.dim),
        "--num-attention-heads",
        str(shape.n_heads),
        "--group-query-attention",
        "--num-query-groups",
        str(shape.n_kv_heads),
        "--kv-channels",
        str(shape.head_dim),
        "--ffn-hidden-size",
        str(shape.moe_hidden_dim),
        "--moe-ffn-hidden-size",
        str(shape.moe_hidden_dim),
        "--num-experts",
        str(shape.num_experts),
        "--moe-router-topk",
        str(shape.top_k),
        "--moe-layer-freq",
        "1",
        "--seq-length",
        str(megatron_seq_length),
        # Megatron asserts that this value is at least --seq-length.
        "--max-position-embeddings",
        str(max(shape.max_seq_len, megatron_seq_length)),
        "--position-embedding-type",
        "rope",
        "--use-rotary-position-embeddings",
        "--rotary-percent",
        "1.0",
        # Megatron parses this flag as an int.
        "--rotary-base",
        str(int(shape.rope_theta)),
        "--normalization",
        "RMSNorm",
        "--norm-epsilon",
        NORM_EPSILON,
        "--swiglu",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--qk-layernorm",
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--init-method-std",
        INIT_METHOD_STD,
    ]


def _engine_flags() -> list[str]:
    """The engine and the stock precision."""
    return [
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--use-mcore-models",
    ]


def _moe_flags() -> list[str]:
    """The mixture-of-experts settings, matched to the TorchTitan config."""
    return [
        "--moe-token-dispatcher-type",
        "alltoall",
        "--moe-grouped-gemm",
        "--moe-router-load-balancing-type",
        "none",
        "--moe-aux-loss-coeff",
        "0.0",
        "--moe-router-dtype",
        "fp32",
    ]


def _optimizer_flags(steps: int, *, global_batch_size: int) -> list[str]:
    """The optimizer and the learning-rate schedule, matched to TorchTitan."""
    return [
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        str(global_batch_size),
        "--train-iters",
        str(steps),
        "--lr",
        LEARNING_RATE,
        "--lr-decay-style",
        "linear",
        "--lr-decay-iters",
        str(steps),
        "--lr-warmup-iters",
        LR_WARMUP_ITERS,
        "--min-lr",
        "0.0",
        "--adam-beta1",
        ADAM_BETA1,
        "--adam-beta2",
        ADAM_BETA2,
        "--adam-eps",
        ADAM_EPS,
        "--weight-decay",
        WEIGHT_DECAY,
        "--clip-grad",
        CLIP_GRAD,
    ]


def _mesh_flags(spec: ParallelismSpec) -> list[str]:
    """The mesh degrees; Megatron derives the data-parallel degree from the world size."""
    return [
        "--tensor-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        str(spec.ep),
        "--pipeline-model-parallel-size",
        str(spec.pp),
    ]


def _data_flags(shape: PiperShape, seed: int) -> list[str]:
    """The tokenizer, the external dataloader, the seed and the logs."""
    return [
        "--tokenizer-type",
        "NullTokenizer",
        "--vocab-size",
        str(shape.vocab_size),
        "--padded-vocab-size",
        str(shape.vocab_size),
        "--dataloader-type",
        "external",
        "--dataloader-inter-document-masking",
        "--no-create-attention-mask-in-dataloader",
        "--num-workers",
        "0",
        "--eval-iters",
        "0",
        "--eval-interval",
        "1000000",
        "--seed",
        str(seed),
        "--rerun-mode",
        "disabled",
        "--log-interval",
        "1",
        "--log-throughput",
    ]


def _bench_flags(
    run: RunSpec, config: MegatronStockConfig, *, arm_dir: str, rows_per_sample: int
) -> list[str]:
    """The harness flags that the driver parses."""
    spec = run.parallelism
    flags = [
        BENCH_ARM_DIR,
        str(arm_dir),
        BENCH_MODEL_SIZE,
        run.shape.name,
        BENCH_LOCAL_BATCH_SIZE,
        str(run.data.local_batch_size),
    ]
    if run.profile:
        flags.extend(
            (
                BENCH_PROFILE,
                BENCH_PROFILE_FREQ,
                str(run.window.freq),
                BENCH_PROFILER_WARMUP,
                str(run.window.warmup),
                BENCH_PROFILER_ACTIVE,
                str(run.window.active),
                BENCH_MIN_TRACE_WINDOWS,
                str(run.window.min_windows),
            )
        )
    flags.extend(
        (
            BENCH_SEQ_LEN,
            str(run.data.seq_len),
            BENCH_ROWS_PER_SAMPLE,
            str(rows_per_sample),
        )
    )
    if spec.pp > 1:
        flags.extend((BENCH_PP_SCHEDULE, str(spec.pp_schedule)))
        if config.p2p_sync == "off":
            flags.extend((BENCH_BATCH_P2P_SYNC, config.p2p_sync))
    return flags


def stock_megatron_flags(
    run: RunSpec, config: MegatronStockConfig, *, arm_dir: str
) -> list[str]:
    """The Megatron flags and then the harness flags of one stock Megatron arm."""
    spec = run.parallelism
    rows_per_sample, microbatches, megatron_seq_length = microbatch_geometry(
        run.data, spec
    )
    return [
        *_geometry_flags(run.shape, megatron_seq_length=megatron_seq_length),
        *_engine_flags(),
        *_moe_flags(),
        *_optimizer_flags(run.data.steps, global_batch_size=microbatches * spec.dp),
        *_mesh_flags(spec),
        *SHARDING_FLAGS_BY_VALUE[spec.zero],
        *PRECISION_FLAGS[config.precision],
        *_data_flags(run.shape, run.seed),
        *(profiler_args(run.data.steps) if run.profile else ()),
        *NAN_GUARD_FLAGS[config.nan_guard],
        *_bench_flags(run, config, arm_dir=arm_dir, rows_per_sample=rows_per_sample),
    ]


OWNED_FLAGS: dict[str, tuple[str, ...]] = {
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
    "--set <arm>.precision": (
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
    "--set <arm>.nan_guard": (NO_CHECK_FOR_NAN_FLAG, "--rerun-mode"),
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
"""The Megatron flags that each harness option or ``--set`` field sets; ``<arm>`` stands for the arm name."""

PINNED_FLAGS: dict[str, tuple[str, ...]] = {
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

PERF_FLAGS: tuple[str, ...] = (
    "--transformer-impl",
    "--moe-token-dispatcher-type",
    "--moe-grouped-gemm",
    "--use-mcore-models",
)
"""The emitted Megatron flags that a passthrough may set; an unlisted Megatron flag also passes."""


def flag_name(token: str) -> str | None:
    """The Megatron flag name in ``token``, or ``None`` for a value."""
    return token.split("=", 1)[0] if token.startswith("--") else None


def refusal(token: str) -> str | None:
    """Why ``token`` cannot pass through to Megatron, or ``None``."""
    name = flag_name(token)
    if name is None:
        return None
    return ownership(name, OWNED_FLAGS, PINNED_FLAGS)


def passthrough_refusals(
    arm_name: str, tokens: tuple[str, ...], zero: int
) -> list[str]:
    """Every refusal of the passthrough ``tokens`` at ZeRO level ``zero``."""
    refusals = []
    offenders = [
        f"{token} ({reason.replace('<arm>', arm_name)})"
        for token in tokens
        if (reason := refusal(token)) is not None
    ]
    if offenders:
        refusals.append(
            f"{arm_name}: {', '.join(offenders)} cannot pass through "
            f"{arm_name}.extra_flags; set the owner instead"
        )
    names = {flag_name(token) for token in tokens}
    if "--overlap-param-gather" in names and zero == 0:
        refusals.append(
            f"{arm_name}: --overlap-param-gather needs --zero 1, because "
            "Megatron asserts a distributed optimizer for it"
        )
    return refusals
