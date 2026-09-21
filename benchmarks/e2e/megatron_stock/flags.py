"""The Megatron-LM command line for the stock scenario, as data.

``benchmarks/e2e/launch.py`` calls ``stock_megatron_flags`` in the parent
process, so this module must stay torch-free. It imports ``dataclasses``,
``PiperShape``, ``ParallelismSpec`` and ``Workload``, and nothing else. A
test can therefore read the whole command line on a host with no GPU.

**Every geometry value comes from ``PiperShape``.** No shape number is
written here. A hardcoded width would build one model and publish it under
the requested size, which no validation rule can see.

**Every flag below exists in the pinned Megatron-LM rev.** Re-run this check
after a submodule bump, from the repository root::

    .venv/bin/python - <<'CHECK'
    import argparse, typing, typing_extensions
    from benchmarks.models.piper_qwen3.megatron_bootstrap import (
        add_megatron_to_path,
    )
    add_megatron_to_path()
    if not hasattr(typing, "override"):
        typing.override = typing_extensions.override
    from megatron.training.arguments import add_megatron_arguments
    from benchmarks.e2e.megatron_stock.flags import stock_megatron_flags
    from benchmarks.e2e.parallelism import TRIVIAL_SPEC
    from benchmarks.e2e.registry import C4_REPLAY_WORKLOAD
    from benchmarks.models.piper_qwen3.shape import PIPER_1B
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_megatron_arguments(parser)
    known = {s for a in parser._actions for s in a.option_strings}
    emitted = stock_megatron_flags(
        PIPER_1B, C4_REPLAY_WORKLOAD, TRIVIAL_SPEC,
        arm_dir="/tmp/x", model_size="1b",
    )
    unknown = [
        t for t in emitted
        if t.startswith("--") and not t.startswith("--bench-")
        and t not in known
    ]
    print(unknown)
    CHECK

**The harness flags start with ``--bench-``.** ``train.py`` adds them
through Megatron's own ``extra_args_provider`` hook, so Megatron's parser
owns them and an unknown one fails at parse time.

The list follows Piper's own stock invocation, with three deliberate
deviations: ``--moe-router-dtype fp32`` (TorchTitan routes fp32 too, so an
unset dtype would make the router a precision difference), no
attention-backend flag (TransformerEngine then resolves to the cuDNN kernel
arm rule 6 pins), and no distributed optimizer under ``--zero 0``
(both engines then replicate their parameters, so the data-parallel axis
carries one change).

**``--zero 1`` sends ONE flag**,
``--use-distributed-optimizer``. Megatron then keeps a plain
``DistributedDataParallel`` and builds a ``DistributedOptimizer``, which
shards the optimizer states over the data-parallel group. It builds no
device mesh, so the level holds a pipeline. See ``ZERO1_FLAGS``.

**``--megatron-p2p-sync off`` adds one harness flag and nothing else.**
Megatron has no CLI flag for ``batch_p2p_sync``, so ``--bench-batch-p2p-sync
off`` carries the value and ``train.py`` sets the field on ``args`` before
Megatron builds its config. The flag is omitted at ``on``, so the default
argv is the argv every published cell ran.

**``--megatron-nan-guard off`` adds one MEGATRON flag and nothing else.**
Megatron has its own switch for this field,
``--no-check-for-nan-in-loss-and-grad``, so the value needs no harness flag:
the argv carries Megatron's own token, a stock user can type the same one,
and ``train.py`` prints the value Megatron parsed. The token is omitted at
``on``, so the default argv does not move here either. See
``NO_CHECK_FOR_NAN_FLAG``.
"""

from __future__ import annotations

from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.schema import Workload
from benchmarks.e2e.registry import (
    DEFAULT_MEGATRON_NAN_GUARD,
    DEFAULT_MEGATRON_P2P_SYNC,
    DEFAULT_MEGATRON_PRECISION,
    MEGATRON_NAN_GUARD_MODES,
    MEGATRON_P2P_SYNC_MODES,
    MEGATRON_PRECISION_MODES,
)
from benchmarks.models.piper_qwen3.shape import PiperShape

SUPPORTED_PP_SCHEDULE = "1F1B"
"""The one pipeline schedule the stock driver runs.

Megatron's ``forward_backward_pipelining_without_interleaving`` is 1F1B and
nothing else.
"""

DATA_PARALLEL_WRAPPERS: dict[int, str] = {
    0: "DistributedDataParallel",
    1: "DistributedDataParallel",
}
"""The wrapper class Megatron builds for each ZeRO level.

``train.py`` prints the wrapper it really got, and validation reads this
table to say what each level must print. Both levels build the same
wrapper, because Megatron picks its sharded one on a flag this suite sends
at no level, so ``DATA_PARALLEL_OPTIMIZERS`` below is what separates them.
"""
NO_SHARD_STRATEGY = "no_shard"
"""The strategy every level of this suite acts on. ``train.py`` prints it."""

SHARDING_STRATEGIES: dict[int, str] = {
    0: NO_SHARD_STRATEGY,
    1: NO_SHARD_STRATEGY,
}
"""The sharding strategy each level acts on.

Megatron's own argparse default is ``optim_grads_params``, but it reads
that field only under the sharded wrapper, which this suite never builds.
A marker built from the raw field would call a replicated run sharded.
"""

DATA_PARALLEL_OVERLAP: dict[int, bool] = {
    level: False for level in SHARDING_STRATEGIES
}
"""What ``overlap_grad_reduce`` reads on the wrapper, per level.

False at both levels, for two independent reasons. The argv omits
``--overlap-grad-reduce`` everywhere (see ``ALWAYS_OMITTED_FLAGS``), and
Megatron mutates that field only inside its sharded wrapper.
"""

DATA_PARALLEL_OPTIMIZERS: dict[int, str] = {
    0: "Float16OptimizerWithFloat16Params",
    1: "DistributedOptimizer",
}
"""The optimizer class Megatron builds for each level.

The wrapper table above separates neither level, but this one does:
Megatron builds ``DistributedOptimizer`` under ``use_distributed_optimizer``
and ``Float16OptimizerWithFloat16Params`` otherwise. The two tables together
pin both levels, and neither does it alone.
"""

CHAINED_OPTIMIZER = "ChainedOptimizer"
"""The outer class Megatron returns for a chain of optimizers.

``get_megatron_optimizer`` ends its standard path with an unconditional
``ChainedOptimizer(optimizers)``, and this suite takes no other path, so
every level prints its optimizer inside a chain. A marker must read
``data_parallel_optimizer`` below, because the raw table cannot state the
line.
"""

LEARNING_RATE = "8e-4"
"""TorchTitan's own optimizer values, replicated flag for flag."""
LR_WARMUP_ITERS = "2"
ADAM_BETA1 = "0.9"
ADAM_BETA2 = "0.95"
ADAM_EPS = "1e-8"
WEIGHT_DECAY = "0.1"
CLIP_GRAD = "1.0"

NORM_EPSILON = "1e-6"
"""The RMSNorm epsilon the TorchTitan config carries."""

INIT_METHOD_STD = "0.01"
"""Piper's own initialization width."""

BENCH_ARM_DIR = "--bench-arm-dir"
"""The harness flag names.

``train.py`` declares the same names, and a test compares the two lists, so
a rename cannot reach only one side.
"""
BENCH_MODEL_SIZE = "--bench-model-size"
BENCH_LOCAL_BATCH_SIZE = "--bench-local-batch-size"

BENCH_PROFILE = "--bench-profile"
"""The profiler switch, and the one token that states it.

The four schedule flags below travel with this token and with nothing else.
The driver refuses a schedule flag without it, so an argv cannot ask for a
window it also declines to collect.
"""
BENCH_PROFILE_FREQ = "--bench-profile-freq"
BENCH_PROFILER_WARMUP = "--bench-profiler-warmup"
BENCH_PROFILER_ACTIVE = "--bench-profiler-active"
BENCH_PP_SCHEDULE = "--bench-pp-schedule"
BENCH_SEQ_LEN = "--bench-seq-len"
BENCH_ROWS_PER_SAMPLE = "--bench-rows-per-sample"
BENCH_MIN_TRACE_WINDOWS = "--bench-min-trace-windows"

BENCH_BATCH_P2P_SYNC = "--bench-batch-p2p-sync"
"""The pipeline point-to-point sync.

The writer emits it only under ``off``, because Megatron has no flag of its
own for the field and ``on`` is its default.
"""

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

BENCH_FLAGS_OMITTED_BY_DEFAULT: tuple[str, ...] = (
    BENCH_PP_SCHEDULE,
    BENCH_BATCH_P2P_SYNC,
)
"""The two harness flags a single-stage argv omits.

Both describe the pipeline: the schedule names a split that does not happen
at pp 1, and the p2p value names a message that does not exist there. A
test reads this tuple, so a third such flag is an edit here.
"""

BENCH_PROFILE_SCHEDULE_FLAGS: tuple[str, ...] = (
    BENCH_PROFILE_FREQ,
    BENCH_PROFILER_WARMUP,
    BENCH_PROFILER_ACTIVE,
    BENCH_MIN_TRACE_WINDOWS,
)
"""The harness flags that describe a profiler window.

``_bench_flags`` emits the whole group under ``--profile`` and none of it
otherwise, and ``train.py`` refuses any member without ``BENCH_PROFILE``.
One tuple keeps the writer and the refusal from disagreeing.
"""

ALWAYS_OMITTED_FLAGS: tuple[str, ...] = (
    "--overlap-grad-reduce",
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
"""The flags this suite declines under every ZeRO level.

The tuple exists so a test asserts their absence by name. The two overlap
flags travel together and move the gradient bucket size, which changes what
the run does rather than what the argv says; the marker reads the resolved
value instead (see ``DATA_PARALLEL_OVERLAP``).
``--use-precision-aware-optimizer`` is absent from this tuple because
``--megatron-precision lean`` sends it, and ``_precision_flags`` declines it
under the default value alone. ``--grad-reduce-in-bf16`` stays under both
precision values, because ``--main-grads-dtype bf16`` already leaves fp32
accumulation off, and two tokens for one fact hide which one did the work.
"""

ZERO1_FLAGS: tuple[str, ...] = ("--use-distributed-optimizer",)
"""The flag names ZeRO level 1 sends.

``--use-distributed-optimizer`` alone gives Megatron a
``DistributedOptimizer`` beside a plain ``DistributedDataParallel``, which
shards the optimizer states over the data-parallel group and nothing else.
It builds no device mesh, so the level holds a pipeline.
"""

SHARDING_FLAGS_BY_VALUE: dict[int, tuple[str, ...]] = {
    0: (),
    1: ZERO1_FLAGS,
}
"""What each ZeRO level sends, as one table.

``_sharding_flags`` builds the tokens and ``omitted_flags`` subtracts this
table from ``ZERO1_FLAGS``, so presence and absence stay one fact.
"""

LEAN_PRECISION_FLAGS: tuple[str, ...] = (
    "--use-precision-aware-optimizer",
    "--main-grads-dtype",
    "--exp-avg-dtype",
    "--exp-avg-sq-dtype",
)
"""The flag names ``--megatron-precision lean`` sends.

``MEGATRON_PRECISION_MODES`` states the whole recipe once. The three dtype
flags take a value token each, so the argv is eight tokens; this tuple holds
the names a test asserts absence by under the default value.
"""

LEAN_PRECISION_DTYPE = "bf16"
"""The dtype every lean flag carries.

One name keeps the three from drifting apart and states the recipe once.
"""

MEGATRON_MAIN_GRADS_DTYPE_DEFAULT = "fp32"
"""Megatron's own default for ``--main-grads-dtype``.

The stock recipe sends no dtype flag, so it runs this one.
"""


def data_parallel_optimizer(zero: int) -> str:
    """The optimizer name the data-parallel line must carry.

    ``DATA_PARALLEL_OPTIMIZERS`` names the class Megatron builds for this
    level. This function wraps that class in the chain the line carries.

    **The standard path always chains.** ``get_megatron_optimizer`` ends it
    with an unconditional ``ChainedOptimizer(optimizers)``, and this suite
    takes no other path, so both levels print a chain. That chain always holds the
    dense optimizer. It holds a second member where TransformerEngine
    marked a weight for the expert process groups. That mark has three
    conditions: an expert degree above 1, an expert tensor degree that
    differs from the dense one, or an expert GTP remat size that differs.
    This argv reaches the first alone, because Megatron defaults the other
    two to the dense values. Both members carry one class, because
    ``use_distributed_optimizer`` is one value for the whole run, so the
    printed string does not move with the expert degree.

    **The model shape decides none of this.** An earlier version of this
    function branched on ``shape.num_experts``, which expected a chain
    Megatron never builds.
    """
    return f"{CHAINED_OPTIMIZER}[{DATA_PARALLEL_OPTIMIZERS[zero]}]"


NO_CHECK_FOR_NAN_FLAG = "--no-check-for-nan-in-loss-and-grad"
"""Megatron's own switch for ``check_for_nan_in_loss_and_grad``.

The flag is ``action='store_false'``, so it turns the field off.
``MEGATRON_NAN_GUARD_MODES`` states what the field gates and why
``--rerun-mode disabled`` is not enough. A test pins the spelling, the dest
and both consumers against the pinned source.
"""


def refuse_unknown_nan_guard(megatron_nan_guard: str) -> None:
    """Raise on a NaN-guard value this module cannot build a command line for.

    A silent fall through would send the default argv under the ``off``
    label, and the run would keep the guard it claims to have removed.
    """
    if megatron_nan_guard not in MEGATRON_NAN_GUARD_MODES:
        raise ValueError(
            f"megatron nan guard {megatron_nan_guard!r} is not one of "
            + ", ".join(repr(mode) for mode in MEGATRON_NAN_GUARD_MODES)
        )


def _nan_guard_flags(megatron_nan_guard: str) -> list[str]:
    """Megatron's own token under ``off``; nothing under ``on``.

    The gate reads the literal ``on`` and never the axis default, so the
    token follows the treatment rather than the default of the day. The
    value is legal at every mesh: the loss check runs on the last stage at
    ``pp`` 1 and the gradient check runs on every rank at ``dp`` 1, so
    there is no degree at which the field is inert.
    """
    refuse_unknown_nan_guard(megatron_nan_guard)
    if megatron_nan_guard == "on":
        return []
    return [NO_CHECK_FOR_NAN_FLAG]


def refuse_unknown_p2p_sync(megatron_p2p_sync: str) -> None:
    """Raise on a p2p value this module cannot build a command line for.

    A silent fall through would send the default argv under the ``off``
    label, which is the wrong record this option exists to prevent.
    """
    if megatron_p2p_sync not in MEGATRON_P2P_SYNC_MODES:
        raise ValueError(
            f"megatron p2p sync {megatron_p2p_sync!r} is not one of "
            + ", ".join(repr(mode) for mode in MEGATRON_P2P_SYNC_MODES)
        )


def refuse_unknown_megatron_precision(megatron_precision: str) -> None:
    """Raise on a precision value this module cannot build a command line for.

    A silent fall through would send the stock argv under the ``lean``
    label. The manifest would then record 10 bytes of optimizer state per
    parameter for a run that held 18.
    """
    if megatron_precision not in MEGATRON_PRECISION_MODES:
        raise ValueError(
            f"megatron precision {megatron_precision!r} is not one of "
            + ", ".join(repr(mode) for mode in MEGATRON_PRECISION_MODES)
        )


def _precision_flags(
    megatron_precision: str, zero: int
) -> list[str]:
    """What Megatron needs to hold the optimizer state this way.

    Empty under ``stock``, which is ``--bf16`` alone and 18 bytes of
    optimizer state per parameter. That is the treatment every published
    cell of this scenario ran, so the default argv does not move.

    Under ``lean`` it is the four flags of ``LEAN_PRECISION_FLAGS`` and
    their three dtype tokens, which reach 10 bytes. The recipe, the byte
    table and the two flags this axis must never send are stated once,
    above ``MEGATRON_PRECISION_MODES`` in ``benchmarks/e2e/registry.py``.

    **``lean`` needs a sharded dense value, and this refuses the rest.**
    ``optimizer_config.py`` asserts ``use_distributed_optimizer`` under
    ``--use-precision-aware-optimizer``, and the zero axis is the
    one owner of that flag. A replicated run would die inside Megatron's
    own config validation, minutes into a subprocess, naming neither this
    axis nor its repair. ``_resolve_run`` refuses the combination first for
    a real run; this refusal is for a caller that builds a command line
    without one.
    """
    refuse_unknown_megatron_precision(megatron_precision)
    if megatron_precision == DEFAULT_MEGATRON_PRECISION:
        return []
    if zero == 0:
        raise ValueError(
            f"megatron precision {megatron_precision!r} needs "
            "--zero 1: Megatron asserts use_distributed_optimizer under "
            "--use-precision-aware-optimizer, and the ZeRO level is the "
            "one owner of that flag"
        )
    return [
        "--use-precision-aware-optimizer",
        "--main-grads-dtype",
        LEAN_PRECISION_DTYPE,
        "--exp-avg-dtype",
        LEAN_PRECISION_DTYPE,
        "--exp-avg-sq-dtype",
        LEAN_PRECISION_DTYPE,
    ]


def main_grads_dtype(megatron_precision: str) -> str:
    """The ``--main-grads-dtype`` value this precision runs.

    ``stock`` sends no dtype flag, so it runs Megatron's own default.
    ``lean`` sends ``LEAN_PRECISION_DTYPE``.
    """
    refuse_unknown_megatron_precision(megatron_precision)
    if megatron_precision == DEFAULT_MEGATRON_PRECISION:
        return MEGATRON_MAIN_GRADS_DTYPE_DEFAULT
    return LEAN_PRECISION_DTYPE


def grad_reduce_in_fp32(megatron_precision: str) -> bool:
    """Whether Megatron reduces the gradients in fp32 at this precision.

    **This field moves with the precision axis, and a marker once pinned
    it to True.** Under ``--bf16`` Megatron turns
    ``accumulate_allreduce_grads_in_fp32`` on only where the main-grad
    dtype is fp32 (``arguments.py``), and ``get_megatron_ddp_config``
    copies that field into ``ddp_config.grad_reduce_in_fp32``. So the
    stock recipe reduces in fp32 and the lean recipe does not. A real
    eight-GPU run failed arm rule 12 on 2026-09-16 for that reason.

    The value is DERIVED from the dtype the recipe sends. A hand-written
    table would keep saying True after somebody moved
    ``LEAN_PRECISION_DTYPE``.
    """
    return (
        main_grads_dtype(megatron_precision)
        == MEGATRON_MAIN_GRADS_DTYPE_DEFAULT
    )


def omitted_flags(zero: int) -> tuple[str, ...]:
    """Every flag this arm declines at ``zero``.

    The roster is a function of the value, because the sharding flags move
    from declined to required as the value changes. A test reads this rather
    than a hand-written list, so every absence is asserted by name.

    **It is SUBTRACTED from what the level sends, never written twice.**
    Level 0 declines ``--use-distributed-optimizer`` and level 1 sends it,
    so a hand-written list here could say a level 1 run declines a flag its
    own argv carries.
    """
    sent = SHARDING_FLAGS_BY_VALUE[zero]
    return ALWAYS_OMITTED_FLAGS + tuple(
        flag for flag in ZERO1_FLAGS if flag not in sent
    )


def microbatch_geometry(
    workload: Workload, spec: ParallelismSpec
) -> tuple[int, int, int]:
    """``(rows per sample, microbatches per step, megatron seq_length)``.

    **One Megatron sample is one packed sequence, never a batch of rows.**
    Megatron's ``flatten_batch_for_packed_sequences`` flattens a ``(m, S)``
    microbatch into ``(1, m*S)`` whenever ``cu_seqlens`` is present, so the
    activation the first stage sends is ``(m*S, 1, H)``. The pipeline allocates its receive
    buffer from ``get_tensor_shapes``, which returns ``(S, m, H)`` and
    validates nothing. The two hold the same number of elements, so the
    transfer succeeds and the next stage reads a **permuted** activation.
    Measured on this rev with ``m`` 4: ``(64, 1, 8)`` sent into a
    ``(16, 4, 8)`` buffer.

    So the harness packs the rows itself and tells Megatron the sample is
    one row of ``rows * S`` tokens. Every shape then agrees, and the attention is unchanged:
    ``cu_seqlens`` already marks every document, and a row boundary is a
    document boundary.

    The row count follows the other engine at both meshes. Under a pipeline
    the microbatch size is the split TorchTitan is given. Without one
    neither engine splits, so the whole batch is one pack, which is what
    every published megatron number was measured on.
    """
    if spec.pp > 1:
        rows_per_sample = spec.pp_microbatch_size
    else:
        rows_per_sample = workload.local_batch_size
    if workload.local_batch_size % rows_per_sample:
        raise ValueError(
            f"local batch size {workload.local_batch_size} does not divide "
            f"into microbatches of {rows_per_sample} row(s): Megatron needs "
            "an exact global-batch-size to micro-batch-size ratio"
        )
    microbatches = workload.local_batch_size // rows_per_sample
    return rows_per_sample, microbatches, rows_per_sample * workload.seq_len


def _geometry_flags(
    shape: PiperShape, *, megatron_seq_length: int
) -> list[str]:
    """The model geometry, every value read from ``shape``.

    ``--rotary-base`` is ``type=int`` in Megatron's parser, so ``1e6`` is
    rejected. The value is rendered as a plain integer.

    ``--group-query-attention`` with ``--num-query-groups`` is how Megatron
    spells grouped-query attention. ``--kv-channels`` is the head width;
    Megatron otherwise derives it as ``hidden_size // num_attention_heads``,
    which is wrong for any shape whose head width is not that ratio.
    """
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
        # Megatron asserts max_position_embeddings >= seq_length, and the
        # packed sample is longer than one titan row. The rope table is
        # indexed by position, so a longer one does not move the entries a
        # shorter one held; every document still starts at position 0.
        "--max-position-embeddings",
        str(max(shape.max_seq_len, megatron_seq_length)),
        "--position-embedding-type",
        "rope",
        "--use-rotary-position-embeddings",
        "--rotary-percent",
        "1.0",
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
    """The engine and the precision.

    ``--bf16`` alone keeps fp32 master parameters, fp32 optimizer moments
    and an fp32 gradient reduction. That is what a stock user gets, and it
    is about 18 bytes of state per parameter against TorchTitan's 8. The
    arm keeps it, and ``train.py`` prints the four fields so every log
    records them.

    ``--use-mcore-models`` is deprecated in this rev and Megatron ignores
    it. Piper passes it, and the flag still parses, so the list keeps it.

    ``--no-gradient-accumulation-fusion`` matches the tuned arm, which
    declines the same fusion. Its fused weight-gradient path needs
    ``main_grad`` buffers that only a data-parallel wrap provides, so the
    fusion would otherwise be a property of the mesh and not of the arm.
    """
    return [
        "--bf16",
        "--transformer-impl",
        "transformer_engine",
        "--use-mcore-models",
        "--no-gradient-accumulation-fusion",
    ]


def _moe_flags() -> list[str]:
    """The mixture-of-experts settings.

    ``--moe-router-load-balancing-type none`` with
    ``--moe-aux-loss-coeff 0.0`` matches the TorchTitan config, whose
    ``load_balance_coeff`` is ``None``.

    ``--moe-router-dtype fp32`` is the one deliberate deviation from Piper.
    TorchTitan wraps its gate GEMM in ``torch.autocast(dtype=float32)``, so
    an unset dtype would make the router a precision difference that no rule
    catches.
    """
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


def _optimizer_flags(
    workload: Workload, *, global_batch_size: int
) -> list[str]:
    """The optimizer and the schedule, matched to TorchTitan.

    ``--micro-batch-size`` is **always 1**: one Megatron sample is one
    packed sequence of ``rows * seq_len`` tokens, for the reason
    ``microbatch_geometry`` gives. ``--global-batch-size`` is therefore
    ``microbatches * dp``, and Megatron computes ``global / (micro * dp)``
    microbatches per step, which is the count ``microbatch_geometry``
    returns and the count TorchTitan runs.

    **Whether Megatron decays over the 38 post-warmup steps, as TorchTitan
    does, or over all 40, is unverified.** Read ``OptimizerParamScheduler``
    before you report the learning rate. The rate does not change the
    throughput, so a mismatch is a reporting defect.
    """
    return [
        "--micro-batch-size",
        "1",
        "--global-batch-size",
        str(global_batch_size),
        "--train-iters",
        str(workload.steps),
        "--lr",
        LEARNING_RATE,
        "--lr-decay-style",
        "linear",
        "--lr-decay-iters",
        str(workload.steps),
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
    """The mesh Megatron resolves for itself.

    The data-parallel degree is absent on purpose. Megatron gives that axis
    every rank the other degrees leave over, so it derives ``dp`` from
    ``WORLD_SIZE``. The driver reads the resolved value back and prints it,
    which is stronger evidence than a degree the harness asserted.

    Both pipeline-split accounting flags stay off, which is their default.
    Megatron then divides ``num_layers`` evenly, and the harness sends
    TorchTitan the two ``less-layers 0`` flags that produce the same split.

    **The expert degree is sent, and it never widens the world.** Megatron
    divides the world by the tensor, pipeline and context degrees and by
    ``gtp_weight_remat_size``, which defaults to 1 and which this arm never
    sets (``arguments.py``). The expert degree is absent from that product,
    so it subdivides the data-parallel axis rather than adding a dimension.
    That is the same arithmetic ``ParallelismSpec.world_size`` states.

    **This flag is the argv, and the argv proves nothing on its own.** The
    driver reads the degree back from the group
    ``initialize_model_parallel`` built and refuses a disagreement, exactly
    as ``data.py`` already does for the data-parallel degree.
    """
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


def _sharding_flags(zero: int) -> list[str]:
    """What Megatron needs to hold the dense parameters this way.

    Empty under ``zero 0``, which is stock Megatron's own default and
    the treatment every published cell of this scenario ran.

    Under ``zero 1`` it is the one flag of ``ZERO1_FLAGS``. Megatron then
    keeps a plain ``DistributedDataParallel`` and builds a
    ``DistributedOptimizer``, which shards the optimizer states over the
    data-parallel group. It builds no device mesh, so the level holds a
    pipeline.

    This module reads no environment variable, which is what keeps a test
    able to build the whole command line without a shell.

    ``SHARDING_FLAGS_BY_VALUE`` answers the question, so an unrecorded
    level raises here rather than falling through to one of two branches.
    Every level ``ZERO_MODES`` names has a row, and
    ``tests/test_axes.py`` pins that.
    """
    return list(SHARDING_FLAGS_BY_VALUE[zero])


def _data_flags(
    shape: PiperShape,
    workload: Workload,
    *,
    profile_step_end: int | None,
) -> list[str]:
    """The data source, the logging and the profiler.

    ``NullTokenizer`` needs no file and adds no token, so
    ``--vocab-size`` alone fixes the table width.
    ``--padded-vocab-size`` sets ``should_pad_vocab`` to False, which pins
    the embedding table at exactly that many rows whatever
    ``--make-vocab-size-divisible-by`` holds. Arm rule 11 then has a
    deterministic target.

    ``--dataloader-type external`` passes the driver's own iterator through.
    Megatron's own sampler would shard a global stream by a different rule
    than TorchTitan's ``split_dataset_by_node``, so the two engines would
    put different tokens on the same rank.

    ``--dataloader-inter-document-masking`` gives Megatron the
    block-diagonal packed-document mask TorchTitan applies. Without it
    Megatron attends across document boundaries, which is more attention
    work, and the comparison would run against Megatron.

    ``--profile-step-end`` is the **last whole profiler cycle**, not the
    step count. The driver replaces the profiler schedule, so this value
    only decides when Megatron calls ``prof.stop()`` -- and a stop inside
    an active window writes a third, short trace that arm rule 5 and the
    per-step metrics would then count. Ending on a cycle boundary puts the
    stop on a step the schedule is idle on, where it is a no-op.

    ``profile_step_end`` is ``None`` when the run collects no traces, and
    the four profiler tokens then leave the argv. Megatron's own default
    for ``--profile`` is off, so the run builds no profiler at all.
    """
    profiler = (
        [
            "--profile",
            "--use-pytorch-profiler",
            "--profile-step-start",
            "1",
            "--profile-step-end",
            str(profile_step_end),
        ]
        if profile_step_end is not None
        else []
    )
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
        str(workload.seed),
        "--rerun-mode",
        "disabled",
        "--log-interval",
        "1",
        "--log-throughput",
        *profiler,
    ]


def _bench_flags(
    workload: Workload,
    spec: ParallelismSpec,
    *,
    arm_dir: str,
    model_size: str,
    rows_per_sample: int,
    megatron_p2p_sync: str,
    profile: bool,
) -> list[str]:
    """The harness group, which ``train.py`` adds to Megatron's own parser.

    ``--bench-seq-len`` is one **titan row**, where ``--seq-length`` is the
    packed sample. The driver needs the row length for the token count, the
    flops denominator and the data builder, and it cannot recover it from
    ``--seq-length`` alone.

    ``--bench-rows-per-sample`` states the packing, so the driver can assert
    ``rows * bench_seq_len == seq_length`` rather than divide and hope.

    ``--bench-min-trace-windows`` carries the workload's own requirement,
    so the driver's window guard reads arm rule 5's number instead of an
    arithmetic that can evaluate to zero.

    ``--bench-pp-schedule`` is omitted at ``pp`` 1, where the driver refuses
    it: a schedule name there would name a split that does not happen.

    ``--bench-batch-p2p-sync`` is omitted at ``on``, which is Megatron's own
    default for the field. The default argv therefore does not move, and a
    reader of a recorded command line sees the flag exactly where the run
    turned the sync off.

    ``--bench-profile`` and the four schedule flags are one group: every
    member of it appears under ``profile`` and none of it otherwise. The
    driver refuses a schedule flag without the token, so a partial group
    cannot install a profiler the run does not declare.
    """
    schedule = (
        [
            BENCH_PROFILE,
            BENCH_PROFILE_FREQ,
            str(workload.profile_freq),
            BENCH_PROFILER_WARMUP,
            str(workload.profiler_warmup),
            BENCH_PROFILER_ACTIVE,
            str(workload.profiler_active),
            BENCH_MIN_TRACE_WINDOWS,
            str(workload.min_trace_windows),
        ]
        if profile
        else []
    )
    flags = [
        BENCH_ARM_DIR,
        str(arm_dir),
        BENCH_MODEL_SIZE,
        model_size,
        BENCH_LOCAL_BATCH_SIZE,
        str(workload.local_batch_size),
        *schedule,
        BENCH_SEQ_LEN,
        str(workload.seq_len),
        BENCH_ROWS_PER_SAMPLE,
        str(rows_per_sample),
    ]
    if spec.pp > 1:
        flags.extend((BENCH_PP_SCHEDULE, str(spec.pp_schedule)))
    # The literal, never the axis default: the driver reads ``on`` when
    # the token is absent, so the token has to follow the treatment.
    #
    # Gated on the pipeline too. There is no pipeline message to
    # synchronize at ``pp`` 1, so the token would name a treatment the run
    # did not have, and the driver refuses it there.
    if spec.pp > 1 and megatron_p2p_sync == "off":
        flags.extend((BENCH_BATCH_P2P_SYNC, megatron_p2p_sync))
    return flags


def stock_megatron_flags(
    shape: PiperShape,
    workload: Workload,
    spec: ParallelismSpec,
    *,
    arm_dir: str,
    model_size: str,
    megatron_p2p_sync: str = DEFAULT_MEGATRON_P2P_SYNC,
    megatron_nan_guard: str = DEFAULT_MEGATRON_NAN_GUARD,
    megatron_precision: str = DEFAULT_MEGATRON_PRECISION,
    profile: bool = True,
) -> list[str]:
    """The whole argument list for one stock Megatron-LM arm.

    The result holds the Megatron flags and then the harness group. It holds
    neither the launcher nor the module name: ``benchmarks/e2e/launch.py``
    puts ``_megatron_launcher(spec)`` and
    ``benchmarks.e2e.megatron_stock.train`` in front of it.

    ``arm_dir`` accepts a string or a ``pathlib.Path``; the function renders
    it with ``str``. This module imports no ``pathlib``, so a caller in a
    torch-free process pays for nothing it does not use.

    ``megatron_p2p_sync`` defaults to ``off``, which adds the harness
    token. ``on`` is refused at ``pp`` 1: the field is inert without a
    pipeline message, and the argv would carry a treatment the run did not
    have.

    ``megatron_nan_guard`` defaults to ``off``, which adds Megatron's own
    ``--no-check-for-nan-in-loss-and-grad`` ahead of the harness group. It
    is legal at every mesh under either value.

    ``megatron_precision`` defaults to ``stock``, which sends nothing and
    is 18 bytes of optimizer state per parameter. ``lean`` adds the four
    flags of ``LEAN_PRECISION_FLAGS`` and reaches 10 bytes. It is refused
    under ``--zero 0``, because Megatron asserts
    ``use_distributed_optimizer`` under the precision-aware optimizer.

    ``profile`` defaults to ``True``, which is the argv this arm has always
    built. Under ``False`` Megatron's own profiler flags and the harness
    schedule group both leave, the run writes no trace, and the whole-cycle
    refusal below does not apply: it exists to keep a profiler window
    whole, and there is no window.

    Raises ``ValueError`` on a request this arm cannot honour. Each refusal
    names the reason, because a caller may build a command line without a
    run and a bare failure names nothing.
    """
    refuse_unknown_p2p_sync(megatron_p2p_sync)
    refuse_unknown_nan_guard(megatron_nan_guard)
    refuse_unknown_megatron_precision(megatron_precision)
    if spec.pp == 1 and megatron_p2p_sync == "on":
        raise ValueError(
            f"megatron p2p sync {megatron_p2p_sync!r} was requested at pp 1, "
            "where there is no pipeline message to synchronize; the argv "
            "would carry a treatment the run did not have"
        )
    if workload.seed is None:
        raise ValueError(
            "the stock megatron arm needs a seeded workload: both engines "
            "must draw the same initial parameters"
        )
    if spec.ep > 1 and spec.zero == 0:
        raise ValueError(
            f"expert-parallel degree {spec.ep} needs --zero 1: TorchTitan "
            "cannot split the experts while it replicates the dense "
            "parameters, so a replicated expert row would compare two "
            "different memory strategies"
        )
    if spec.pp > 1 and spec.pp_schedule != SUPPORTED_PP_SCHEDULE:
        raise ValueError(
            f"pipeline schedule {spec.pp_schedule!r} is not implemented by "
            f"the stock driver; it runs {SUPPORTED_PP_SCHEDULE!r} alone"
        )
    if profile and workload.steps % workload.profile_freq:
        # **This arm rides Megatron's own loop, and that loop keeps calling
        # prof.step() after it has called prof.stop().** The stop is guarded
        # on ``iteration == --profile-step-end``; the step at the top of
        # the loop body is guarded only on ``--profile``. So every iteration
        # after the stop transits a dead Kineto session.
        #
        # Ending the profiler on the last whole cycle does not fix that. It
        # only delays the first bad transit: measured against the pinned
        # torch, a 50-step run reaches ``NONE -> WARMUP`` on a stopped
        # profiler, and a 57-step run reaches eight such transits. Ending it
        # at ``--train-iters`` instead trades them for a **truncated**
        # window -- 3 recorded steps rather than 5 at 57 steps -- which
        # ``assert_windows_written`` does not catch, because it refuses a
        # count below the requirement and not a short window. A truncated
        # window is pooled with the full ones and moves every per-step
        # figure.
        #
        # A whole number of cycles removes the case: ``--profile-step-end``
        # is then ``--train-iters``, no iteration follows the stop, and
        # every window holds its full ``profiler_active`` steps. Verified
        # against the pinned torch at 40, 60, 80, 100 and 200 steps.
        raise ValueError(
            f"steps ({workload.steps}) must be a whole number of profiler "
            f"cycles of {workload.profile_freq} for the stock megatron arm: "
            "megatron steps the profiler after it stops it, so a partial "
            "cycle either transits a dead session or writes a short "
            "profiler window that the per-step metrics would pool"
        )
    rows_per_sample, microbatches, megatron_seq_length = (
        microbatch_geometry(workload, spec)
    )
    # A whole number of cycles, refused above, so this is workload.steps.
    # See _data_flags, and the refusal above for why it may not be less.
    # ``None`` under no profile, which drops the profiler tokens.
    profile_step_end = (
        (workload.steps // workload.profile_freq) * workload.profile_freq
        if profile
        else None
    )
    return [
        *_geometry_flags(shape, megatron_seq_length=megatron_seq_length),
        *_engine_flags(),
        *_moe_flags(),
        *_optimizer_flags(
            workload, global_batch_size=microbatches * spec.dp
        ),
        *_mesh_flags(spec),
        *_sharding_flags(spec.zero),
        *_precision_flags(megatron_precision, spec.zero),
        *_data_flags(
            shape, workload, profile_step_end=profile_step_end
        ),
        *_nan_guard_flags(megatron_nan_guard),
        *_bench_flags(
            workload,
            spec,
            arm_dir=arm_dir,
            model_size=model_size,
            rows_per_sample=rows_per_sample,
            megatron_p2p_sync=megatron_p2p_sync,
            profile=profile,
        ),
    ]
