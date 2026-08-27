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
    import argparse, sys, typing, typing_extensions
    sys.path.insert(0, "third_party/Megatron-LM")
    if not hasattr(typing, "override"):
        typing.override = typing_extensions.override
    from megatron.training.arguments import add_megatron_arguments
    from benchmarks.e2e.megatron_stock.flags import stock_megatron_flags
    from benchmarks.e2e.parallelism import TRIVIAL_SPEC
    from benchmarks.e2e.registry import PIPER_1B_MEGATRON_WORKLOAD
    from benchmarks.models.piper_qwen3.shape import PIPER_1B
    parser = argparse.ArgumentParser(allow_abbrev=False)
    add_megatron_arguments(parser)
    known = {s for a in parser._actions for s in a.option_strings}
    emitted = stock_megatron_flags(
        PIPER_1B, PIPER_1B_MEGATRON_WORKLOAD, TRIVIAL_SPEC,
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
deviations that section 7 of ``PIPER_STOCK_MEGATRON_PLAN.md`` records:
``--moe-router-dtype fp32`` (TorchTitan routes fp32 too, so an unset dtype
would make the router a precision difference), no attention-backend flag
(TransformerEngine then resolves to the same cuDNN kernel the tuned arm
runs), and no distributed optimizer under ``--dense-sharding replicate``
(both engines then replicate their parameters, so the data-parallel axis
carries one change).

**``--dense-sharding shard`` moves that third deviation and nothing else.**
Both engines then shard the dense parameters, so the axis still carries one
change, and Megatron needs its own distributed optimizer to do it. See
``SHARDING_FLAGS``.
"""

from __future__ import annotations

from benchmarks.e2e.parallelism import (
    DENSE_SHARDING_MODES,
    ParallelismSpec,
)
from benchmarks.e2e.registry import Workload
from benchmarks.models.piper_qwen3.shape import PiperShape

# The one compile mode this arm accepts. Megatron compiles no whole
# transformer layer, so there is no treatment to turn off, and the scenario
# declares this mode alone.
SUPPORTED_MODE = "default"

# The one pipeline schedule the stock driver runs. Megatron's
# forward_backward_pipelining_without_interleaving is 1F1B and nothing else.
SUPPORTED_PP_SCHEDULE = "1F1B"

# **Megatron-FSDP runs at version 1, and version 2 is impossible here.**
# FullyShardedDataParallelV2._validate_config raises on a pipeline degree, an
# expert degree, a tensor degree, a context degree, and on any config whose
# num_moe_experts is set (mcore_fsdp_adapter.py). Every shape in
# PIPER_SHAPES is a mixture of experts, so v2 refuses this suite even at
# pp 1 and ep 1. Megatron already defaults this value to 1. The flag list
# states the fact rather than inherits it. A submodule bump that moves the
# default then changes a recorded argv rather than a silent run.
MEGATRON_FSDP_VERSION = "1"

# What Megatron-FSDP shards. It pairs with what TorchTitan shards under
# --dense-sharding shard: the parameters, the gradients and the optimizer
# state. Megatron already defaults this value too. The flag list states it
# for the reason above.
MEGATRON_SHARDING_STRATEGY = "optim_grads_params"

# arguments.py asserts ckpt_format == "fsdp_dtensor" under
# --use-megatron-fsdp. The arm saves no checkpoint, so the flag is inert
# except for that assert.
MEGATRON_CHECKPOINT_FORMAT = "fsdp_dtensor"

# The wrapper class Megatron builds for each value, and the sharding
# strategy that wrapper then acts on. train.py prints both off the wrapper
# it really got; benchmarks/e2e/validation.py reads this table to say what
# each value must print. **The class name follows the version constant**,
# because training.py picks FullyShardedDataParallel and that factory picks
# the class from ddp_config.megatron_fsdp_version.
#
# **The strategy is "no_shard" under replicate, and Megatron's own argparse
# default is not.** args.data_parallel_sharding_strategy defaults to
# "optim_grads_params" and reaches the DDP config whatever the wrapper is,
# but megatron/core/optimizer/__init__.py reads it only under
# use_megatron_fsdp. So the value the run acts on is "no_shard" here, and
# the raw field is inert. A marker built from the raw field would say a
# replicated run sharded.
DATA_PARALLEL_WRAPPERS: dict[str, str] = {
    "replicate": "DistributedDataParallel",
    "shard": f"FullyShardedDataParallelV{MEGATRON_FSDP_VERSION}",
}
SHARDING_STRATEGIES: dict[str, str] = {
    "replicate": "no_shard",
    "shard": MEGATRON_SHARDING_STRATEGY,
}

# The strategies under which Megatron-FSDP turns the gradient overlap on
# by itself. ``MegatronFSDP.__init__`` reads this exact list
# (``megatron_fsdp.py``), and it mutates the very object
# ``get_megatron_ddp_config`` built, because it keeps the reference rather
# than a copy (see ``mcore_fsdp_adapter.py``'s own
# ``self.ddp_config = ddp_config``). So a sharded run reports
# ``overlap_grad_reduce=True`` even though the argv omits the flag.
MEGATRON_FSDP_GRAD_OVERLAP_STRATEGIES: tuple[str, ...] = (
    "optim_grads_params",
    "optim_grads",
)

# What ``overlap_grad_reduce`` reads on the wrapper, per value.
#
# **It is DERIVED from the strategy, and it may not restate it.** Megatron
# keys the mutation on ``data_parallel_sharding_strategy``, not on this
# repo's dense-sharding value. ``MEGATRON_SHARDING_STRATEGY`` is a
# documented reversal target, so a hand-written table here would keep
# saying True after somebody moved that constant to a strategy the list
# above does not hold -- and arm rule 12 would then fail the first run
# after the flip. One statement of the fact, in one place.
#
# **``replicate`` reads False for two independent reasons.** Its strategy
# is ``no_shard``, which the list above does not hold; and no
# Megatron-FSDP wrapper exists at all under that value, so the mutation
# never runs. Do not read that row as a coincidence of the derivation.
DATA_PARALLEL_OVERLAP: dict[str, bool] = {
    value: strategy in MEGATRON_FSDP_GRAD_OVERLAP_STRATEGIES
    for value, strategy in SHARDING_STRATEGIES.items()
}

# TorchTitan's own optimizer values, replicated flag for flag. The source is
# benchmarks/e2e/megatron/train.py, which replicates the TorchTitan trainer.
LEARNING_RATE = "8e-4"
LR_WARMUP_ITERS = "2"
ADAM_BETA1 = "0.9"
ADAM_BETA2 = "0.95"
ADAM_EPS = "1e-8"
WEIGHT_DECAY = "0.1"
CLIP_GRAD = "1.0"

# The RMSNorm epsilon the TorchTitan config carries.
NORM_EPSILON = "1e-6"

# Piper's own initialization width.
INIT_METHOD_STD = "0.01"

# The harness flag names. train.py declares the same names, and a test
# compares the two lists, so a rename cannot reach only one side.
BENCH_ARM_DIR = "--bench-arm-dir"
BENCH_MODEL_SIZE = "--bench-model-size"
BENCH_LOCAL_BATCH_SIZE = "--bench-local-batch-size"
BENCH_PROFILE_FREQ = "--bench-profile-freq"
BENCH_PROFILER_WARMUP = "--bench-profiler-warmup"
BENCH_PROFILER_ACTIVE = "--bench-profiler-active"
BENCH_MODE = "--bench-mode"
BENCH_PP_SCHEDULE = "--bench-pp-schedule"
BENCH_SEQ_LEN = "--bench-seq-len"
BENCH_ROWS_PER_SAMPLE = "--bench-rows-per-sample"
BENCH_MIN_TRACE_WINDOWS = "--bench-min-trace-windows"

BENCH_FLAGS: tuple[str, ...] = (
    BENCH_ARM_DIR,
    BENCH_MODEL_SIZE,
    BENCH_LOCAL_BATCH_SIZE,
    BENCH_PROFILE_FREQ,
    BENCH_PROFILER_WARMUP,
    BENCH_PROFILER_ACTIVE,
    BENCH_MODE,
    BENCH_PP_SCHEDULE,
    BENCH_SEQ_LEN,
    BENCH_ROWS_PER_SAMPLE,
    BENCH_MIN_TRACE_WINDOWS,
)

# Flags this suite declines under EVERY dense-sharding value, each for a
# reason section 7 of PIPER_STOCK_MEGATRON_PLAN.md states. The tuple exists
# so a test can assert their absence by name rather than by a hand-written
# list that can drift from the reason.
#
# **--overlap-grad-reduce and --overlap-param-gather stay here under both
# values, and the reason is not the reason --use-distributed-optimizer
# leaves.** Megatron-FSDP turns all three on itself, so the "an argv that
# omits it would deny a fact the run has" argument appears to reach all
# three. It does not.
#
# ``arguments.py`` sets ``args.use_distributed_optimizer = True`` inside
# the Megatron-FSDP block whatever the argv said. Sending that flag
# therefore changes nothing the run does, and the argv gains a fact.
#
# The two overlap flags travel together: ``arguments.py`` asserts
# ``--overlap-param-gather`` needs ``--overlap-grad-reduce``, so neither
# can be sent alone. And ``training.py`` reads ``overlap_grad_reduce``
# **before** it builds the wrapper, in ``resolve_ddp_bucket_size``, which
# returns ``None`` when the value is False. So sending the pair moves the
# gradient bucket size. That changes what the run does rather than what
# the argv says, and this arm runs stock Megatron at its own defaults.
#
# The marker reads the resolved value instead. See DATA_PARALLEL_OVERLAP.
#
# **The consequence under ``shard`` is a caption obligation, not a defect.**
# ``resolve_ddp_bucket_size`` runs before the wrapper exists and reads the
# argument, which is False, so a sharded run enters Megatron-FSDP with
# ``bucket_size = None``. The wrapper then flips ``overlap_grad_reduce`` to
# True on the config it holds. So the sharded arm overlaps its gradient
# reduction and buckets it at Megatron's unbucketed default, and **whether
# that costs anything is unmeasured**. Say so beside any sharded number.
# Sending the pair to bucket it would change the run rather than the record,
# which is what the paragraph above refuses.
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
    "--use-precision-aware-optimizer",
    "--profile-ranks",
)

# The five flags --dense-sharding shard sends and replicate declines. The
# same tuple states both facts, so the two cannot drift apart.
#
# **--use-distributed-optimizer is one of them, and it must be.**
# Megatron-FSDP v1 turns it on itself and warns (arguments.py). An argv that
# omitted it would deny a fact the run has.
SHARDING_FLAGS: tuple[str, ...] = (
    "--use-megatron-fsdp",
    "--megatron-fsdp-version",
    "--data-parallel-sharding-strategy",
    "--use-distributed-optimizer",
    "--ckpt-format",
)


def refuse_unknown_dense_sharding(dense_sharding: str) -> None:
    """Raise on a value this module cannot build a command line for.

    ``ParallelismSpec`` is meant to refuse one first, and this module does
    not depend on that: a caller may build a command line without a run,
    and a silent fall through to the ``replicate`` branch would send the
    replicated argv under the sharded label.
    """
    if dense_sharding not in DENSE_SHARDING_MODES:
        raise ValueError(
            f"dense sharding {dense_sharding!r} is not one of "
            + ", ".join(repr(mode) for mode in DENSE_SHARDING_MODES)
        )


def omitted_flags(dense_sharding: str) -> tuple[str, ...]:
    """Every flag this arm declines at ``dense_sharding``.

    The roster is a function of the value because all five
    sharding flags move from declined to required under ``shard``. A test
    reads this rather than a hand-written list, so the absence under
    ``replicate`` and the presence under ``shard`` are both asserted by
    name.
    """
    refuse_unknown_dense_sharding(dense_sharding)
    if dense_sharding == "shard":
        return ALWAYS_OMITTED_FLAGS
    return ALWAYS_OMITTED_FLAGS + SHARDING_FLAGS


def microbatch_geometry(
    workload: Workload, spec: ParallelismSpec
) -> tuple[int, int, int]:
    """``(rows per sample, microbatches per step, megatron seq_length)``.

    **One Megatron sample is one packed sequence, never a batch of rows.**
    Megatron flattens a ``(m, S)`` microbatch into ``(1, m*S)`` whenever
    ``cu_seqlens`` is present (``megatron/core/utils.py``'s
    ``flatten_batch_for_packed_sequences``), so the activation the first
    stage sends is ``(m*S, 1, H)``. The pipeline allocates its receive
    buffer from ``get_tensor_shapes``, which returns ``(S, m, H)`` and
    validates nothing. The two hold the same number of elements, so the
    transfer succeeds and the next stage reads a **permuted** activation.
    Measured on this rev with ``m`` 4: ``(64, 1, 8)`` sent into a
    ``(16, 4, 8)`` buffer.

    So the harness packs the rows itself, exactly as the tuned megatron
    driver does, and tells Megatron the sample is one row of ``rows * S``
    tokens. Every shape then agrees, and the attention is unchanged:
    ``cu_seqlens`` already marks every document, and a row boundary is a
    document boundary.

    The row count follows the other engine at both meshes. Under a pipeline
    the microbatch size is the split TorchTitan is given. Without one
    neither engine splits, so the whole batch is one pack -- which is what
    ``benchmarks/e2e/megatron/train.py``'s ``pipeline_settings`` returns and
    what every published megatron number was measured on.
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


def _sharding_flags(dense_sharding: str) -> list[str]:
    """What Megatron needs to hold the dense parameters this way.

    Empty under ``replicate``, which is stock Megatron's own default and
    the treatment every published cell of this scenario ran.

    Under ``shard`` it is the five flags of ``SHARDING_FLAGS``. Two of them
    restate a Megatron default on purpose, so a submodule bump that moves
    either default changes a recorded argv rather than a silent run.

    **Megatron-FSDP v1 accepts a pipeline degree and an expert degree.**
    Its distributed index reads ``expt_dp_group`` and ``ep_group``
    (``mcore_fsdp_adapter.py``), and no assert in ``arguments.py`` forbids
    the combination for v1. Version 2 refuses every shape here; see
    ``MEGATRON_FSDP_VERSION``.

    **One precondition lives outside this module, and it is enforced.**
    ``arguments.py`` asserts ``CUDA_DEVICE_MAX_CONNECTIONS != "1"`` under
    ``--use-megatron-fsdp``. Nothing under ``benchmarks/`` sets that
    variable, but the child inherits the operator's own shell, so an
    ambient ``"1"`` would kill every rank at argument parsing.
    ``benchmarks/execution/environment.py``'s
    ``refuse_megatron_fsdp_connection_limit`` refuses such a host, and
    ``_resolve_run`` calls it after it builds the argv -- reading the flag
    off this list rather than re-deriving the condition, so the two cannot
    drift apart. This module reads no environment variable, which is what
    keeps a test able to build the whole command line without a shell.
    """
    refuse_unknown_dense_sharding(dense_sharding)
    if dense_sharding != "shard":
        return []
    return [
        "--use-megatron-fsdp",
        "--megatron-fsdp-version",
        MEGATRON_FSDP_VERSION,
        "--data-parallel-sharding-strategy",
        MEGATRON_SHARDING_STRATEGY,
        "--use-distributed-optimizer",
        "--ckpt-format",
        MEGATRON_CHECKPOINT_FORMAT,
    ]


def _data_flags(
    shape: PiperShape, workload: Workload, *, profile_step_end: int
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
    """
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
        "--profile",
        "--use-pytorch-profiler",
        "--profile-step-start",
        "1",
        "--profile-step-end",
        str(profile_step_end),
    ]


def _bench_flags(
    workload: Workload,
    spec: ParallelismSpec,
    *,
    arm_dir: str,
    model_size: str,
    compile_mode: str,
    rows_per_sample: int,
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
    """
    flags = [
        BENCH_ARM_DIR,
        str(arm_dir),
        BENCH_MODEL_SIZE,
        model_size,
        BENCH_LOCAL_BATCH_SIZE,
        str(workload.local_batch_size),
        BENCH_PROFILE_FREQ,
        str(workload.profile_freq),
        BENCH_PROFILER_WARMUP,
        str(workload.profiler_warmup),
        BENCH_PROFILER_ACTIVE,
        str(workload.profiler_active),
        BENCH_MODE,
        compile_mode,
        BENCH_SEQ_LEN,
        str(workload.seq_len),
        BENCH_ROWS_PER_SAMPLE,
        str(rows_per_sample),
        BENCH_MIN_TRACE_WINDOWS,
        str(workload.min_trace_windows),
    ]
    if spec.pp > 1:
        flags.extend((BENCH_PP_SCHEDULE, str(spec.pp_schedule)))
    return flags


def stock_megatron_flags(
    shape: PiperShape,
    workload: Workload,
    spec: ParallelismSpec,
    *,
    arm_dir: str,
    model_size: str,
    compile_mode: str = SUPPORTED_MODE,
) -> list[str]:
    """The whole argument list for one stock Megatron-LM arm.

    The result holds the Megatron flags and then the harness group. It holds
    neither the launcher nor the module name: ``benchmarks/e2e/launch.py``
    puts ``_megatron_launcher(spec)`` and
    ``benchmarks.e2e.megatron_stock.train`` in front of it.

    ``arm_dir`` accepts a string or a ``pathlib.Path``; the function renders
    it with ``str``. This module imports no ``pathlib``, so a caller in a
    torch-free process pays for nothing it does not use.

    Raises ``ValueError`` on a request this arm cannot honour. Each refusal
    names the reason, because a caller may build a command line without a
    run and a bare failure names nothing.
    """
    if workload.seed is None:
        raise ValueError(
            "the stock megatron arm needs a seeded workload: both engines "
            "must draw the same initial parameters"
        )
    if compile_mode != SUPPORTED_MODE:
        raise ValueError(
            f"compile mode {compile_mode!r} names a whole-block "
            "torch.compile treatment, and Megatron never has one; the stock "
            f"arm runs at {SUPPORTED_MODE!r} alone"
        )
    refuse_unknown_dense_sharding(spec.dense_sharding)
    if spec.ep > 1 and spec.dense_sharding != "shard":
        raise ValueError(
            f"expert-parallel degree {spec.ep} needs "
            "--dense-sharding shard: TorchTitan cannot split the experts "
            "while it replicates the dense parameters, so a replicated "
            "expert row would compare two different memory strategies"
        )
    if spec.pp > 1 and spec.pp_schedule != SUPPORTED_PP_SCHEDULE:
        raise ValueError(
            f"pipeline schedule {spec.pp_schedule!r} is not implemented by "
            f"the stock driver; it runs {SUPPORTED_PP_SCHEDULE!r} alone"
        )
    if workload.steps % workload.profile_freq:
        # **This arm rides Megatron's own loop, and that loop keeps calling
        # prof.step() after it has called prof.stop().** The stop is guarded
        # on ``iteration == --profile-step-end``
        # (``megatron/training/training.py``); the step at the top of the
        # loop body is guarded only on ``--profile``. So every iteration
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
    profile_step_end = (
        workload.steps // workload.profile_freq
    ) * workload.profile_freq
    return [
        *_geometry_flags(shape, megatron_seq_length=megatron_seq_length),
        *_engine_flags(),
        *_moe_flags(),
        *_optimizer_flags(
            workload, global_batch_size=microbatches * spec.dp
        ),
        *_mesh_flags(spec),
        *_sharding_flags(spec.dense_sharding),
        *_data_flags(
            shape, workload, profile_step_end=profile_step_end
        ),
        *_bench_flags(
            workload,
            spec,
            arm_dir=arm_dir,
            model_size=model_size,
            compile_mode=compile_mode,
            rows_per_sample=rows_per_sample,
        ),
    ]
