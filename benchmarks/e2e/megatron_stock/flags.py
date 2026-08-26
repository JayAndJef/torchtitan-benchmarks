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
runs), and no distributed optimizer (both engines replicate their
parameters, so the data-parallel axis carries one change).
"""

from __future__ import annotations

from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.registry import Workload
from benchmarks.models.piper_qwen3.shape import PiperShape

# The one compile mode this arm accepts. Megatron compiles no whole
# transformer layer, so there is no treatment to turn off, and the scenario
# declares this mode alone.
SUPPORTED_MODE = "default"

# The one pipeline schedule the stock driver runs. Megatron's
# forward_backward_pipelining_without_interleaving is 1F1B and nothing else.
SUPPORTED_PP_SCHEDULE = "1F1B"

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
)

# Flags this suite declines, each for a reason section 7 of the plan states.
# The tuple exists so a test can assert their absence by name rather than by
# a hand-written list that can drift from the reason.
OMITTED_FLAGS: tuple[str, ...] = (
    "--use-distributed-optimizer",
    "--overlap-grad-reduce",
    "--overlap-param-gather",
    "--data-parallel-sharding-strategy",
    "--use-megatron-fsdp",
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
    """
    return [
        "--tensor-model-parallel-size",
        "1",
        "--context-parallel-size",
        "1",
        "--expert-model-parallel-size",
        "1",
        "--pipeline-model-parallel-size",
        str(spec.pp),
    ]


def _data_flags(shape: PiperShape, workload: Workload) -> list[str]:
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

    ``--profile-step-start 1`` and ``--profile-step-end <steps>`` bracket
    the whole run. The driver replaces the profiler schedule, so these two
    values only decide when Megatron calls ``prof.stop()``.
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
        str(workload.steps),
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
    if spec.ep > 1:
        raise ValueError(
            f"expert-parallel degree {spec.ep} is not delivered to the stock "
            "megatron arm: --expert-model-parallel-size stays 1, so the "
            "manifest could not record the run it really did"
        )
    if spec.pp > 1 and spec.pp_schedule != SUPPORTED_PP_SCHEDULE:
        raise ValueError(
            f"pipeline schedule {spec.pp_schedule!r} is not implemented by "
            f"the stock driver; it runs {SUPPORTED_PP_SCHEDULE!r} alone"
        )
    rows_per_sample, microbatches, megatron_seq_length = (
        microbatch_geometry(workload, spec)
    )
    return [
        *_geometry_flags(shape, megatron_seq_length=megatron_seq_length),
        *_engine_flags(),
        *_moe_flags(),
        *_optimizer_flags(
            workload, global_batch_size=microbatches * spec.dp
        ),
        *_mesh_flags(spec),
        *_data_flags(shape, workload),
        *_bench_flags(
            workload,
            spec,
            arm_dir=arm_dir,
            model_size=model_size,
            compile_mode=compile_mode,
            rows_per_sample=rows_per_sample,
        ),
    ]
