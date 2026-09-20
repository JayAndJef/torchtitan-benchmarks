"""Stock Megatron-LM, driven through its own ``pretrain`` entry point.

``pretrain_gpt.py``'s ``__main__`` block calls ``parse_and_validate_args``,
``gpt_config_from_args``, ``pretrain_cfg_container_from_args`` and then
``pretrain``. Every one of those is importable, and every provider it hands
to ``pretrain`` is a module-level function. This driver reproduces that
block and substitutes **one** argument: the dataset provider, so that both
engines read the same c4_test stream rank for rank.

The model builder, the optimizer, the learning-rate schedule, the
distributed setup, the forward step, the embedding-rank rule and the
training loop all stay Megatron's. That is the point of the arm: the
driver replicates none of the TorchTitan treatment.

**No file under ``third_party/`` is edited.** Three shims run in this
process instead:

* ``bootstrap.install_typing_override`` adds one name Python 3.10 lacks;
* ``profiling.install_profiler_shim`` gives Megatron the trace schedule and
  the trace path the harness reads; and
* ``install_step_log_shim`` below adds the step line
  ``benchmarks/e2e/results.py`` parses.

**One config field takes a value Megatron has no flag for.**
``--bench-batch-p2p-sync off`` makes ``apply_p2p_sync`` set
``args.batch_p2p_sync = False`` after Megatron has parsed its arguments.
``core_transformer_config_from_args`` copies every ``args`` attribute whose
name is a config field, so the value reaches ``TransformerConfig`` through
Megatron's own path. The driver then prints a ``Megatron-LM stock p2p:``
line from the config it really built, on every rank.

**One Megatron field is printed back, so the log records the guard.**
``--megatron-nan-guard off`` reaches this driver as Megatron's own
``--no-check-for-nan-in-loss-and-grad``, and nothing here reads the harness
value. ``nan_guard_line`` prints ``args.check_for_nan_in_loss_and_grad`` as
Megatron PARSED it, on every rank, so a run whose argv lost the token, or
whose Megatron turned the field off by itself, prints a value the requested
one does not match. Arm rule 12 reads the line at every mesh.

**This arm is not plain bf16.** ``--bf16`` alone keeps fp32 master
parameters, fp32 optimizer moments and an fp32 gradient reduction, which is
about 18 bytes of state per parameter against TorchTitan's 8. The arm keeps
the stock defaults, because Piper ran them and a stock user gets them. The
mode line prints those fields so every log records the difference, and
the manifest's ``execution_model`` -- which is composed from the harness
spec and says ``plain-bf16`` -- describes the TorchTitan arm and not this
one.

**Importing this module costs no torch and no megatron.** Every heavy
import sits inside ``main``. That keeps the module readable from the
parent, and it is what lets a test read the marker strings on a host with
no Megatron-LM checkout. It does **not** make ``--help`` cheap: the parser
is Megatron's own, so a help run pays for the ML stack like any other.
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any, Callable, MutableMapping

from benchmarks.e2e.megatron_stock import bootstrap
from benchmarks.e2e.megatron_stock.flags import (
    BENCH_ARM_DIR,
    BENCH_BATCH_P2P_SYNC,
    BENCH_LOCAL_BATCH_SIZE,
    BENCH_MIN_TRACE_WINDOWS,
    BENCH_MODE,
    BENCH_MODEL_SIZE,
    BENCH_PP_SCHEDULE,
    BENCH_PROFILE_FREQ,
    BENCH_PROFILER_ACTIVE,
    BENCH_PROFILER_WARMUP,
    BENCH_ROWS_PER_SAMPLE,
    BENCH_SEQ_LEN,
    SUPPORTED_MODE,
    SUPPORTED_PP_SCHEDULE,
)
from benchmarks.e2e.registry import (
    DEFAULT_MEGATRON_P2P_SYNC,
    MEGATRON_P2P_SYNC_MODES,
)

# --------------------------------------------------------------------------
# The log-line contract with benchmarks/e2e/validation.py's megatron_stock
# validation profile and benchmarks/e2e/results.py's STEP_METRICS regex.
# A one-character difference fails a real run at validation time. A test
# compares these constants against the profile's own strings.
# --------------------------------------------------------------------------

# Arm rule 8. The profile matches the prefix up to the first comma.
#
# **The four precision fields are the --megatron-precision half of arm rule
# 12.** Megatron resolves every one of them before it builds the optimizer,
# and it maps each dtype string to a torch.dtype (arguments.py's dtype_map),
# so this line prints "torch.bfloat16" where the flag says "bf16". The
# profile asks for the four under both values: a run whose argv lost the
# lean flags prints fp32 under a lean label, and a run that gained them
# prints bf16 under a stock label. Either fails.
#
# main_params_dtype stays torch.float32 under both values on purpose. The
# recipe never sends --main-params-dtype, which Megatron restricts to fp32
# and fp16, and store_param_remainders already holds the master copy at 2
# bytes for each parameter.
MODE_LINE = (
    "Megatron-LM stock training loop (mode={mode}, "
    "main_params_dtype={main_params_dtype}, "
    "main_grads_dtype={main_grads_dtype}, "
    "use_precision_aware_optimizer={precision_aware}, "
    "exp_avg_dtype={exp_avg_dtype}, "
    "exp_avg_sq_dtype={exp_avg_sq_dtype}, "
    "accumulate_allreduce_grads_in_fp32={accumulate}, "
    "cross_entropy_loss_fusion={cross_entropy_loss_fusion}, "
    "moe_token_dispatcher_type={dispatcher})"
)

# Arm rule 12. Every rank prints it above world size 1.
#
# ``dp`` and ``microbatches`` are what Megatron itself resolved. ``pp`` and
# ``ep`` are the two degrees it takes verbatim from the command line, and
# ``schedule`` and ``stages`` are literals the driver refuses every other
# value of. The titan mesh line names ``ep`` too, so arm rule 12 reads the
# same three degrees on both engines.
#
# **``ep`` here restates the argv, and the data-parallel line below proves
# it.** ``install_data_parallel_marker`` reads the expert group
# ``initialize_model_parallel`` really built and refuses a disagreement.
# This line cannot do that itself: it prints before ``pretrain()`` runs, so
# no process group exists yet.
PARALLELISM_LINE = (
    "Megatron-LM stock parallelism: dp={dp} pp={pp} ep={ep} "
    "schedule={schedule} microbatches={microbatches} stages={stages}"
)

# The data-parallel half of arm rule 12, printed above dp 1.
#
# **Every value in it observes the wrapper or the group.**
# install_data_parallel_marker reads the object Megatron really built, and
# its real ddp_config, after setup_model_and_optimizer returns. A run whose
# wrapper went missing raises there rather than printing the line.
#
# **{wrapper} is the class name, and it names the mechanism.** Megatron
# picks one of three wrapper classes from the arguments (training.py).
# Each of the three derives directly from _BaseDataParallel. None of them
# derives from another. So the class name is what says which memory
# strategy ran. The word "DistributedDataParallel" cannot be hardcoded
# here, because it is wrong under --dense-sharding zero3.
#
# **{sharding} is the strategy the run acts on, not the raw field.**
# Megatron's argparse defaults data_parallel_sharding_strategy to
# "optim_grads_params" and copies it into every ddp_config, but
# megatron/core/optimizer/__init__.py reads it only under
# use_megatron_fsdp. So the raw field says "optim_grads_params" on a
# replicated run that shards nothing, and this line reports "no_shard"
# there. Both halves are read off the wrapper's own config.
#
# **{expert} is the expert group's real width**, from
# mpu.get_expert_model_parallel_world_size(). The expert degree carves its
# ranks out of the data-parallel axis, so this is its line.
#
# Arm rule 13 cannot cover for a declared line, which is why every value
# here is an observation: stock Megatron all-reduces the loss over its
# data-parallel group on every last-stage rank every step (training.py's
# train_step), so ncclDevKernel_AllReduce appears whether or not a gradient
# was reduced.
# **{optimizer} is the optimizer class Megatron really built**, and it is
# what separates the two sharded values. --use-distributed-optimizer alone
# gives ZeRO-1, where the wrapper stays DistributedDataParallel exactly as
# it is under replicate; the wrapper class therefore cannot tell replicate
# from zero1, and this field can. Megatron picks the class in
# megatron/core/optimizer/__init__.py: DistributedOptimizer under the
# distributed optimizer, and Float16OptimizerWithFloat16Params without it.
#
# **replicate and zero1 get a CHAIN, and the chain's own name proves no
# ZeRO level.** get_megatron_optimizer ends its standard path with an
# unconditional ChainedOptimizer(optimizers), so both values carry one.
# zero3 takes the Megatron-FSDP branch instead, which returns its one
# optimizer bare at a single model chunk. A chain of
# Float16OptimizerWithFloat16Params and a
# chain of DistributedOptimizer print the same word, so
# optimizer_class_name names the members. A real eight-GPU run failed this
# rule on 2026-09-16, because the line said "ChainedOptimizer" alone.
DATA_PARALLEL_LINE = (
    "Megatron-LM stock data parallel: {wrapper} over {dp} "
    "ranks (overlap_grad_reduce={overlap}, grad_reduce_in_fp32={fp32}, "
    "sharding_strategy={sharding}, expert_parallel={expert}, "
    "optimizer={optimizer})"
)

# The pipeline point-to-point sync treatment, printed on every rank at every
# mesh. Both values come from the BUILT config, never from ``args``:
# megatron guards its per-message torch.cuda.synchronize() on
# ``batch_p2p_comm and batch_p2p_sync`` (p2p_communication.py), and
# ``batch_p2p_comm`` is derived from ``overlap_p2p_comm`` inside
# ``core_transformer_config_from_args``. A line built from the arguments
# could not show either.
P2P_LINE = (
    "Megatron-LM stock p2p: batch_p2p_comm={comm} batch_p2p_sync={sync}"
)

# Stock Megatron's NaN/Inf guard, printed on every rank at every mesh from
# the value Megatron's parser resolved. The field is a bool, so ``on`` reads
# True and ``off`` reads False. It is read off ``args`` and not off a built
# config, because Megatron copies it into two places -- pretrain_gpt.py's
# loss_func reads ``args`` directly, and training.py copies it into
# ddp_config.check_for_nan_in_grad -- and ``args`` is the one value both
# consumers descend from. Megatron itself sets the field False under fp16
# with dynamic loss scaling and under a fake process group
# (arguments.py); this arm runs neither, and the line would show it.
NAN_GUARD_LINE = (
    "Megatron-LM stock nan guard: check_for_nan_in_loss_and_grad={value}"
)

# Arm rule 1. Megatron's own completion line is rank 0 only
# (training.py's "after training is done"), and the rule runs per rank.
TRAINING_COMPLETED = "Training completed"

# The two parameter lines CountingGPTModelBuilder prints. They live here,
# with the other three marker groups, because this module is the log-line
# contract and because it imports no megatron: a test can read all four
# groups without a Megatron-LM checkout. ``model_builder.py`` imports them
# from here. Arm rule 11 matches the second one, which is why the total
# carries a thousands separator.
STAGE_SIZE_LINE = (
    "stock-megatron stage {stage}/{stages} local size: {count} parameters"
)
MODEL_SIZE_LINE = (
    "Model qwen3_piper_{size} stock-megatron size: {total} total parameters"
)

# The step line benchmarks/e2e/results.py parses, in the shape the tuned
# megatron driver and the TorchTitan trainer both print.
STEP_LINE = (
    "step: {step:2}  loss: {loss:8.5f}  grad_norm: {grad_norm:7.4f}  "
    "memory: {memory:5.2f}GiB({percent:.2f}%)  tps: {tps:,}  "
    "tflops: {tflops:,.2f}  mfu: {mfu:.2f}%"
)
# The same line on a rank that holds no loss. Under a pipeline split only
# the last stage computes one, and broadcast_pipeline_loss gives it to every
# rank of that pipeline, so a training step never reaches this line. It
# stays because the shim must print a step line whatever Megatron hands it:
# the field is then absent rather than filled with a sentinel, and a
# sentinel is what benchmarks/e2e/results.py would parse as a loss.
STEP_LINE_NO_LOSS = (
    "step: {step:2}  grad_norm: {grad_norm:7.4f}  "
    "memory: {memory:5.2f}GiB({percent:.2f}%)  tps: {tps:,}  "
    "tflops: {tflops:,.2f}  mfu: {mfu:.2f}%"
)

# The peak the mfu column divides by. The same constant the tuned driver
# uses, so the two megatron arms report one definition.
H100_CLASS_BF16_PEAK_FLOPS = 989e12

# The first eight parameters of megatron's training_log, in order. The step
# shim forwards them positionally, so a submodule bump that reorders them
# must fail loudly rather than print the wrong number in the wrong column.
TRAINING_LOG_HEAD = (
    "loss_dict",
    "total_loss_dict",
    "learning_rate",
    "iteration",
    "loss_scale",
    "report_memory_flag",
    "skipped_iter",
    "grad_norm",
)


def add_bench_args(parser: Any) -> Any:
    """Add the harness group to Megatron's own parser.

    Megatron parses these, so an unknown one fails at parse time rather
    than being ignored. The flag names live in ``flags.py``, which builds
    the command line, so one rename reaches both sides.

    The data-parallel and pipeline degrees are **not** here. The driver
    reads them back from ``args`` after Megatron resolved them, and a degree
    the engine resolved is stronger evidence than a degree the harness
    asserted.
    """
    group = parser.add_argument_group(title="torchtitan-benchmarks harness")
    group.add_argument(BENCH_ARM_DIR, type=Path, required=True)
    group.add_argument(BENCH_MODEL_SIZE, type=str, required=True)
    group.add_argument(BENCH_LOCAL_BATCH_SIZE, type=int, required=True)
    group.add_argument(BENCH_PROFILE_FREQ, type=int, required=True)
    group.add_argument(BENCH_PROFILER_WARMUP, type=int, required=True)
    group.add_argument(BENCH_PROFILER_ACTIVE, type=int, required=True)
    group.add_argument(BENCH_MODE, type=str, required=True)
    group.add_argument(BENCH_PP_SCHEDULE, type=str, default=None)
    group.add_argument(BENCH_SEQ_LEN, type=int, required=True)
    group.add_argument(BENCH_ROWS_PER_SAMPLE, type=int, required=True)
    group.add_argument(BENCH_MIN_TRACE_WINDOWS, type=int, required=True)
    # Defaulted rather than required: the flag list omits it at ``on``, so
    # the default argv reaches Megatron's own default for the field.
    group.add_argument(
        BENCH_BATCH_P2P_SYNC,
        type=str,
        choices=MEGATRON_P2P_SYNC_MODES,
        default=DEFAULT_MEGATRON_P2P_SYNC,
    )
    return parser


def apply_p2p_sync(args: Any) -> Any:
    """Set ``args.batch_p2p_sync`` False under ``--bench-batch-p2p-sync off``.

    Megatron lists ``batch_p2p_sync`` under the config fields no CLI
    argument exists for, so its parser never sets the attribute.
    ``core_transformer_config_from_args`` copies every ``args`` attribute
    whose name is a config field, which is what makes this assignment reach
    ``TransformerConfig`` through Megatron's own path.

    Under ``on`` the attribute is left absent, so Megatron's dataclass
    default rules exactly as it did before the option existed. Returns
    ``args`` for the caller's convenience.
    """
    if args.bench_batch_p2p_sync != DEFAULT_MEGATRON_P2P_SYNC:
        args.batch_p2p_sync = False
    return args


def p2p_line(model_cfg: Any) -> str:
    """The p2p line, from the BUILT config and never from ``args``.

    ``model_cfg`` is what ``gpt_config_from_args`` returns; its
    ``transformer`` field is the ``TransformerConfig`` the model is built
    from. Both fields are read off it, because megatron's guard reads both.
    """
    transformer = model_cfg.transformer
    return P2P_LINE.format(
        comm=transformer.batch_p2p_comm, sync=transformer.batch_p2p_sync
    )


def nan_guard_line(args: Any) -> str:
    """The NaN-guard line, from the value Megatron parsed.

    ``args`` is what ``parse_and_validate_args`` returned, after every
    adjustment Megatron's own validation makes to the field.
    """
    return NAN_GUARD_LINE.format(value=args.check_for_nan_in_loss_and_grad)


def refuse_unsupported_run(args: Any) -> None:
    """Reject a run this driver cannot honour, before it builds anything.

    Each refusal lands where the missing work is. A driver that ignored one
    of these would train something the manifest does not name, which is a
    wrong number rather than a crash.
    """
    if args.bench_mode != SUPPORTED_MODE:
        raise ValueError(
            f"{BENCH_MODE} {args.bench_mode!r} is not implemented by the "
            f"stock driver; it runs {SUPPORTED_MODE!r} alone, because "
            "Megatron compiles no whole transformer layer and there is no "
            "treatment to turn off"
        )
    pipeline_degree = args.pipeline_model_parallel_size
    if pipeline_degree > 1:
        if args.bench_pp_schedule != SUPPORTED_PP_SCHEDULE:
            raise ValueError(
                f"{BENCH_PP_SCHEDULE} {args.bench_pp_schedule!r} is not "
                f"implemented by the stock driver; it runs "
                f"{SUPPORTED_PP_SCHEDULE!r} alone"
            )
    elif args.bench_pp_schedule is not None:
        raise ValueError(
            f"{BENCH_PP_SCHEDULE} {args.bench_pp_schedule!r} was given at "
            "pipeline degree 1, where there is no pipeline to schedule"
        )
    if (
        pipeline_degree == 1
        and args.bench_batch_p2p_sync != DEFAULT_MEGATRON_P2P_SYNC
    ):
        # The field is inert without a pipeline message, so the run would
        # print a treatment it did not have.
        raise ValueError(
            f"{BENCH_BATCH_P2P_SYNC} {args.bench_batch_p2p_sync!r} was given "
            "at pipeline degree 1, where there is no pipeline message to "
            "synchronize"
        )
    if args.virtual_pipeline_model_parallel_size is not None:
        raise ValueError(
            "the stock driver builds one model chunk per rank, so a virtual "
            f"pipeline degree of {args.virtual_pipeline_model_parallel_size} "
            "is refused; the parameter check compares a whole stage"
        )
    # One Megatron sample is one packed sequence. flags.py's
    # microbatch_geometry gives the reason: at a micro batch size above 1
    # the pipeline receive buffer is (S, m, H) where the activation is
    # (m*S, 1, H), so the next stage reads a permuted tensor and nothing
    # raises. Assert the packing rather than divide and hope.
    if args.micro_batch_size != 1:
        raise ValueError(
            f"--micro-batch-size {args.micro_batch_size} would send the next "
            "pipeline stage a permuted activation; the stock arm packs its "
            "rows itself and always runs a micro batch size of 1"
        )
    packed = args.bench_rows_per_sample * args.bench_seq_len
    if packed != args.seq_length:
        raise ValueError(
            f"{args.bench_rows_per_sample} row(s) of {args.bench_seq_len} "
            f"tokens is {packed}, not the --seq-length {args.seq_length} "
            "Megatron will size its pipeline buffers from"
        )


def mode_line(args: Any) -> str:
    """The line arm rule 8 matches, plus the precision this arm really runs.

    The eight fields after the mode are what makes the arm a configured
    engine rather than a plain-bf16 one. They are read from the resolved
    arguments, so the log records what Megatron built. Four of them carry
    the --megatron-precision treatment, and the validation profile asks
    for those four under both values.
    """
    return MODE_LINE.format(
        mode=args.bench_mode,
        main_params_dtype=args.main_params_dtype,
        main_grads_dtype=args.main_grads_dtype,
        precision_aware=args.use_precision_aware_optimizer,
        exp_avg_dtype=args.exp_avg_dtype,
        exp_avg_sq_dtype=args.exp_avg_sq_dtype,
        accumulate=args.accumulate_allreduce_grads_in_fp32,
        cross_entropy_loss_fusion=args.cross_entropy_loss_fusion,
        dispatcher=args.moe_token_dispatcher_type,
    )


def parallelism_lines(args: Any, *, microbatches: int) -> list[str]:
    """The mesh line arm rule 12 matches, for this run.

    Empty at world size 1, where there is no mesh to get wrong and rule 12
    is not consulted.

    **This function never prints the data-parallel line.**
    ``install_data_parallel_marker`` is the only source of that line, and it
    prints it from the wrapper Megatron really built. A second copy derived
    from ``args`` would satisfy arm rule 12 on its own, so a run that lost
    the wrapper shim would pass the rule the shim exists to enforce.
    """
    if args.world_size <= 1:
        return []
    schedule = args.bench_pp_schedule or SUPPORTED_PP_SCHEDULE
    return [
        PARALLELISM_LINE.format(
            dp=args.data_parallel_size,
            pp=args.pipeline_model_parallel_size,
            ep=args.expert_model_parallel_size,
            schedule=schedule,
            microbatches=microbatches,
            stages=args.pipeline_model_parallel_size,
        )
    ]


def tokens_per_second(
    local_tokens_per_step: int,
    elapsed_seconds: float,
    pipeline_degree: int,
) -> int:
    """Tokens per second PER DEVICE, which is the published figure.

    The same definition the tuned megatron driver and TorchTitan use: one
    rank's own token count divided by ``cp * tp * pp``. The ranks of one
    pipeline share a batch; each data-parallel rank reads a batch of its
    own, so the data-parallel degree is absent from the divisor.
    """
    if pipeline_degree < 1:
        raise ValueError(f"pipeline degree {pipeline_degree} must be >= 1")
    return round(local_tokens_per_step / (elapsed_seconds * pipeline_degree))


def loss_value(loss_dict: dict) -> float | None:
    """The batch loss this rank holds, or None where it holds none.

    Only the last pipeline stage computes a loss, and this driver
    broadcasts nothing, so a middle stage returns None and prints no loss
    field. ``benchmarks/e2e/results.py`` reads the trajectory from one rank
    and ``loss_visible_rank`` picks a last-stage rank.
    """
    for key, value in loss_dict.items():
        if "loss" not in key:
            continue
        try:
            return float(value)
        except Exception:
            # A value that is not one number is not a loss. Megatron's own
            # entry is a 0-dim tensor, so this is a guard and not a path.
            return None
    return None


def step_log_line(
    *,
    step: int,
    loss: float | None,
    grad_norm: float | None,
    memory_bytes: int,
    device_total_bytes: int,
    tps: int,
    tflops: float,
    mfu: float,
) -> str:
    """One step line, in the shape ``STEP_METRICS`` parses.

    A pure function of numbers, so a test reads the format without a GPU.
    ``grad_norm`` is None on a skipped step, and prints as ``nan`` -- which
    is what ``GRAD_NORM_METRIC`` already accepts.
    """
    fields = {
        "step": step,
        "grad_norm": float("nan") if grad_norm is None else grad_norm,
        "memory": memory_bytes / 2**30,
        "percent": 100 * memory_bytes / device_total_bytes,
        "tps": tps,
        "tflops": tflops,
        "mfu": mfu,
    }
    if loss is None:
        return STEP_LINE_NO_LOSS.format(**fields)
    return STEP_LINE.format(loss=loss, **fields)


def install_step_log_shim(
    *,
    local_tokens_per_step: int,
    pipeline_degree: int,
    num_flops_per_token: int,
) -> Callable[[], None]:
    """Make Megatron print the step line the harness parses.

    Megatron's own ``training_log`` prints an ``iteration ... elapsed time
    per iteration (ms)`` line, on the last rank only, in a format
    ``benchmarks/e2e/results.py``'s three regexes do not match. Without a
    step line this arm publishes no throughput, no peak memory and no loss
    trajectory, so ``evaluate`` has nothing to compare.

    **The plan for this scenario does not name this shim.** It is added
    because the arm is otherwise unmeasurable. It prints one extra line per
    rank per step and changes nothing Megatron computes.

    The loss is broadcast from the last pipeline stage, so every rank
    prints the real number rather than an absent field. ``training_log``
    runs on every rank once per iteration, so the collective is safe: every
    member of the pipeline group reaches it the same number of times.

    **Step 1 carries the setup.** The clock starts when this function runs,
    which is before ``pretrain()`` builds the model, so step 1's rate and
    its peak memory include the build. ``results.py``'s ``stable_tps``
    excludes step 1 of every cycle by construction, and peak memory is a
    maximum over the run, so the published throughput is unaffected and the
    published memory is the run peak rather than a step peak.

    The device's total memory is read on the first step, not here.
    ``pretrain()`` binds this rank's device, and reading the property
    earlier would build a CUDA context on device 0 from every rank.

    Returns a callable that puts Megatron's own ``training_log`` back.
    """
    import inspect

    import torch

    import megatron.training.training as megatron_training

    original = megatron_training.training_log
    head = tuple(inspect.signature(original).parameters)[
        : len(TRAINING_LOG_HEAD)
    ]
    if head != TRAINING_LOG_HEAD:
        raise RuntimeError(
            "megatron's training_log takes "
            f"{head} where this driver forwards {TRAINING_LOG_HEAD}; the "
            "step line would print the wrong value in a column, so the "
            "shim refuses to install"
        )
    state: dict[str, Any] = {"last": time.perf_counter(), "total": None}

    def replacement(
        loss_dict: dict,
        total_loss_dict: dict,
        learning_rate: Any,
        iteration: int,
        loss_scale: Any,
        report_memory_flag: Any,
        skipped_iter: Any,
        grad_norm: Any,
        *rest: Any,
        **keywords: Any,
    ) -> Any:
        result = original(
            loss_dict,
            total_loss_dict,
            learning_rate,
            iteration,
            loss_scale,
            report_memory_flag,
            skipped_iter,
            grad_norm,
            *rest,
            **keywords,
        )
        now = time.perf_counter()
        elapsed = now - state["last"]
        state["last"] = now
        if state["total"] is None:
            state["total"] = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
        reserved = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        tps = tokens_per_second(
            local_tokens_per_step, elapsed, pipeline_degree
        )
        tflops = num_flops_per_token * tps / 1e12
        print(
            step_log_line(
                step=iteration,
                loss=broadcast_pipeline_loss(loss_value(loss_dict)),
                grad_norm=None if grad_norm is None else float(grad_norm),
                memory_bytes=reserved,
                device_total_bytes=state["total"],
                tps=tps,
                tflops=tflops,
                mfu=100 * tflops / (H100_CLASS_BF16_PEAK_FLOPS / 1e12),
            ),
            flush=True,
        )
        return result

    megatron_training.training_log = replacement

    def uninstall() -> None:
        megatron_training.training_log = original

    return uninstall


# Megatron's own attribute for the members of a chain
# (megatron/core/optimizer/optimizer.py: ChainedOptimizer.__init__ sets
# self.chained_optimizers). The name is read off Megatron rather than
# guessed. A submodule bump that renames it makes a chained run print the
# outer class alone. Arm rule 12 then refuses the run, because the
# expected string names the members.
CHAINED_OPTIMIZERS_ATTRIBUTE = "chained_optimizers"


def optimizer_class_name(optimizer: Any) -> str:
    """The optimizer name the data-parallel line states.

    A bare optimizer states its own class. The Megatron-FSDP branch
    returns one, because it collapses its optimizer list where the list
    holds one member, and this harness builds one group of model chunks
    (``megatron/core/optimizer/__init__.py``).

    **A chain states its members too.** The standard path ends with an
    unconditional ``ChainedOptimizer(optimizers)``, so ``replicate`` and
    ``zero1`` both reach this function with a chain. That path always
    holds the dense optimizer. It adds a second member where
    TransformerEngine marked a weight for the expert process groups, which
    this argv reaches above expert degree 1 alone. Every member takes
    ``DistributedOptimizer`` under
    ``use_distributed_optimizer`` and
    ``Float16OptimizerWithFloat16Params`` without it, because that flag is
    one value for the whole run. So a ZeRO-0 chain and a ZeRO-1 chain
    carry the same outer class, and the members are what separate them.

    The member names are deduplicated and sorted, so one string covers
    both cases: ``ChainedOptimizer[DistributedOptimizer]`` where every
    member agrees, and ``ChainedOptimizer[A+B]`` where they do not.

    **An empty chain raises.** ``ChainedOptimizer`` accepts an empty list,
    for a rank that holds no trainable parameter. No path in this arm
    builds one. Such a chain names no class, so the line would state a
    ZeRO level nothing observed.
    """
    name = type(optimizer).__name__
    members = getattr(optimizer, CHAINED_OPTIMIZERS_ATTRIBUTE, None)
    if members is None:
        return name
    if not members:
        raise RuntimeError(
            f"megatron returned a {name} with no member optimizer, so no "
            "line can name the class that holds the optimizer state; the "
            "run would record a ZeRO level it did not have"
        )
    inner = "+".join(sorted({type(member).__name__ for member in members}))
    return f"{name}[{inner}]"


def install_data_parallel_marker(
    *, data_parallel_size: int, expert_parallel_size: int = 1
) -> None:
    """Print the data-parallel line from the wrapper Megatron really built.

    Below two data-parallel ranks there is nothing to observe and nothing
    is printed: arm rule 12's data-parallel marker applies above ``dp`` 1.
    Spec rule 9 makes ``ep`` divide ``dp``, so every expert split above 1
    also has ``dp`` above 1 and reaches this line.

    Above it, this wraps ``setup_model_and_optimizer`` and reads the model
    it returns. **It raises when no chunk carries a data-parallel
    wrapper.** Two ranks that never reduce their gradients train two models
    and report about twice the true throughput, and every other rule passes
    -- including arm rule 13, because stock Megatron all-reduces the loss
    over the same group every step.

    **The check accepts ``_BaseDataParallel``, and the line names the
    class.** Megatron picks the wrapper from the arguments
    (``training.py``): ``DistributedDataParallel``,
    ``FullyShardedDataParallel`` or the torch FSDP2 wrapper. Each of them
    derives directly from ``_BaseDataParallel``
    (``data_parallel_base.py``). None of them derives from another. So a
    check against ``DistributedDataParallel`` alone raises on an honest
    sharded run.
    **``FullyShardedDataParallel`` is a factory function and not a class**
    (``mcore_fsdp_adapter.py``, whose own docstring says so), so
    ``isinstance`` against it raises ``TypeError``; the version classes are
    what a type check may name.

    Accepting the base class alone would prove less than the narrow check
    did. Printing ``type(chunk).__name__`` restores that and adds what the
    dense-sharding value needs: the class name says which mechanism ran.

    **The sharding strategy printed is the one the run acts on.**
    ``data_parallel_sharding_strategy`` reaches every ``ddp_config``,
    because Megatron's argparse defaults it, but
    ``megatron/core/optimizer/__init__.py`` reads it only under
    ``use_megatron_fsdp``. A line built from the raw field would say a
    replicated run sharded. Both fields come from the wrapper's own config.

    **The expert degree is read from the group, and a disagreement
    raises.** ``initialize_model_parallel`` has run by the time this
    replacement fires, so ``mpu.get_expert_model_parallel_world_size()``
    reports the group that really exists. The argv states the degree twice
    -- once as ``--expert-model-parallel-size`` and once on the mesh line
    -- and neither statement proves it. This is the same cross-check
    ``data.py`` already applies to the data-parallel degree.

    ``overlap_grad_reduce`` and ``grad_reduce_in_fp32`` come from the
    wrapper's own ``ddp_config``, not from the arguments.

    **The optimizer class is read here too, and it raises when it is
    absent.** ``setup_model_and_optimizer`` returns the optimizer beside
    the model, and its class is the one observation that separates
    ``zero1`` from ``replicate``: both keep the ``DistributedDataParallel``
    wrapper, so the wrapper name proves nothing about the optimizer state.
    A run that lost ``--use-distributed-optimizer`` builds
    ``Float16OptimizerWithFloat16Params`` and fails arm rule 12 here,
    rather than publishing ZeRO-0 memory under a ZeRO-1 label.

    **A mixture of experts gets a ``ChainedOptimizer``, and the line names
    its members.** ``optimizer_class_name`` does that. The outer class is
    the same under every dense-sharding value, so the chain's own name
    separates none of them.
    """
    if data_parallel_size <= 1:
        return

    import megatron.core.parallel_state as mpu
    import megatron.training.training as megatron_training
    from megatron.core.distributed.data_parallel_base import (
        _BaseDataParallel,
    )

    original = megatron_training.setup_model_and_optimizer

    def replacement(*args: Any, **keywords: Any) -> Any:
        result = original(*args, **keywords)
        model = result[0]
        chunks = model if isinstance(model, list) else [model]
        wrapped = [
            chunk for chunk in chunks if isinstance(chunk, _BaseDataParallel)
        ]
        if not wrapped:
            raise RuntimeError(
                f"the data-parallel degree is {data_parallel_size} and no "
                "model chunk carries a data-parallel wrapper, so "
                "no gradient is reduced; the ranks would train separate "
                "models and report about "
                f"{data_parallel_size}x the true throughput"
            )
        # The optimizer Megatron really built, beside the model. Its class
        # is the one observation that separates zero1 from replicate: both
        # keep the DistributedDataParallel wrapper, so the wrapper name
        # cannot tell them apart. An absent optimizer leaves the line
        # unable to name the class, so it raises rather than printing a
        # ZeRO level nothing observed.
        optimizer = result[1] if len(result) > 1 else None
        if optimizer is None:
            raise RuntimeError(
                "megatron returned no optimizer from "
                "setup_model_and_optimizer, so no line can name the class "
                "that holds the optimizer state; the run would record a "
                "ZeRO level it did not have"
            )
        config = wrapped[0].ddp_config
        built_expert_size = mpu.get_expert_model_parallel_world_size()
        if built_expert_size != expert_parallel_size:
            raise RuntimeError(
                "megatron built an expert-model-parallel group of "
                f"{built_expert_size} rank(s) and the arguments say "
                f"{expert_parallel_size}; the log would name one expert "
                "split and the run would do another"
            )
        print(
            DATA_PARALLEL_LINE.format(
                wrapper=type(wrapped[0]).__name__,
                dp=data_parallel_size,
                overlap=config.overlap_grad_reduce,
                fp32=config.grad_reduce_in_fp32,
                sharding=(
                    config.data_parallel_sharding_strategy
                    if config.use_megatron_fsdp
                    else "no_shard"
                ),
                expert=built_expert_size,
                optimizer=optimizer_class_name(optimizer),
            ),
            flush=True,
        )
        megatron_training.setup_model_and_optimizer = original
        return result

    megatron_training.setup_model_and_optimizer = replacement


def broadcast_pipeline_loss(loss: "Any") -> "Any":
    """The last pipeline stage's loss, on every rank.

    Only the last stage computes one, and stock Megatron leaves the other
    stages with an empty ``loss_dict``. A rank with no loss would print no
    loss field, and ``loss_visible_rank`` -- TorchTitan's own arithmetic,
    ``(world_size // pp) * (pp - 1)`` -- does not always name a last-stage
    rank of Megatron's own layout. Broadcasting removes the question: every
    rank prints the same real number, exactly as the tuned driver does.

    The value the last stage holds is already the mean over the
    data-parallel group, because ``train_step`` all-reduces it there. So
    this broadcast alone makes the printed loss mean what TorchTitan's
    ``global_avg_loss`` means.

    Returns ``None`` when no rank held a loss, which is not a state a
    training step reaches.
    """
    import torch

    if not torch.distributed.is_initialized():
        return loss

    from megatron.core import mpu

    group = mpu.get_pipeline_model_parallel_group()
    if torch.distributed.get_world_size(group=group) == 1:
        return loss
    # The last stage is the source. Its global rank is the last entry of
    # this rank's own pipeline group, which every rank of that group agrees
    # on, so no rank has to guess.
    ranks = torch.distributed.get_process_group_ranks(group)
    source = ranks[-1]
    device = torch.cuda.current_device()
    payload = torch.tensor(
        [0.0 if loss is None else float(loss), 0.0 if loss is None else 1.0],
        dtype=torch.float32,
        device=device,
    )
    torch.distributed.broadcast(payload, source, group=group)
    return float(payload[0]) if float(payload[1]) > 0 else None


def install_rendezvous_defaults(environ: "MutableMapping[str, str]" = os.environ) -> None:
    """Fill the rendezvous variables at one rank, and keep the launcher's.

    Megatron's ``_initialize_distributed`` calls ``init_process_group`` with
    no store, so torch reads ``MASTER_ADDR`` and ``MASTER_PORT`` from the
    environment. Above one rank ``torch.distributed.run`` sets both. At one
    rank nothing did, and the arm died before it trained a step. The two
    ``setdefault`` calls keep a value the launcher chose and fill the
    single-rank case only, which is what the tuned driver does.
    """
    environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in environ:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            environ["MASTER_PORT"] = str(sock.getsockname()[1])


def main(argv: list[str] | None = None) -> int:
    """Run one stock Megatron-LM arm.

    ``argv`` is accepted so a caller can drive the driver; Megatron's own
    parser reads ``sys.argv`` when it is None, which is what ``python -m``
    does.
    """
    import sys

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]

    # The allocator setting the TorchTitan arms get from run_train.sh, and
    # the tuned megatron driver sets for itself. It must be set before torch
    # initializes CUDA, and torch is not imported yet. Both engines of this
    # scenario then run one allocator policy, which is the comparability
    # property that matters here.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    # torchrun sets both variables above one rank. At one rank nothing
    # does, and Megatron's env:// rendezvous fails before a step trains.
    install_rendezvous_defaults()

    bootstrap.prepare()

    # Deferred, and it has to be: both modules import torch at module
    # scope, and ``python -m benchmarks.e2e.megatron_stock.train --help``
    # must not pay for the ML stack.
    from benchmarks.e2e.megatron_stock import data, profiling

    import pretrain_gpt
    from megatron.core.enums import ModelType
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.training import pretrain
    from megatron.training.argument_utils import (
        gpt_config_from_args,
        pretrain_cfg_container_from_args,
    )
    from megatron.training.arguments import parse_and_validate_args

    from benchmarks.e2e.megatron_stock.model_builder import BenchGPTModelConfig
    from benchmarks.models.piper_qwen3.shape import shape_by_name

    args = parse_and_validate_args(extra_args_provider=add_bench_args)
    refuse_unsupported_run(args)
    # After the refusal, and before Megatron builds its config from args.
    apply_p2p_sync(args)

    shape = shape_by_name(args.bench_model_size)
    # The engine's own count, not the harness's arithmetic. Megatron
    # resolved it from --global-batch-size, --micro-batch-size and the
    # data-parallel degree it derived from WORLD_SIZE.
    microbatches = get_num_microbatches()
    print(mode_line(args), flush=True)
    # From the parsed value, on every rank. Arm rule 12 reads it at every
    # mesh, against the requested --megatron-nan-guard value.
    print(nan_guard_line(args), flush=True)
    for line in parallelism_lines(args, microbatches=microbatches):
        print(line, flush=True)

    # Megatron's own resolved rank, read from RANK by its parser. The
    # profiler shim names the trace file after it, and torch.distributed is
    # not up until pretrain() starts.
    shim = profiling.install_profiler_shim(
        arm_dir=args.bench_arm_dir,
        rank=args.rank,
        profile_freq=args.bench_profile_freq,
        profiler_warmup=args.bench_profiler_warmup,
        profiler_active=args.bench_profiler_active,
    )
    install_step_log_shim(
        # The titan row length, not --seq-length: that one is the packed
        # sample, and this rank reads local_batch_size rows per step.
        local_tokens_per_step=(
            args.bench_local_batch_size * args.bench_seq_len
        ),
        pipeline_degree=args.pipeline_model_parallel_size,
        num_flops_per_token=shape.num_flops_per_token(args.bench_seq_len),
    )
    install_data_parallel_marker(
        data_parallel_size=args.data_parallel_size,
        expert_parallel_size=args.expert_model_parallel_size,
    )

    model_cfg = gpt_config_from_args(
        args, model_config_cls=BenchGPTModelConfig
    )
    # From the built config, on every rank. Arm rule 12 reads it above pp 1.
    print(p2p_line(model_cfg), flush=True)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        data.train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        pretrain_gpt.forward_step,
        get_embedding_ranks=pretrain_gpt.get_embedding_ranks,
    )

    # The windows the shimmed schedule must have written. Megatron calls
    # prof.step() once per training iteration, and the schedule flushes one
    # window every --bench-profile-freq steps, so the count is exact. The
    # workload's own requirement is the floor, and it is never below it:
    # workload_with_overrides refuses steps below
    # profile_freq * min_trace_windows. Taking the larger of the two keeps
    # the guard from ever evaluating to "any count is acceptable".
    expected_windows = max(
        args.bench_min_trace_windows,
        args.train_iters // args.bench_profile_freq,
    )
    profiling.assert_windows_written(
        shim, min_trace_windows=expected_windows
    )
    # Every rank prints it: arm rule 1 runs per rank, and megatron's own
    # completion line is rank 0 only.
    print(TRAINING_COMPLETED, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
