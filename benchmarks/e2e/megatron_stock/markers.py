"""The log lines the stock Megatron arm prints, and the validator matches.

Every string here is one half of a contract: ``benchmarks.e2e.validation``
holds the other half, and a test compares the two rosters. The formatters
build one line from resolved Megatron arguments, so a log records what
Megatron built rather than what the harness asked for.

``benchmarks.e2e.megatron_stock.model_builder`` prints the two parameter
lines, ``dp_marker`` the data-parallel line and ``step_log`` the step line.
The rest are printed by ``train.main``.
"""

from __future__ import annotations

from typing import Any

from benchmarks.e2e.megatron_stock.flags import SUPPORTED_PP_SCHEDULE


MODE_LINE = (
    "Megatron-LM stock training loop ("
    "main_params_dtype={main_params_dtype}, "
    "main_grads_dtype={main_grads_dtype}, "
    "use_precision_aware_optimizer={precision_aware}, "
    "exp_avg_dtype={exp_avg_dtype}, "
    "exp_avg_sq_dtype={exp_avg_sq_dtype}, "
    "accumulate_allreduce_grads_in_fp32={accumulate}, "
    "cross_entropy_loss_fusion={cross_entropy_loss_fusion}, "
    "moe_token_dispatcher_type={dispatcher})"
)
"""Arm rule 8, and the head of this module's log-line contract.

The constants below carry the contract with the ``megatron_stock``
validation profile and with the ``STEP_METRICS`` regex of the evaluation. A
one-character difference fails a real run at validation time, so a test
compares these constants against the profile's own strings.

The profile matches this line's prefix up to the first field. The four
precision fields are the ``--megatron-precision`` half of arm rule 12:
Megatron resolves each one before it builds the optimizer, and it maps each
dtype string to a ``torch.dtype``, so the line prints ``torch.bfloat16``
where the flag says ``bf16``. The profile asks for the four under both
values, so an argv that lost or gained the lean flags fails.
``main_params_dtype`` stays ``torch.float32`` under both values, because
the recipe never sends ``--main-params-dtype`` and
``store_param_remainders`` already holds the master copy at 2 bytes for
each parameter.
"""

PARALLELISM_LINE = (
    "Megatron-LM stock parallelism: dp={dp} pp={pp} ep={ep} "
    "schedule={schedule} microbatches={microbatches} stages={stages}"
)
"""Arm rule 12. Every rank prints it above world size 1.

``dp`` and ``microbatches`` are what Megatron resolved. ``pp`` and ``ep``
come verbatim from the command line, and ``schedule`` and ``stages`` are
literals the driver refuses every other value of. The titan mesh line names
``ep`` too, so arm rule 12 reads the same three degrees on both engines.

``ep`` here restates the argv, and the data-parallel line below proves it:
``install_data_parallel_marker`` reads the expert group that
``initialize_model_parallel`` really built. This line cannot do that
itself, because it prints before ``pretrain()`` runs and no process group
exists yet.
"""

DATA_PARALLEL_LINE = (
    "Megatron-LM stock data parallel: {wrapper} over {dp} "
    "ranks (overlap_grad_reduce={overlap}, grad_reduce_in_fp32={fp32}, "
    "sharding_strategy={sharding}, expert_parallel={expert}, "
    "optimizer={optimizer})"
)
"""The data-parallel half of arm rule 12, printed above dp 1.

Every value observes the wrapper or the group, because arm rule 13 cannot
cover for a declared line: stock Megatron all-reduces the loss over its
data-parallel group on every last-stage rank every step, so an all-reduce
kernel appears whether or not a gradient was reduced.
``install_data_parallel_marker`` reads the object Megatron really built,
and its real ``ddp_config``, after ``setup_model_and_optimizer`` returns.

``{wrapper}`` names the mechanism. Megatron picks one of three wrapper
classes, each derived directly from ``_BaseDataParallel``, so the class
name says which memory strategy ran.

``{sharding}`` is the strategy the run acts on, not the raw field.
Megatron's argparse default reaches every ``ddp_config``, but Megatron
reads it only under a sharded wrapper, and this suite builds none.

``{expert}`` is the expert group's real width. The expert degree carves its
ranks out of the data-parallel axis, so this is its line.

``{optimizer}`` is the optimizer class Megatron really built, and it
separates the two ZeRO levels, which the wrapper class cannot. Both levels
get a chain, and the chain's own name proves no level, so
``optimizer_class_name`` names the members. A real eight-GPU run failed
this rule on 2026-09-16, because the line said ``ChainedOptimizer`` alone.
"""

P2P_LINE = (
    "Megatron-LM stock p2p: batch_p2p_comm={comm} batch_p2p_sync={sync}"
)
"""The pipeline point-to-point sync treatment, on every rank at every mesh.

Both values come from the built config, never from ``args``. Megatron
guards its per-message ``torch.cuda.synchronize()`` on ``batch_p2p_comm and
batch_p2p_sync``, and it derives ``batch_p2p_comm`` from
``overlap_p2p_comm`` while it builds the transformer config. A line built
from the arguments could show neither.
"""

NAN_GUARD_LINE = (
    "Megatron-LM stock nan guard: check_for_nan_in_loss_and_grad={value}"
)
"""Stock Megatron's NaN/Inf guard, on every rank at every mesh.

The field is a bool, so ``on`` reads True and ``off`` reads False. The line
reads ``args`` rather than a built config, because Megatron copies the
field into two consumers and ``args`` is the value both descend from.
Megatron itself sets the field False under fp16 with dynamic loss scaling
and under a fake process group; this arm runs neither, and the line would
show it.
"""

TRAINING_COMPLETED = "Training completed"
"""Arm rule 1.

Megatron's own completion line is rank 0 only, and the rule runs per rank.
"""

STAGE_SIZE_LINE = (
    "stock-megatron stage {stage}/{stages} local size: {count} parameters"
)
MODEL_SIZE_LINE = (
    "Model qwen3_piper_{size} stock-megatron size: {total} total parameters"
)
"""The two parameter lines ``CountingGPTModelBuilder`` prints.

They live here with the other marker groups, because this module is the
log-line contract and imports no megatron: a test reads every group without
a Megatron-LM checkout. ``model_builder.py`` imports them from here. Arm
rule 11 matches the second line, which is why the total carries a thousands
separator.
"""

STEP_LINE = (
    "step: {step:2}  loss: {loss:8.5f}  grad_norm: {grad_norm:7.4f}  "
    "memory: {memory:5.2f}GiB({percent:.2f}%)  tps: {tps:,}  "
    "tflops: {tflops:,.2f}  mfu: {mfu:.2f}%"
)
"""The step line the evaluation parses, in the shape TorchTitan prints."""

STEP_LINE_NO_LOSS = (
    "step: {step:2}  grad_norm: {grad_norm:7.4f}  "
    "memory: {memory:5.2f}GiB({percent:.2f}%)  tps: {tps:,}  "
    "tflops: {tflops:,.2f}  mfu: {mfu:.2f}%"
)
"""The same line on a rank that holds no loss.

Under a pipeline split only the last stage computes a loss, and
``broadcast_pipeline_loss`` gives it to every rank of that pipeline, so a
training step never reaches this line. It stays because the shim must print
a step line whatever Megatron hands it. The field is then absent rather
than a sentinel, which the evaluation would parse as a loss.
"""

H100_CLASS_BF16_PEAK_FLOPS = 989e12
"""The peak the mfu column divides by."""

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
"""The first eight parameters of megatron's ``training_log``, in order.

The step shim forwards them positionally, so a submodule bump that reorders
them must fail loudly rather than print the wrong number in the wrong
column.
"""


def mode_line(args: Any) -> str:
    """The line arm rule 8 matches, plus the precision this arm really runs.

    The eight fields make the arm a configured engine rather than a
    plain-bf16 one. They are read from the resolved
    arguments, so the log records what Megatron built. Four of them carry
    the --megatron-precision treatment, and the validation profile asks
    for those four under both values.
    """
    return MODE_LINE.format(
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
