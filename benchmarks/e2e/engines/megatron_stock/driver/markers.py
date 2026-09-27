"""The log lines that the driver prints, and the functions that format them.

``benchmarks/e2e/engines/megatron_stock/validate.py`` states the same lines, and a test compares the two.
"""

from __future__ import annotations

from typing import Any

from benchmarks.e2e.engines.megatron_stock.flags import PP_SCHEDULE


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
"""The training-loop line: the precision and the loss path that Megatron resolved."""

PARALLELISM_LINE = (
    "Megatron-LM stock parallelism: dp={dp} pp={pp} ep={ep} "
    "schedule={schedule} microbatches={microbatches} stages={stages}"
)
"""The mesh line that every rank prints above one rank; ``dp`` and ``microbatches`` are Megatron's resolved values."""

DATA_PARALLEL_LINE = (
    "Megatron-LM stock data parallel: {wrapper} over {dp} "
    "ranks (overlap_grad_reduce={overlap}, grad_reduce_in_fp32={fp32}, "
    "sharding_strategy={sharding}, expert_parallel={expert}, "
    "optimizer={optimizer})"
)
"""The data-parallel line that every rank prints above one data-parallel rank, from the wrapper and the optimizer that Megatron built."""

P2P_LINE = (
    "Megatron-LM stock p2p: batch_p2p_comm={comm} batch_p2p_sync={sync}"
)
"""The p2p line, from the ``TransformerConfig`` that Megatron built."""

NAN_GUARD_LINE = (
    "Megatron-LM stock nan guard: check_for_nan_in_loss_and_grad={value}"
)
"""The NaN-guard line, from the value that Megatron parsed."""

TRAINING_COMPLETED = "Training completed"
"""The completion line that every rank prints; Megatron prints its own on rank 0 alone."""

STAGE_SIZE_LINE = (
    "stock-megatron stage {stage}/{stages} local size: {count} parameters"
)
MODEL_SIZE_LINE = (
    "Model qwen3_piper_{size} stock-megatron size: {total} total parameters"
)
"""The whole-model parameter line that the model fact reads; ``STAGE_SIZE_LINE`` gives the count of one stage."""

H100_CLASS_BF16_PEAK_FLOPS = 989e12
"""The peak FLOPS that the mfu figure of the step record divides by."""

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
"""The first eight parameters of Megatron's ``training_log``, which the step shim passes by position."""


def mode_line(args: Any) -> str:
    """The training-loop line, from the resolved ``args``."""
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
    """The mesh line, or nothing at one rank; ``install_data_parallel_marker`` prints the data-parallel line."""
    if args.world_size <= 1:
        return []
    schedule = args.bench_pp_schedule or PP_SCHEDULE
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
    """The p2p line of ``model_cfg``, which ``gpt_config_from_args`` returns."""
    transformer = model_cfg.transformer
    return P2P_LINE.format(
        comm=transformer.batch_p2p_comm, sync=transformer.batch_p2p_sync
    )


def nan_guard_line(args: Any) -> str:
    """The NaN-guard line of the parsed ``args``."""
    return NAN_GUARD_LINE.format(value=args.check_for_nan_in_loss_and_grad)
