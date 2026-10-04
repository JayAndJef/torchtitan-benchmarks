"""The driver entry point: ``pretrain_gpt``'s main block with our dataset provider and four shims.

This module imports no torch and no megatron until ``main`` runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.e2e.engines.megatron_stock.driver import bootstrap
from benchmarks.e2e.engines.megatron_stock.driver.dp_marker import (
    install_data_parallel_marker,
)
from benchmarks.e2e.engines.megatron_stock.driver.markers import (
    TRAINING_COMPLETED,
    mode_line,
    nan_guard_line,
    p2p_line,
    parallelism_lines,
)
from benchmarks.e2e.engines.megatron_stock.driver.step_log import (
    install_step_log_shim,
)
from benchmarks.e2e.engines.megatron_stock.flags import (
    BENCH_ARM_DIR,
    BENCH_BATCH_P2P_SYNC,
    BENCH_LOCAL_BATCH_SIZE,
    BENCH_MIN_TRACE_WINDOWS,
    BENCH_MODEL_SIZE,
    BENCH_PP_SCHEDULE,
    BENCH_PROFILE,
    BENCH_PROFILE_FREQ,
    BENCH_PROFILE_SCHEDULE_FLAGS,
    BENCH_PROFILER_ACTIVE,
    BENCH_PROFILER_WARMUP,
    BENCH_ROWS_PER_SAMPLE,
    BENCH_SEQ_LEN,
    MEGATRON_P2P_SYNC_MODES,
    PP_SCHEDULE,
)


def add_bench_args(parser: Any) -> Any:
    """Add the harness flags to Megatron's own parser, so that Megatron refuses an unknown one."""
    group = parser.add_argument_group(title="torchtitan-benchmarks harness")
    group.add_argument(BENCH_ARM_DIR, type=Path, required=True)
    group.add_argument(BENCH_MODEL_SIZE, type=str, required=True)
    group.add_argument(BENCH_LOCAL_BATCH_SIZE, type=int, required=True)
    group.add_argument(BENCH_PROFILE, action="store_true")
    group.add_argument(BENCH_PROFILE_FREQ, type=int, default=None)
    group.add_argument(BENCH_PROFILER_WARMUP, type=int, default=None)
    group.add_argument(BENCH_PROFILER_ACTIVE, type=int, default=None)
    group.add_argument(BENCH_PP_SCHEDULE, type=str, default=None)
    group.add_argument(BENCH_SEQ_LEN, type=int, required=True)
    group.add_argument(BENCH_ROWS_PER_SAMPLE, type=int, required=True)
    group.add_argument(BENCH_MIN_TRACE_WINDOWS, type=int, default=None)
    # Megatron's own default, so that an absent flag keeps it.
    group.add_argument(
        BENCH_BATCH_P2P_SYNC,
        type=str,
        choices=MEGATRON_P2P_SYNC_MODES,
        default="on",
    )
    return parser


def apply_p2p_sync(args: Any) -> Any:
    """Set ``args.batch_p2p_sync`` to False under ``--bench-batch-p2p-sync off``, and return ``args``.

    Megatron copies each ``args`` attribute that has the name of a config field into ``TransformerConfig``.
    """
    if args.bench_batch_p2p_sync == "off":
        args.batch_p2p_sync = False
    return args


def refuse_unsupported_run(args: Any) -> None:
    """Raise ``ValueError`` on parsed arguments that the driver cannot run as the manifest records them."""
    schedule_given = {
        flag: getattr(args, flag[2:].replace("-", "_"))
        for flag in BENCH_PROFILE_SCHEDULE_FLAGS
    }
    if args.bench_profile:
        missing = sorted(
            flag for flag, value in schedule_given.items() if value is None
        )
        if missing:
            raise ValueError(
                f"{BENCH_PROFILE} needs the whole profiler schedule, and "
                f"{', '.join(missing)} is absent; the shim cannot build a "
                "schedule from a partial group"
            )
    else:
        extra = sorted(
            flag for flag, value in schedule_given.items() if value is not None
        )
        if extra:
            raise ValueError(
                f"{', '.join(extra)} was given without {BENCH_PROFILE}; the "
                "run collects no trace, so a schedule would name a window "
                "nothing writes"
            )
    pipeline_degree = args.pipeline_model_parallel_size
    if pipeline_degree > 1:
        if args.bench_pp_schedule != PP_SCHEDULE:
            raise ValueError(
                f"{BENCH_PP_SCHEDULE} {args.bench_pp_schedule!r} is not "
                f"implemented by the stock driver; it runs "
                f"{PP_SCHEDULE!r} alone"
            )
    elif args.bench_pp_schedule is not None:
        raise ValueError(
            f"{BENCH_PP_SCHEDULE} {args.bench_pp_schedule!r} was given at "
            "pipeline degree 1, where there is no pipeline to schedule"
        )
    if pipeline_degree == 1 and args.bench_batch_p2p_sync == "off":
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


def main(argv: list[str] | None = None) -> int:
    """Run one stock Megatron-LM arm; ``None`` reads ``sys.argv``."""
    import sys

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]

    bootstrap.prepare()

    from benchmarks.e2e.engines.megatron_stock.driver import data, profiling

    import pretrain_gpt
    from megatron.core.enums import ModelType
    from megatron.core.num_microbatches_calculator import get_num_microbatches
    from megatron.training import pretrain
    from megatron.training.argument_utils import (
        gpt_config_from_args,
        pretrain_cfg_container_from_args,
    )
    from megatron.training.arguments import parse_and_validate_args

    from benchmarks.e2e.engines.megatron_stock.driver.model_builder import (
        BenchGPTModelConfig,
    )
    from benchmarks.models.piper_qwen3.shape import shape_by_name

    # Every rank builds the data, as the stock entry point asks.
    setattr(data.train_valid_test_datasets_provider, "is_distributed", True)

    args = parse_and_validate_args(extra_args_provider=add_bench_args)
    refuse_unsupported_run(args)
    apply_p2p_sync(args)

    shape = shape_by_name(args.bench_model_size)
    microbatches = get_num_microbatches()
    print(mode_line(args), flush=True)
    print(nan_guard_line(args), flush=True)
    for line in parallelism_lines(args, microbatches=microbatches):
        print(line, flush=True)

    shim = (
        profiling.install_profiler_shim(
            arm_dir=args.bench_arm_dir,
            rank=args.rank,
            profile_freq=args.bench_profile_freq,
            profiler_warmup=args.bench_profiler_warmup,
            profiler_active=args.bench_profiler_active,
        )
        if args.bench_profile
        else None
    )
    install_step_log_shim(
        # One TorchTitan row, because --seq-length is the packed sample.
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
    print(p2p_line(model_cfg), flush=True)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    pretrain(
        full_config,
        data.train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        pretrain_gpt.forward_step,
        get_embedding_ranks=pretrain_gpt.get_embedding_ranks,
    )

    if shim is not None:
        expected_windows = max(
            args.bench_min_trace_windows,
            args.train_iters // args.bench_profile_freq,
        )
        profiling.assert_windows_written(
            shim, min_trace_windows=expected_windows
        )
    # Megatron prints its own completion line on rank 0 alone.
    print(TRAINING_COMPLETED, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
