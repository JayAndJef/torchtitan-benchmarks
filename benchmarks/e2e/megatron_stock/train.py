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

**No file of the Megatron-LM checkout is edited.** Three shims run in this
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

from benchmarks.e2e.megatron_stock import bootstrap
from benchmarks.e2e.megatron_stock.dp_marker import install_data_parallel_marker
from benchmarks.e2e.megatron_stock.flags import (
    add_bench_args,
    apply_p2p_sync,
    refuse_unsupported_run,
)
from benchmarks.e2e.megatron_stock.markers import (
    TRAINING_COMPLETED,
    mode_line,
    nan_guard_line,
    p2p_line,
    parallelism_lines,
)
from benchmarks.e2e.megatron_stock.step_log import install_step_log_shim


def main(argv: list[str] | None = None) -> int:
    """Run one stock Megatron-LM arm.

    ``argv`` is accepted so a caller can drive the driver; Megatron's own
    parser reads ``sys.argv`` when it is None, which is what ``python -m``
    does.
    """
    import sys

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]

    bootstrap.install_allocator_defaults()
    bootstrap.install_rendezvous_defaults()
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
    # Installed only under --bench-profile. Without it Megatron builds no
    # profiler at all, because the argv carries no --profile, so a shim
    # would replace an attribute nothing calls and
    # ``assert_windows_written`` would refuse the run for a window it never
    # asked for.
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
    if shim is not None:
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
