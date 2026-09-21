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

import os
import socket
from typing import Any, MutableMapping

from benchmarks.e2e.megatron_stock import bootstrap
from benchmarks.e2e.megatron_stock.flags import (
    NO_SHARD_STRATEGY,
    add_bench_args,
    apply_p2p_sync,
    refuse_unsupported_run,
)
from benchmarks.e2e.megatron_stock.markers import (
    DATA_PARALLEL_LINE,
    TRAINING_COMPLETED,
    mode_line,
    nan_guard_line,
    p2p_line,
    parallelism_lines,
)
from benchmarks.e2e.megatron_stock.step_log import install_step_log_shim
























CHAINED_OPTIMIZERS_ATTRIBUTE = "chained_optimizers"
"""Megatron's own attribute for the members of a chain.

``ChainedOptimizer.__init__`` sets it, so the name is read off Megatron
rather than guessed. A submodule bump that renames it makes a chained run
print the outer class alone, and arm rule 12 then refuses the run, because
the expected string names the members.
"""


def optimizer_class_name(optimizer: Any) -> str:
    """The optimizer name the data-parallel line states.

    A bare optimizer states its own class.

    **A chain states its members too.** The standard path ends with an
    unconditional ``ChainedOptimizer(optimizers)``, so ``zero 0`` and
    ``zero 1`` both reach this function with a chain. That path always
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
    ZeRO level needs: the class name says which mechanism ran.

    **The sharding strategy printed is the one the run acts on.**
    ``data_parallel_sharding_strategy`` reaches every ``ddp_config``,
    because Megatron's argparse defaults it, but Megatron reads it only
    under a sharded wrapper. This suite builds none, so the line states ``no_shard``. A
    line built from the raw field would say a replicated run sharded.

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
    ``zero 1`` from ``zero 0``: both keep the ``DistributedDataParallel``
    wrapper, so the wrapper name proves nothing about the optimizer state.
    A run that lost ``--use-distributed-optimizer`` builds
    ``Float16OptimizerWithFloat16Params`` and fails arm rule 12 here,
    rather than publishing ZeRO-0 memory under a ZeRO-1 label.

    **A mixture of experts gets a ``ChainedOptimizer``, and the line names
    its members.** ``optimizer_class_name`` does that. The outer class is
    the same under every ZeRO level, so the chain's own name
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
                sharding=NO_SHARD_STRATEGY,
                expert=built_expert_size,
                optimizer=optimizer_class_name(optimizer),
            ),
            flush=True,
        )
        megatron_training.setup_model_and_optimizer = original
        return result

    megatron_training.setup_model_and_optimizer = replacement




def install_rendezvous_defaults(environ: "MutableMapping[str, str]" = os.environ) -> None:
    """Fill the rendezvous variables at one rank, and keep the launcher's.

    Megatron's ``_initialize_distributed`` calls ``init_process_group`` with
    no store, so torch reads ``MASTER_ADDR`` and ``MASTER_PORT`` from the
    environment. Above one rank ``torch.distributed.run`` sets both. At one
    rank nothing did, and the arm died before it trained a step. The two
    ``setdefault`` calls keep a value the launcher chose and fill the
    single-rank case only.
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

    # The allocator setting the TorchTitan arms get from run_train.sh, which
    # this driver sets for itself. It must be set before torch initializes
    # CUDA, and torch is not imported yet. Both engines of this
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
