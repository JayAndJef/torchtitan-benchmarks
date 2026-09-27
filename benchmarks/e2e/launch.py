"""Build the training launch of one end-to-end benchmark arm.

This module holds one builder per engine. Each builder reads the run and
the arm's config and returns a ``Launch``, and each engine in
``benchmarks.e2e.engines`` calls its own builder.

At the trivial parallelism spec no ``--parallelism.*`` token appears.
``tests/test_megatron_stock_launch.py`` asserts that over every arm of every
scenario.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.e2e.engines.api import Arm, CompileMode, Launch, RunSpec
from benchmarks.e2e.parallelism import (
    PP_SCHEDULES,
    ParallelismSpec,
    titan_mesh,
    titan_reshard_after_forward,
)
from benchmarks.e2e.passthrough import refuse_passthrough


STOCK_MEGATRON_DRIVER_MODULE = "benchmarks.e2e.megatron_stock.train"
"""What ``python -m`` starts for the stock Megatron arm.

Named once, so a test and the command builder cannot drift apart.
"""

STOCK_MEGATRON_PP_SCHEDULE = "1F1B"
"""The one pipeline schedule the stock driver implements.

Megatron-LM implements more, but this driver builds no model-chunk list, so
it runs ``forward_backward_pipelining_without_interleaving`` alone.
"""


def _titan_parallelism_flags(spec: ParallelismSpec) -> tuple[str, ...]:
    """The ``--parallelism.*`` block TorchTitan needs for this spec.

    Empty at the trivial spec, which is what keeps a single-GPU argv exactly
    the argv it was before this axis existed.

    Two of the flags are sent whenever they apply, and neither is optional:

    ``--parallelism.data-parallel-shard-degree`` -- TorchTitan defaults
    ``data_parallel_shard_degree`` to ``-1`` (``config/configs.py``), which
    resolves to "every remaining rank". An omitted flag therefore turns a
    dp 2 run into full sharding rather than the intended replication, and
    neither the log nor the manifest would say so. ``titan_mesh`` decides
    the pair and both halves are delivered.

    **The pair is gated on the mesh, not on ``dp``.** ``titan_mesh`` reads
    ``spec.zero``: it returns ``(1, dp)`` at level 1 and ``(dp, 1)`` at
    level 0, at every expert degree. So the shard degree moves without
    ``dp`` moving, and a ``dp``-gated test would send a sharded run no
    shard degree at all -- which is the silent substitution this whole
    paragraph exists to prevent. Gating on the mesh also keeps the trivial
    spec's argv empty, because ``titan_mesh`` returns ``(1, 1)`` there at
    every level.

    **``--parallelism.fsdp-reshard-after-forward`` is what makes level 1
    ZeRO-1 here.** The mesh flags alone would reshard every forward. The
    fork
    types the field as ``Literal["default", "always", "never"]`` on its
    ``ParallelismConfig`` (``config/configs.py``), and
    ``get_fsdp_reshard_after_forward_policy`` reads it. Under ``never``
    FSDP2 gathers the parameters at the first microbatch forward and holds
    them for the whole step, which shards the optimizer states and keeps
    whole parameters.

    **The token is sent only when ``titan_reshard_after_forward`` returns a
    value.** Every other spec sends nothing, so TorchTitan keeps its own
    default and no recorded argv moves. That one function decides it, so
    this argv and the tests cannot disagree about which value forces the
    policy.

    **``--parallelism.expert-parallel-degree`` needs no gate of its own.**
    Spec rule 14 refuses ``ep > 1`` under ``zero 0``, so every spec that
    reaches here with an expert degree also asks for a sharded value and
    therefore already carries the pair above. The expert mesh degree TorchTitan derives
    is ``efsdp = dp_shard * cp * tp // ep``, which needs the shard degree the
    pair delivers.

    ``--parallelism.pipeline-parallel-first-stage-less-layers 0`` and its
    ``last`` twin -- both default to **1**, which counts the embedding and
    the output head as layers. At 16 layers and 2 stages the two conventions
    agree, but at 4 stages the default splits [4, 5, 4, 3] where weight 0
    splits [4, 4, 4, 4], and Megatron always divides ``config.num_layers``
    evenly. Rule 7 of ``benchmarks/e2e/parallelism.py`` checks
    ``n_layers % (pp * stages_per_rank)``, which is the arithmetic weight 0
    produces; these two flags are what make that assumption true, and
    ``_golden_titan_pp2_command`` freezes them so an upstream default change
    breaks a test rather than a split.
    """
    flags: list[str] = []
    if spec.pp > 1:
        # Restated here, because a caller may build an argv without a run.
        if spec.pp_schedule not in PP_SCHEDULES:
            raise ValueError(
                f"pp {spec.pp} needs a registered pipeline schedule, got "
                f"{spec.pp_schedule!r}; choose one of "
                + ", ".join(PP_SCHEDULES)
            )
        schedule = PP_SCHEDULES[spec.pp_schedule]
        flags.extend(
            (
                "--parallelism.pipeline-parallel-degree",
                str(spec.pp),
                "--parallelism.pipeline-parallel-schedule",
                schedule.titan_name,
                "--parallelism.pipeline-parallel-microbatch-size",
                str(spec.pp_microbatch_size),
                "--parallelism.pipeline-parallel-first-stage-less-layers",
                "0",
                "--parallelism.pipeline-parallel-last-stage-less-layers",
                "0",
            )
        )
    replicate, shard = titan_mesh(spec)
    if (replicate, shard) != (1, 1):
        flags.extend(
            (
                "--parallelism.data-parallel-replicate-degree",
                str(replicate),
                "--parallelism.data-parallel-shard-degree",
                str(shard),
            )
        )
    reshard_after_forward = titan_reshard_after_forward(spec)
    if reshard_after_forward is not None:
        flags.extend(
            (
                "--parallelism.fsdp-reshard-after-forward",
                reshard_after_forward,
            )
        )
    if spec.ep > 1:
        flags.extend(("--parallelism.expert-parallel-degree", str(spec.ep)))
    return tuple(flags)


TORCHTITAN_TRAIN_MODULE = "torchtitan.train"
"""What ``python -m`` starts for a TorchTitan arm."""


def titan_launch(run: RunSpec, arm: Arm, arm_dir: Path) -> Launch:
    """The ``torchtitan.train`` launch of one TorchTitan arm."""
    config = arm.config
    refuse_passthrough(
        "torchtitan", arm.name, config.extra_flags, run.parallelism.zero
    )
    # The fork defaults it off, so an eager arm passes no negation.
    compile_flags = (
        ("--compile.enable",) if config.compile is CompileMode.TORCH else ()
    )
    # Off unless these tokens ask for it, so an unprofiled run drops them.
    profiler_flags = (
        (
            "--profiler.enable_profiling",
            "--profiler.profile_freq",
            str(run.window.freq),
            "--profiler.profiler_active",
            str(run.window.active),
            "--profiler.profiler_warmup",
            str(run.window.warmup),
        )
        if run.profile
        else ()
    )
    args = [
        "--module",
        config.module,
        "--config",
        config.config,
        # The fork forwards a --config-arg pair as a config keyword.
        "--config-arg",
        f"size={run.shape.name}",
        "--training.seq-len",
        str(run.data.seq_len),
        "--training.steps",
        str(run.data.steps),
        "--training.local-batch-size",
        str(run.data.local_batch_size),
        *compile_flags,
        *profiler_flags,
        *_titan_parallelism_flags(run.parallelism),
        # The replay loader refuses a run longer than the steps it holds.
        "--dataloader.replay-steps",
        str(run.data.steps),
    ]
    if run.seed is not None:
        args.extend(("--debug.seed", str(run.seed)))
    if config.override_imports:
        args.extend(("--override.imports", ",".join(config.override_imports)))
    args = args + list(config.extra_flags) + ["--dump-folder", str(arm_dir)]
    if run.ac_mode == "none":
        # A tyro subcommand token, which has to come last.
        args.append("activation-checkpoint:none")
    return Launch(
        target=("-m", TORCHTITAN_TRAIN_MODULE, *args),
        processes="per_rank",
        pin=True,
    )


def megatron_stock_launch(run: RunSpec, arm: Arm, arm_dir: Path) -> Launch:
    """The launch of one stock Megatron-LM arm.

    ``benchmarks/e2e/megatron_stock/flags.py`` builds every flag, and this
    function adds the driver module and the passthrough tokens. The
    refusals repeat the run checks, so a caller that builds a launch
    without a run gets a message that names the arm.
    """
    config = arm.config
    spec = run.parallelism
    if run.ac_mode != "none":
        raise ValueError(
            f"{arm.name}: the stock megatron arm runs without recompute; "
            f"ac mode {run.ac_mode!r} has no Megatron parity (use --ac none)"
        )
    if run.seed is None:
        raise ValueError(
            f"{arm.name}: megatron arms require a seeded workload"
        )
    if spec.pp > 1 and spec.pp_schedule != STOCK_MEGATRON_PP_SCHEDULE:
        raise ValueError(
            f"{arm.name}: the stock megatron driver implements "
            f"{STOCK_MEGATRON_PP_SCHEDULE!r} alone, and this run asks for "
            f"{spec.pp_schedule!r}"
        )
    # Below the refusals, so a refused request fails with its own message.
    from benchmarks.e2e.megatron_stock.flags import stock_megatron_flags

    refuse_passthrough("megatron", arm.name, config.extra_flags, spec.zero)
    return Launch(
        target=(
            "-m",
            STOCK_MEGATRON_DRIVER_MODULE,
            *stock_megatron_flags(run, config, arm_dir=str(arm_dir)),
            # Last, because Megatron's parser is last-wins.
            *config.extra_flags,
        ),
        processes="per_rank",
        pin=True,
    )
