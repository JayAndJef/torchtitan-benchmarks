"""Build the training launch command for one end-to-end benchmark arm.

Command construction is the seam between a declarative arm and the engine
that actually trains it. This module holds one builder per engine, and each
delivers the scenario workload, the global run axes, and the arm's own
overrides in that engine's own spelling. Every builder takes the same
parameters; ``benchmarks.e2e.engines`` holds the records that pair a builder
with a validation profile, and it owns the dispatch.

**At the trivial parallelism spec the argv is the argv this repo has always
built.** ``_titan_parallelism_flags`` returns an empty tuple there, so no
``--parallelism.*`` token appears and every published command line is
reproduced token for token. ``tests/test_migration_contract.py`` asserts that
mechanically over every arm of every scenario, not by reading a golden list.
"""

from __future__ import annotations

import sys
from pathlib import Path

from benchmarks.e2e.schema import Arm, ParallelismSpec, Workload
from benchmarks.e2e.parallelism import (
    PP_SCHEDULES,
    TRIVIAL_SPEC,
    titan_mesh,
    titan_reshard_after_forward,
)
from benchmarks.e2e.registry import (
    DEFAULT_MEGATRON_NAN_GUARD,
    DEFAULT_MEGATRON_PRECISION,
    DEFAULT_MEGATRON_P2P_SYNC,
)
from benchmarks.models.piper_qwen3.shape import shape_by_name


# What ``python -m`` starts for the stock Megatron arm. Named once, so a test
# and the command builder cannot drift apart.
STOCK_MEGATRON_DRIVER_MODULE = "benchmarks.e2e.megatron_stock.train"

# The one pipeline schedule the stock driver implements. Megatron-LM itself
# implements more, and this driver builds no model-chunk list, so it runs
# ``forward_backward_pipelining_without_interleaving`` alone.
STOCK_MEGATRON_PP_SCHEDULE = "1F1B"


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
        # Rules 3 and 4 already refuse both cases for a run. Restated here
        # because a caller may build a command line without a run, and a
        # bare KeyError names neither the flag nor the reason.
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


def _refuse_parallelism_passthrough(
    arm: Arm, extra_args: list[str] | tuple[str, ...]
) -> None:
    """Refuse a ``--parallelism.*`` token in the TorchTitan passthrough.

    The passthrough lands **after** the block above, and tyro is last-wins on
    a repeated flag: measured through the fork's own ``ConfigManager``, a
    trailing ``--parallelism.data-parallel-shard-degree 4`` beats the ``1``
    the harness delivered. The manifest's ``parallelism`` block would still
    record ``dp_shard: 1``, so the run would carry a recorded fact its own
    argv contradicts -- exactly the wrongness this axis exists to prevent.

    The same route also reaches ``--parallelism.tensor-parallel-degree``,
    which ``ParallelismSpec`` deliberately cannot express, so the spec would
    no longer describe the mesh at all.

    **Refused, not repaired.** Silently dropping the token would run a
    command the operator did not ask for, and honoring it would publish a
    mesh the manifest does not name. The megatron branch already refuses the
    whole passthrough for a comparable reason.

    Nothing is refused at the trivial spec that was accepted before: the
    block is empty there, and TorchTitan's own ``ParallelDims`` assert
    rejects any degree product other than 1 at world size 1. So this closes a
    path rather than narrowing a working one.
    """
    offenders = [
        token for token in extra_args if token.startswith("--parallelism.")
    ]
    if offenders:
        raise ValueError(
            f"{arm.name}: {', '.join(offenders)} cannot be passed through: "
            "the parallelism block is built from --dp/--pp/--ep and "
            "--zero, and recorded "
            "in the manifest, and a trailing flag would override it while "
            "the record still named the requested mesh"
        )


def titan_command(
    workload: Workload,
    arm: Arm,
    arm_dir: Path,
    extra_args: list[str] | tuple[str, ...],
    ac_mode: str,
    model_size: str = "1b",
    parallelism: ParallelismSpec = TRIVIAL_SPEC,
    megatron_p2p_sync: str = DEFAULT_MEGATRON_P2P_SYNC,
    megatron_nan_guard: str = DEFAULT_MEGATRON_NAN_GUARD,
    megatron_precision: str = DEFAULT_MEGATRON_PRECISION,
    profile: bool = True,
) -> list[str]:
    """Launch command for one TorchTitan arm.

    ``parallelism`` defaults to ``TRIVIAL_SPEC``, which is the identity:
    ``_titan_parallelism_flags`` returns an empty tuple there, so the argv
    below it is the argv this repo has always built.

    The three megatron values are accepted and ignored. Every engine builder
    takes the same parameters, so ``benchmarks.e2e.engines`` dispatches one
    call and no caller branches on the engine. A TorchTitan arm sends no
    pipeline message through Megatron and holds no Megatron optimizer, so
    none of the three can change this argv; ``_resolve_run``
    (``benchmarks.e2e.runner``) is what refuses a non-default value for a
    run that holds no megatron arm.

    ``profile`` decides the profiler block alone. Under ``False`` the arm
    passes no profiler token and writes no trace.
    """
    _refuse_parallelism_passthrough(arm, extra_args)
    # CompileConfig.enable is False in the fork, so an eager arm omits the
    # flag: there is no negation to pass. The flag keeps its position in the
    # list, so a compiled arm builds the command line it built before
    # compile became an arm property.
    compile_flags = ("--compile.enable",) if arm.compile == "torch" else ()
    # TorchTitan's profiler is off unless these tokens ask for it
    # (``CompileConfig``'s sibling ``ProfilingConfig``), so an unprofiled
    # run drops the block and passes no negation. The flags keep their
    # position, so a profiled arm builds the command line it built before
    # the axis existed.
    profiler_flags = (
        (
            "--profiler.enable_profiling",
            "--profiler.profile_freq",
            str(workload.profile_freq),
            "--profiler.profiler_active",
            str(workload.profiler_active),
            "--profiler.profiler_warmup",
            str(workload.profiler_warmup),
        )
        if profile
        else ()
    )
    args = [
        "./run_train.sh",
        "--module",
        workload.module,
        "--config",
        arm.config or workload.config,
        # The fork's ConfigManager forwards --config-arg pairs as keyword
        # arguments to the config function, which resolves the name through
        # benchmarks.models.piper_qwen3.shape.shape_by_name.
        "--config-arg",
        f"size={model_size}",
        "--training.seq-len",
        str(workload.seq_len),
        "--training.steps",
        str(workload.steps),
        "--training.local-batch-size",
        str(workload.local_batch_size),
        *compile_flags,
        *profiler_flags,
        # Empty at the trivial spec, so the argv below it is unchanged.
        *_titan_parallelism_flags(parallelism),
    ]
    if workload.replay_dataloader:
        # The replay loader materializes exactly this many steps of samples
        # and hard-fails when the run asks for more, so it must track --steps.
        args.extend(("--dataloader.replay-steps", str(workload.steps)))
    if workload.seed is not None:
        args.extend(("--debug.seed", str(workload.seed)))
    if arm.override_imports:
        args.extend(("--override.imports", ",".join(arm.override_imports)))
    args = args + list(extra_args) + ["--dump-folder", str(arm_dir)]
    if ac_mode == "none":
        # tyro subcommand token selecting activation_checkpoint=None; the
        # flag-style spelling does not exist for subcommand unions, and tyro
        # attributes any flags after the token to the (fieldless) subcommand,
        # so the token must come last.
        args.append("activation-checkpoint:none")
    return args


def _megatron_launcher(spec: ParallelismSpec) -> list[str]:
    """What starts the megatron driver's processes.

    **At the trivial spec this is ``[sys.executable, "-m"]``**, so the argv
    is the argv this repo has always built and every recorded command line
    is reproduced token for token.

    Above one rank it is ``torch.distributed.run`` -- torchrun under its
    module name, run by this same interpreter, so the CLI and the training
    processes keep sharing one environment. The flags are the ones
    TorchTitan's own ``run_train.sh`` passes, for one reason each:

    * ``--nproc-per-node`` starts the ranks. It reads the spec rather than
      the device count, and parallelism rule 1 is what makes the two agree.
    * ``--rdzv-backend``/``--rdzv-endpoint`` let the kernel pick the port,
      so two runs on one host cannot collide.
    * ``--local-ranks-filter`` names every rank. **torchrun's own default is
      every rank**, not rank 0 -- the empty default resolves to no filter at
      all. Rank 0 alone is ``run_train.sh``'s ``LOG_RANK`` default, which is
      a TorchTitan fact and reaches only the titan arms. The flag is passed
      here so the megatron launcher states the set rather than inheriting a
      default from either side; a kernel that degraded on rank 1 must not be
      invisible.
    * ``--role rank`` with ``--tee 3`` is what puts a rank prefix on every
      line. ``TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE`` (set by
      ``benchmarks/execution/environment.py``) decides its shape, and
      ``benchmarks/artifacts/layout.py``'s ``logs_by_rank`` reads it back.

    The titan arms need no equivalent: ``run_train.sh`` already calls
    torchrun with these flags and reads ``NGPU`` and ``LOG_RANK`` from the
    environment.
    """
    if spec.world_size == 1:
        return [sys.executable, "-m"]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={spec.world_size}",
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        "localhost:0",
        "--local-ranks-filter",
        ",".join(str(rank) for rank in range(spec.world_size)),
        "--role",
        "rank",
        "--tee",
        "3",
        "-m",
    ]


def megatron_stock_command(
    workload: Workload,
    arm: Arm,
    arm_dir: Path,
    extra_args: list[str] | tuple[str, ...],
    ac_mode: str,
    model_size: str = "1b",
    parallelism: ParallelismSpec = TRIVIAL_SPEC,
    megatron_p2p_sync: str = DEFAULT_MEGATRON_P2P_SYNC,
    megatron_nan_guard: str = DEFAULT_MEGATRON_NAN_GUARD,
    megatron_precision: str = DEFAULT_MEGATRON_PRECISION,
    profile: bool = True,
) -> list[str]:
    """Launch command for the stock Megatron-LM driver.

    This function owns three things and no more: the launcher, the ``python
    -m`` target, and the six values ``stock_megatron_flags`` cannot read
    off a workload. ``benchmarks/e2e/megatron_stock/flags.py`` builds every
    flag, both the Megatron group Megatron's own parser reads and the
    ``--bench-*`` group the driver adds through Megatron's
    ``extra_args_provider`` hook. ``tests/test_megatron_stock_launch.py``
    refuses a repeated flag name in the result: Megatron's parser is
    last-wins, so a duplicate would change a value with nothing to see it.

    **At the trivial spec the launcher is the plain interpreter.**
    ``_megatron_launcher`` starts torchrun only above one rank.

    The four refusals below restate what a run already refuses, and
    ``flags.py`` restates two of them again for a caller that reaches it
    directly. A caller may build a command line without a run, and a bare
    Megatron failure minutes into a subprocess names neither the flag nor
    the reason.

    **The schedule refusal has a second cause, and it is not a
    restatement.** Parallelism rule 5 asks what Megatron-LM implements, so
    a schedule the library implements and this driver does not passes every
    parallelism rule. The refusal lands here instead, and ``flags.py``
    repeats it.
    """
    if extra_args:
        raise ValueError(
            f"{arm.name}: TorchTitan passthrough arguments cannot apply to a "
            f"megatron arm: {list(extra_args)}"
        )
    if ac_mode != "none":
        raise ValueError(
            f"{arm.name}: the stock megatron arm runs without recompute; "
            f"ac mode {ac_mode!r} has no Megatron parity (use --ac none)"
        )
    # ``flags.py`` refuses this one too, with its own message. Refused here
    # as well, so a caller that never reaches the flag module still gets the
    # arm's name.
    if workload.seed is None:
        raise ValueError(
            f"{arm.name}: megatron arms require a seeded workload"
        )
    # ``flags.py`` refuses this one too. See the docstring for why the
    # harness keeps its own copy.
    if (
        parallelism.pp > 1
        and parallelism.pp_schedule != STOCK_MEGATRON_PP_SCHEDULE
    ):
        raise ValueError(
            f"{arm.name}: the stock megatron driver implements "
            f"{STOCK_MEGATRON_PP_SCHEDULE!r} alone, and this run asks for "
            f"{parallelism.pp_schedule!r}"
        )
    # Imported here, and below every refusal above. Two reasons. Only this
    # branch of the dispatch needs the stock flag list, so no other arm's
    # command line imports it. And a refused request must fail with its own
    # message, not with whatever the flag module raises first.
    from benchmarks.e2e.megatron_stock.flags import stock_megatron_flags

    # ``model_size`` is passed on as the operator typed it.
    # ``_resolve_run`` canonicalizes
    # the name before it reaches here, and ``shape_by_name`` resolves an
    # alias either way, so the flag list and the shape cannot disagree.
    return [
        *_megatron_launcher(parallelism),
        STOCK_MEGATRON_DRIVER_MODULE,
        *stock_megatron_flags(
            shape_by_name(model_size),
            workload,
            parallelism,
            arm_dir=str(arm_dir),
            model_size=model_size,
            # flags.py refuses off at pp 1 and an unknown value, with its
            # own messages; a run never reaches either, because
            # _resolve_run refuses both first.
            megatron_p2p_sync=megatron_p2p_sync,
            # Megatron's own token under off, nothing under on; flags.py
            # refuses an unknown value with its own message.
            megatron_nan_guard=megatron_nan_guard,
            # Four flags under lean, nothing under stock. flags.py refuses
            # an unknown value, and lean under a replicated dense value,
            # each with its own message; _resolve_run refuses both first.
            megatron_precision=megatron_precision,
            # Megatron's own profiler flags and the harness schedule group
            # under True, and the --bench-profile token with them; nothing
            # at all under False.
            profile=profile,
        ),
    ]
