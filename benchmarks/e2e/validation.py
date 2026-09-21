"""Reject partial or wrongly configured end-to-end runs before analysis.

``validate_arm`` is the gate every arm passes before its numbers are
published. Engine differences live in two named profiles,
``TORCHTITAN_PROFILE`` and ``MEGATRON_STOCK_PROFILE``, chosen by the arm's
engine; the structural rules -- trace-window count, kernel markers, override
counting, and the parameter-count line -- are shared. ``ValidationProfile``
itself is declared in ``benchmarks.e2e.schema``; this module holds the
profiles, and ``benchmarks.e2e.engines`` puts each on its engine record.

**The log rules run once per rank.** One ``<arm>.log`` holds every rank's
output, so a rule read against the whole file asks "did some rank do this".
Arm rule 4 is the sharpest case: a kernel that silently degraded on rank 1
alone leaves rank 0's log clean. ``benchmarks.artifacts.layout.logs_by_rank``
does the split and returns a single-rank log whole, so a one-GPU arm is
checked against exactly the text it was checked against before.

A rank that wrote nothing is in neither the log split nor the trace
grouping, so no rule fires for it. The declared world size is what makes
that visible, and it is why this module takes a ``ParallelismSpec``.

**Arm rule 12 is the other half of that spec.** Both engines log what mesh
they really built, and a run that ignored the ``--parallelism.*`` flags --
or a driver that read no ``RANK`` -- passes every other rule while training
something else. The rule is consulted only above one rank, where there is a
mesh to get wrong.

**Arm rule 13 is the data-parallel axis's own hazard, read from the
traces.** Two ranks that never reduce their gradients train two models and
report roughly twice the true throughput, and every other rule passes. The
rule asks each rank's traces for an all-reduce kernel. It does not stand
alone: a mesh can carry an all-reduce that reduces no gradient -- TorchTitan
reduces the loss over its own mesh on every logged step -- so arm rule 12's
per-engine data-parallel log line is what names the mechanism, and this rule
is what says a collective really ran on every rank.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from benchmarks.artifacts.layout import logs_by_rank, trace_files_by_rank
from benchmarks.e2e.megatron_stock.flags import (
    DATA_PARALLEL_OVERLAP,
    DATA_PARALLEL_WRAPPERS,
    SHARDING_STRATEGIES,
    data_parallel_optimizer,
    grad_reduce_in_fp32,
    microbatch_geometry,
)
from benchmarks.e2e.parallelism import (
    PP_SCHEDULES,
    ParallelismSpec,
    TRIVIAL_SPEC,
    n_microbatches,
    titan_mesh,
)
from benchmarks.e2e.registry import (
    DEFAULT_MEGATRON_NAN_GUARD,
    DEFAULT_MEGATRON_P2P_SYNC,
    DEFAULT_MEGATRON_PRECISION,
)
from benchmarks.e2e.schema import Arm, Workload
from benchmarks.models.piper_qwen3.shape import shape_by_name


@dataclass(frozen=True)
class ValidationProfile:
    """Engine-specific pieces of validate_arm, carried by an Engine.

    The engine-neutral rules (trace-window count, kernel markers, override
    counting when declared) are shared; these fields carry what differs:
    the completion marker, the log line that proves the arm's compile
    treatment, the phrases that mean a silent fallback, and whether the
    SelectiveAC line is expected at all.

    ``compile_marker`` is rule 8, and it is read both ways: the line must
    be *present* when the arm declares ``compile="torch"``, and *absent*
    when it declares ``compile="none"``. A profile leaves it ``None`` when
    the engine compiles no whole block and exposes no switch for one. Rule
    8 then checks nothing for that engine, so no arm of it may declare
    ``"torch"``. That pairing is a property of the registry, and
    ``tests/test_engines.py`` pins it over every arm.

    ``parallelism_markers`` is arm rule 12: the log lines that prove this
    engine really ran the requested mesh. It is a callable rather than a
    string because every value in those lines comes from the spec, the
    workload and the precision. Every profile must name at least the mesh
    line for a mesh above one rank, or the run would be published under a
    mesh nothing checked. ``tests/test_engines.py`` pins that too.

    **It takes the precision because one field of the stock data-parallel
    line moves with it, and a real run proved it.**
    ``--megatron-precision lean`` sends ``--main-grads-dtype bf16``, so
    Megatron reduces the gradients in bf16 and the wrapper reports it. The
    value reaches every profile, because one call site serves all three.
    The other two profiles read it and state nothing for it.

    ``pipelined_pattern`` is the other half of arm rule 12, and it reads the
    other way. ``parallelism_markers`` proves the engine built the mesh that
    was asked for. It says nothing when nothing was asked for, so a log from
    a real pipeline passed validation against the trivial spec: the run would
    have been published as single-GPU. This pattern matches only a log that
    built a pipeline, and at ``pp`` 1 its presence fails the arm.

    This is the inversion an eager arm already uses on ``compile_marker``:
    a run that silently compiled cannot be published as eager, and a run
    that silently pipelined cannot be published as one GPU.
    Measured before it was added: of 296 arm logs under ``out/``, exactly one
    matches, and it is a deliberate ``--pp 2`` run.

    ``data_parallel_pattern`` is the same inversion on the other axis, and
    it guards a worse mistake. A pipeline rank and a single-GPU rank publish
    the same per-device throughput, so a pipeline published as one GPU
    misstates the mesh and not the rate. A **data-parallel** rank reads a
    batch of its own, so a ``dp 2`` run published under the trivial spec
    reads as roughly twice the true rate, and every other rule passes. Each
    engine's pattern names two witnesses: the mesh line the engine logs
    whatever this repo's code does, and the wrapper line this repo prints.
    Neither matches a ``pp 2, dp 1`` log, which was checked against a real
    one.

    Measured before it was added, the way ``pipelined_pattern`` was: of the
    532 arm logs under ``out/`` -- every run this repo has ever done, all of
    them single-GPU or pipeline-only -- **none** matches either pattern. So
    the rule refuses nothing that has already happened.

    ``p2p_markers`` is the ``--megatron-p2p-sync`` half of arm rule 12. It
    takes the spec and the requested value, and returns the line the engine
    prints from its BUILT config, above ``pp`` 1 alone: below a pipeline
    there is no message to synchronize, and the option is refused there
    before a GPU is claimed. It is a second callable rather than a third
    argument of ``parallelism_markers``, because the value is not part of
    the mesh and a TorchTitan arm never receives it. An empty tuple here is
    therefore not a refusal: the titan profile returns one at every mesh,
    and ``validate_arm`` reads only the mesh markers for that.

    ``nan_guard_markers`` is the ``--megatron-nan-guard`` half of the same
    rule. It takes the requested value alone, because the guard runs at
    every mesh: the loss check runs on the last stage at ``pp`` 1 and the
    gradient check runs on every rank at ``dp`` 1, so there is no spec
    below which the line is not asked. The stock profile returns the line
    its driver prints from the value Megatron parsed. The titan profile
    returns nothing, because the option never reaches it.

    ``precision_markers`` is the ``--megatron-precision`` half of the same
    rule, and it takes the requested value alone for the reason
    ``nan_guard_markers`` does: the optimizer state has a precision at
    every mesh. The stock profile asks for the four fields its driver
    prints from the arguments Megatron resolved, under BOTH values. A
    ``stock`` label is a claim about the precision exactly as a ``lean``
    label is, so a run that gained the lean flags must fail a stock label
    as surely as a run that lost them fails a lean one. The titan profile
    returns nothing, because the option never reaches it.
    """

    completion_marker: str
    compile_marker: str | None
    failure_markers: tuple[str, ...]
    check_ac_line: bool
    parallelism_markers: Callable[
        [ParallelismSpec, Workload, str], tuple[str, ...]
    ]
    pipelined_pattern: re.Pattern[str]
    data_parallel_pattern: re.Pattern[str]
    p2p_markers: Callable[[ParallelismSpec, str], tuple[str, ...]]
    nan_guard_markers: Callable[[str], tuple[str, ...]]
    precision_markers: Callable[[str], tuple[str, ...]]


_SAC_APPLIED_LINE = "Applied SelectiveAC activation checkpointing"

ALL_REDUCE_MARKER = "ncclDevKernel_AllReduce"
"""Arm rule 13's marker: the device kernel a gradient all-reduce runs.

Both engines reduce over NCCL, so one string serves both. It names the
all-reduce and not NCCL in general, because a pipeline emits send-receive
and broadcast kernels of its own and a bare ``nccl`` would pass a
data-parallel run that reduced nothing. The algorithm and protocol suffix
is left off, because NCCL picks those per message size and topology.

The marker is a necessary condition and not a sufficient one, measured
rather than assumed. A real ``pp 2, dp 1`` trace carries an all-reduce five
times per window from the gradient-norm reduction over the pipeline group,
and above ``dp`` 1 both engines also reduce the loss. So an all-reduce
kernel proves a collective ran, never which one. Arm rule 12's per-engine
data-parallel log line is what names the mechanism; read the two rules
together and do not strengthen this one by guessing at a count.

If the megatron arm fails arm rule 13, read it as a question about this
string first. Megatron issues its bucket reductions inside a coalescing
manager, and a grouped NCCL launch can surface under a generic kernel name.
Settle it by looking at the arm's own trace. Widening this to a bare
``nccl`` is not the repair.
"""


def _titan_parallelism_markers(
    spec: ParallelismSpec,
    workload: Workload,
    megatron_precision: str,
) -> tuple[str, ...]:
    """What TorchTitan logs about the mesh it really built.

    ``megatron_precision`` reaches every profile, because one call site
    serves all three. This one reads it and states nothing for it: no
    Megatron optimizer holds a TorchTitan arm's state.

    The first line comes from ``ParallelDims``, which TorchTitan builds from
    the command line it was given, so it states the degrees that took effect
    rather than the ones the harness asked for -- and it names ``cp`` and
    ``tp``, which ``ParallelismSpec`` cannot express and which must therefore
    both read 1.

    The second is the pipeline schedule and the microbatch count, and it is
    the one that catches the hazard this axis carries: two engines that agree
    on the split but disagree on how many microbatches they move through it
    would publish two different schedules under one label.

    **The third is ours, and it is the only one that proves a reduction.**
    The mesh line above says a mesh was built, not that anything was wrapped
    in it: TorchTitan logs it from ``ParallelDims`` before ``parallelize_fn``
    runs, so a run whose ``parallelize_piper1b`` skipped the data-parallel
    path prints it and reduces nothing. ``parallelize_piper1b`` therefore
    counts the FSDP units the delegate really built and prints
    ``DATA_PARALLEL_LINE`` after the count. **This module cannot import that
    constant**: ``parallelize.py`` imports torch and torchtitan, and this
    module is parent-side. The string is stated twice and
    ``tests/test_parallel_validation.py`` pins the two against each other,
    exactly as it does for the megatron driver's own line.

    A titan *loss* all-reduce is not evidence of a gradient all-reduce, which
    is why this rule reads a log line rather than only the NCCL trace marker
    arm rule 13 adds. TorchTitan reduces the loss over its ``loss`` mesh on
    every logged step whenever ``dp_cp_enabled`` (``trainer.py``), so an
    ``ncclDevKernel_AllReduce`` appears under dp 2 even with the gradients
    never reduced.
    """
    replicate, shard = titan_mesh(spec)
    markers = [
        f"Building device mesh with parallelism: pp={spec.pp}, "
        f"dp_replicate={replicate}, dp_shard={shard}, cp=1, tp=1, "
        f"ep={spec.ep}"
    ]
    if replicate * shard > 1:
        markers.append(
            "piper1b data parallel: fully_shard applied "
            f"(dp_replicate={replicate}, dp_shard={shard})"
        )
    if spec.pp > 1:
        schedule = PP_SCHEDULES[spec.pp_schedule]
        microbatches = n_microbatches(
            spec, local_batch_size=workload.local_batch_size
        )
        markers.append(
            f"Using pipeline schedule {schedule.titan_name} with "
            f"{microbatches} microbatches and "
            f"{spec.pp * schedule.stages_per_rank} stages"
        )
    return tuple(markers)


def _megatron_stock_parallelism_markers(
    spec: ParallelismSpec,
    workload: Workload,
    megatron_precision: str,
) -> tuple[str, ...]:
    """``benchmarks.e2e.megatron_stock.markers``'s two lines. Keep in sync.

    ``dp``, ``pp``, ``ep`` and the microbatch count come from Megatron's
    own resolved arguments. ``schedule=1F1B`` and ``stages`` do not:
    the schedule is a literal on both sides, and ``stages`` repeats the
    pipeline degree, which is the real stage count only while no virtual
    pipeline exists. The driver refuses a virtual pipeline degree, and the
    harness cannot express one. The schedule is a literal because the driver
    refuses every other one, so the name cannot vary, and because the spec
    carries ``None`` at ``pp`` 1, which no line may state.

    **The microbatch count comes from ``microbatch_geometry``, and it is
    not ``n_microbatches``.** Stock Megatron derives the count as
    ``global_batch_size // (micro_batch_size * data_parallel_size)``, in
    ``ConstantNumMicroBatchesCalculator``. That calculator is the one this
    arm gets: ``rampup_batch_size`` defaults to ``None`` and
    the flag list never sets it, so the count is fixed for the whole run.
    ``decrease_batch_size_if_needed`` defaults to ``False``, so Megatron
    asserts the division instead of rounding it.

    The harness sends ``--micro-batch-size 1`` and ``--global-batch-size
    microbatches * dp``, because one Megatron sample is one packed sequence
    of ``rows * seq_len`` tokens rather than a batch of rows
    (``benchmarks/e2e/megatron_stock/flags.py``'s ``microbatch_geometry``
    gives the reason). So the ``dp`` term cancels and the count Megatron
    derives is the count that function returns. **It is 1 at ``pp`` 1**,
    where neither engine splits the batch. Reading the count from that one
    function is what keeps
    this marker and the argv from drifting apart: both come from it.

    **The data-parallel line observes the wrapper.**
    ``install_data_parallel_marker`` in the driver replaces
    ``setup_model_and_optimizer``, reads the model it returns, and raises
    when no chunk carries a data-parallel wrapper. It prints the wrapper's
    class name, ``overlap_grad_reduce``, ``grad_reduce_in_fp32`` and the
    sharding strategy from the wrapper's own ``ddp_config``, and the expert
    degree from the group ``initialize_model_parallel`` built. So a run
    whose wrapper went missing dies there and prints no line.

    **The class name states the mechanism.** Megatron picks one of three
    wrapper classes from the arguments (``training.py``), and the three are
    siblings rather than one a subclass of another. This suite builds
    ``DistributedDataParallel`` at both levels, so a run that reached
    another wrapper prints another class name here and fails this rule.
    ``DATA_PARALLEL_WRAPPERS`` is the table, and it lives beside the flags
    that produce it so the two cannot drift.

    **The optimizer class is what proves level 1, and the wrapper class
    cannot.** ``--use-distributed-optimizer`` alone gives ZeRO-1, and it
    leaves the wrapper at ``DistributedDataParallel`` -- the same class
    level 0 gets. So the two levels print the same wrapper name, and
    only the optimizer separates them: Megatron builds
    ``DistributedOptimizer`` under that flag and
    ``Float16OptimizerWithFloat16Params`` without it. The driver reads the
    optimizer ``setup_model_and_optimizer`` returned, so a run that lost
    the flag fails this rule rather than publishing ZeRO-0 memory under a
    ZeRO-1 label. ``DATA_PARALLEL_OPTIMIZERS`` is the table.

    **The sharding strategy is the value the run acts on**, which is not
    the raw argument. Megatron defaults
    ``data_parallel_sharding_strategy`` to ``optim_grads_params`` and
    copies it into every ``ddp_config``, but its optimizer reads it only
    under a sharded wrapper, which this suite never builds.
    ``SHARDING_STRATEGIES`` therefore names ``no_shard`` at both levels,
    and the driver prints the same value.

    **The expert degree here is an observation and the mesh line's is
    not.** The mesh line prints before ``pretrain()`` runs, where no
    process group exists. The driver reads the built group inside the shim
    and refuses a disagreement, so the two statements of the degree cannot
    differ in a run that reaches this rule.

    **``overlap_grad_reduce`` reads False at both levels.** The argv omits
    ``--overlap-grad-reduce``, so ``args`` carries False, and no wrapper
    this suite builds mutates the field. ``DATA_PARALLEL_OVERLAP`` is the
    table, and it lives beside the flags for the reason the other two
    tables do.

    **``grad_reduce_in_fp32`` moves with ``--megatron-precision``, and this
    rule once pinned it True.** ``--bf16`` with the default
    ``--main-grads-dtype fp32`` sets ``accumulate_allreduce_grads_in_fp32``,
    which ``get_megatron_ddp_config`` copies into ``grad_reduce_in_fp32``,
    and no wrapper touches that field after that. ``--megatron-precision
    lean`` sends ``--main-grads-dtype bf16``, so the field reads False
    there. A real eight-GPU run failed this rule on 2026-09-16 for that
    reason. ``grad_reduce_in_fp32`` in ``flags.py`` is the one statement of
    the derivation, and the driver prints what the wrapper really carries.

    **The optimizer field names the members of a CHAIN.**
    ``get_megatron_optimizer`` ends its standard path with an unconditional
    ``ChainedOptimizer(optimizers)``, and this suite takes no other path.
    A chain's own name proves no ZeRO level, because both levels carry it,
    so the marker names the members. ``data_parallel_optimizer`` is the
    one statement of that string.

    A Megatron bump that moves any of these values fails this rule rather
    than publishing a treatment the log does not state.

    **Arm rule 13 cannot carry this axis alone.** Stock Megatron
    all-reduces the reported loss over the data-parallel group on every
    last-stage rank, on every step, inside its own train step. At
    ``pp`` 1 every rank is a last stage, so every rank emits
    ``ncclDevKernel_AllReduce`` whether or not a gradient moved. Above
    ``pp`` 1 the gradient-norm reduction over the pipeline group already
    puts one on every rank, which this repo measured on a real ``pp 2, dp
    1`` trace. So rule 13 says a collective ran, and the line above says
    which mechanism built it. Cite the two together.
    """
    # The one authority on this count. flags.py builds the argv from it, so
    # a marker taken from anything else could disagree with the run.
    _, microbatches, _ = microbatch_geometry(workload, spec)
    markers = [
        f"Megatron-LM stock parallelism: dp={spec.dp} pp={spec.pp} "
        f"ep={spec.ep} schedule=1F1B microbatches={microbatches} "
        f"stages={spec.pp}"
    ]
    if spec.dp > 1:
        markers.append(
            "Megatron-LM stock data parallel: "
            f"{DATA_PARALLEL_WRAPPERS[spec.zero]} over "
            f"{spec.dp} ranks (overlap_grad_reduce="
            f"{DATA_PARALLEL_OVERLAP[spec.zero]}, "
            "grad_reduce_in_fp32="
            f"{grad_reduce_in_fp32(megatron_precision)}, "
            "sharding_strategy="
            f"{SHARDING_STRATEGIES[spec.zero]}, "
            f"expert_parallel={spec.ep}, optimizer="
            f"{data_parallel_optimizer(spec.zero)})"
        )
    return tuple(markers)


def _p2p_sync_token(megatron_p2p_sync: str) -> str:
    """The ``batch_p2p_sync`` token a built config prints for a value.

    Both drivers format the field with ``str`` on the config's own bool,
    so ``on`` reads ``True`` and ``off`` reads ``False``.

    ``_resolve_run`` refuses a value outside ``MEGATRON_P2P_SYNC_MODES``
    before any arm starts, and ``tests/test_axes.py`` pins the CLI choice
    list equal to that tuple, so no other value reaches this function.
    """
    return str(megatron_p2p_sync == "on")


def _no_p2p_markers(
    spec: ParallelismSpec, megatron_p2p_sync: str
) -> tuple[str, ...]:
    """TorchTitan has no p2p sync to prove.

    ``--megatron-p2p-sync`` reaches the megatron command alone, and a
    TorchTitan argv is the same under either value. So no line is asked of
    a titan rank, and its absence is not a failure.
    """
    return ()


def _megatron_stock_p2p_markers(
    spec: ParallelismSpec, megatron_p2p_sync: str
) -> tuple[str, ...]:
    """``benchmarks.e2e.megatron_stock.markers``'s p2p line. Keep in sync.

    The driver prints the two fields off the config ``gpt_config_from_args``
    built, which is the config ``pretrain`` trains with. Stock Megatron
    derives ``batch_p2p_comm`` as ``not overlap_p2p_comm``
    (``arguments.py``), and forces ``overlap_p2p_comm`` off for the
    non-interleaved schedule this arm runs, so the first field reads True.
    Megatron's guard is ``batch_p2p_comm and batch_p2p_sync``
    (``p2p_communication.py``), so a run with the first field False skips
    the synchronize under either label.
    """
    sync = _p2p_sync_token(megatron_p2p_sync)
    if spec.pp == 1:
        return ()
    return (
        "Megatron-LM stock p2p: batch_p2p_comm=True "
        f"batch_p2p_sync={sync}",
    )


def _nan_guard_token(megatron_nan_guard: str) -> str:
    """The ``check_for_nan_in_loss_and_grad`` token a parsed value prints.

    The stock driver formats Megatron's own bool with ``str``, so ``on``
    reads ``True`` and ``off`` reads ``False``.

    ``_resolve_run`` refuses a value outside ``MEGATRON_NAN_GUARD_MODES``
    before any arm starts, and ``tests/test_axes.py`` pins the CLI choice
    list equal to that tuple, so no other value reaches this function.
    """
    return str(megatron_nan_guard == "on")


def _no_nan_guard_markers(megatron_nan_guard: str) -> tuple[str, ...]:
    """TorchTitan has no Megatron NaN guard to prove.

    The value reaches the stock megatron command alone, and a TorchTitan
    argv is the same under either value.
    """
    return ()


def _megatron_stock_nan_guard_markers(
    megatron_nan_guard: str,
) -> tuple[str, ...]:
    """``benchmarks.e2e.megatron_stock.markers``'s nan guard line. Keep in sync.

    The driver prints ``args.check_for_nan_in_loss_and_grad`` as Megatron
    parsed it, on every rank at every mesh, and nothing in the driver sets
    the field. So a run whose argv lost the token prints ``True`` under an
    ``off`` label, and a run whose Megatron turned the field off by itself
    prints ``False`` under ``on``; either fails here.
    """
    token = _nan_guard_token(megatron_nan_guard)
    return (
        "Megatron-LM stock nan guard: "
        f"check_for_nan_in_loss_and_grad={token}",
    )


def _precision_tokens(megatron_precision: str) -> tuple[str, str]:
    """The two tokens the resolved arguments print for a value.

    Megatron maps every dtype string to a ``torch.dtype`` before it builds
    the optimizer (``arguments.py``'s ``dtype_map``), so the log prints
    ``torch.bfloat16`` where the flag says ``bf16``. The first token is the
    ``use_precision_aware_optimizer`` bool and the second is the dtype the
    three precision fields carry.

    ``_resolve_run`` refuses a value outside ``MEGATRON_PRECISION_MODES``
    before any arm starts, and ``tests/test_axes.py`` pins the CLI choice
    list equal to that tuple, so no other value reaches this function.
    """
    lean = megatron_precision == "lean"
    return str(lean), "torch.bfloat16" if lean else "torch.float32"


def _no_precision_markers(megatron_precision: str) -> tuple[str, ...]:
    """TorchTitan has no Megatron optimizer precision to prove.

    The value reaches the stock megatron command alone, and a TorchTitan
    argv is the same under either value.
    """
    return ()


def _megatron_stock_precision_markers(
    megatron_precision: str,
) -> tuple[str, ...]:
    """``benchmarks.e2e.megatron_stock.markers``'s precision fields. Keep in sync.

    The driver prints all four off the arguments Megatron resolved, on
    every rank at every mesh, and nothing in the driver sets one. So a run
    whose argv lost the lean flags prints ``torch.float32`` under a
    ``lean`` label, and a run that gained them prints ``torch.bfloat16``
    under a ``stock`` label; either fails. Each field is its own marker, so
    a failure names the field that disagreed.

    ``main_params_dtype`` is deliberately absent. The recipe never sends
    ``--main-params-dtype``, so it reads ``torch.float32`` under both
    values and could separate neither.

    The first marker is the line the four fields are printed on. The driver
    prints it on every rank, where Megatron's own "after training is done"
    line is rank 0 only. It carries no compile treatment, so rule 8 cannot
    hold it; it stays here, with the fields it introduces. The open bracket
    leaves those fields free to be read rather than matched twice.
    """
    precision_aware, dtype = _precision_tokens(megatron_precision)
    return (
        "Megatron-LM stock training loop (",
        f"use_precision_aware_optimizer={precision_aware}",
        f"main_grads_dtype={dtype}",
        f"exp_avg_dtype={dtype}",
        f"exp_avg_sq_dtype={dtype}",
    )


TORCHTITAN_PROFILE = ValidationProfile(
    completion_marker="Training completed",
    # Both of TorchTitan's compile lines carry it, in both directions.
    compile_marker="with torch.compile",
    failure_markers=("falling back to the PyTorch",),
    check_ac_line=True,
    parallelism_markers=_titan_parallelism_markers,
    # Logged only above one pipeline rank.
    pipelined_pattern=re.compile(r"Using pipeline schedule"),
    # Two witnesses, because the resolved mesh line is not ours.
    data_parallel_pattern=re.compile(
        r"dp_replicate=(?!1\b)\d+"
        r"|dp_shard=(?!1\b)\d+"
        r"|piper1b data parallel:"
    ),
    # The option never reaches a TorchTitan arm.
    p2p_markers=_no_p2p_markers,
    # Nor does this one.
    nan_guard_markers=_no_nan_guard_markers,
    precision_markers=_no_precision_markers,
)
"""The TorchTitan profile.

The ``torchtitan`` engine record carries it; nothing else reads it.
"""

MEGATRON_STOCK_PROFILE = ValidationProfile(
    completion_marker="Training completed",
    # None on purpose: no log line proves a whole-block treatment here.
    compile_marker=None,
    failure_markers=(),
    check_ac_line=False,
    parallelism_markers=_megatron_stock_parallelism_markers,
    # The driver's own resolved degrees, which the trivial spec forbids.
    pipelined_pattern=re.compile(
        r"Megatron-LM stock parallelism: dp=\d+ pp=(?!1\b)\d+"
    ),
    # The same line's other degree, and the wrapper's own line.
    data_parallel_pattern=re.compile(
        r"Megatron-LM stock parallelism: dp=(?!1\b)\d+"
        r"|Megatron-LM stock data parallel:"
    ),
    # The stock driver's own p2p line, and it carries the word "stock".
    p2p_markers=_megatron_stock_p2p_markers,
    # The stock driver's nan guard line, from the value Megatron parsed.
    nan_guard_markers=_megatron_stock_nan_guard_markers,
    precision_markers=_megatron_stock_precision_markers,
)
"""The megatron arm of the ``engines`` scenario.

It runs Megatron's own ``pretrain()`` through the stock providers. Every
marker carries the word "stock", and the driver prints the same strings.
"""


_PROFILE_BY_ENGINE = {
    "torchtitan": TORCHTITAN_PROFILE,
    "megatron_stock": MEGATRON_STOCK_PROFILE,
}
"""Which profile validates an arm of which engine.

``benchmarks/e2e/engines.py`` builds the same pairing on its own records,
from the same two constants, and ``tests/test_engines.py`` pins the two
against each other. The duplication is the price of the import direction:
this module sits below ``engines.py``, because an engine record carries a
profile, so it cannot read the records back.
"""


def profile_for_engine(engine: str) -> ValidationProfile:
    """The validation profile for one engine name."""
    try:
        return _PROFILE_BY_ENGINE[engine]
    except KeyError as error:
        raise ValueError(
            f"Unknown engine {engine!r}. Available: "
            + ", ".join(sorted(_PROFILE_BY_ENGINE))
        ) from error


def _trace_contains(trace_path: Path, marker: str) -> bool:
    try:
        with gzip.open(trace_path, "rt", errors="replace") as trace_file:
            overlap = ""
            while chunk := trace_file.read(1024 * 1024):
                text = overlap + chunk
                if marker in text:
                    return True
                overlap = text[-len(marker) :] if marker else ""
            return False
    except OSError:
        return False


def _validate_log(
    arm: Arm,
    log: str,
    where: str,
    *,
    profile: ValidationProfile,
    shape,
    ac_mode: str,
    model_size: str,
    parallelism_markers: tuple[str, ...] = (),
    spec_pp: int = 1,
    spec_dp: int = 1,
    nan_guard_markers: tuple[str, ...] = (),
    precision_markers: tuple[str, ...] = (),
) -> None:
    """The rules one rank's own output answers: 1, 2, 3, 4, 8, 10, 11 and 12.

    Every one of them is a statement about a process. Read against the whole
    file they become "some rank did this", which is the weaker question -- a
    kernel that silently degraded on rank 1 alone, or a rank that never
    reached the end of training, passes it. ``where`` names the rank in the
    message and is empty at one rank, so a single-GPU failure reads exactly
    as it read before.

    The expected values do not change per rank, and that is a property of
    both engines rather than an assumption. TorchTitan applies the overrides
    and prints the parameter count while building the whole model, before
    ``pipelining_fn`` splits it, so every rank states the same counts. The
    megatron driver prints the declared total on every rank and puts its own
    stage's count on a separate line.
    """
    if profile.completion_marker not in log:
        raise RuntimeError(f"{arm.name}: training did not complete{where}")
    # Arm rule 8, both ways. Never relax the absence half, or an arm that
    # silently compiled publishes as eager.
    if profile.compile_marker is not None:
        if arm.compile == "torch":
            if profile.compile_marker not in log:
                raise RuntimeError(
                    f"{arm.name}: the arm asks for torch.compile and the "
                    f"engine did not apply it{where}"
                )
        elif profile.compile_marker in log:
            raise RuntimeError(
                f"{arm.name}: the arm runs eager and the engine compiled the "
                f"model{where}"
            )
    if profile.check_ac_line:
        # The AC policy logs its application; its presence must match the
        # requested mode or the run measured the wrong recompute treatment.
        sac_applied = _SAC_APPLIED_LINE in log
        if ac_mode == "sac" and not sac_applied:
            raise RuntimeError(
                f"{arm.name}: ac mode 'sac' requested but SelectiveAC was not "
                f"applied{where}"
            )
        if ac_mode == "none" and sac_applied:
            raise RuntimeError(
                f"{arm.name}: ac mode 'none' requested but SelectiveAC was "
                f"applied{where}"
            )
    # Without it, a shape that silently fell back publishes under the
    # wrong size.
    size_marker = f"size: {shape.param_count:,} total parameters"
    if size_marker not in log:
        raise RuntimeError(
            f"{arm.name}: model size {model_size!r} "
            f"({shape.param_count:,} parameters) did not apply{where}"
        )
    if arm.overrides_per_block:
        expected_overrides = arm.overrides_per_block * shape.n_layers
        override_count = len(re.findall(r"\[Override\]", log))
        if override_count != expected_overrides:
            raise RuntimeError(
                f"{arm.name}: expected {expected_overrides} override "
                "applications, "
                f"found {override_count}{where}"
            )
        for override_import in arm.override_imports:
            if f"[Override] {override_import}:" not in log:
                raise RuntimeError(
                    f"{arm.name}: override {override_import!r} did not "
                    f"apply{where}"
                )
    for marker in profile.failure_markers:
        if marker in log:
            raise RuntimeError(
                f"{arm.name}: silent fallback marker {marker!r} found in the "
                f"log{where}"
            )
    # Arm rule 12, per rank, and empty at the trivial spec.
    for marker in parallelism_markers:
        if marker not in log:
            raise RuntimeError(
                f"{arm.name}: the requested parallelism did not apply; the "
                f"engine never logged {marker!r}{where}"
            )
    # The --megatron-nan-guard half of arm rule 12, asked at every mesh.
    # Empty for an engine the value never reaches.
    for marker in nan_guard_markers:
        if marker not in log:
            raise RuntimeError(
                f"{arm.name}: the requested megatron nan guard did not "
                f"apply; the engine never logged {marker!r}{where}"
            )
    # The --megatron-precision half of arm rule 12, asked at every mesh and
    # under both values. Empty for an engine the value never reaches.
    for marker in precision_markers:
        if marker not in log:
            raise RuntimeError(
                f"{arm.name}: the requested megatron precision did not "
                f"apply; the engine never logged {marker!r}{where}"
            )
    # The other half: the trivial spec asks the question the other way.
    if spec_pp == 1:
        found = profile.pipelined_pattern.search(log)
        if found is not None:
            raise RuntimeError(
                f"{arm.name}: the run declares no pipeline, and the log "
                f"records one: {found.group(0)!r}{where}"
            )
    # The worse mistake: a dp 2 run published as one GPU reads as roughly
    # twice the true rate.
    if spec_dp == 1:
        found = profile.data_parallel_pattern.search(log)
        if found is not None:
            raise RuntimeError(
                f"{arm.name}: the run declares no data parallelism, and the "
                f"log records some: {found.group(0)!r}{where}"
            )


def validate_arm(
    arm: Arm,
    arm_dir: Path,
    log_path: Path,
    workload: Workload,
    *,
    ac_mode: str = "sac",
    model_size: str = "1b",
    parallelism: ParallelismSpec = TRIVIAL_SPEC,
    megatron_p2p_sync: str = DEFAULT_MEGATRON_P2P_SYNC,
    megatron_nan_guard: str = DEFAULT_MEGATRON_NAN_GUARD,
    megatron_precision: str = DEFAULT_MEGATRON_PRECISION,
    profile: bool = True,
) -> None:
    """Reject partial or wrongly configured runs before analysis.

    ``parallelism`` is the run's declared mesh, and it is what turns "the
    ranks that wrote something" into "the ranks this run asked for". Without
    it a rank that died before it opened its log or its first trace file is
    invisible: no rule fires for a rank that left nothing behind, and the
    evaluation then publishes a maximum over the survivors. It defaults to
    the trivial spec, under which every check below is the check this
    function has always made.

    ``megatron_p2p_sync`` is the requested ``--megatron-p2p-sync`` value.
    Above ``pp`` 1 each megatron profile asks every rank for the p2p line
    its driver prints from the built config, with the ``batch_p2p_sync``
    token the value implies, so a run that ignored the flag cannot be
    published under the label it was asked for. It defaults to ``on``,
    which is stock Megatron and the treatment of every run before the
    option existed.

    ``megatron_nan_guard`` is the requested ``--megatron-nan-guard`` value,
    and it defaults to ``on`` for the same reason. The stock profile asks
    every rank, at every mesh, for the line its driver prints from the
    value Megatron parsed.

    ``megatron_precision`` is the requested ``--megatron-precision`` value,
    and it defaults to ``stock``, which every published number of this
    scenario was measured under. The stock profile asks every rank, at
    every mesh, for the four precision fields its driver prints from the
    arguments Megatron resolved, under both values.

    ``profile`` says whether the run collected traces. Under ``False`` the
    arm writes none, so arm rules 5, 6 and 13 and the per-rank trace count
    have nothing to read and are skipped; every log rule still runs on
    every rank. It defaults to ``True``, which is the reading every run
    before the axis existed got, and it is the strict direction: a caller
    that forgets the argument fails an unprofiled arm rather than passing a
    profiled one unchecked.

    **Rule 13 leaves with the traces, and arm rule 12 then carries the
    data-parallel axis alone.** Rule 12 reads each engine's own
    data-parallel log line, which the engine prints after it has built the
    reduction path, so a ``dp`` run that reduced nothing still fails. What
    is lost is the second witness -- the all-reduce kernel -- so cite a
    ``dp`` number from an unprofiled run as resting on the log line.
    """
    engine_profile = profile_for_engine(arm.engine)
    shape = shape_by_name(model_size)
    # At every world size, and resolved before any log is read.
    nan_guard_markers = engine_profile.nan_guard_markers(megatron_nan_guard)
    # The optimizer state has a precision at every mesh too, and both
    # values are a claim the log must carry.
    precision_markers = engine_profile.precision_markers(megatron_precision)
    if not log_path.is_file():
        raise RuntimeError(f"{arm.name}: training log is missing: {log_path}")
    logs = logs_by_rank(log_path.read_text(errors="replace"))
    expected_ranks = set(range(parallelism.world_size))
    # Arm rule 12 is consulted only where there is a mesh to prove.
    parallelism_markers: tuple[str, ...] = ()
    if parallelism.world_size > 1:
        parallelism_markers = engine_profile.parallelism_markers(
            parallelism, workload, megatron_precision
        )
        # The p2p half may be empty and that is honest: a TorchTitan arm
        # never receives the value.
        parallelism_markers += engine_profile.p2p_markers(
            parallelism, megatron_p2p_sync
        )
    if parallelism.world_size > 1 and set(logs) != expected_ranks:
        raise RuntimeError(
            f"{arm.name}: the run declares {parallelism.world_size} ranks and "
            f"{log_path} carries output from {sorted(logs)}; a rank that "
            "wrote nothing is a rank no rule can check"
        )
    for rank, rank_log in logs.items():
        _validate_log(
            arm,
            rank_log,
            f" on rank {rank}; see {log_path}"
            if parallelism.world_size > 1
            else f"; see {log_path}",
            profile=engine_profile,
            shape=shape,
            ac_mode=ac_mode,
            model_size=model_size,
            parallelism_markers=parallelism_markers,
            spec_pp=parallelism.pp,
            spec_dp=parallelism.dp,
            nan_guard_markers=nan_guard_markers,
            precision_markers=precision_markers,
        )

    # Every rule below reads a trace file, and an unprofiled run writes
    # none, so the block is skipped whole rather than rule by rule.
    if not profile:
        return

    # Arm rules 5 and 7 are per rank, because every rank runs the same
    # number of profiler windows.
    traces_by_rank = trace_files_by_rank(arm_dir)
    traces = [path for paths in traces_by_rank.values() for path in paths]
    # A rank that wrote nothing is no key, so the world size closes the gap.
    if parallelism.world_size > 1 and set(traces_by_rank) != expected_ranks:
        raise RuntimeError(
            f"{arm.name}: the run declares {parallelism.world_size} ranks and "
            f"only {sorted(traces_by_rank)} wrote profiler traces under "
            f"{arm_dir}; a rank with no trace is a rank no per-step figure "
            "measures"
        )
    for rank, rank_traces in (traces_by_rank or {0: []}).items():
        if len(rank_traces) < workload.min_trace_windows:
            where = f"under {arm_dir}" if len(traces_by_rank) <= 1 else (
                f"for rank {rank} under {arm_dir}"
            )
            raise RuntimeError(
                f"{arm.name}: expected at least {workload.min_trace_windows} "
                "profiler windows, "
                f"found {len(rank_traces)} {where}"
            )
    # Arm rule 6 reads every rank's traces as one set, because under a
    # pipeline a stage may legitimately lack a marker kernel.
    for marker in arm.trace_kernel_markers:
        if not any(_trace_contains(path, marker) for path in traces):
            raise RuntimeError(
                f"{arm.name}: marker kernel {marker!r} absent from profiler traces"
            )
    # Arm rule 9 is DELETED with graph capture.

    # Arm rule 13, every rank, with the spec's own marker.
    if parallelism.dp > 1:
        for rank in sorted(expected_ranks):
            if not any(
                _trace_contains(path, ALL_REDUCE_MARKER)
                for path in traces_by_rank.get(rank, ())
            ):
                raise RuntimeError(
                    f"{arm.name}: dp {parallelism.dp} was requested and rank "
                    f"{rank}'s profiler traces under {arm_dir} carry no "
                    f"{ALL_REDUCE_MARKER!r}; a rank that reduced no gradient "
                    "reports roughly twice the true throughput"
                )
    # Arm rule 7 is DELETED with the compiled-region measurements.
