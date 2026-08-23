"""Reject partial or wrongly configured end-to-end runs before analysis.

``validate_arm`` is the gate every arm passes before its numbers are
published. Engine differences live in the ``VALIDATION_PROFILES`` registry,
selected by ``Arm.validation``; the structural rules -- trace-window count,
kernel markers, ``cudaGraphLaunch`` under cuda-graph mode, override counting,
and the parameter-count line -- are shared. Compiled-region structure is
*not*: it is a per-profile field (``check_regions``), because the megatron
arm has no Inductor graph annotations to match.

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
from benchmarks.e2e.parallelism import (
    PP_SCHEDULES,
    ParallelismSpec,
    TRIVIAL_SPEC,
    n_microbatches,
    titan_mesh,
)
from benchmarks.e2e.registry import (
    CUDAGRAPH_COMPILE_MODES,
    TORCH_COMPILE_MODE,
    UNCOMPILED_COMPILE_MODES,
    Arm,
    Workload,
)
from benchmarks.models.piper_qwen3.shape import shape_by_name
from benchmarks.traces.extraction import per_rank_pooled_metrics
from benchmarks.traces.schema import Region


_SAC_APPLIED_LINE = "Applied SelectiveAC activation checkpointing"

# Arm rule 13's marker: the device kernel a gradient all-reduce runs. Both
# engines reduce over NCCL, so one string serves both.
#
# **It names the all-reduce and not NCCL in general, deliberately.** A
# pipeline emits ``ncclDevKernel_SendRecv`` and
# ``ncclDevKernel_Broadcast_RING_LL`` for its own point-to-point traffic, and
# neither reduces a gradient, so a bare ``nccl`` would pass a dp run that
# reduced nothing.
#
# **This marker is a necessary condition and not a sufficient one, and that
# was measured rather than assumed.** A real ``pp 2, dp 1`` trace from this
# harness carries ``ncclDevKernel_AllReduce_Sum_bf16_RING_LL`` five times per
# window on both ranks -- one per active step, from the gradient-norm
# reduction over the pipeline group. Above ``dp`` 1 both engines also reduce
# the loss over the data-parallel group on every logged step. So an
# all-reduce kernel proves a collective ran, never which one. What names the
# mechanism is arm rule 12's per-engine data-parallel log line, which each
# engine prints only after it has really built the path: TorchTitan after
# counting its FSDP units, megatron after the DDP wrapper exists. Read the
# two rules together, and do not strengthen this one by guessing at a count.
#
# **The algorithm and protocol suffix is deliberately left off.** NCCL picks
# those per message size and topology, so the ``_Sum_bf16_RING_LL`` spelling
# above is one of several a correct run can produce, and pinning it whole
# would fail an honest run whose buckets chose another. What is fixed is the
# operation in the name.
ALL_REDUCE_MARKER = "ncclDevKernel_AllReduce"


@dataclass(frozen=True)
class ValidationProfile:
    """Engine-specific pieces of validate_arm, selected by Arm.validation.

    The engine-neutral rules (trace-window count, kernel markers,
    cudaGraphLaunch under cuda-graph mode, override counting when declared)
    are shared; these fields carry what differs: the completion marker, the
    log line that proves the requested mode actually applied, the phrases
    that mean a silent fallback, and whether the SelectiveAC line and the
    compiled-region structure are expected at all.

    ``compiled_marker`` is the other half of rule 8, and it is read the
    other way round: ``mode_line`` must be *present* under a compiled mode,
    and ``compiled_marker`` must be *absent* under an uncompiled one. A
    profile leaves it ``None`` when the engine compiles regions it exposes no
    switch for, which is a statement that the engine cannot run uncompiled at
    all; ``validate_arm`` then refuses such a run rather than publishing a
    treatment nothing checked.

    ``parallelism_markers`` is arm rule 12: the log lines that prove this
    engine really ran the requested mesh. It is a callable rather than a
    string because every value in those lines comes from the spec and the
    workload. An empty tuple means this engine logs nothing that proves this
    spec, and ``validate_arm`` then refuses the run rather than publishing a
    mesh nothing checked -- the same shape as ``compiled_marker`` above.

    ``pipelined_pattern`` is the other half of arm rule 12, and it reads the
    other way. ``parallelism_markers`` proves the engine built the mesh that
    was asked for. It says nothing when nothing was asked for, so a log from
    a real pipeline passed validation against the trivial spec: the run would
    have been published as single-GPU. This pattern matches only a log that
    built a pipeline, and at ``pp`` 1 its presence fails the arm.

    This is the inversion ``--compile-mode none`` already uses on
    ``compiled_marker``: a run that silently compiled cannot be published as
    eager, and a run that silently pipelined cannot be published as one GPU.
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
    """

    completion_marker: str
    mode_line: Callable[[str], str]
    compiled_marker: str | None
    failure_markers: tuple[str, ...]
    check_ac_line: bool
    check_regions: bool
    parallelism_markers: Callable[
        [ParallelismSpec, Workload], tuple[str, ...]
    ]
    pipelined_pattern: re.Pattern[str]
    data_parallel_pattern: re.Pattern[str]


def _titan_parallelism_markers(
    spec: ParallelismSpec, workload: Workload
) -> tuple[str, ...]:
    """What TorchTitan logs about the mesh it really built.

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


def _megatron_parallelism_markers(
    spec: ParallelismSpec, workload: Workload
) -> tuple[str, ...]:
    """``benchmarks.e2e.megatron.train``'s two lines. Keep in sync.

    The driver prints what it resolved: the degrees from its own arguments,
    checked against what ``initialize_model_parallel`` gave it, and the
    microbatch count from its own ``pipeline_settings``.

    **The count is 1 at ``pp`` 1, and ``n_microbatches`` is not.** Neither
    engine splits a batch without a pipeline, so the driver runs one pack of
    every row, while ``n_microbatches`` describes the split a pipeline would
    make. This rule is reachable at ``pp`` 1 now that a data-parallel run
    exists, so the condition is written out here rather than left to a
    comment saying it cannot happen. ``pipeline_settings`` is the authority
    and a test compares the two.

    **The second line is the one that proves a reduction.** The line above
    states the mesh, and ``initialize_model_parallel`` builds a
    data-parallel group whether or not anything reduces over it -- so a
    driver that lost its wrapper would print it and publish roughly twice
    the true throughput. The driver prints the second line only after the
    wrapper exists.
    """
    microbatches = (
        n_microbatches(spec, local_batch_size=workload.local_batch_size)
        if spec.pp > 1
        else 1
    )
    markers = [
        f"Megatron-LM parallelism: dp={spec.dp} pp={spec.pp} "
        f"schedule={spec.pp_schedule} microbatches={microbatches} "
        f"stages={spec.pp}"
    ]
    if spec.dp > 1:
        markers.append(
            f"Megatron-LM data parallel: DistributedDataParallel over "
            f"{spec.dp} ranks (overlap_grad_reduce=True, "
            "grad_reduce_in_fp32=False)"
        )
    return tuple(markers)


VALIDATION_PROFILES = {
    "torchtitan": ValidationProfile(
        completion_marker="Training completed",
        # apply_compile logs the torch-level mode name.
        mode_line=lambda mode: (
            f"with torch.compile (mode={TORCH_COMPILE_MODE[mode]})"
        ),
        # Carried by both of TorchTitan's compile log lines -- apply_compile's
        # per-block line and the loss function's -- so one absence check
        # covers every component --compile.enable switches on.
        compiled_marker="with torch.compile",
        failure_markers=("falling back to the PyTorch",),
        check_ac_line=True,
        check_regions=True,
        parallelism_markers=_titan_parallelism_markers,
        # TorchTitan logs this from _build_pipeline_schedule, which
        # runs only when the pipeline degree is above 1.
        pipelined_pattern=re.compile(r"Using pipeline schedule"),
        # Two independent witnesses of a data-parallel degree, because one
        # of them is not ours. ``ParallelDims.build_mesh`` logs the resolved
        # mesh on every rank whatever this repo's code does, so a degree
        # above 1 shows there even in a run that never reached
        # ``parallelize_piper1b``; the second alternative is our own line.
        # A ``pp 2, dp 1`` run logs ``dp_replicate=1, dp_shard=1`` and
        # matches neither, which was checked against a real one.
        data_parallel_pattern=re.compile(
            r"dp_replicate=(?!1\b)\d+"
            r"|dp_shard=(?!1\b)\d+"
            r"|piper1b data parallel:"
        ),
    ),
    "megatron": ValidationProfile(
        completion_marker="Training completed",
        # benchmarks.e2e.megatron.train.MODE_LINE; the trailing comma pins
        # the mode token without pinning which graph implementation ran.
        mode_line=lambda mode: f"Megatron-LM training loop (mode={mode},",
        # None on purpose: megatron-core sets jit_fuser = torch.compile at
        # import and decorates 41 functions with it, so no log line proves a
        # megatron arm ran uncompiled, and disable_jit_fuser() cannot make
        # one true (see the piper1b_megatron declaration). That scenario
        # declines the uncompiled modes, and validate_arm refuses one here if
        # it ever reaches this profile.
        compiled_marker=None,
        failure_markers=(),
        check_ac_line=False,
        check_regions=False,
        parallelism_markers=_megatron_parallelism_markers,
        # The driver prints its own degrees. Any pipeline degree
        # other than 1 is what this must not see at the trivial spec.
        pipelined_pattern=re.compile(
            r"Megatron-LM parallelism: dp=\d+ pp=(?!1\b)\d+"
        ),
        # The same line's other degree, plus the wrapper's own line.
        data_parallel_pattern=re.compile(
            r"Megatron-LM parallelism: dp=(?!1\b)\d+"
            r"|Megatron-LM data parallel:"
        ),
    ),
}


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
    compile_mode: str,
    ac_mode: str,
    model_size: str,
    parallelism_markers: tuple[str, ...] = (),
    spec_pp: int = 1,
    spec_dp: int = 1,
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
    if compile_mode in UNCOMPILED_COMPILE_MODES:
        # Arm rule 8 inverts here: an uncompiled arm prints no compile line,
        # so the proof is the absence of one. Never relax this into "skip the
        # check" -- a run that silently compiled would then publish as eager.
        if profile.compiled_marker is None:
            raise RuntimeError(
                f"{arm.name}: validation profile {arm.validation!r} cannot "
                f"prove compile mode {compile_mode!r}; that engine compiles "
                "regions it exposes no switch for"
            )
        if profile.compiled_marker in log:
            raise RuntimeError(
                f"{arm.name}: compile mode {compile_mode!r} requested but the "
                f"engine compiled the model{where}"
            )
    # The engine reports which mode it actually applied.
    elif profile.mode_line(compile_mode) not in log:
        raise RuntimeError(
            f"{arm.name}: compile mode {compile_mode!r} did not apply{where}"
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
    # Both engines print this line; without the check a run whose --config
    # or --model-size silently fell back to another shape would pass every
    # other rule and be published under the wrong size.
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
    # Arm rule 12. Empty at the trivial spec, where there are no parallelism
    # flags to ignore. Every rank logs these, because neither engine guards
    # the line on the rank.
    for marker in parallelism_markers:
        if marker not in log:
            raise RuntimeError(
                f"{arm.name}: the requested parallelism did not apply; the "
                f"engine never logged {marker!r}{where}"
            )
    # The other half of arm rule 12. A positive marker cannot speak for a
    # spec that asked for nothing, so the trivial spec asks the question the
    # other way round: this log must not show a pipeline nobody requested.
    if spec_pp == 1:
        found = profile.pipelined_pattern.search(log)
        if found is not None:
            raise RuntimeError(
                f"{arm.name}: the run declares no pipeline, and the log "
                f"records one: {found.group(0)!r}{where}"
            )
    # The data-parallel half of the same inversion, and it guards a worse
    # mistake than the pipeline half. A pipeline rank and a single-GPU rank
    # publish the same per-device throughput; a data-parallel rank does not,
    # so a dp 2 run published under the trivial spec would read as roughly
    # twice the true rate with every other rule satisfied. The positive
    # markers cannot ask this, because the trivial spec declares none.
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
    regions: tuple[Region, ...] = (),
    compile_mode: str = "default",
    ac_mode: str = "sac",
    model_size: str = "1b",
    parallelism: ParallelismSpec = TRIVIAL_SPEC,
) -> None:
    """Reject partial or wrongly configured runs before analysis.

    ``parallelism`` is the run's declared mesh, and it is what turns "the
    ranks that wrote something" into "the ranks this run asked for". Without
    it a rank that died before it opened its log or its first trace file is
    invisible: no rule fires for a rank that left nothing behind, and the
    evaluation then publishes a maximum over the survivors. It defaults to
    the trivial spec, under which every check below is the check this
    function has always made.
    """
    profile = VALIDATION_PROFILES[arm.validation]
    shape = shape_by_name(model_size)
    if not log_path.is_file():
        raise RuntimeError(f"{arm.name}: training log is missing: {log_path}")
    logs = logs_by_rank(log_path.read_text(errors="replace"))
    expected_ranks = set(range(parallelism.world_size))
    # Arm rule 12 is consulted only where there is a mesh to prove. A profile
    # that names no marker for a non-trivial spec cannot prove the run ran
    # what it claims, and the run is refused rather than published -- the
    # same reading arm rule 8 gives an engine that cannot prove eager
    # execution.
    parallelism_markers: tuple[str, ...] = ()
    if parallelism.world_size > 1:
        parallelism_markers = profile.parallelism_markers(
            parallelism, workload
        )
        if not parallelism_markers:
            raise RuntimeError(
                f"{arm.name}: validation profile {arm.validation!r} logs "
                f"nothing that proves dp {parallelism.dp} x pp "
                f"{parallelism.pp}; the run cannot be published under a mesh "
                "no rule checked"
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
            profile=profile,
            shape=shape,
            compile_mode=compile_mode,
            ac_mode=ac_mode,
            model_size=model_size,
            parallelism_markers=parallelism_markers,
            spec_pp=parallelism.pp,
            spec_dp=parallelism.dp,
        )

    # Arm rules 5 and 7 are per rank. Every rank runs the same number of
    # profiler windows, so a rank short of them is as broken as a run short of
    # them, and pooling two ranks' windows for arm rule 7 would ask one
    # structural question of two different processes' graphs.
    traces_by_rank = trace_files_by_rank(arm_dir)
    traces = [path for paths in traces_by_rank.values() for path in paths]
    # A rank that wrote no trace file at all is not a key in that mapping, so
    # neither of those two rules would fire for it and the evaluation would
    # publish a maximum over the survivors. The declared world size is what
    # closes that, and it is the reason this function takes one.
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
    # Arm rules 6 and 9 read every rank's traces as one set, which is what
    # they already did when one rank was all there was. They are deliberately
    # NOT per rank, and this stage does not change either.
    #
    # Arm rule 9 is unreachable: parallelism rule 13 refuses cuda-graph for
    # every parallel run, and this rule fires only under cuda-graph. Arm rule
    # 6 IS reachable, and its reading stays "any rank" for now. Under PP a
    # stage holds some of the layers, so a marker kernel can be legitimately
    # absent from a rank: "every rank" would fail an honest run, and "any
    # rank" passes a run where one stage silently degraded.
    #
    # **One PP2 megatron trace has now been read, and it says "every rank"
    # would cost that arm nothing.** Both stages of a 16-layer ``pp 2`` run
    # carry the cuDNN fused-attention kernel, ``_mul_silu_split`` and
    # ``_permute_kernel``. That is one arm at one shape: a stage that holds
    # no layer of the kind a marker names would still lack it, so the general
    # rule needs a per-arm declaration of which ranks carry which marker
    # rather than a blanket "every rank". Do not weaken this rule to make a
    # hypothetical run pass, and do not tighten it on one arm's evidence.
    for marker in arm.trace_kernel_markers:
        if not any(_trace_contains(path, marker) for path in traces):
            raise RuntimeError(
                f"{arm.name}: marker kernel {marker!r} absent from profiler traces"
            )
    if compile_mode in CUDAGRAPH_COMPILE_MODES and not any(
        _trace_contains(path, "cudaGraphLaunch") for path in traces
    ):
        raise RuntimeError(
            f"{arm.name}: compile mode {compile_mode!r} enables CUDA graphs but "
            f"no cudaGraphLaunch appears in the profiler traces under {arm_dir}"
        )
    # Arm rule 13. **Every rank**, and that reading is provable here where
    # arm rule 6's is not: at dp above 1 every rank sits in a data-parallel
    # group of that size, so every rank reduces. A rank whose traces carry no
    # all-reduce did not.
    #
    # The marker is the SPEC's, never an ``Arm.trace_kernel_markers`` entry.
    # Data parallelism is a run axis, so a static declaration would fail
    # every single-GPU run of the same arm -- there is no all-reduce there at
    # all.
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
    if regions and profile.check_regions:
        try:
            per_rank_pooled_metrics(traces_by_rank, regions)
        except ValueError as error:
            raise RuntimeError(
                f"{arm.name}: profiler traces failed structural validation: {error}"
            ) from error
