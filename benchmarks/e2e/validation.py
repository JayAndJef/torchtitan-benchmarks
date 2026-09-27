"""The log and trace rules that every engine's validation applies before the harness publishes an arm."""

from __future__ import annotations

import gzip
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from benchmarks.artifacts.layout import logs_by_rank, trace_files_by_rank
from benchmarks.e2e.engines.api import Arm, CompileMode, DataSpec, Engine, RunSpec
from benchmarks.e2e.megatron_stock.flags import (
    DATA_PARALLEL_WRAPPERS,
    SHARDING_STRATEGIES,
    data_parallel_optimizer,
    data_parallel_overlap,
    grad_reduce_in_fp32,
    microbatch_geometry,
)
from benchmarks.e2e.parallelism import ParallelismSpec


@dataclass(frozen=True)
class ValidationProfile:
    """The log lines of one engine that ``validate_against_profile`` reads."""

    completion_marker: str
    """The line that every rank of a finished run prints."""
    compile_marker: str | None
    """The line that proves whole-block ``torch.compile``; ``None`` when the engine has no such treatment."""
    failure_markers: tuple[str, ...]
    """The phrases that mean a silent fallback."""
    ac_line: str | None
    """The line that proves selective activation checkpointing; ``None`` when the engine never applies it."""
    pipelined_pattern: re.Pattern[str]
    """A line that records a pipeline, which a run at one pipeline rank must not print."""
    data_parallel_pattern: re.Pattern[str]
    """A line that records data parallelism, which a run at one data-parallel rank must not print."""


ALL_REDUCE_MARKER = "ncclDevKernel_AllReduce"
"""The kernel name that rule 13 asks each rank's traces for above one data-parallel rank."""


def _megatron_stock_parallelism_markers(
    spec: ParallelismSpec,
    data: DataSpec,
    megatron_precision: str | None,
    megatron_args: tuple[str, ...] = (),
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

    **``overlap_grad_reduce`` follows ``megatron_args``.** The stock argv
    omits ``--overlap-grad-reduce``, so the field reads False unless the
    run's own passthrough sends it. ``data_parallel_overlap`` states that.

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
    _, microbatches, _ = microbatch_geometry(data, spec)
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
            f"{data_parallel_overlap(megatron_args)}, "
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


MEGATRON_STOCK_PROFILE = ValidationProfile(
    completion_marker="Training completed",
    # None on purpose: no log line proves a whole-block treatment here.
    compile_marker=None,
    failure_markers=(),
    ac_line=None,
    # The driver's own resolved degrees, which the trivial spec forbids.
    pipelined_pattern=re.compile(
        r"Megatron-LM stock parallelism: dp=\d+ pp=(?!1\b)\d+"
    ),
    # The same line's other degree, and the wrapper's own line.
    data_parallel_pattern=re.compile(
        r"Megatron-LM stock parallelism: dp=(?!1\b)\d+"
        r"|Megatron-LM stock data parallel:"
    ),
)
"""The stock Megatron profile; every marker carries the word "stock"."""


def _trace_contains(trace_path: Path, marker: str) -> bool:
    """Whether the gzipped trace holds ``marker``; an unreadable trace holds nothing."""
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
    arm_name: str,
    log: str,
    where: str,
    *,
    profile: ValidationProfile,
    shape,
    ac_mode: str,
    compile: CompileMode | None,
    overrides_per_block: int,
    override_imports: tuple[str, ...],
    required_lines: Mapping[str, tuple[str, ...]],
    spec_pp: int,
    spec_dp: int,
) -> None:
    """Raise when one rank's log breaks a log rule; ``where`` names the rank."""
    if profile.completion_marker not in log:
        raise RuntimeError(f"{arm_name}: training did not complete{where}")
    # Rule 8 reads both ways, so an arm that compiled silently cannot publish as eager.
    if profile.compile_marker is not None:
        if compile is None:
            raise ValueError(
                f"{arm_name}: the profile proves torch.compile, and the engine "
                "passed no compile value to compare"
            )
        if compile is CompileMode.TORCH:
            if profile.compile_marker not in log:
                raise RuntimeError(
                    f"{arm_name}: the arm asks for torch.compile and the "
                    f"engine did not apply it{where}"
                )
        elif profile.compile_marker in log:
            raise RuntimeError(
                f"{arm_name}: the arm runs eager and the engine compiled the "
                f"model{where}"
            )
    if profile.ac_line is not None:
        sac_applied = profile.ac_line in log
        if ac_mode == "sac" and not sac_applied:
            raise RuntimeError(
                f"{arm_name}: ac mode 'sac' requested but SelectiveAC was not "
                f"applied{where}"
            )
        if ac_mode == "none" and sac_applied:
            raise RuntimeError(
                f"{arm_name}: ac mode 'none' requested but SelectiveAC was "
                f"applied{where}"
            )
    size_marker = f"size: {shape.param_count:,} total parameters"
    if size_marker not in log:
        raise RuntimeError(
            f"{arm_name}: model size {shape.name!r} "
            f"({shape.param_count:,} parameters) did not apply{where}"
        )
    if overrides_per_block:
        expected_overrides = overrides_per_block * shape.n_layers
        override_count = len(re.findall(r"\[Override\]", log))
        if override_count != expected_overrides:
            raise RuntimeError(
                f"{arm_name}: expected {expected_overrides} override "
                "applications, "
                f"found {override_count}{where}"
            )
        for override_import in override_imports:
            if f"[Override] {override_import}:" not in log:
                raise RuntimeError(
                    f"{arm_name}: override {override_import!r} did not "
                    f"apply{where}"
                )
    for marker in profile.failure_markers:
        if marker in log:
            raise RuntimeError(
                f"{arm_name}: silent fallback marker {marker!r} found in the "
                f"log{where}"
            )
    for treatment, markers in required_lines.items():
        for marker in markers:
            if marker not in log:
                raise RuntimeError(
                    f"{arm_name}: the requested {treatment} did not apply; "
                    f"the engine never logged {marker!r}{where}"
                )
    if spec_pp == 1:
        found = profile.pipelined_pattern.search(log)
        if found is not None:
            raise RuntimeError(
                f"{arm_name}: the run declares no pipeline, and the log "
                f"records one: {found.group(0)!r}{where}"
            )
    # A dp 2 run published as one GPU reads as about twice the true rate.
    if spec_dp == 1:
        found = profile.data_parallel_pattern.search(log)
        if found is not None:
            raise RuntimeError(
                f"{arm_name}: the run declares no data parallelism, and the "
                f"log records some: {found.group(0)!r}{where}"
            )


def validate_arm(
    run: RunSpec, arm: Arm, engine: Engine, arm_dir: Path, log_path: Path
) -> None:
    """Raise ``RuntimeError`` when the engine refuses the arm's log or traces."""
    engine.validate(run, arm, arm_dir, log_path)


def validate_against_profile(
    run: RunSpec,
    arm_name: str,
    arm_dir: Path,
    log_path: Path,
    *,
    profile: ValidationProfile,
    required_lines: Mapping[str, tuple[str, ...]],
    compile: CompileMode | None = None,
    overrides_per_block: int = 0,
    override_imports: tuple[str, ...] = (),
    trace_kernel_markers: tuple[str, ...] = (),
) -> None:
    """Raise ``RuntimeError`` when an arm's log or traces break a rule of ``profile``.

    ``required_lines`` maps each requested treatment to the lines that every
    rank must print for it. An unprofiled run writes no trace, so the trace
    rules 5, 6 and 13 apply to a profiled run alone.
    """
    shape = run.shape
    parallelism = run.parallelism
    if not log_path.is_file():
        raise RuntimeError(f"{arm_name}: training log is missing: {log_path}")
    logs = logs_by_rank(log_path.read_text(errors="replace"))
    expected_ranks = set(range(parallelism.world_size))
    if parallelism.world_size > 1 and set(logs) != expected_ranks:
        raise RuntimeError(
            f"{arm_name}: the run declares {parallelism.world_size} ranks and "
            f"{log_path} carries output from {sorted(logs)}; a rank that "
            "wrote nothing is a rank no rule can check"
        )
    for rank, rank_log in logs.items():
        _validate_log(
            arm_name,
            rank_log,
            f" on rank {rank}; see {log_path}"
            if parallelism.world_size > 1
            else f"; see {log_path}",
            profile=profile,
            shape=shape,
            ac_mode=run.ac_mode,
            compile=compile,
            overrides_per_block=overrides_per_block,
            override_imports=override_imports,
            required_lines=required_lines,
            spec_pp=parallelism.pp,
            spec_dp=parallelism.dp,
        )

    if not run.profile:
        return

    traces_by_rank = trace_files_by_rank(arm_dir)
    traces = [path for paths in traces_by_rank.values() for path in paths]
    # A rank that wrote no trace holds no key.
    if parallelism.world_size > 1 and set(traces_by_rank) != expected_ranks:
        raise RuntimeError(
            f"{arm_name}: the run declares {parallelism.world_size} ranks and "
            f"only {sorted(traces_by_rank)} wrote profiler traces under "
            f"{arm_dir}; a rank with no trace is a rank no per-step figure "
            "measures"
        )
    for rank, rank_traces in (traces_by_rank or {0: []}).items():
        if len(rank_traces) < run.window.min_windows:
            where = f"under {arm_dir}" if len(traces_by_rank) <= 1 else (
                f"for rank {rank} under {arm_dir}"
            )
            raise RuntimeError(
                f"{arm_name}: expected at least {run.window.min_windows} "
                "profiler windows, "
                f"found {len(rank_traces)} {where}"
            )
    # Rule 6 reads all ranks as one set, because a pipeline stage can lack a marker kernel.
    for marker in trace_kernel_markers:
        if not any(_trace_contains(path, marker) for path in traces):
            raise RuntimeError(
                f"{arm_name}: marker kernel {marker!r} absent from profiler traces"
            )
    # Rule 13: every rank reduces over NCCL above one data-parallel rank.
    if parallelism.dp > 1:
        for rank in sorted(expected_ranks):
            if not any(
                _trace_contains(path, ALL_REDUCE_MARKER)
                for path in traces_by_rank.get(rank, ())
            ):
                raise RuntimeError(
                    f"{arm_name}: dp {parallelism.dp} was requested and rank "
                    f"{rank}'s profiler traces under {arm_dir} carry no "
                    f"{ALL_REDUCE_MARKER!r}; a rank that reduced no gradient "
                    "reports roughly twice the true throughput"
                )
