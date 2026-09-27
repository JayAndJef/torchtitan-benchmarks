"""The log and trace rules that every engine's validation applies before the harness publishes an arm."""

from __future__ import annotations

import gzip
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from benchmarks.artifacts.layout import logs_by_rank, trace_files_by_rank
from benchmarks.e2e.engines.api import Arm, CompileMode, Engine, RunSpec


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
