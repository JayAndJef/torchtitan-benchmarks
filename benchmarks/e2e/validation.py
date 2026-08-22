"""Reject partial or wrongly configured end-to-end runs before analysis.

``validate_arm`` is the gate every arm passes before its numbers are
published. Engine differences live in the ``VALIDATION_PROFILES`` registry,
selected by ``Arm.validation``; the structural rules -- trace-window count,
kernel markers, ``cudaGraphLaunch`` under cuda-graph mode, override counting,
and the parameter-count line -- are shared. Compiled-region structure is
*not*: it is a per-profile field (``check_regions``), because the megatron
arm has no Inductor graph annotations to match.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from benchmarks.artifacts.layout import trace_files
from benchmarks.e2e.registry import (
    CUDAGRAPH_COMPILE_MODES,
    TORCH_COMPILE_MODE,
    UNCOMPILED_COMPILE_MODES,
    Arm,
    Workload,
)
from benchmarks.models.piper_qwen3.shape import shape_by_name
from benchmarks.traces.extraction import pooled_window_metrics
from benchmarks.traces.schema import Region


_SAC_APPLIED_LINE = "Applied SelectiveAC activation checkpointing"


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
    """

    completion_marker: str
    mode_line: Callable[[str], str]
    compiled_marker: str | None
    failure_markers: tuple[str, ...]
    check_ac_line: bool
    check_regions: bool


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
) -> None:
    """Reject partial or wrongly configured runs before analysis."""
    profile = VALIDATION_PROFILES[arm.validation]
    shape = shape_by_name(model_size)
    if not log_path.is_file():
        raise RuntimeError(f"{arm.name}: training log is missing: {log_path}")
    log = log_path.read_text(errors="replace")
    if profile.completion_marker not in log:
        raise RuntimeError(f"{arm.name}: training did not complete; see {log_path}")
    if compile_mode in UNCOMPILED_COMPILE_MODES:
        # Rule 8 inverts here: an uncompiled arm prints no compile line, so
        # the proof is the absence of one. Never relax this into "skip the
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
                f"engine compiled the model; see {log_path}"
            )
    # The engine reports which mode it actually applied.
    elif profile.mode_line(compile_mode) not in log:
        raise RuntimeError(
            f"{arm.name}: compile mode {compile_mode!r} did not apply; "
            f"see {log_path}"
        )
    if profile.check_ac_line:
        # The AC policy logs its application; its presence must match the
        # requested mode or the run measured the wrong recompute treatment.
        sac_applied = _SAC_APPLIED_LINE in log
        if ac_mode == "sac" and not sac_applied:
            raise RuntimeError(
                f"{arm.name}: ac mode 'sac' requested but SelectiveAC was not "
                f"applied; see {log_path}"
            )
        if ac_mode == "none" and sac_applied:
            raise RuntimeError(
                f"{arm.name}: ac mode 'none' requested but SelectiveAC was "
                f"applied; see {log_path}"
            )
    # Both engines print this line; without the check a run whose --config
    # or --model-size silently fell back to another shape would pass every
    # other rule and be published under the wrong size.
    size_marker = f"size: {shape.param_count:,} total parameters"
    if size_marker not in log:
        raise RuntimeError(
            f"{arm.name}: model size {model_size!r} "
            f"({shape.param_count:,} parameters) did not apply; see {log_path}"
        )
    if arm.overrides_per_block:
        expected_overrides = arm.overrides_per_block * shape.n_layers
        override_count = len(re.findall(r"\[Override\]", log))
        if override_count != expected_overrides:
            raise RuntimeError(
                f"{arm.name}: expected {expected_overrides} override "
                "applications, "
                f"found {override_count}; see {log_path}"
            )
        for override_import in arm.override_imports:
            if f"[Override] {override_import}:" not in log:
                raise RuntimeError(
                    f"{arm.name}: override {override_import!r} did not apply; "
                    f"see {log_path}"
                )
    for marker in profile.failure_markers:
        if marker in log:
            raise RuntimeError(
                f"{arm.name}: silent fallback marker {marker!r} found in the "
                f"log; see {log_path}"
            )

    traces = trace_files(arm_dir)
    if len(traces) < workload.min_trace_windows:
        raise RuntimeError(
            f"{arm.name}: expected at least {workload.min_trace_windows} "
            "profiler windows, "
            f"found {len(traces)} under {arm_dir}"
        )
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
    if regions and profile.check_regions:
        try:
            pooled_window_metrics(traces, regions)
        except ValueError as error:
            raise RuntimeError(
                f"{arm.name}: profiler traces failed structural validation: {error}"
            ) from error
