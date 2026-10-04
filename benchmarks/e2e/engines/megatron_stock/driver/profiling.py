"""The shim that gives Megatron's profiler the schedule and the trace path that the harness reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

PROFILER_STEP_OFFSET = 1
"""The steps that the schedule skips first, because Megatron steps the profiler one step before TorchTitan does."""

TRACE_SUBDIR = "profiling/traces"
"""The trace directory under the arm directory, as both engines write it."""

WINDOW_DIR = "iteration_{step}"
TRACE_NAME = "rank{rank}_trace.json.gz"


@dataclass
class ProfilerShim:
    """The replaced ``torch.profiler.profile``, its replacement, and the calls and the windows that the replacement recorded."""

    replacement: Callable[..., Any]
    original: Callable[..., Any]
    arm_dir: Path
    rank: int
    calls: int = 0
    windows: list[Path] = field(default_factory=list)

    def uninstall(self) -> None:
        """Put the real ``torch.profiler.profile`` back."""
        torch.profiler.profile = self.original

    def written_windows(self) -> list[Path]:
        """The trace files of this rank, read from the disk."""
        return sorted(
            self.arm_dir.glob(
                f"{TRACE_SUBDIR}/iteration_*/"
                + TRACE_NAME.format(rank=self.rank)
            )
        )


def install_profiler_shim(
    *,
    arm_dir: Path,
    rank: int,
    profile_freq: int,
    profiler_warmup: int,
    profiler_active: int,
) -> ProfilerShim:
    """Replace ``torch.profiler.profile`` for the rest of this process with one that uses the harness schedule and trace path.

    Call this function once, before ``pretrain``.
    """
    wait = profile_freq - profiler_warmup - profiler_active
    if wait < 0:
        raise ValueError(
            f"a profiler cycle of {profile_freq} steps cannot hold "
            f"{profiler_warmup} warmup plus {profiler_active} active steps"
        )
    original = torch.profiler.profile
    schedule = torch.profiler.schedule(
        wait=wait,
        warmup=profiler_warmup,
        active=profiler_active,
        repeat=0,
        skip_first=PROFILER_STEP_OFFSET,
    )
    shim = ProfilerShim(
        replacement=lambda *a, **k: None,
        original=original,
        arm_dir=Path(arm_dir),
        rank=rank,
    )

    def trace_handler(prof: Any) -> None:
        window = (
            shim.arm_dir
            / TRACE_SUBDIR
            / WINDOW_DIR.format(step=prof.step_num)
        )
        window.mkdir(parents=True, exist_ok=True)
        path = window / TRACE_NAME.format(rank=shim.rank)
        prof.export_chrome_trace(str(path))
        shim.windows.append(path)
        print(f"Dumping profiler traces at step {prof.step_num}", flush=True)

    def replacement(*args: Any, **kwargs: Any) -> Any:
        shim.calls += 1
        kwargs["schedule"] = schedule
        kwargs["on_trace_ready"] = trace_handler
        return original(*args, **kwargs)

    shim.replacement = replacement
    torch.profiler.profile = replacement
    return shim


def assert_windows_written(
    shim: ProfilerShim, *, min_trace_windows: int
) -> list[Path]:
    """Raise when the shim did not run once, or when this rank wrote fewer than ``min_trace_windows`` windows; return the windows."""
    if min_trace_windows < 1:
        raise ValueError(
            f"a window requirement of {min_trace_windows} accepts a run "
            "that wrote nothing; a guard against a shim that did not "
            "install cannot evaluate to 'any count is acceptable'"
        )
    if shim.calls != 1:
        raise RuntimeError(
            f"the torch.profiler shim ran {shim.calls} time(s), not once; "
            "either megatron built no profiler (check --profile and "
            "--use-pytorch-profiler) or the shim was installed too late"
        )
    windows = shim.written_windows()
    if len(windows) < min_trace_windows:
        raise RuntimeError(
            f"rank {shim.rank} wrote {len(windows)} profiler window(s) under "
            f"{shim.arm_dir}, below the {min_trace_windows} this workload "
            "declares; do not lower the requirement, find the missing window"
        )
    return windows
