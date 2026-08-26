"""Give stock Megatron the profiler layout the harness reads.

Stock Megatron writes **one** window per rank, to
``{args.tensorboard_dir}/../torch_profile/rank-<rank>.json.gz``
(``megatron/training/training.py``). Its schedule carries ``repeat=1``.

The harness reads
``<arm_dir>/profiling/traces/iteration_*/rank<n>_trace.json.gz`` and arm
rule 5 needs ``min_trace_windows`` files per rank, which is 2. Two windows
are what every published figure in this repository rests on, so the repair
is to give Megatron the right schedule and the right path, never to lower
the rule.

``training.py`` calls ``torch.profiler.profile`` by attribute at run time,
so replacing that attribute before ``pretrain()`` reaches the call is
enough. The replacement keeps every keyword the caller passed except
``schedule`` and ``on_trace_ready``.

**The shim is loud in both failure directions.** It counts its own calls,
and ``assert_windows_written`` refuses a run whose call count is not 1 and a
run that wrote fewer windows than the workload declares. A shim that did not
install writes nothing under ``arm_dir`` and reports zero calls, so both
guards fire.

``record_shapes`` stays whatever Megatron asked for, which is
``--pytorch-profiler-collect-shapes`` and therefore False. No metric in
``benchmarks/traces/`` reads a shape; ``tools/analyze.py`` does, so add that
flag before you run that diagnostic on this arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

# The trace layout both engines write and benchmarks/artifacts/layout.py
# reads back.
TRACE_SUBDIR = "profiling/traces"
WINDOW_DIR = "iteration_{step}"
TRACE_NAME = "rank{rank}_trace.json.gz"


@dataclass
class ProfilerShim:
    """What the shim replaced, what it built, and what it recorded."""

    replacement: Callable[..., Any]
    original: Callable[..., Any]
    arm_dir: Path
    rank: int
    calls: int = 0
    windows: list[Path] = field(default_factory=list)

    def uninstall(self) -> None:
        """Put the real ``torch.profiler.profile`` back.

        The driver does not need this; a test does, because a replaced
        attribute outlives the function that set it.
        """
        torch.profiler.profile = self.original

    def written_windows(self) -> list[Path]:
        """This rank's trace files, read from the disk.

        Reads the disk rather than the recorded list, because the recorded
        list says what the handler believes and the rule reads what the run
        produced.
        """
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
    """Replace ``torch.profiler.profile`` for the rest of this process.

    The schedule is the one both other engines run: ``wait`` fills the cycle
    to ``profile_freq``, then ``profiler_warmup`` untimed steps and
    ``profiler_active`` timed ones, repeating for the whole run
    (``repeat=0``). Megatron calls ``prof.step()`` once per training
    iteration, so a 40-step run at the default schedule writes two windows,
    named ``iteration_20`` and ``iteration_40`` after the profiler's own
    step number.

    Measured against the real ``torch.profiler.schedule`` of the pinned
    torch: the schedule returns ``RECORD_AND_SAVE`` at step 19 and 39 and
    ``NONE`` at 20 and 40, so both windows flush through ``prof.step()``
    alone. The flush count is exactly ``train_iters // profile_freq``,
    because the cycle length is ``profile_freq`` by construction.

    Megatron's own ``prof.stop()`` at ``--profile-step-end`` then transits
    ``NONE -> None``, which its action map does not hold and which is
    therefore a no-op. **That holds only because ``flags.py`` ends the
    profiler on a cycle boundary.** torch's map does hold
    ``(RECORD, None)`` and ``(RECORD_AND_SAVE, None)``, and both write a
    window -- so a stop inside an active window adds a third, short trace
    that arm rule 5 and the per-step metrics would count.

    Call this once per process, before ``pretrain()``.
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
    )
    shim = ProfilerShim(
        replacement=lambda *a, **k: None,  # replaced below
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
    """Refuse a run that did not write the windows the workload declares.

    Two failures reach this function, and each has its own message:

    * the shim was not called exactly once, which means Megatron built no
      profiler or built more than one; and
    * this rank wrote fewer windows than ``min_trace_windows``, which is
      what arm rule 5 needs and what region pooling and the trace metrics
      rest on.

    Returns the windows this rank wrote.
    """
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
