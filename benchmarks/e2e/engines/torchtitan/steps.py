"""The step samples that one TorchTitan rank's log states, read from the fork's own step line."""

from __future__ import annotations

import re
from types import MappingProxyType

from benchmarks.e2e.engines.api import StepSample
from benchmarks.execution.launcher import RANK_PREFIX


COLOR_CODE = re.compile(r"\x1b\[[0-9;]*m")
"""The terminal color codes that the fork puts into its step line."""

STEP_MARKER = re.compile(r"(?<!validate )step:\s*\d")
"""The first field of the step line; the validation line starts with ``validate step:``."""

_NUMBER = r"(nan|-?inf|-?[0-9.]+)"

TORN_TAIL = rf"\s*(?:{RANK_PREFIX}.*)?"
"""The text that may follow a step line: another rank's line, which a torn write appended."""

STEP_LINE = re.compile(
    rf"step:\s*(\d+)\s+loss:\s*{_NUMBER}\s+grad_norm:\s*{_NUMBER}\s+"
    r"memory:\s*([0-9.]+)GiB\([0-9.]+%\)\s+tps:\s*([0-9,]+)\s+"
    rf"tflops:\s*([0-9,.]+)\s+mfu:\s*(?:([0-9.]+)%|N/A){TORN_TAIL}"
)
"""The fields of the step line, from ``step:`` to the end of the line."""

NO_LOSS = -1.0
"""The loss that the fork logs on a pipeline rank that holds no loss."""


def _step_sample(rank: int, line: str) -> StepSample | None:
    """The sample of one log line, or ``None`` when the line is not a step line."""
    plain = COLOR_CODE.sub("", line)
    marker = STEP_MARKER.search(plain)
    if marker is None:
        return None
    match = STEP_LINE.fullmatch(plain, marker.start())
    if match is None:
        raise ValueError(
            f"rank {rank} logs a TorchTitan step line that does not parse: "
            f"{plain.strip()!r}"
        )
    step, loss, grad_norm, memory, tps, tflops, mfu = match.groups()
    extras = {"tflops": float(tflops.replace(",", ""))}
    if mfu is not None:
        extras["mfu"] = float(mfu)
    return StepSample(
        rank=rank,
        step=int(step),
        tokens_per_second=int(tps.replace(",", "")),
        peak_memory_gib=float(memory),
        loss=None if float(loss) == NO_LOSS else float(loss),
        grad_norm=float(grad_norm),
        extras=MappingProxyType(extras),
    )


def read_steps(rank: int, text: str) -> list[StepSample]:
    """The step samples in one rank's log, in log order."""
    return [
        sample
        for line in text.splitlines()
        if (sample := _step_sample(rank, line)) is not None
    ]
