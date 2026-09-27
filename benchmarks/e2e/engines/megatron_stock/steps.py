"""The step record that the stock Megatron driver prints, and the reader of the step samples in one rank's log.

The reader also reads the text step line, which the stored run directories
hold.
"""

from __future__ import annotations

import json
import re
from types import MappingProxyType
from typing import Any

from benchmarks.e2e.engines.api import DroppedLine, StepRead, StepSample
from benchmarks.execution.launcher import RANK_PREFIX, holds_rank_prefix


STEP_PREFIX = "bench-step: "
"""The start of the step record; a JSON object follows it."""

STEP_RECORD = re.compile(rf"^(?:{RANK_PREFIX})?{re.escape(STEP_PREFIX)}(.*)$")
"""The step record, after the rank prefix that a log of one rank keeps."""

TORN_TAIL = re.compile(rf"\s*(?:{RANK_PREFIX}.*)?")
"""The text that may follow a step line: another rank's line, which a torn write appended."""

RECORD_KEYS = frozenset(
    {"step", "tokens_per_second", "peak_memory_gib", "loss", "grad_norm", "extras"}
)
"""The keys of the step record."""

TEXT_MARKER = re.compile(rf"^(?:{RANK_PREFIX})?step:")
"""The start of the text step line."""

RECORD_STEP = re.compile(r'^\{"step": (\d+)')
"""The step at the start of a step record, which a cut record can still show."""

TEXT_STEP = re.compile(r"\s*(\d+)")
"""The step after the text marker."""

_NUMBER = r"(nan|-?inf|-?[0-9.]+)"

TEXT_LINE = re.compile(
    rf"^(?:{RANK_PREFIX})?step:\s*(\d+)\s+(?:loss:\s*{_NUMBER}\s+)?"
    rf"grad_norm:\s*{_NUMBER}\s+memory:\s*([0-9.]+)GiB\([0-9.]+%\)\s+"
    rf"tps:\s*([0-9,]+)\s+tflops:\s*([0-9,.]+)\s+mfu:\s*([0-9.]+)%{TORN_TAIL.pattern}$"
)
"""The text step line; a rank that holds no loss prints no loss field."""


def _rounded(value: float | None, digits: int) -> float | None:
    """``value`` at the precision of TorchTitan's step line, so both engines publish the same digits."""
    return None if value is None else float(f"{value:.{digits}f}")


def step_record(
    *,
    step: int,
    tokens_per_second: int,
    peak_memory_gib: float,
    loss: float | None,
    grad_norm: float,
    tflops: float,
    mfu: float,
) -> str:
    """The step record of one rank and one step."""
    return STEP_PREFIX + json.dumps(
        {
            "step": step,
            "tokens_per_second": tokens_per_second,
            "peak_memory_gib": _rounded(peak_memory_gib, 2),
            "loss": _rounded(loss, 5),
            "grad_norm": _rounded(grad_norm, 4),
            "extras": {"tflops": _rounded(tflops, 2), "mfu": _rounded(mfu, 2)},
        }
    )


def _number(rank: int, record: dict[str, Any], key: str) -> float:
    """The number under ``key``; any other value raises."""
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"rank {rank} logs a step record whose {key} is {value!r}, not a number"
        )
    return value


def _whole_record(text: str) -> bool:
    """Whether ``text`` starts with a whole JSON value, followed by nothing but an appended rank line."""
    try:
        _, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return False
    return TORN_TAIL.fullmatch(text, end) is not None


def _record_sample(rank: int, text: str) -> StepSample:
    """The sample of one whole step record."""
    record, _ = json.JSONDecoder().raw_decode(text)
    if not isinstance(record, dict) or set(record) != RECORD_KEYS:
        found = sorted(record) if isinstance(record, dict) else type(record).__name__
        raise ValueError(
            f"rank {rank} logs a step record with keys {found}; a record "
            f"holds {sorted(RECORD_KEYS)}"
        )
    step = _number(rank, record, "step")
    if not isinstance(step, int) or step < 1:
        raise ValueError(f"rank {rank} logs a step record with step {step!r}")
    extras = record["extras"]
    if not isinstance(extras, dict):
        raise ValueError(
            f"rank {rank} logs a step record whose extras is {extras!r}, not an object"
        )
    return StepSample(
        rank=rank,
        step=step,
        tokens_per_second=_number(rank, record, "tokens_per_second"),
        peak_memory_gib=_number(rank, record, "peak_memory_gib"),
        loss=None if record["loss"] is None else _number(rank, record, "loss"),
        grad_norm=None
        if record["grad_norm"] is None
        else _number(rank, record, "grad_norm"),
        extras=MappingProxyType(
            {key: _number(rank, extras, key) for key in extras}
        ),
    )


def _text_sample(rank: int, match: re.Match[str]) -> StepSample:
    """The sample of one text step line that parses."""
    step, loss, grad_norm, memory, tps, tflops, mfu = match.groups()
    return StepSample(
        rank=rank,
        step=int(step),
        tokens_per_second=int(tps.replace(",", "")),
        peak_memory_gib=float(memory),
        loss=None if loss is None else float(loss),
        grad_norm=float(grad_norm),
        extras=MappingProxyType(
            {"tflops": float(tflops.replace(",", "")), "mfu": float(mfu)}
        ),
    )


def _dropped(rank: int, number: int, step: re.Match[str] | None) -> DroppedLine:
    """The dropped step line at ``number``, with the step that ``step`` matched."""
    return DroppedLine(
        rank=rank, line=number, step=int(step.group(1)) if step else None
    )


def read_steps(rank: int, text: str) -> StepRead:
    """The step samples of one rank's log, from step records or text step lines, and the step lines that a rank prefix cut."""
    samples = []
    dropped = []
    for number, line in enumerate(text.splitlines(), start=1):
        record = STEP_RECORD.match(line)
        marker = TEXT_MARKER.match(line)
        if record is not None:
            body = record.group(1)
            if _whole_record(body):
                samples.append(_record_sample(rank, body))
            elif holds_rank_prefix(body):
                dropped.append(_dropped(rank, number, RECORD_STEP.match(body)))
            else:
                raise ValueError(
                    f"rank {rank} logs a step record that does not parse: {body!r}"
                )
        elif marker is not None:
            match = TEXT_LINE.match(line)
            fields = line[marker.end() :]
            if match is not None:
                samples.append(_text_sample(rank, match))
            elif holds_rank_prefix(fields):
                dropped.append(_dropped(rank, number, TEXT_STEP.match(fields)))
            else:
                raise ValueError(
                    f"rank {rank} logs a Megatron step line that does not parse: "
                    f"{line!r}"
                )
    return StepRead(samples=tuple(samples), dropped=tuple(dropped))
