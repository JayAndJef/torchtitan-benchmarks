"""The progress events that a runner emits, and the process runner that it starts processes through."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RunEvent:
    """One progress message of a runner; the CLI prints it by its ``kind``."""

    kind: str
    message: str
    arm_name: str | None = None


EventHandler = Callable[[RunEvent], None]
"""The receiver of the events of a runner."""

ProcessRunner = Callable[..., subprocess.CompletedProcess]
"""A callable with the signature of ``subprocess.run``, which a test replaces."""


def _emit(
    handler: EventHandler | None,
    kind: str,
    message: str,
    arm_name: str | None = None,
) -> None:
    """Send one event to ``handler``; a ``None`` handler drops it."""
    if handler is not None:
        handler(RunEvent(kind=kind, message=message, arm_name=arm_name))
