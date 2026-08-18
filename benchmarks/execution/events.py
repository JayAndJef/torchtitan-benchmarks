"""The seam between a runner and whatever is driving it.

Both measurement systems' runners are parameterized over two callables, and
neither has anything to do with the subprocess *environment* they used to be
filed under:

* ``ProcessRunner`` -- the ``subprocess.run``-shaped callable a runner
  launches through. ``benchmarks.e2e.runner.execute_run`` defaults it to
  ``subprocess.run`` and the CPU tests substitute a fake, which is the only
  reason the whole resume/archive/validation path can be exercised on a box
  with no GPU.
* ``RunEvent`` / ``EventHandler`` / ``_emit`` -- structured progress out. A
  runner never prints; it emits, and ``benchmarks.cli.rendering`` decides how
  an event is rendered. That inversion is what keeps the runners callable
  from ``tools/`` and from tests, and it is why ``RunEvent`` is the only name
  the whole ``benchmarks.cli`` package takes from ``execution/`` -- one
  module, one import, and no ``click`` on this side of the seam.

``_emit`` accepts a ``None`` handler so that the fourteen call sites in
``e2e/runner.py`` stay single expressions instead of conditionals; a runner
with no observer is the normal case for a test, not an error.

The types are deliberately loose. ``RunEvent.kind`` is a free string rather
than an enum because the renderer treats unknown kinds as plain lines, so
adding an event never requires a coordinated change in the CLI, and
``ProcessRunner`` is ``Callable[..., ...]`` because the runners pass
``subprocess.run``'s keyword arguments straight through.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RunEvent:
    kind: str
    message: str
    arm_name: str | None = None


EventHandler = Callable[[RunEvent], None]
ProcessRunner = Callable[..., subprocess.CompletedProcess]


def _emit(
    handler: EventHandler | None,
    kind: str,
    message: str,
    arm_name: str | None = None,
) -> None:
    if handler is not None:
        handler(RunEvent(kind=kind, message=message, arm_name=arm_name))
