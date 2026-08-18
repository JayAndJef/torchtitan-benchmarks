"""How a runner's progress events reach the terminal.

Neither measurement system's runner prints anything. Both are parameterized
over an ``EventHandler`` and emit ``RunEvent``s instead, and
``benchmarks/execution/events.py`` states why: that inversion is what keeps
``execute_run`` and ``execute_kernel_run`` callable from ``tools/`` and from
the CPU tests, which pass no handler at all. This module is the CLI half of
that contract -- the only place in the repo that decides an event's
appearance -- and it is four lines of ``click.echo`` because that is the
entire policy: a new arm gets a blank line before it, an ``error`` event goes
to stderr, everything else is a plain line on stdout. ``RunEvent.kind`` is a
free string precisely so that adding an event kind needs no change here.

It is a module rather than a function in ``main.py`` for a structural reason.
``main.py`` owns the ``click.Group`` and attaches the commands defined in
``e2e.py`` and ``kernel.py`` with ``cli.add_command``, so the import edge runs
main -> command module and never back. This renderer is shared by
``e2e._execute`` and ``kernel.kernel_bench_command``, so putting it in
``main.py`` would turn that edge into a cycle; putting it in ``e2e.py`` would
make the kernel command depend on the end-to-end one for a line of output.
It therefore sits below both, and imports no other CLI module.

Folding it into ``benchmarks.execution.events`` instead was rejected. That
module is imported by both runners and reachable from ``tools/``, and it is
deliberately ``click``-free: a runner that could reach the renderer is a
runner one edit away from printing.
"""

from __future__ import annotations

import click

from benchmarks.execution.events import RunEvent


def _show_event(event: RunEvent) -> None:
    if event.kind == "arm":
        click.echo()
    click.echo(event.message, err=event.kind == "error")
