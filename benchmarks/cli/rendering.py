"""Print the progress events of a runner on the terminal."""

from __future__ import annotations

import click

from benchmarks.execution.events import RunEvent


def _show_event(event: RunEvent) -> None:
    """Print one event: a blank line before an arm, and an ``error`` event on stderr."""
    if event.kind == "arm":
        click.echo()
    click.echo(event.message, err=event.kind == "error")
