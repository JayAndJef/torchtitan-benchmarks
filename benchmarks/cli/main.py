"""The group, the one command that spans both systems, and the wiring.

What is left of the CLI once each measurement system's commands own their own
module: the ``click.Group`` itself, ``scenarios`` -- which lists both
registries and is the only command that legitimately knows about both, so it
binds with ``@cli.command`` right here -- and the four ``add_command`` calls
that attach the rest. This was the last module in the repo that had to know
about end-to-end runs and kernel isolation at once; now it is the only one,
and knowing about both is its whole job.

**The registration direction is load-bearing.** ``run``, ``evaluate`` and
``kernel-bench`` are declared with plain ``@click.command``
in ``e2e.py`` and ``kernel.py`` and attached here, so the import edge runs
main -> command module and never back. Had they kept ``@cli.command``, each
module would have to import the group from here, and ``from
benchmarks.cli.main import cli`` -- exactly what ``__main__.py`` and both CLI
test modules do -- would return a group holding only the commands whose
modules some earlier import had happened to load. That failure is silent: a
CLI missing ``evaluate`` still starts and still prints a usage message.
``tests/test_cli.py`` asserts that importing this module alone yields all
four commands.

``_show_event`` lives in ``rendering.py`` rather than here for the same
reason: both command families need it, and a command module importing it from
here would close the cycle this layout exists to avoid.

Torch-free, like the rest of the parent side, so ``./run_bench.sh scenarios``
and ``--help`` reach both registries in well under a second and never
initialize CUDA (``tests/test_import_boundaries.py``).
"""

from __future__ import annotations

import click

from benchmarks.cli.e2e import evaluate_command, run_command
from benchmarks.cli.kernel import kernel_bench_command
from benchmarks.e2e.registry import SCENARIOS
from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.spans import KERNEL_SPANS


@click.group()
def cli() -> None:
    """Run and evaluate declarative TorchTitan benchmarks."""


@cli.command("scenarios")
def scenarios_command() -> None:
    """List benchmark scenarios and their arms."""
    click.echo("end-to-end scenarios (run --scenario):")
    for scenario in SCENARIOS.values():
        click.echo(f"{scenario.name}: {scenario.description}")
        for arm in scenario.arms:
            click.echo(f"  {arm.name} — {arm.description}")
    click.echo("\nkernel scenarios (kernel-bench --scenario):")
    for scenario in KERNEL_SCENARIOS.values():
        click.echo(f"{scenario.name}: {scenario.description}")
        for arm in scenario.arms:
            click.echo(f"  {arm.name} — {arm.description}")

    # Listed even when the roster is empty, or nobody learns --span exists.
    click.echo("\nkernel spans (kernel-bench --span):")
    if not KERNEL_SPANS:
        click.echo("(none declared)")
    for span in KERNEL_SPANS.values():
        click.echo(
            f"{span.name} [replaces {' + '.join(span.scenarios)}]: "
            f"{span.description}"
        )
        for arm in span.arms:
            parts = " + ".join(
                f"{scenario}/{part}" for scenario, part in span.parts_for(arm.name)
            )
            click.echo(f"  {arm.name} — {arm.description}")
            click.echo(f"    against {parts}")


cli.add_command(run_command)
cli.add_command(evaluate_command)
cli.add_command(kernel_bench_command)
