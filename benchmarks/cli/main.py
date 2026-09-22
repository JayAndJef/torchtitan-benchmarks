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

import re

from benchmarks.cli.e2e import evaluate_command, run_command
from benchmarks.cli.kernel import kernel_bench_command
from benchmarks.e2e.registry import SCENARIOS
from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.spans import KERNEL_SPANS


def _single_sentence(text: str) -> str:
    """Extract the first concise sentence from a description."""
    text = text.strip()
    if not text:
        return ""
    for m in re.finditer(r"([.!?])(?=\s+[A-Z]|\s*$)", text):
        idx = m.start()
        prefix = text[: idx + 1]
        if prefix.endswith("e.g.") or prefix.endswith("i.e.") or prefix.endswith("vs."):
            continue
        return prefix.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else text


def _echo_e2e_scenarios() -> None:
    click.echo("end-to-end scenarios (run --scenario):")
    for scenario in SCENARIOS.values():
        click.echo(f"{scenario.name}: {_single_sentence(scenario.description)}")
        for arm in scenario.arms:
            click.echo(f"  {arm.name} — {_single_sentence(arm.description)}")


def _echo_kernel_scenarios() -> None:
    click.echo("kernel scenarios (kernel-bench --scenario):")
    for scenario in KERNEL_SCENARIOS.values():
        click.echo(f"{scenario.name}: {_single_sentence(scenario.description)}")
        for arm in scenario.arms:
            click.echo(f"  {arm.name} — {_single_sentence(arm.description)}")


def _echo_kernel_spans() -> None:
    click.echo("kernel spans (kernel-bench --span):")
    if not KERNEL_SPANS:
        click.echo("(none declared)")
    for span in KERNEL_SPANS.values():
        replaces = " + ".join(span.scenarios)
        click.echo(
            f"{span.name} [replaces {replaces}]: {_single_sentence(span.description)}"
        )
        for arm in span.arms:
            parts = " + ".join(
                f"{scenario}/{part}" for scenario, part in span.parts_for(arm.name)
            )
            click.echo(f"  {arm.name} — {_single_sentence(arm.description)}")
            click.echo(f"    against {parts}")


def _is_known_target(name: str) -> bool:
    if name in SCENARIOS or name in KERNEL_SCENARIOS or name in KERNEL_SPANS:
        return True
    for scen in SCENARIOS.values():
        if target_matches_arm(scen.name, scen.arms, name):
            return True
    for scen in KERNEL_SCENARIOS.values():
        if target_matches_arm(scen.name, scen.arms, name):
            return True
    for span in KERNEL_SPANS.values():
        if target_matches_arm(span.name, span.arms, name):
            return True
    return False


def target_matches_arm(parent_name: str, arms: tuple | list, target: str) -> bool:
    if target.startswith(f"{parent_name}/"):
        suffix = target[len(parent_name) + 1 :]
        return any(a.name == suffix for a in arms)
    return False


def _show_detail(target: str) -> None:
    if target in SCENARIOS:
        scen = SCENARIOS[target]
        click.echo(f"End-to-end scenario: {scen.name}")
        click.echo(f"Command: ./run_bench.sh run <gpu> --scenario {scen.name}\n")
        click.echo(f"Description:\n  {scen.description}\n")
        click.echo("Arms:")
        for arm in scen.arms:
            click.echo(f"  {arm.name}:")
            click.echo(f"    {arm.description}")
        return

    if target in KERNEL_SCENARIOS:
        scen = KERNEL_SCENARIOS[target]
        click.echo(f"Kernel scenario: {scen.name}")
        click.echo(f"Command: ./run_bench.sh kernel-bench <gpu> --scenario {scen.name}\n")
        click.echo(f"Description:\n  {scen.description}\n")
        click.echo("Arms:")
        for arm in scen.arms:
            click.echo(f"  {arm.name}:")
            click.echo(f"    {arm.description}")
        return

    if target in KERNEL_SPANS:
        span = KERNEL_SPANS[target]
        click.echo(f"Kernel span: {span.name}")
        replaces = " + ".join(span.scenarios)
        click.echo(f"Replaces: {replaces}")
        click.echo(f"Command: ./run_bench.sh kernel-bench <gpu> --span {span.name}\n")
        click.echo(f"Description:\n  {span.description}\n")
        click.echo("Arms:")
        for arm in span.arms:
            parts = " + ".join(
                f"{s}/{p}" for s, p in span.parts_for(arm.name)
            )
            click.echo(f"  {arm.name}:")
            click.echo(f"    {arm.description}")
            click.echo(f"    Against: {parts}")
        return

    for scen_name, scen in SCENARIOS.items():
        if target.startswith(f"{scen_name}/"):
            arm_name = target[len(scen_name) + 1 :]
            arm = next((a for a in scen.arms if a.name == arm_name), None)
            if arm:
                click.echo(f"End-to-end arm: {scen.name}/{arm.name}")
                click.echo(f"Scenario: {scen.name}\n")
                click.echo(f"Description:\n  {arm.description}")
                return

    for scen_name, scen in KERNEL_SCENARIOS.items():
        if target.startswith(f"{scen_name}/"):
            arm_name = target[len(scen_name) + 1 :]
            arm = next((a for a in scen.arms if a.name == arm_name), None)
            if arm:
                click.echo(f"Kernel arm: {scen.name}/{arm.name}")
                click.echo(f"Scenario: {scen.name}\n")
                click.echo(f"Description:\n  {arm.description}")
                return

    for span_name, span in KERNEL_SPANS.items():
        if target.startswith(f"{span_name}/"):
            arm_name = target[len(span_name) + 1 :]
            arm = next((a for a in span.arms if a.name == arm_name), None)
            if arm:
                parts = " + ".join(
                    f"{s}/{p}" for s, p in span.parts_for(arm.name)
                )
                click.echo(f"Kernel span arm: {span.name}/{arm.name}")
                click.echo(f"Span: {span.name}")
                click.echo(f"Against: {parts}\n")
                click.echo(f"Description:\n  {arm.description}")
                return

    avail_e2e = ", ".join(SCENARIOS)
    avail_kernel = ", ".join(KERNEL_SCENARIOS)
    avail_spans = ", ".join(KERNEL_SPANS)
    raise click.ClickException(
        f"Unknown scenario, span, or arm '{target}'.\n\n"
        f"Available e2e scenarios: {avail_e2e}\n"
        f"Available kernel scenarios: {avail_kernel}\n"
        f"Available kernel spans: {avail_spans}"
    )


class ScenarioGroup(click.Group):
    """A Click Group that lists scenarios or dispatches to detail."""

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        if _is_known_target(cmd_name):
            @click.command(name=cmd_name)
            def _dynamic_detail() -> None:
                _show_detail(cmd_name)

            return _dynamic_detail
        return None


@click.group()
def cli() -> None:
    """Run and evaluate declarative TorchTitan benchmarks."""


@cli.group("scenarios", cls=ScenarioGroup, invoke_without_command=True)
@click.option(
    "--detail",
    "-d",
    "detail_target",
    default=None,
    help="Show full detail for a scenario, span, or arm.",
)
@click.pass_context
def scenarios_command(ctx: click.Context, detail_target: str | None = None) -> None:
    """List benchmark scenarios and their arms, or inspect scenario details."""
    if ctx.invoked_subcommand is not None:
        return
    if detail_target is not None:
        _show_detail(detail_target)
        return
    _echo_e2e_scenarios()
    click.echo()
    _echo_kernel_scenarios()
    click.echo()
    _echo_kernel_spans()
    click.echo(
        "\nTip:\n"
        "  run_bench.sh scenarios e2e             List end-to-end scenarios only\n"
        "  run_bench.sh scenarios kernel          List kernel scenarios and spans only\n"
        "  run_bench.sh scenarios detail <name>   Show full details for a scenario, span, or arm"
    )


@scenarios_command.command("e2e")
def scenarios_e2e_command() -> None:
    """List end-to-end benchmark scenarios and their arms."""
    _echo_e2e_scenarios()


@scenarios_command.command("kernel")
def scenarios_kernel_command() -> None:
    """List kernel-isolation benchmark scenarios and spans."""
    _echo_kernel_scenarios()
    click.echo()
    _echo_kernel_spans()


@scenarios_command.command("detail")
@click.argument("name")
def scenarios_detail_command(name: str) -> None:
    """Show full details and notes for a scenario, span, or arm."""
    _show_detail(name)


cli.add_command(run_command)
cli.add_command(evaluate_command)
cli.add_command(kernel_bench_command)
