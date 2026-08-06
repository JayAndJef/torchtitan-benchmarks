"""Click command-line interface for benchmark execution and evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import click

from benchmarks.artifacts import record_evaluation_status
from benchmarks.kernel_runner import KernelRunRequest, execute_kernel_run
from benchmarks.kernels import KERNEL_SCENARIOS
from benchmarks.metrics import evaluate_run, write_results
from benchmarks.reporting import render_evaluation, render_kernel_results
from benchmarks.runner import (
    RunEvent,
    RunRequest,
    RunResult,
    execute_run,
    run_timestamp,
)
from benchmarks.scenarios import SCENARIOS


PASSTHROUGH_CONTEXT = {
    "ignore_unknown_options": True,
    "allow_extra_args": True,
}


def _execution_options(command: Callable[..., Any]) -> Callable[..., Any]:
    options = [
        click.option("--scenario", help="Declarative benchmark scenario name."),
        click.option(
            "--hardware",
            default="auto",
            show_default=True,
            help="Stable output/provenance label; auto uses the GPU name.",
        ),
        click.option(
            "--out",
            "out_dir",
            type=click.Path(path_type=Path),
            envvar="OUT",
            show_envvar=True,
        ),
        click.option("--seq-len", type=int, envvar="SEQ", show_envvar=True),
        click.option("--steps", type=int, envvar="STEPS", show_envvar=True),
        click.option("--batch", type=int, envvar="BATCH", show_envvar=True),
        click.option(
            "--cache-root",
            type=click.Path(path_type=Path),
            envvar="BENCHMARK_CACHE_ROOT",
            show_envvar=True,
        ),
        click.option(
            "--compiler-env",
            type=click.Path(path_type=Path),
            envvar="BENCH_COMPILER_ENV",
            show_envvar=True,
            help="Shell script that enables the host compiler for CUDA extensions.",
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def _request(
    gpu: str,
    torchtitan_args: tuple[str, ...],
    *,
    arm_name: str | None = None,
    resume_dir: Path | None = None,
    **options: Any,
) -> RunRequest:
    scenario_name = options.pop("scenario")
    if scenario_name is None and resume_dir is None:
        scenario_name = "piper1b_rope"
    return RunRequest(
        gpu=gpu,
        scenario_name=scenario_name,
        arm_name=arm_name,
        resume_dir=resume_dir,
        extra_args=(
            None if resume_dir is not None and not torchtitan_args else torchtitan_args
        ),
        **options,
    )


def _show_event(event: RunEvent) -> None:
    if event.kind == "arm":
        click.echo()
    click.echo(event.message)


def _execute(request: RunRequest) -> RunResult:
    try:
        return execute_run(request, event_handler=_show_event)
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error


def _evaluate(out_dir: Path, arms: tuple[str, ...], results_path: Path | None) -> None:
    try:
        result = evaluate_run(out_dir, arms or None)
        destination = write_results(result, results_path)
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(render_evaluation(result))
    click.echo(f"\nmachine-readable results: {destination}")


@click.group()
def cli() -> None:
    """Run and evaluate declarative TorchTitan benchmarks."""


@cli.command("scenarios")
def scenarios_command() -> None:
    """List benchmark scenarios and their arms."""
    click.echo("end-to-end scenarios (run / run-all):")
    for scenario in SCENARIOS.values():
        arms = ", ".join(arm.name for arm in scenario.arms)
        click.echo(f"{scenario.name}: {scenario.description}")
        click.echo(f"  arms: {arms}")
    click.echo("\nkernel scenarios (kernel-bench):")
    for scenario in KERNEL_SCENARIOS.values():
        arms = ", ".join(arm.name for arm in scenario.arms)
        click.echo(f"{scenario.name}: {scenario.description}")
        click.echo(f"  arms: {arms}")


@cli.command("run", context_settings=PASSTHROUGH_CONTEXT)
@click.argument("gpu")
@click.option("--arm", "arm_name", help="Run one arm instead of every arm.")
@_execution_options
@click.argument("torchtitan_args", nargs=-1, type=click.UNPROCESSED)
def run_command(
    gpu: str,
    arm_name: str | None,
    torchtitan_args: tuple[str, ...],
    **options: Any,
) -> None:
    """Run and validate selected arms; pass TorchTitan arguments after --."""
    result = _execute(_request(gpu, torchtitan_args, arm_name=arm_name, **options))
    click.echo(f"\nAll selected arms validated: {result.out_dir}")
    if "baseline" in (arm.name for arm in result.selected_arms):
        click.echo(f"Evaluate with: run_bench.sh evaluate {result.out_dir}")


@cli.command("evaluate")
@click.argument(
    "out_dir", type=click.Path(path_type=Path, exists=True, file_okay=False)
)
@click.option("--arm", "arms", multiple=True, help="Arm subset; repeat per arm.")
@click.option(
    "--results",
    "results_path",
    type=click.Path(path_type=Path),
    help="JSON destination; defaults to <output-dir>/results.json.",
)
def evaluate_command(
    out_dir: Path, arms: tuple[str, ...], results_path: Path | None
) -> None:
    """Report throughput, compiled-region timings, and significance tests."""
    _evaluate(out_dir, arms, results_path)


@cli.command("run-all", context_settings=PASSTHROUGH_CONTEXT)
@click.argument("gpu")
@click.option(
    "--all-scenarios",
    "all_scenarios",
    is_flag=True,
    help="Run every scenario in sequence, stopping at the first failure.",
)
@click.option(
    "--resume",
    "resume_dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Resume an interrupted output directory and retry incomplete arms.",
)
@click.option(
    "--results",
    "results_path",
    type=click.Path(path_type=Path),
    help="JSON destination; defaults to <output-dir>/results.json.",
)
@_execution_options
@click.argument("torchtitan_args", nargs=-1, type=click.UNPROCESSED)
def run_all_command(
    gpu: str,
    all_scenarios: bool,
    resume_dir: Path | None,
    results_path: Path | None,
    torchtitan_args: tuple[str, ...],
    **options: Any,
) -> None:
    """Run, validate, and evaluate every arm in one or every scenario."""
    if not all_scenarios:
        _run_and_evaluate(
            _request(gpu, torchtitan_args, resume_dir=resume_dir, **options),
            results_path,
        )
        return

    for flag, value in (
        ("--scenario", options["scenario"]),
        ("--out", options["out_dir"]),
        ("--resume", resume_dir),
        ("--results", results_path),
    ):
        if value is not None:
            raise click.UsageError(f"--all-scenarios cannot be combined with {flag}")

    # One stamp for the sweep so every scenario lands under out/<stamp>/.
    timestamp = run_timestamp()
    for name in SCENARIOS:
        click.echo(f"\n===== scenario: {name} =====")
        scenario_options: dict[str, Any] = {**options, "scenario": name}
        _run_and_evaluate(
            _request(gpu, torchtitan_args, timestamp=timestamp, **scenario_options),
            None,
        )


@cli.command("kernel-bench")
@click.argument("gpu")
@click.option(
    "--scenario",
    "scenario_names",
    multiple=True,
    type=click.Choice(list(KERNEL_SCENARIOS)),
    help="Kernel scenario subset; repeat per scenario. Default: all.",
)
@click.option("--n", default=200, show_default=True, help="Interleaved cycles.")
@click.option(
    "--warmup", default=30, show_default=True, help="Warmup cycles per mode."
)
@click.option(
    "--burst",
    is_flag=True,
    help="Add the 1/4/16/64 burst dispatch-cost diagnostic.",
)
@click.option("--batch", type=int, help="Override the Piper-1B batch size.")
@click.option("--seq-len", type=int, help="Override the Piper-1B sequence length.")
@click.option("--seed", default=0, show_default=True, help="Input seed.")
@click.option(
    "--hardware",
    default="auto",
    show_default=True,
    help="Stable output/provenance label; auto uses the GPU name.",
)
@click.option(
    "--out",
    "out_dir",
    type=click.Path(path_type=Path),
    help="Output directory; only valid with a single --scenario.",
)
@click.option(
    "--cache-root",
    type=click.Path(path_type=Path),
    envvar="BENCHMARK_CACHE_ROOT",
    show_envvar=True,
)
@click.option(
    "--compiler-env",
    type=click.Path(path_type=Path),
    envvar="BENCH_COMPILER_ENV",
    show_envvar=True,
    help="Shell script that enables the host compiler for CUDA extensions.",
)
def kernel_bench_command(
    gpu: str,
    scenario_names: tuple[str, ...],
    out_dir: Path | None,
    **options: Any,
) -> None:
    """Benchmark kernel implementations head-to-head in isolation."""
    selected = scenario_names or tuple(KERNEL_SCENARIOS)
    if out_dir is not None and len(selected) != 1:
        raise click.UsageError(
            "--out requires exactly one --scenario; otherwise scenarios would "
            "overwrite each other"
        )
    request = KernelRunRequest(
        gpu=gpu, scenario_names=selected, out_dir=out_dir, **options
    )
    try:
        outcomes = execute_kernel_run(request, event_handler=_show_event)
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    for outcome in outcomes:
        click.echo()
        if outcome.result is not None:
            click.echo(render_kernel_results(outcome.result))
            click.echo(f"\nmachine-readable results: {outcome.out_dir}/results.json")
        if outcome.error:
            click.echo(f"ERROR {outcome.scenario}: {outcome.error}", err=True)

    failures = [outcome for outcome in outcomes if outcome.failed]
    if failures:
        summary = ", ".join(
            f"{outcome.scenario}"
            f"{' (correctness)' if outcome.correctness_failed else ''}"
            for outcome in failures
        )
        raise click.ClickException(f"kernel scenarios failed: {summary}")


def _run_and_evaluate(request: RunRequest, results_path: Path | None) -> None:
    result = _execute(request)
    click.echo("\nAll arms validated. Evaluating...")
    try:
        _evaluate(result.out_dir, (), results_path)
    except click.ClickException as error:
        record_evaluation_status(
            result.out_dir, completed=False, error=error.format_message()
        )
        raise
    record_evaluation_status(result.out_dir, completed=True)


if __name__ == "__main__":
    cli(prog_name="run_bench.sh")
