"""Click command-line interface for benchmark execution and evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import click

from benchmarks.metrics import evaluate_run, write_results
from benchmarks.reporting import render_evaluation
from benchmarks.runner import RunEvent, RunRequest, RunResult, execute_run
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
        click.option("--out", "out_dir", type=click.Path(path_type=Path)),
        click.option("--seq-len", type=int, envvar="SEQ"),
        click.option("--steps", type=int, envvar="STEPS"),
        click.option("--batch", type=int, envvar="BATCH"),
        click.option(
            "--titan-dir",
            type=click.Path(path_type=Path),
            envvar="TITAN_DIR",
        ),
        click.option(
            "--cache-root",
            type=click.Path(path_type=Path),
            envvar="BENCHMARK_CACHE_ROOT",
        ),
        click.option(
            "--compiler-env",
            type=click.Path(path_type=Path),
            envvar="BENCH_COMPILER_ENV",
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
        extra_args=torchtitan_args,
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
    for scenario in SCENARIOS.values():
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
    resume_dir: Path | None,
    results_path: Path | None,
    torchtitan_args: tuple[str, ...],
    **options: Any,
) -> None:
    """Run, validate, and evaluate every arm in one scenario."""
    result = _execute(
        _request(gpu, torchtitan_args, resume_dir=resume_dir, **options)
    )
    click.echo("\nAll arms validated. Evaluating...")
    _evaluate(result.out_dir, (), results_path)


def _legacy_args(args: list[str]) -> list[str]:
    if args[:1] == ["--help"]:
        return args
    if "--list-scenarios" in args:
        return ["scenarios"]
    if args[:1] and args[0] in {"scenarios", "run", "evaluate", "run-all"}:
        return args
    return ["run", *args]


def legacy_main(
    args: list[str] | None = None,
    *,
    prog_name: str = "run_bench.sh",
) -> None:
    """Accept the original positional-GPU form while using Click underneath."""
    actual_args = list(sys.argv[1:] if args is None else args)
    cli.main(args=_legacy_args(actual_args), prog_name=prog_name)


def legacy_evaluate_main(args: list[str] | None = None) -> None:
    """Translate the old ``--arms A B`` analyzer syntax to repeated options."""
    tokens = list(sys.argv[1:] if args is None else args)
    translated: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token != "--arms":
            translated.append(token)
            index += 1
            continue
        index += 1
        while index < len(tokens) and not tokens[index].startswith("--"):
            translated.extend(("--arm", tokens[index]))
            index += 1
    cli.main(args=["evaluate", *translated], prog_name="analysis/compare_arms.py")


if __name__ == "__main__":
    legacy_main(prog_name="python -m benchmarks.cli")
