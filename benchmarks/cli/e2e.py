"""The end-to-end command family: ``run``, ``run-all`` and ``evaluate``.

Everything the declarative scenario runs need from the CLI, and nothing the
kernel benchmarks do. ``_execution_options`` is the reason this is a module:
the eleven options every execution command shares are the largest single
block in the CLI, they are applied to exactly two commands, and both are
here. Sitting two hundred lines above ``kernel-bench``'s own option stack it
was not readable which surface a given ``click.option`` belonged to.

**The duplicate ``--model-size`` is deliberate and must stay duplicated.**
This module declares it with ``envvar="MODEL_SIZE"`` and no default, so an
unrequested size reaches ``RunRequest`` as ``None`` -- which is what lets
``run-all --resume`` tell "inherit the size recorded in the manifest" from
"the caller asked for ``1b``". ``benchmarks/cli/kernel.py`` declares its
own with ``default="1b"`` and no envvar, because ``kernel-bench`` takes
flags only, so that an ``OUT``/``SEQ``/``BATCH`` environment exported for an
end-to-end shell cannot leak into a kernel measurement (CLAUDE.md,
"Kernel-isolation benchmarks"). They are two different options that share a
spelling, and unifying them would change behavior on one side or the other.
Separating the modules is what makes the asymmetry visible instead of
hiding it behind two hundred lines.

The commands are declared with plain ``@click.command`` and attached to the
group by ``benchmarks/cli/main.py`` with ``cli.add_command``. Binding them
with ``@cli.command`` instead would require importing the group from
``main``, and then ``from benchmarks.cli.main import cli`` -- what
``__main__.py`` and both CLI test modules do -- would yield a group holding
only whichever commands some other import had already loaded.

The private helpers are the CLI's whole share of run logic: ``_request``
turns option keywords into a ``RunRequest`` and supplies the default scenario
only when this is not a resume, ``_execute`` narrows the runner's exceptions
to ``ClickException``, ``_evaluate`` renders an evaluation and says where the
machine-readable copy landed, and ``_run_and_evaluate`` records an evaluation
failure in ``run_state.json`` so a later ``--resume`` retries it. The tests
patch ``execute_run`` and ``_evaluate`` at *this* module, because that is
where those names are bound.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import click

from benchmarks.artifacts.layout import run_timestamp
from benchmarks.artifacts.run_state import record_evaluation_status
from benchmarks.cli.rendering import _show_event
from benchmarks.e2e.registry import AC_MODES, COMPILE_MODES, SCENARIOS
from benchmarks.e2e.results import evaluate_run, render_evaluation, write_results
from benchmarks.e2e.runner import RunRequest, RunResult, execute_run
from benchmarks.models.piper_qwen3.shape import MODEL_SIZE_CHOICES


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
        click.option(
            "--compile-mode",
            type=click.Choice(COMPILE_MODES),
            envvar="COMPILE_MODE",
            show_envvar=True,
            help=(
                "Compile mode applied to every arm in the run [default: "
                "default]. cuda-graph maps to torch.compile reduce-overhead "
                "for TorchTitan arms. Results are only comparable within one "
                "mode."
            ),
        ),
        click.option(
            "--ac",
            "ac_mode",
            type=click.Choice(AC_MODES),
            envvar="AC_MODE",
            show_envvar=True,
            help=(
                "Activation checkpointing applied to every arm in the run "
                "[default: sac]. Results are only comparable within one mode."
            ),
        ),
        click.option(
            "--model-size",
            "model_size",
            type=click.Choice(MODEL_SIZE_CHOICES),
            envvar="MODEL_SIZE",
            show_envvar=True,
            help=(
                "Model shape applied to every arm in the run [default: 1b]. "
                "See benchmarks/models/piper_qwen3/shape.py. Results are only "
                "comparable within one size."
            ),
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


@click.command("run", context_settings=PASSTHROUGH_CONTEXT)
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


@click.command("evaluate")
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


@click.command("run-all", context_settings=PASSTHROUGH_CONTEXT)
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
    ac_mode = options.get("ac_mode") or "sac"
    for name, scenario in SCENARIOS.items():
        if ac_mode not in scenario.supported_ac_modes:
            click.echo(
                f"\n===== scenario: {name} ====="
                f"\nskipped: does not support ac mode {ac_mode!r} "
                f"(supported: {', '.join(scenario.supported_ac_modes)})"
            )
            continue
        click.echo(f"\n===== scenario: {name} =====")
        scenario_options: dict[str, Any] = {**options, "scenario": name}
        _run_and_evaluate(
            _request(gpu, torchtitan_args, timestamp=timestamp, **scenario_options),
            None,
        )


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
