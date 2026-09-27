"""The end-to-end commands ``run`` and ``evaluate``; ``benchmarks/cli/main.py`` attaches them to the group."""

from __future__ import annotations

import shlex
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import click

from benchmarks.artifacts.layout import run_timestamp
from benchmarks.artifacts.run_state import record_evaluation_status
from benchmarks.cli.rendering import _show_event
from benchmarks.e2e.axes import RequestedAxes, RunRequest
from benchmarks.e2e.overrides import Override, parse_override
from benchmarks.e2e.parallelism import (
    DEFAULT_ZERO,
    PP_SCHEDULE_CHOICES,
    ParallelismSpec,
    ZERO_MODES,
)
from benchmarks.e2e.registry import (
    AC_MODES,
    DEFAULT_AC_MODE,
    DEFAULT_MODEL_SIZE,
    DEFAULT_PROFILE,
    DEFAULT_WARMUP_STEPS,
    SCENARIOS,
)
from benchmarks.e2e.results import evaluate_run, render_evaluation, write_results
from benchmarks.e2e.runner import RunResult, execute_run
from benchmarks.models.piper_qwen3.shape import MODEL_SIZE_CHOICES


REMOVED_OPTIONS: dict[str, tuple[str, ...]] = {
    "--megatron-p2p-sync": ("megatron_stock.p2p_sync={}",),
    "--megatron-nan-guard": ("megatron_stock.nan_guard={}",),
    "--megatron-precision": ("megatron_stock.precision={}",),
    "--megatron-arg": ("megatron_stock.extra_flags+={}",),
    "--torchtitan-arg": (
        "titan_compiled.extra_flags+={}",
        "titan_eager.extra_flags+={}",
    ),
}
"""Each removed ``run`` option, and the ``--set`` values that replace one of its values."""


def replacement(flag: str, value: str) -> str:
    """The ``--set`` spelling of one value of the removed option ``flag``."""
    return " ".join(
        f"--set {shlex.quote(template.format(value))}"
        for template in REMOVED_OPTIONS[flag]
    )


def _refuse_removed(
    context: click.Context, parameter: click.Parameter, values: tuple[str, ...]
) -> None:
    """Refuse a removed option, and name the ``--set`` spelling of each value."""
    if not values:
        return
    flag = parameter.opts[0]
    raise click.UsageError(
        "; ".join(
            f"{flag} {shlex.quote(value)} is now {replacement(flag, value)}"
            for value in values
        ),
        context,
    )


def _parse_overrides(
    context: click.Context, parameter: click.Parameter, values: tuple[str, ...]
) -> tuple[Override, ...]:
    """The ``--set`` values, parsed; a malformed value is a usage error."""
    try:
        return tuple(parse_override(value) for value in values)
    except ValueError as error:
        raise click.BadParameter(str(error), context, parameter) from error


def _execution_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """The options that ``run`` takes beside its own; the parallelism options and ``--set`` read no environment variable."""
    options = [
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
            "--ac",
            "ac_mode",
            type=click.Choice(AC_MODES),
            envvar="AC_MODE",
            show_envvar=True,
            help=(
                "Activation checkpointing applied to every arm in the run "
                f"[default: {DEFAULT_AC_MODE}]. Results are only comparable "
                "within one mode."
            ),
        ),
        click.option(
            "--model-size",
            "model_size",
            type=click.Choice(MODEL_SIZE_CHOICES),
            envvar="MODEL_SIZE",
            show_envvar=True,
            help=(
                "Model shape applied to every arm in the run "
                f"[default: {DEFAULT_MODEL_SIZE}]. "
                "See benchmarks/models/piper_qwen3/shape.py. Results are only "
                "comparable within one size."
            ),
        ),
        click.option(
            "--dp",
            "dp",
            type=click.IntRange(min=1),
            help=(
                "Data-parallel degree [default: 1]. dp x pp must equal the "
                "number of devices in the <gpu> argument."
            ),
        ),
        click.option(
            "--pp",
            "pp",
            type=click.IntRange(min=1),
            help=(
                "Pipeline-parallel degree [default: 1]. Above 1 it needs "
                "--pp-schedule."
            ),
        ),
        click.option(
            "--ep",
            "ep",
            type=click.IntRange(min=1),
            help=(
                "Expert-parallel degree [default: 1]. Needs --zero 1, "
                "because TorchTitan cannot split the experts and keep the "
                "dense parameters replicated. ep takes its ranks out of "
                "the dp axis."
            ),
        ),
        click.option(
            "--pp-schedule",
            "pp_schedule",
            type=click.Choice(PP_SCHEDULE_CHOICES),
            help="Pipeline schedule; required at --pp above 1.",
        ),
        click.option(
            "--pp-microbatch-size",
            "pp_microbatch_size",
            type=click.IntRange(min=1),
            help=(
                "Rows per pipeline microbatch [default: 1]. The local batch "
                "size must divide by it."
            ),
        ),
        click.option(
            "--zero",
            "zero",
            type=click.Choice(ZERO_MODES),
            help=(
                "The ZeRO level the run holds the dense parameters at "
                f"[default: {DEFAULT_ZERO}]. 0 keeps a whole copy on every "
                "rank. 1 shards the optimizer states. An expert degree "
                "needs 1. Results are only comparable within one level."
            ),
        ),
        # A boolean pair, so an omitted option reaches RunRequest as None.
        click.option(
            "--profile/--no-profile",
            "profile",
            default=None,
            help=(
                "Whether the run collects profiler traces [default: "
                f"{'on' if DEFAULT_PROFILE else 'off'}]. On, both engines "
                "write <arm>/profiling/traces/iteration_*/ and every trace "
                "rule applies, and --steps must be at least 40. Off, the "
                "run writes no trace. Results are only comparable within "
                "one value."
            ),
        ),
        click.option(
            "--warmup-steps",
            "warmup_steps",
            type=click.IntRange(min=0),
            envvar="WARMUP_STEPS",
            show_envvar=True,
            help=(
                "Steps an unprofiled run discards before it measures "
                f"[default: {DEFAULT_WARMUP_STEPS}]. --steps must be more "
                "than it. Refused beside --profile, which samples around "
                "the profiler schedule instead. Results are only comparable "
                "within one value."
            ),
        ),
        click.option(
            "--set",
            "overrides",
            multiple=True,
            metavar="ARM.FIELD=VALUE",
            callback=_parse_overrides,
            help=(
                "Set one field of one arm's engine config; repeat per field. "
                "ARM.FIELD+=VALUE appends to a list field, and shell rules "
                "split VALUE, so 'megatron_stock.extra_flags+=--moe-permute-"
                "fusion' adds one flag. Results are only comparable within "
                "one config."
            ),
        ),
        *(
            click.option(
                flag,
                multiple=True,
                hidden=True,
                expose_value=False,
                callback=_refuse_removed,
            )
            for flag in REMOVED_OPTIONS
        ),
    ]
    for option in reversed(options):
        command = option(command)
    return command


_PARALLELISM_OPTIONS = (
    ("dp", 1),
    ("pp", 1),
    ("ep", 1),
    ("pp_schedule", None),
    ("pp_microbatch_size", 1),
    ("zero", DEFAULT_ZERO),
)
"""The six option names of one ``ParallelismSpec``, each with its default."""


def _parallelism(options: dict[str, Any]) -> ParallelismSpec | None:
    """Pop the six parallelism options; ``None`` when the operator gave none of them."""
    given = {
        name: options.pop(name, None) for name, _ in _PARALLELISM_OPTIONS
    }
    if all(value is None for value in given.values()):
        return None
    return ParallelismSpec(
        **{
            name: (given[name] if given[name] is not None else default)
            for name, default in _PARALLELISM_OPTIONS
        }
    )


def _refuse_warmup_under_profile(options: dict[str, Any]) -> None:
    """Refuse ``--warmup-steps`` beside ``--profile``; the two name two sample rules."""
    if options.get("profile") and options.get("warmup_steps") is not None:
        raise click.UsageError(
            "--profile uses the profiler schedule; --warmup-steps applies "
            "only without it"
        )


_AXIS_OPTIONS = ("ac_mode", "model_size", "profile", "warmup_steps")
"""The option names that ``RequestedAxes`` takes directly."""


def _axes(options: dict[str, Any]) -> RequestedAxes:
    """Pop the axis options; an omitted option stays ``None``."""
    parallelism = _parallelism(options)
    _refuse_warmup_under_profile(options)
    return RequestedAxes(
        parallelism=parallelism,
        **{name: options.pop(name, None) for name in _AXIS_OPTIONS},
    )


def _request(
    gpu: str,
    *,
    scenario_name: str | None,
    arm_names: tuple[str, ...] = (),
    resume_dir: Path | None = None,
    **options: Any,
) -> RunRequest:
    axes = _axes(options)
    return RunRequest(
        gpu=gpu,
        scenario_name=scenario_name,
        arm_names=arm_names,
        resume_dir=resume_dir,
        axes=axes,
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


@click.command("run")
@click.argument("gpu")
@click.option(
    "--scenario",
    "scenario_names",
    multiple=True,
    help=(
        "Scenario to run; repeat per scenario. Omit to run every scenario "
        "in sequence."
    ),
)
@click.option(
    "--arm",
    "arm_names",
    multiple=True,
    help=(
        "Arm subset; repeat per arm. It applies to every selected scenario, "
        "so each one must declare each name. Omit to run every arm."
    ),
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
def run_command(
    gpu: str,
    scenario_names: tuple[str, ...],
    arm_names: tuple[str, ...],
    resume_dir: Path | None,
    results_path: Path | None,
    **options: Any,
) -> None:
    """Run, validate and evaluate arms.

    Every scenario runs unless ``--scenario`` narrows the set. A name may
    repeat, and each repeat is another run. The first failing arm stops the
    command.
    """
    requested = bool(scenario_names)
    selected = scenario_names if requested else tuple(SCENARIOS)
    _refuse_unknown_scenarios(selected)
    _refuse_a_missing_arm(
        selected,
        (
            ("--arm", arm_names),
            ("--set", tuple(override.arm for override in options["overrides"])),
        ),
    )
    _refuse_a_single_run_option(
        selected,
        (
            ("--out", options["out_dir"]),
            ("--resume", resume_dir),
            ("--results", results_path),
        ),
    )

    # One stamp above every scenario of a multi-scenario run.
    timestamp = run_timestamp() if len(selected) > 1 else None
    skipped: list[str] = []
    executed = False
    occurrences: Counter[str] = Counter()
    for name in selected:
        occurrences[name] += 1
        if not requested:
            reason = _skip_reason(name, options)
            if reason is not None:
                click.echo(f"\n===== scenario: {name} =====\nskipped: {reason}")
                skipped.append(f"{name}: {reason}")
                continue
        if len(selected) > 1:
            click.echo(f"\n===== scenario: {name} =====")
        # A copy per scenario, because _axes pops the axis options out.
        scenario_options: dict[str, Any] = dict(options)
        _run_and_evaluate(
            _request(
                gpu,
                # A resume reads the scenario from the manifest.
                scenario_name=(
                    name if requested or resume_dir is None else None
                ),
                arm_names=arm_names,
                resume_dir=resume_dir,
                timestamp=timestamp,
                occurrence=occurrences[name],
                **scenario_options,
            ),
            results_path,
        )
        executed = True
    if not executed:
        raise click.ClickException(
            "every selected scenario declines one of the run axes, so "
            "nothing ran:\n  "
            + "\n  ".join(skipped)
            + "\nChange the axis, or name a scenario with --scenario to get "
            "the refusal for it."
        )


def _refuse_unknown_scenarios(selected: tuple[str, ...]) -> None:
    unknown = [name for name in selected if name not in SCENARIOS]
    if unknown:
        raise click.UsageError(
            "no such scenario: "
            + ", ".join(repr(name) for name in unknown)
            + f". Available: {', '.join(SCENARIOS)}"
        )


def _refuse_a_missing_arm(
    selected: tuple[str, ...], given: tuple[tuple[str, tuple[str, ...]], ...]
) -> None:
    """Refuse an arm name in ``--arm`` or ``--set`` that a selected scenario lacks."""
    for name in selected:
        available = {arm.name for arm in SCENARIOS[name].arms}
        for flag, arm_names in given:
            missing = [arm for arm in dict.fromkeys(arm_names) if arm not in available]
            if missing:
                raise click.UsageError(
                    f"{flag}: scenario {name!r} has no arm(s) "
                    + ", ".join(repr(arm) for arm in missing)
                    + f". Available: {', '.join(sorted(available))}"
                )


def _refuse_a_single_run_option(
    selected: tuple[str, ...], given: tuple[tuple[str, Any], ...]
) -> None:
    """Refuse ``--out``, ``--resume`` or ``--results`` above one scenario; each names one path."""
    if len(selected) == 1:
        return
    for flag, value in given:
        if value is not None:
            raise click.UsageError(
                f"{flag} names one directory, and this run selects "
                f"{len(selected)} scenarios ({', '.join(selected)}); "
                "pass --scenario once"
            )


def _skip_reason(name: str, options: dict[str, Any]) -> str | None:
    """Why a scenario declines the requested ac mode, or ``None``; a run of every scenario skips that scenario."""
    scenario = SCENARIOS[name]
    ac_mode = options.get("ac_mode") or DEFAULT_AC_MODE
    if ac_mode not in scenario.supported_ac_modes:
        return (
            f"does not support ac mode {ac_mode!r} "
            f"(supported: {', '.join(scenario.supported_ac_modes)})"
        )
    return None


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
    """Report throughput, step time and peak memory for a finished run."""
    _evaluate(out_dir, arms, results_path)


def _run_and_evaluate(request: RunRequest, results_path: Path | None) -> None:
    """Run, then evaluate, and record an evaluation failure so that a resume retries it."""
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
