"""The end-to-end command family: ``run`` and ``evaluate``.

Everything the declarative scenario runs need from the CLI, and nothing the
kernel benchmarks do. ``_execution_options`` is the reason this is a module:
the nineteen options ``run`` takes are the largest single block in the CLI.
Sitting two hundred lines above ``kernel-bench``'s own option stack it was
not readable which surface a given ``click.option`` belonged to.

**The duplicate ``--model-size`` is deliberate and must stay duplicated.**
This module declares it with ``envvar="MODEL_SIZE"`` and no default, so an
unrequested size reaches ``RunRequest`` as ``None`` -- which is what lets
``run --resume`` tell "inherit the size recorded in the manifest" from
"the caller asked for the default". ``benchmarks/cli/kernel.py`` declares
its own with an explicit default and no envvar, because ``kernel-bench`` takes
flags only, so that an ``OUT``/``SEQ``/``BATCH`` environment exported for an
end-to-end shell cannot leak into a kernel measurement. They are two
different options that share a spelling, and unifying them would change
behavior on one side or the other. Separating the modules is what makes the
asymmetry visible instead of hiding it behind two hundred lines.

The commands are declared with plain ``@click.command`` and attached to the
group by ``benchmarks/cli/main.py`` with ``cli.add_command``. Binding them
with ``@cli.command`` instead would require importing the group from
``main``, and then ``from benchmarks.cli.main import cli`` -- what
``__main__.py`` and both CLI test modules do -- would yield a group holding
only whichever commands some other import had already loaded.

**``run`` is one command, and it always evaluates.** ``--scenario``
narrows a default that is every scenario, so an omitted flag runs the whole
roster rather than one scenario under a label the operator assumed. Three
options name a single directory -- ``--out``, ``--resume`` and
``--results`` -- so each one needs exactly one selected scenario.
``benchmarks/e2e/runner.py`` still holds the rule that a run without a
scenario name is refused: a resume reads the name from the manifest, and
this module passes ``None`` for that case alone.

The private helpers are the CLI's whole share of run logic: ``_request``
turns option keywords into a ``RunRequest``, ``_axes`` builds the run-axis
record inside it, ``_execute`` narrows the runner's exceptions to
``ClickException``, ``_evaluate`` renders an evaluation and says where the
machine-readable copy landed, and ``_run_and_evaluate`` records an evaluation
failure in ``run_state.json`` so a later ``--resume`` retries it. The tests
patch ``execute_run`` and ``_evaluate`` at *this* module, because that is
where those names are bound.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

import click

from benchmarks.artifacts.layout import run_timestamp
from benchmarks.artifacts.run_state import record_evaluation_status
from benchmarks.cli.rendering import _show_event
from benchmarks.e2e.axes import RequestedAxes, RunRequest
from benchmarks.e2e.schema import (
    DEFAULT_ZERO,
    ParallelismSpec,
    ZERO_MODES,
)
from benchmarks.e2e.parallelism import (
    MEGATRON_ENGINES,
    PP_SCHEDULE_CHOICES,
)
from benchmarks.e2e.registry import (
    AC_MODES,
    DEFAULT_AC_MODE,
    DEFAULT_MEGATRON_NAN_GUARD,
    DEFAULT_MEGATRON_PRECISION,
    DEFAULT_MEGATRON_P2P_SYNC,
    DEFAULT_MODEL_SIZE,
    DEFAULT_PROFILE,
    DEFAULT_WARMUP_STEPS,
    MEGATRON_NAN_GUARD_MODES,
    MEGATRON_PRECISION_MODES,
    MEGATRON_P2P_SYNC_MODES,
    SCENARIOS,
)
from benchmarks.e2e.results import evaluate_run, render_evaluation, write_results
from benchmarks.e2e.runner import (
    RunResult,
    execute_run,
    megatron_nan_guard_refusal,
    megatron_precision_refusal,
)
from benchmarks.models.piper_qwen3.shape import MODEL_SIZE_CHOICES


PASSTHROUGH_CONTEXT = {
    "ignore_unknown_options": True,
    "allow_extra_args": True,
}


def _execution_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """The option block ``run`` takes beside its own four options.

    **The six parallelism options take no environment variable, and the
    three older axes do.** The asymmetry is deliberate and the reason is
    specific: each parallelism value has to agree with the ``<gpu>``
    positional, which names the device set, and a positional has no
    environment form. An exported ``PP=2`` would therefore make a plain
    ``run 0 --scenario X`` fail its own world-size check -- rule 1 of
    ``benchmarks/e2e/parallelism.py`` compares ``dp * pp`` against the number
    of devices requested -- and the operator would see a refusal naming a
    flag they did not pass. ``AC_MODE`` and ``MODEL_SIZE`` have no such
    partner and stay exported.

    ``--zero`` joins them for the same reason, one step removed. A sharded
    level says something only above ``dp`` 1, so the level has to agree
    with the ``<gpu>`` positional too. An exported ``ZERO=3`` would make a
    plain ``run 0 --scenario X`` carry a level the mesh cannot hold, and
    the warning would name a flag the operator never passed.

    Each of the six defaults to ``None``, meaning "not requested", exactly
    as ``--model-size`` does: ``_request`` builds a ``ParallelismSpec`` only
    when at least one was given, so an untouched command line reaches
    ``_resolve_run`` with ``parallelism=None`` and resolves to the trivial
    spec.

    ``--megatron-p2p-sync`` takes no environment variable either, for the
    reason ``--zero`` gives. Its ``off`` value is legal only above
    ``pp`` 1 and only beside a megatron arm, so it has to agree with the
    ``<gpu>`` positional and with ``--arm``. An exported value would make a
    plain ``run 0 --scenario X`` fail a refusal naming a flag the operator
    never passed. It defaults to ``None`` for the reason ``--ac`` does: a
    resume inherits the recorded value, and a fresh run takes ``on``.

    ``--megatron-nan-guard`` takes no environment variable for the same
    reason. Its ``off`` value is legal beside a stock megatron arm alone,
    so it has to agree with ``--scenario`` and with ``--arm``, and an
    exported value would fail a plain titan run on a flag nobody passed.
    It defaults to ``None`` as the option above does.

    ``--megatron-precision`` takes no environment variable for the same
    reason again, and it has one more agreement to keep: ``lean`` needs a
    sharded ``--zero`` level, so an exported value would fail every
    replicated run on a flag nobody passed.

    ``--profile`` takes no environment variable for the reason the five
    above give, and its partner is ``--steps``. A profiled run needs at
    least 40 steps, so an exported ``PROFILE=1`` would refuse a plain
    ``run 0 --steps 12`` on a flag the operator never passed. It defaults
    to ``None`` as the options above do: a resume inherits the recorded
    value, and a fresh run takes ``DEFAULT_PROFILE``.

    ``--warmup-steps`` is the one option of this block that *does* take an
    environment variable, and the asymmetry costs something: an exported
    ``WARMUP_STEPS`` refuses every ``--profile`` run, on a flag the
    operator did not pass. Unset it before a profiled run. It is exported
    because the matrix supervisor drives a sweep of unprofiled cells and
    one value serves the whole sweep, which is the case ``AC_MODE`` and
    ``MODEL_SIZE`` are exported for.
    """
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
        # The six parallelism options. No envvar on any of them; the
        # docstring above gives the reason.
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
        # No envvar; the docstring above gives the reason.
        click.option(
            "--megatron-p2p-sync",
            "megatron_p2p_sync",
            type=click.Choice(MEGATRON_P2P_SYNC_MODES),
            help=(
                "Whether Megatron synchronizes the device after every "
                f"pipeline message [default: {DEFAULT_MEGATRON_P2P_SYNC}]. "
                "on is stock Megatron, and it needs --pp above 1 and a "
                "megatron arm; TorchTitan arms receive nothing. Results "
                "are only comparable within one value."
            ),
        ),
        # No envvar; the docstring above gives the reason.
        click.option(
            "--megatron-nan-guard",
            "megatron_nan_guard",
            type=click.Choice(MEGATRON_NAN_GUARD_MODES),
            help=(
                "Whether stock Megatron checks every loss and gradient for "
                f"NaN and Inf [default: {DEFAULT_MEGATRON_NAN_GUARD}]. on "
                "is stock Megatron. off sends Megatron's own "
                "--no-check-for-nan-in-loss-and-grad to the stock megatron "
                "arm, and it needs one; TorchTitan arms receive nothing. "
                "Results are only comparable within one value."
            ),
        ),
        # No envvar; the docstring above gives the reason.
        click.option(
            "--megatron-precision",
            "megatron_precision",
            type=click.Choice(MEGATRON_PRECISION_MODES),
            help=(
                "How stock Megatron holds the optimizer state [default: "
                f"{DEFAULT_MEGATRON_PRECISION}]. stock is --bf16 alone, "
                "which is 18 bytes per parameter. lean adds the "
                "precision-aware optimizer with bf16 gradients and bf16 "
                "Adam moments, which is 10. lean needs --zero 1, because "
                "Megatron asserts the distributed optimizer under it, and "
                "it reaches the stock megatron arm alone. Results are only "
                "comparable within one value."
            ),
        ),
        # No envvar; the docstring above gives the reason.
        # A boolean pair rather than a bare flag, so an omitted option
        # reaches ``RunRequest`` as ``None`` and a resume can inherit the
        # recorded value.
        click.option(
            "--profile/--no-profile",
            "profile",
            default=None,
            help=(
                "Whether the run collects profiler traces [default: "
                f"{'on' if DEFAULT_PROFILE else 'off'}]. On, both engines "
                "write <arm>/profiling/traces/iteration_*/ and every trace "
                "rule applies, and --steps must be at least 40. Off, the "
                "run writes no trace and the evaluation publishes no kernel "
                "time. Results are only comparable within one value."
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
"""The six option names of one ``ParallelismSpec``, each with its default.

``_parallelism`` pops all six, so a renamed option here is a renamed keyword
there and nowhere else. Five defaults are a second copy of the spec's own,
which can drift; a test compares every row against ``ParallelismSpec()``.
"""


def _parallelism(options: dict[str, Any]) -> ParallelismSpec | None:
    """Pop the six parallelism options and build the spec they describe.

    Returns ``None`` when the operator gave none of them, which is what
    ``RunRequest.parallelism`` reads as "not requested". A spec built from
    all six defaults would be the same object, but ``None`` is what lets a
    later reader tell an untouched command line from one that asked for the
    trivial spec by name.
    """
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
    """Refuse ``--warmup-steps`` beside ``--profile``.

    The two name two sample rules for one figure. A profiled run samples
    the steps of each cycle the profiler is idle on
    (``benchmarks/e2e/results.py``'s ``stable_tps``), so a warmup count
    would reach no reader and the manifest would record a rule the
    evaluation did not use. Refused rather than ignored: a silently
    dropped option publishes a number the operator did not ask for.
    """
    if options.get("profile") and options.get("warmup_steps") is not None:
        raise click.UsageError(
            "--profile uses the profiler schedule; --warmup-steps applies "
            "only without it"
        )


_AXIS_OPTIONS = (
    "ac_mode",
    "model_size",
    "megatron_p2p_sync",
    "megatron_nan_guard",
    "megatron_precision",
    "profile",
    "warmup_steps",
)
"""The seven option names ``RequestedAxes`` takes directly.

``_parallelism`` builds the spec from six more. ``_axes`` pops all seven, so
an option renamed here is a keyword renamed there and nowhere else.
"""


def _axes(options: dict[str, Any]) -> RequestedAxes:
    """Pop the axis options and build the record they describe.

    Every value stays as Click delivered it, so an omitted option reaches
    ``_resolve_run`` as ``None`` and a resume inherits the recorded value.
    """
    parallelism = _parallelism(options)
    _refuse_warmup_under_profile(options)
    return RequestedAxes(
        parallelism=parallelism,
        **{name: options.pop(name, None) for name in _AXIS_OPTIONS},
    )


def _request(
    gpu: str,
    torchtitan_args: tuple[str, ...],
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
@click.argument("torchtitan_args", nargs=-1, type=click.UNPROCESSED)
def run_command(
    gpu: str,
    scenario_names: tuple[str, ...],
    arm_names: tuple[str, ...],
    resume_dir: Path | None,
    results_path: Path | None,
    torchtitan_args: tuple[str, ...],
    **options: Any,
) -> None:
    """Run, validate and evaluate arms; pass TorchTitan arguments after --.

    Every scenario runs unless ``--scenario`` narrows the set. Named
    scenarios run one at a time, in the order given, and a name may repeat:
    each repeat is another run of it, under an output directory of its own.
    The run is fail-fast: the first failing arm stops it, and the scenarios
    behind it never start.
    """
    requested = bool(scenario_names)
    selected = scenario_names if requested else tuple(SCENARIOS)
    _refuse_unknown_scenarios(selected)
    _refuse_a_missing_arm(selected, arm_names)
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
        # A copy per scenario: ``_axes`` pops the axis options out of it.
        scenario_options: dict[str, Any] = dict(options)
        _run_and_evaluate(
            _request(
                gpu,
                torchtitan_args,
                # A resume reads the scenario from the manifest, so an
                # operator who named none leaves the question to it.
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
            "every scenario of this sweep declines one of the run axes, so "
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
    selected: tuple[str, ...], arm_names: tuple[str, ...]
) -> None:
    """Refuse an ``--arm`` name that any selected scenario lacks.

    The option applies to every selected scenario, so a name one of them
    does not declare would run a smaller matrix than the operator asked
    for. Refused here, before a GPU is claimed.
    """
    for name in selected:
        available = {arm.name for arm in SCENARIOS[name].arms}
        missing = [arm for arm in arm_names if arm not in available]
        if missing:
            raise click.UsageError(
                f"scenario {name!r} has no arm(s) "
                + ", ".join(repr(arm) for arm in missing)
                + f". Available: {', '.join(sorted(available))}"
            )


def _refuse_a_single_run_option(
    selected: tuple[str, ...], given: tuple[tuple[str, Any], ...]
) -> None:
    """Refuse the options that name one directory, above one scenario.

    ``--out``, ``--resume`` and ``--results`` each name a single path. Two
    scenarios sharing one would overwrite each other's manifest and
    results, so the count has to be one.
    """
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
    """Why a swept scenario declines one of the global axes, or ``None``.

    A sweep skips such a scenario and says why, rather than aborting: the
    restriction is a declaration, not a fault. A ``--scenario`` that names
    the scenario gets the matching refusal from ``_resolve_run`` instead.

    A sweep that skips every scenario is a different case, and ``run``
    refuses it: the command ran no arm, and an exit code of 0 would report
    a measurement that never happened.
    """
    scenario = SCENARIOS[name]
    ac_mode = options.get("ac_mode") or DEFAULT_AC_MODE
    megatron_p2p_sync = (
        options.get("megatron_p2p_sync") or DEFAULT_MEGATRON_P2P_SYNC
    )
    megatron_nan_guard = (
        options.get("megatron_nan_guard") or DEFAULT_MEGATRON_NAN_GUARD
    )
    megatron_precision = (
        options.get("megatron_precision") or DEFAULT_MEGATRON_PRECISION
    )
    # Read rather than popped: ``_axes`` pops it from the per-scenario copy,
    # and this needs the value alone.
    zero = options.get("zero")
    if zero is None:
        zero = DEFAULT_ZERO

    if ac_mode not in scenario.supported_ac_modes:
        reason = (
            f"does not support ac mode {ac_mode!r} "
            f"(supported: {', '.join(scenario.supported_ac_modes)})"
        )
    elif megatron_p2p_sync == "on" and not any(
        arm.engine in MEGATRON_ENGINES for arm in scenario.arms
    ):
        reason = (
            f"--megatron-p2p-sync {megatron_p2p_sync!r} reaches no arm of "
            "this scenario (every arm runs on TorchTitan)"
        )
    else:
        reason = megatron_nan_guard_refusal(
            scenario.arms, megatron_nan_guard
        ) or megatron_precision_refusal(
            scenario.arms, megatron_precision, zero
        )
    return reason


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
