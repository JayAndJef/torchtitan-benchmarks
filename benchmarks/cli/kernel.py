"""The ``kernel-bench`` command: the kernel-isolation family, on its own.

One command and its own option stack. The option count that used to open this
paragraph is gone: it was stale two commits after it was written, and a
reader can list the options from the file below. With the end-to-end family
the command shares the group, the event renderer and the model-shape
registry, and nothing else -- different request type, different runner,
different results schema, different reporter -- so its options no longer sit
two hundred lines below ``_execution_options``, an eleven-option block that
never applied to it.

**Environment variables are declined here on purpose.** Only ``--cache-root``
and ``--compiler-env`` read one; ``--out``, ``--seq-len``, ``--batch`` and
``--model-size`` are flags only, so an ``OUT``/``SEQ``/``BATCH`` environment
exported for an end-to-end session cannot leak into a kernel measurement
(CLAUDE.md, "Kernel-isolation benchmarks"). That is also why ``--model-size``
is declared twice in this package rather than shared: this one defaults to
``normal`` and shows it, while ``benchmarks/cli/e2e.py``'s carries
``envvar="MODEL_SIZE"`` and no default so a resume can tell an unrequested
size from an explicit ``normal``. Two options that share a spelling; see that
module's docstring for the other half.

The single ``--out`` guard is here rather than in ``KernelRunRequest``
because it is a usage error about flags, not a property of a request: without
it, several scenarios would resolve to the same directory and overwrite each
other's ``results.json``. It refuses ``--span`` outright, because a span run
always measures the scenarios the span replaces as well and is therefore
never one unit. ``--arm`` refuses ``--span`` for the other half of the same
fact: an arm name belongs to one scenario's roster, and a span run has
several rosters, so the selection cannot say which one it names. Everything
else this function does after the runner returns is reporting -- errors were
already streamed through the shared renderer as they happened, so only the
successful scenarios' reports are printed, and a nonzero exit summarizes the
failures.

Two import facts. ``from benchmarks.kernel...`` inside a module named
``benchmarks.cli.kernel`` resolves to the top-level package, not to this one:
Python 3 imports are absolute. And the command is declared with a plain
``@click.command`` and attached by ``benchmarks/cli/main.py`` via
``cli.add_command``, so importing ``main`` is what populates the group and no
command module imports it back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from benchmarks.cli.rendering import _show_event
from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.results.reporting import (
    render_kernel_results,
    render_kernel_span_results,
)
from benchmarks.kernel.results.schema import KernelSpanResult
from benchmarks.kernel.runner import KernelRunRequest, execute_kernel_run
from benchmarks.kernel.spans import KERNEL_SPANS
from benchmarks.models.piper_qwen3.shape import MODEL_SIZE_CHOICES


@click.command("kernel-bench")
@click.argument("gpu")
@click.option(
    "--scenario",
    "scenario_names",
    multiple=True,
    type=click.Choice(list(KERNEL_SCENARIOS)),
    help="Kernel scenario subset; repeat per scenario. Default: all.",
)
@click.option(
    "--arm",
    "arm_names",
    multiple=True,
    help=(
        "Arm subset within the one selected scenario; repeat per arm. "
        "Default: every arm. The selection must name the anchor arm, "
        "because every comparison is a ratio against it, and every "
        "correctness reference the selected arms use. A selection that "
        "omits either is refused, not repaired: adding an arm the operator "
        "did not ask for changes what the run measures."
    ),
)
@click.option(
    "--span",
    "span_names",
    multiple=True,
    type=click.Choice(list(KERNEL_SPANS)),
    help=(
        "Kernel span to measure; repeat per span. Default: none. A span is "
        "compared against the SUM of the scenarios it replaces, and those "
        "scenarios are measured in the same run -- so asking for one span "
        "can add several scenarios to the run."
    ),
)
@click.option(
    "--replicates",
    default=5,
    show_default=True,
    # Each of the three counts something a run cannot have none of, and none
    # of the three refused a zero here. ``--replicates 0`` reached median()
    # with an empty list and died on a StatisticsError from the standard
    # library; ``--samples-per-replicate 0`` timed one burst and discarded
    # it, so every arm produced an empty mode map; ``--burst-k 0`` divided a
    # burst by no calls and was caught only inside the worker. The range
    # check states the requirement where the operator reads it, before a GPU
    # is claimed.
    type=click.IntRange(min=1),
    help="Sweeps of every arm; the repetition unit the CI is taken over.",
)
@click.option(
    "--replicates-per-process",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Consecutive replicates of one arm per worker process. 1 rebuilds the "
        "arm for every replicate and keeps the replicate-major sweep; higher "
        "values buy wall-clock and cost the drift cancellation the ratio "
        "relies on, so the results file publishes the interval as "
        "within_process_ratio_ci_* instead. Two arms never share a process at "
        "any value. Use 1 for anything published."
    ),
)
@click.option(
    "--samples-per-replicate",
    default=40,
    show_default=True,
    type=click.IntRange(min=1),
    help="Timed bursts per arm per mode, within one replicate.",
)
@click.option(
    "--burst-k",
    default=16,
    show_default=True,
    type=click.IntRange(min=1),
    help="Calls per timed burst. One value for every arm in the scenario.",
)
@click.option(
    "--warmup-calls",
    default=30,
    show_default=True,
    help="Untimed calls per arm per mode, before each replicate.",
)
@click.option(
    "--burst",
    is_flag=True,
    help="Add the 1/4/16/64 burst dispatch-cost diagnostic.",
)
@click.option(
    "--model-size",
    default="1b",
    show_default=True,
    type=click.Choice(MODEL_SIZE_CHOICES),
    help=(
        "Model shape from benchmarks/models/piper_qwen3/shape.py; sizes the "
        "geometry only."
    ),
)
@click.option(
    "--batch",
    type=int,
    help="Batch size run through the model; not a model property.",
)
@click.option(
    "--seq-len",
    type=int,
    help="Sequence length run through the model; not a model property.",
)
@click.option(
    "--max-seq-len",
    type=int,
    help=(
        "Raise the shape's max_seq_len ceiling (default 2048); needed to "
        "sweep attention_core past 2048. Also sizes the RoPE cos/sin "
        "tables."
    ),
)
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
    arm_names: tuple[str, ...],
    span_names: tuple[str, ...],
    out_dir: Path | None,
    **options: Any,
) -> None:
    """Benchmark kernel implementations head-to-head in isolation."""
    # ``--scenario`` defaults to every scenario; ``--span`` defaults to none.
    # A span drags every scenario it encloses into the run, so a default of
    # "all spans" would silently change what a bare invocation costs. An
    # explicit ``--span`` with no ``--scenario`` measures that span and its
    # range, and nothing else.
    selected = scenario_names or (() if span_names else tuple(KERNEL_SCENARIOS))
    if out_dir is not None and (len(selected) != 1 or span_names):
        # A span is never allowed here, whatever else was asked for: it
        # measures the scenarios it replaces in the same run, so a span run
        # is always several units and they would all resolve to this one
        # directory.
        raise click.UsageError(
            "--out requires exactly one --scenario and no --span; a span "
            "also measures every scenario it replaces, so the units would "
            "overwrite each other"
        )
    # ``--arm`` and ``--span`` do not combine. A span run always measures
    # several scenarios -- the span itself and every scenario it replaces --
    # so one arm selection cannot say which roster it names, and a per-
    # scenario selection has no meaning across a range.
    if arm_names and span_names:
        raise click.UsageError(
            "--arm does not combine with --span; a span run measures several "
            "scenarios and an arm selection belongs to one roster"
        )
    # An arm name belongs to one roster. Two scenarios share neither their
    # arms nor their anchor, so a selection applied to both would mean a
    # different thing in each, and a name valid in one would be a typo in the
    # other. The roster itself is checked in resolve_arm_skips.
    if arm_names and len(selected) != 1:
        raise click.UsageError(
            "--arm requires exactly one --scenario; arm names are per "
            "scenario and no roster is shared"
        )
    request = KernelRunRequest(
        gpu=gpu,
        scenario_names=selected,
        arm_names=arm_names,
        span_names=span_names,
        out_dir=out_dir,
        **options,
    )
    try:
        outcomes = execute_kernel_run(request, event_handler=_show_event)
    except (OSError, ValueError, RuntimeError) as error:
        raise click.ClickException(str(error)) from error

    # Errors were streamed as they happened; only reports are rendered here.
    for outcome in outcomes:
        if outcome.result is not None:
            click.echo()
            click.echo(
                render_kernel_span_results(outcome.result)
                if isinstance(outcome.result, KernelSpanResult)
                else render_kernel_results(outcome.result)
            )
            click.echo(f"\nmachine-readable results: {outcome.out_dir}/results.json")

    failures = [outcome for outcome in outcomes if outcome.failed]
    if failures:
        summary = ", ".join(
            f"{outcome.scenario}"
            f"{' (correctness)' if outcome.correctness_failed else ''}"
            for outcome in failures
        )
        raise click.ClickException(f"kernel scenarios failed: {summary}")
