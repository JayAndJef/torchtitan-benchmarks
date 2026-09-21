"""The ``kernel-bench`` command: the kernel-isolation family, on its own.

``--out``, ``--seq-len``, ``--batch`` and ``--model-size`` read no
environment variable, so an ``OUT``, ``SEQ`` or ``BATCH`` exported for an
end-to-end session cannot reach a kernel measurement. Only ``--cache-root``
and ``--compiler-env`` read one. That is why ``--model-size`` is declared
twice in this package; ``benchmarks/cli/e2e.py`` holds the other half.

``benchmarks/cli/main.py`` attaches this plain ``@click.command`` through
``cli.add_command``, so no command module imports the group back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from benchmarks.cli.rendering import _show_event
from benchmarks.execution.devices import parse_devices
from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.results.reporting import (
    render_kernel_results,
    render_kernel_span_results,
)
from benchmarks.kernel.results.schema import KernelSpanResult
from benchmarks.kernel.runner import KernelRunRequest, execute_kernel_run
from benchmarks.kernel.schema import DEFAULT_MODEL_SIZE
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
    # min=1: a zero count was caught only deep in the run, never at the flag.
    type=click.IntRange(min=1),
    help="Passes over every arm; the repetition unit the CI is taken over.",
)
@click.option(
    "--replicates-per-process",
    default=1,
    show_default=True,
    type=click.IntRange(min=1),
    help=(
        "Consecutive replicates of one arm per worker process. 1 rebuilds the "
        "arm for every replicate and keeps the replicate-major order; higher "
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
    default=DEFAULT_MODEL_SIZE,
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
        "Raise the shape's max_seq_len ceiling (default 4096); needed to "
        "run attention_core past 4096. Also sizes the RoPE cos/sin "
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
    # One device, always: a worker builds one arm, so a second device idles.
    try:
        devices = parse_devices(gpu)
    except ValueError as error:
        raise click.UsageError(str(error)) from error
    if len(devices) != 1:
        raise click.UsageError(
            f"kernel-bench measures one device; {gpu!r} names {len(devices)}"
        )
    # ``--span`` defaults to none, because one span adds every scenario it
    # encloses to the run.
    selected = scenario_names or (() if span_names else tuple(KERNEL_SCENARIOS))
    if out_dir is not None and (len(selected) != 1 or span_names):
        # A span run is always several units, and they share this directory.
        raise click.UsageError(
            "--out requires exactly one --scenario and no --span; a span "
            "also measures every scenario it replaces, so the units would "
            "overwrite each other"
        )
    # A span run has several rosters, so one arm selection cannot pick one.
    if arm_names and span_names:
        raise click.UsageError(
            "--arm does not combine with --span; a span run measures several "
            "scenarios and an arm selection belongs to one roster"
        )
    # An arm name belongs to one roster; resolve_arm_skips checks the roster.
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
