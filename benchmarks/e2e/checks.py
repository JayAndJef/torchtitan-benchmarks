"""The checks that refuse a run before any host probe: the shared rules, then each arm's engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.artifacts.manifests import resume_mismatches
from benchmarks.e2e.engines.api import Arm, RunSpec
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.e2e.parallelism import parallelism_refusals, zero_warnings
from benchmarks.e2e.registry import AC_MODES
from benchmarks.e2e.schema import Scenario


def step_floor_refusals(run: RunSpec) -> list[str]:
    """Why the run takes too few steps to publish a number, or nothing."""
    steps = run.data.steps
    if run.profile:
        minimum = run.window.freq * run.window.min_windows
        if steps < minimum:
            return [
                f"steps ({steps}) must be at least {minimum} to collect "
                f"{run.window.min_windows} profiler windows"
            ]
    elif steps <= run.warmup_steps:
        return [
            f"steps ({steps}) must be more than the {run.warmup_steps} "
            "warmup step(s); a run that measures no step publishes no "
            "throughput"
        ]
    return []


def ac_mode_refusals(run: RunSpec, scenario: Scenario) -> list[str]:
    """Why the scenario cannot run at the run's activation checkpointing mode, or nothing."""
    if run.ac_mode not in AC_MODES:
        return [f"unknown ac mode {run.ac_mode!r}. Available: {', '.join(AC_MODES)}"]
    if run.ac_mode not in scenario.supported_ac_modes:
        return [
            f"scenario {scenario.name!r} does not support ac mode "
            f"{run.ac_mode!r} (supported: "
            f"{', '.join(scenario.supported_ac_modes)})"
        ]
    return []


def check_run(
    run: RunSpec,
    scenario: Scenario,
    arms: tuple[Arm, ...],
    *,
    device_count: int,
    resumed: Mapping[str, Any] | None,
) -> None:
    """Raise one ``ValueError`` that lists every shared refusal and every engine refusal; ``resumed`` is the manifest that a resume continues."""
    refusals = [
        *parallelism_refusals(
            run.parallelism,
            shape=run.shape,
            local_batch_size=run.data.local_batch_size,
            device_count=device_count,
        ),
        *step_floor_refusals(run),
        *ac_mode_refusals(run, scenario),
    ]
    if resumed is not None:
        mismatches = resume_mismatches(resumed, run=run, arms=arms)
        if mismatches:
            refusals.append(
                "resume request does not match the existing manifest: "
                + ", ".join(mismatches)
            )
    for arm in arms:
        refusals.extend(engine_for(arm).check(run, arm))
    if refusals:
        raise ValueError("; ".join(refusals))


def run_warnings(run: RunSpec, arms: tuple[Arm, ...]) -> tuple[str, ...]:
    """The shared ZeRO warning, then each distinct warning of the arms' engines."""
    engine_warnings = (
        warning for arm in arms for warning in engine_for(arm).warnings(run, arm)
    )
    return (*zero_warnings(run.parallelism), *dict.fromkeys(engine_warnings))
