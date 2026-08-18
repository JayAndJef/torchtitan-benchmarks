"""Running one kernel scenario end to end, inside the pinned GPU worker.

The orchestrator, and the only module in ``benchmarks.kernel.engine`` that
knows the order of operations: resolve the scenario's dotted builder paths,
seed the device, build every arm, gate them on correctness *before* any
timing (``engine.correctness``), sweep every arm once per replicate
(``engine.measurement``), derive the per-arm columns, pair each arm against
its opponent (``engine.statistics``) and assemble the ``KernelScenarioResult``
that the worker writes. The methodology itself lives in the two modules it
calls; what is here is sequencing and bookkeeping.

The sweep is **replicate-major**: one replicate times every arm once, in
declaration order, and the replicate is repeated. Drift is therefore shared
across arms rather than charged to whichever arm was running when it
happened, which is what survives of the old round-robin's paired-sample
property once an arm is timed as a burst rather than a single call. It is
also the ordering a later commit reproduces when each (arm, replicate)
becomes its own process.

Nothing in this package imports ``benchmarks.kernel.operations`` or a model
package. Arms arrive as already-resolved ``BuiltArm`` values through
``resolve_symbol``, and the scenario declarations arrive as
``benchmarks.kernel.schema`` types rather than through
``benchmarks.kernel.registry``, so the engine's import graph stays
independent of which kernel families happen to exist and of everything they
depend on -- which is what will let a later change run each arm in its own
process. ``tests/test_import_boundaries.py`` asserts both edges are absent.

``run_kernel_scenario`` re-asserts the balanced-routing invariant that
``benchmarks.kernel.runner`` also checks, and does so before the CUDA check:
the runner's loud skip is the friendly path, not the guard, and a direct
caller (``python -m benchmarks.kernel.worker``, the GPU smoke test) must not
be able to measure an expert split that does not cover the rows it built.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from statistics import median
from typing import TYPE_CHECKING, Any

import torch

from benchmarks.artifacts.summaries import summarize
from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.engine.correctness import run_correctness
from benchmarks.kernel.engine.measurement import (
    burst_ladder,
    burst_samples,
    memory_pass,
)
from benchmarks.kernel.engine.statistics import (
    KERNEL_SIGNIFICANCE_METHODOLOGY,
    kernel_comparison,
)
from benchmarks.kernel.results.schema import (
    ArmResult,
    KernelScenarioResult,
    ModeResult,
)
from benchmarks.kernel.schema import (
    KernelScenario,
    KernelWorkload,
    MODES,
    routing_divides_evenly,
    shape_summary,
)

if TYPE_CHECKING:
    # Annotation-only. The engine must not import a model package: arms reach
    # it as already-resolved BuiltArm values via resolve_symbol, never as
    # imports, which is what will let a later commit run each arm in its own
    # process without the engine dragging in every model's dependencies.
    from benchmarks.models.piper_qwen3.shape import PiperShape


@dataclass(frozen=True)
class RunOptions:
    """Everything the measurement passes need that the scenario does not say.

    ``replicates`` / ``burst_k`` / ``warmup_calls`` replace the former
    ``n`` / ``warmup`` pair rather than reinterpreting it. The old fields
    counted round-robin *cycles*, and results.json printed them as such, so
    reusing the names for a different quantity would put a false statement in
    every published table.
    """

    replicates: int = 5
    samples_per_replicate: int = 40
    burst_k: int = 16
    warmup_calls: int = 30
    burst: bool = False
    seed: int = 0
    memory_iters: int = 5
    bursts: tuple[int, ...] = (1, 4, 16, 64)
    burst_iters: int = 50


def resolve_symbol(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _flatten(replicates: list[list[float]]) -> list[float]:
    return [value for replicate in replicates for value in replicate]


def _heaviest_mode(arm: BuiltArm) -> str:
    for mode in reversed(MODES):
        if mode in arm.calls:
            return mode
    raise ValueError(f"{arm.name}: no modes")


def run_kernel_scenario(
    scenario: KernelScenario,
    shape: PiperShape,
    workload: KernelWorkload,
    options: RunOptions,
    hardware: str,
) -> KernelScenarioResult:
    # Asserted here as well as in the runner, and before the device check so
    # it holds on any host: the runner's loud skip is the friendly path, not
    # the guard. `python -m benchmarks.kernel.worker` and direct callers reach
    # this function without passing through it, and an unbalanced split is the
    # violation that can measure rather than fail -- swiglu_inputs hands every
    # expert an equal slice that does not cover the rows it just built.
    if scenario.requires_balanced_routing and not routing_divides_evenly(
        shape, workload
    ):
        rows = workload.batch * workload.seq_len * shape.top_k
        raise ValueError(
            f"{scenario.name}: {rows} routed rows (batch {workload.batch} x "
            f"seq {workload.seq_len} x top_k {shape.top_k}) do not divide "
            f"evenly among {shape.num_experts} experts"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("kernel benchmarks require a CUDA device")
    device = torch.device("cuda")
    torch.manual_seed(options.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(options.seed)

    inputs = resolve_symbol(scenario.inputs_builder)(
        shape, workload, device, generator
    )
    built = {
        arm.name: resolve_symbol(arm.builder)(shape, workload, inputs)
        for arm in scenario.arms
    }
    fp64_reference = (
        resolve_symbol(scenario.reference_builder)(shape, workload, inputs)
        if scenario.reference_builder
        else None
    )
    correctness, all_passed = run_correctness(scenario, built, fp64_reference)

    # Replicate-major, so drift is shared across arms rather than charged to
    # whichever arm was timed while it happened. Within one replicate the
    # arms run in declaration order; the next replicate repeats the sweep.
    # This is what survives of the old round-robin's paired-sample property
    # once an arm is timed as a burst instead of a single call -- and it is
    # the ordering the parent reproduces when each (arm, replicate) becomes
    # its own process.
    # samples[arm][mode] is a list of replicates, each a list of samples. The
    # boundaries are load-bearing: engine.statistics estimates the ratio once
    # per replicate and bootstraps across them, so flattening here would
    # destroy the only repetition unit the design has.
    samples: dict[str, dict[str, list[list[float]]]] = {
        arm.name: {} for arm in scenario.arms
    }
    for _ in range(options.replicates):
        for arm in scenario.arms:
            built_arm = built[arm.name]
            for mode in MODES:
                values = burst_samples(
                    built_arm,
                    mode,
                    options.burst_k,
                    options.samples_per_replicate,
                    options.warmup_calls,
                )
                if values:
                    samples[arm.name].setdefault(mode, []).append(values)

    floor_medians = {
        mode: median(_flatten(samples[arm.name][mode]))
        for arm in scenario.arms
        if built[arm.name].floor
        for mode in samples[arm.name]
    }

    arm_results: dict[str, ArmResult] = {}
    for arm in scenario.arms:
        built_arm = built[arm.name]
        modes: dict[str, ModeResult] = {}
        for mode, replicates in samples[arm.name].items():
            values = _flatten(replicates)
            derived: dict[str, float] = {}
            mode_median = median(values)
            if built_arm.bytes_moved and mode_median:
                derived["gbps"] = (
                    built_arm.bytes_moved / (mode_median * 1e-6) / 1e9
                )
            if (
                mode in floor_medians
                and not built_arm.floor
                and floor_medians[mode]
            ):
                derived["x_floor"] = mode_median / floor_medians[mode]
            modes[mode] = ModeResult(
                summary=summarize(values),
                replicates_us=tuple(tuple(r) for r in replicates),
                derived=derived,
            )
        arm_results[arm.name] = ArmResult(
            name=arm.name,
            modes=modes,
            peak_memory_gib=(
                None
                if built_arm.floor
                else memory_pass(
                    built_arm, _heaviest_mode(built_arm), options.memory_iters
                )
            ),
            # Every declared mode, not just "forward". The old pass read
            # arm.calls["forward"] directly, which left lm_head -- whose arms
            # declare forward_backward only -- with no way to run the
            # diagnostic at all.
            burst_us_per_call=(
                {
                    mode: burst_ladder(
                        built_arm, mode, options.bursts, options.burst_iters
                    )
                    for mode in MODES
                    if mode in built_arm.calls
                }
                if options.burst
                else None
            ),
        )

    # An arm that serves as another arm's opponent is itself a reference and
    # gets no comparison row; compare_to exists so an arm measured at a
    # different scope than the scenario baseline can face a same-scope
    # opponent instead of an apples-to-oranges baseline ratio.
    references = {scenario.baseline_arm} | {
        arm.compare_to for arm in scenario.arms if arm.compare_to
    }
    comparisons: list[dict[str, Any]] = []
    for arm in scenario.arms:
        if built[arm.name].floor or arm.name in references:
            continue
        opponent = arm.compare_to or scenario.baseline_arm
        for mode in samples[arm.name]:
            if mode not in samples[opponent]:
                continue
            row: dict[str, Any] = {
                "arm": arm.name,
                "opponent": opponent,
                "mode": mode,
            }
            row.update(
                kernel_comparison(samples[opponent][mode], samples[arm.name][mode])
            )
            comparisons.append(row)

    return KernelScenarioResult(
        scenario=scenario.name,
        hardware=hardware,
        model_size=shape.name,
        model_shape=shape.describe(seq_len=workload.seq_len),
        workload=asdict(workload),
        shapes=shape_summary(scenario.name, shape, workload),
        replicates=options.replicates,
        samples_per_replicate=options.samples_per_replicate,
        burst_k=options.burst_k,
        warmup_calls=options.warmup_calls,
        seed=options.seed,
        arms=arm_results,
        comparisons=comparisons,
        correctness=correctness,
        all_correctness_passed=all_passed,
        methodology={
            **KERNEL_SIGNIFICANCE_METHODOLOGY,
            "measurand": "burst_amortized_per_call_device_time",
            "l2_flush": False,
            "l2_flush_rationale": (
                "no flush, and no equalization is claimed: one arm is timed "
                "at a time, so an arm whose working set fits in L2 benefits "
                "from bursting more than one whose does not"
            ),
            "gc_paused_during_timing": True,
            "first_burst_discarded": True,
            "units": "microseconds",
        },
        environment={
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    )
