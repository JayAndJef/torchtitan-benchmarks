"""Measurement engine for kernel-isolation benchmarks.

Runs inside the pinned GPU worker. Arms are timed round-robin: every cycle
runs each arm's closure once between adjacent entries of a preallocated CUDA
event matrix, with a single device synchronize after all cycles, so clock
and thermal drift hit every arm equally and per-cycle deltas are paired.
Correctness gates run before any timing.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Callable

import torch

from benchmarks.kernel_results import (
    ArmResult,
    CorrectnessResult,
    KernelScenarioResult,
    ModeResult,
)
from benchmarks.kernel_stats import (
    KERNEL_SIGNIFICANCE_METHODOLOGY,
    kernel_comparison,
)
from benchmarks.kernels import (
    CorrectnessCheck,
    KernelScenario,
    MODES,
    Piper1BSpec,
    shape_summary,
)
from benchmarks.metrics import summarize


@dataclass
class BuiltArm:
    """A constructed arm: per-mode timed closures plus its validity hook.

    Each ``calls`` closure performs exactly one timed operation over
    prebuilt tensors; ``correctness_outputs`` runs the arm once on the
    shared seeded inputs and returns the named tensors its declared
    correctness checks compare.
    """

    name: str
    calls: dict[str, Callable[[], object]]
    correctness_outputs: Callable[[], dict[str, torch.Tensor]]
    bytes_moved: int | None = None
    floor: bool = False
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunOptions:
    n: int = 200
    warmup: int = 30
    burst: bool = False
    seed: int = 0
    memory_iters: int = 5
    bursts: tuple[int, ...] = (1, 4, 16, 64)
    burst_iters: int = 50


def resolve_symbol(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


@contextlib.contextmanager
def _gc_paused():
    """Keep collection cycles out of the timed region.

    The measured arms only stay ahead of the GPU if the host keeps enqueuing;
    a collection pause starves the stream and lands as idle time inside
    whichever arm's interval was open. Measured on the swiglu modules: sd
    fell from ~63 us to ~1.4 us and every 2x outlier disappeared, with the
    median unchanged. ``timeit`` disables the collector for the same reason.
    """
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def interleaved_samples(
    arms: list[BuiltArm], mode: str, n: int, warmup: int
) -> dict[str, list[float]]:
    """Time all arms that support ``mode``, one cycle at a time."""
    active = [arm for arm in arms if mode in arm.calls]
    if not active:
        return {}
    for _ in range(warmup):
        for arm in active:
            arm.calls[mode]()
    torch.cuda.synchronize()
    # One extra cycle absorbs the post-sync cold start: the queue is empty
    # after the warmup synchronize, so the first cycle cannot overlap host
    # dispatch with device work and reads systematically high.
    events = [
        [torch.cuda.Event(enable_timing=True) for _ in range(len(active) + 1)]
        for _ in range(n + 1)
    ]
    with _gc_paused():
        for row in events:
            row[0].record()
            for slot, arm in enumerate(active):
                arm.calls[mode]()
                row[slot + 1].record()
        torch.cuda.synchronize()
    return {
        arm.name: [
            row[slot].elapsed_time(row[slot + 1]) * 1e3 for row in events[1:]
        ]
        for slot, arm in enumerate(active)
    }


def memory_pass(arm: BuiltArm, mode: str, iters: int) -> float:
    """Peak allocated GiB across ``iters`` isolated calls of ``mode``."""
    peak = 0.0
    for _ in range(iters):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        arm.calls[mode]()
        torch.cuda.synchronize()
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
    return peak


def burst_pass(
    arm: BuiltArm, bursts: tuple[int, ...], iters: int
) -> dict[str, float]:
    """Median us per call at increasing back-to-back burst sizes.

    Distinguishes kernel cost from dispatch cost: if per-call time collapses
    as the burst grows, single-call timing was host-dispatch-bound.
    """
    call = arm.calls["forward"]
    result = {}
    with _gc_paused():
        for burst in bursts:
            for _ in range(20):
                call()
            torch.cuda.synchronize()
            samples = []
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(burst):
                    call()
                end.record()
                torch.cuda.synchronize()
                samples.append(start.elapsed_time(end) * 1e3 / burst)
            result[str(burst)] = median(samples)
    return result


def _tensor_pair(
    arm_outputs: dict[str, torch.Tensor],
    reference_outputs: dict[str, torch.Tensor],
    arm_name: str,
    output: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output not in arm_outputs:
        raise RuntimeError(
            f"{arm_name}: correctness_outputs did not produce {output!r}"
        )
    if output not in reference_outputs:
        raise RuntimeError(
            f"{arm_name}: reference did not produce {output!r}"
        )
    return arm_outputs[output], reference_outputs[output]


def _check_rows(
    arm_name: str,
    check: CorrectnessCheck,
    arm_outputs: dict[str, torch.Tensor],
    reference_outputs: dict[str, torch.Tensor],
) -> list[CorrectnessResult]:
    rows = []
    for output in check.outputs:
        value, truth = _tensor_pair(
            arm_outputs, reference_outputs, arm_name, output
        )
        if check.kind == "bitwise":
            equal = value.shape == truth.shape and torch.equal(value, truth)
            difference = (
                (value.float() - truth.float()).abs().max().item()
                if value.shape == truth.shape
                else float("inf")
            )
            rows.append(
                CorrectnessResult(
                    arm=arm_name,
                    reference=check.reference,
                    kind=check.kind,
                    output=output,
                    metric="max_abs",
                    value=difference,
                    threshold=0.0,
                    passed=None if check.informational else equal,
                    informational=check.informational,
                )
            )
        elif check.kind == "fp64_ulp":
            truth64 = truth.double()
            ulp = torch.ldexp(
                torch.ones_like(truth64),
                torch.floor(
                    torch.log2(truth64.abs().clamp_min(1e-30))
                ).long()
                - 7,
            )
            mean_ulp = ((value.double() - truth64).abs() / ulp).mean().item()
            rows.append(
                CorrectnessResult(
                    arm=arm_name,
                    reference=check.reference,
                    kind=check.kind,
                    output=output,
                    metric="mean_ulp",
                    value=mean_ulp,
                    threshold=check.max_mean_ulp,
                    passed=None
                    if check.informational
                    else mean_ulp <= float(check.max_mean_ulp),
                    informational=check.informational,
                )
            )
        elif check.kind == "tolerance":
            delta = (value.float() - truth.float()).abs()
            if check.max_abs is not None:
                max_abs = delta.max().item()
                rows.append(
                    CorrectnessResult(
                        arm=arm_name,
                        reference=check.reference,
                        kind=check.kind,
                        output=output,
                        metric="max_abs",
                        value=max_abs,
                        threshold=check.max_abs,
                        passed=None
                        if check.informational
                        else max_abs <= float(check.max_abs),
                        informational=check.informational,
                    )
                )
            if check.max_rel is not None:
                max_rel = (
                    (delta / truth.float().abs().clamp_min(1e-6))
                    .max()
                    .item()
                )
                rows.append(
                    CorrectnessResult(
                        arm=arm_name,
                        reference=check.reference,
                        kind=check.kind,
                        output=output,
                        metric="max_rel",
                        value=max_rel,
                        threshold=check.max_rel,
                        passed=None
                        if check.informational
                        else max_rel <= float(check.max_rel),
                        informational=check.informational,
                    )
                )
            if check.max_rel_l2 is not None:
                norm = truth.float().norm()
                rel_l2 = (
                    (delta.norm() / norm).item()
                    if norm
                    else delta.norm().item()
                )
                rows.append(
                    CorrectnessResult(
                        arm=arm_name,
                        reference=check.reference,
                        kind=check.kind,
                        output=output,
                        metric="rel_l2",
                        value=rel_l2,
                        threshold=check.max_rel_l2,
                        passed=None
                        if check.informational
                        else rel_l2 <= float(check.max_rel_l2),
                        informational=check.informational,
                    )
                )
        else:
            raise ValueError(f"unknown correctness kind {check.kind!r}")
    return rows


def run_correctness(
    scenario: KernelScenario,
    built: dict[str, BuiltArm],
    fp64_reference: dict[str, torch.Tensor] | None,
) -> tuple[list[CorrectnessResult], bool]:
    outputs_cache: dict[str, dict[str, torch.Tensor]] = {}

    def outputs_for(name: str) -> dict[str, torch.Tensor]:
        if name == "fp64":
            if fp64_reference is None:
                raise RuntimeError(
                    f"{scenario.name}: fp64 reference requested but the "
                    f"scenario declares no reference_builder"
                )
            return fp64_reference
        if name not in outputs_cache:
            outputs_cache[name] = built[name].correctness_outputs()
        return outputs_cache[name]

    rows: list[CorrectnessResult] = []
    for arm in scenario.arms:
        for check in arm.correctness:
            rows.extend(
                _check_rows(
                    arm.name,
                    check,
                    outputs_for(arm.name),
                    outputs_for(check.reference),
                )
            )
    all_passed = all(row.passed for row in rows if row.passed is not None)
    return rows, all_passed


def _heaviest_mode(arm: BuiltArm) -> str:
    for mode in reversed(MODES):
        if mode in arm.calls:
            return mode
    raise ValueError(f"{arm.name}: no modes")


def run_kernel_scenario(
    scenario: KernelScenario,
    spec: Piper1BSpec,
    options: RunOptions,
    hardware: str,
) -> KernelScenarioResult:
    if not torch.cuda.is_available():
        raise RuntimeError("kernel benchmarks require a CUDA device")
    device = torch.device("cuda")
    torch.manual_seed(options.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(options.seed)

    inputs = resolve_symbol(scenario.inputs_builder)(spec, device, generator)
    built = {
        arm.name: resolve_symbol(arm.builder)(spec, inputs)
        for arm in scenario.arms
    }
    fp64_reference = (
        resolve_symbol(scenario.reference_builder)(spec, inputs)
        if scenario.reference_builder
        else None
    )
    correctness, all_passed = run_correctness(scenario, built, fp64_reference)

    ordered = [built[arm.name] for arm in scenario.arms]
    samples: dict[str, dict[str, list[float]]] = {arm.name: {} for arm in scenario.arms}
    for mode in MODES:
        for name, values in interleaved_samples(
            ordered, mode, options.n, options.warmup
        ).items():
            samples[name][mode] = values

    floor_medians = {
        mode: median(samples[arm.name][mode])
        for arm in scenario.arms
        if built[arm.name].floor
        for mode in samples[arm.name]
    }

    arm_results: dict[str, ArmResult] = {}
    for arm in scenario.arms:
        built_arm = built[arm.name]
        modes: dict[str, ModeResult] = {}
        for mode, values in samples[arm.name].items():
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
                samples_us=tuple(values),
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
            burst_us_per_call=(
                burst_pass(built_arm, options.bursts, options.burst_iters)
                if options.burst and "forward" in built_arm.calls
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
        spec=asdict(spec),
        shapes=shape_summary(scenario.name, spec),
        n=options.n,
        warmup=options.warmup,
        seed=options.seed,
        arms=arm_results,
        comparisons=comparisons,
        correctness=correctness,
        all_correctness_passed=all_passed,
        methodology={
            **KERNEL_SIGNIFICANCE_METHODOLOGY,
            "l2_flush": False,
            "l2_flush_rationale": (
                "arms are interleaved, so cache state is equalized across "
                "arms rather than cleared"
            ),
            "gc_paused_during_timing": True,
            "units": "microseconds",
        },
        environment={
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        },
    )
