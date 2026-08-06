"""Human-readable rendering of benchmark evaluation results."""

from __future__ import annotations

import math

from benchmarks.kernel_results import KernelScenarioResult
from benchmarks.metrics import EvaluationResult


def _value(value: float | None, width: int, precision: int = 1) -> str:
    if value is None:
        return f"{'n/a':>{width}s}"
    return f"{value:{width}.{precision}f}"


def _pvalue(value: float | None, width: int) -> str:
    """Significant-digit form; p-values span many orders of magnitude."""
    if value is None:
        return f"{'n/a':>{width}s}"
    return f"{value:{width}.3g}"


def _trajectory(values: list[tuple[int, float]], nonfinite_label: str) -> str:
    if not values:
        return "(no log)"
    picks = [values[0]] + [values[i] for i in (9, 19, 29, 39) if i < len(values)]
    rendered = "  ".join(f"s{step}:{value:.5f}" for step, value in picks)
    if not all(math.isfinite(value) for _, value in values):
        rendered += f"   NON-FINITE {nonfinite_label}"
    return rendered


def render_evaluation(result: EvaluationResult) -> str:
    """Render the complete stable-throughput and compiled-region report."""
    lines = [
        f"== {result.output_dir} ==",
        f"scenario: {result.scenario}   hardware: {result.hardware}",
    ]
    lines.extend(f"WARNING: {warning}" for warning in result.warnings)

    lines.extend(
        [
            "",
            "benchmark summary:",
            "  "
            + f"{'arm':22s} {'stable tokens/s':>15s} {'n':>4s} {'ratio':>8s} "
            + f"{'peak GiB':>9s}",
        ]
    )
    for arm in result.arms:
        training = result.training[arm]
        lines.append(
            f"  {arm:22s} "
            f"{_value(training.stable_tokens_per_second, 15)} "
            f"{training.stable_sample_count:4d} "
            f"{_value(training.baseline_ratio, 8, 4)} "
            f"{_value(training.peak_memory_gib, 9, 2)}"
        )

    lines.extend(
        [
            "",
            "gpu kernel time (host-speed-immune; compare kernels with this):",
            "  "
            + f"{'arm':22s} {'kernel ms/step':>14s} {'vs base':>8s} "
            + f"{'regions ms':>11s} {'other ms':>9s} {'fwd kernel us':>14s} "
            + f"{'bwd kernel us':>14s} {'launch us':>10s}",
        ]
    )
    for arm in result.arms:
        gpu = result.gpu_time[arm]
        forward = result.regions[arm].get("forward_block")
        backward = result.regions[arm].get("backward_block")
        lines.append(
            f"  {arm:22s} "
            f"{_value(gpu.kernel_ms_per_step, 14, 2)} "
            f"{_value(gpu.baseline_kernel_ratio, 8, 4)} "
            f"{_value(gpu.region_kernel_ms_per_step, 11, 2)} "
            f"{_value(gpu.other_kernel_ms_per_step, 9, 2)} "
            f"{_value(forward.kernel.mean_us if forward else None, 14)} "
            f"{_value(backward.kernel.mean_us if backward else None, 14)} "
            f"{_value(gpu.launch_latency_us, 10, 2)}"
        )

    for arm in result.arms:
        if arm == "baseline":
            continue
        lines.extend(
            [
                "",
                f"compiled-region distributions, baseline vs {arm} "
                f"(pooled over {result.trace_windows['baseline']}+"
                f"{result.trace_windows[arm]} windows):",
                "  "
                + f"{'region':16s} {'n':>4} | {'base kern':>10} {'arm kern':>9} "
                + f"{'ratio':>7} | {'base span':>10} {'arm span':>9} "
                + f"{'ratio':>7} | {'diag Welch p':>12} {'diag MWU p':>10} "
                + f"{'d':>6}",
            ]
        )
        for row in result.comparisons[arm]:
            lines.append(
                f"  {row['region']:16s} {row['n_arm']:>4} | "
                f"{row['base_kernel_mean_us']:10.1f} "
                f"{row['arm_kernel_mean_us']:9.1f} "
                f"{_value(row['kernel_ratio'], 7, 4)} | "
                f"{row['base_span_mean_us']:10.1f} "
                f"{row['arm_span_mean_us']:9.1f} "
                f"{_value(row['span_ratio'], 7, 4)} | "
                f"{row['span_welch_p']:12.3g} {row['span_mwu_p']:10.3g} "
                f"{row['span_cohens_d']:6.2f}"
            )

    lines.extend(
        [
            "",
            "Significance limitation: pooled compiled-region invocations share",
            "training steps and layer structure. Welch/MWU p-values and Cohen's d",
            "describe span distributions; they are not inference from",
            "independent benchmark repetitions.",
        ]
    )

    lines.extend(["", "loss trajectories (sanity check, not a measurement):"])
    for arm in result.arms:
        lines.append(f"  {arm:22s} {_trajectory(result.losses[arm], 'LOSS')}")
    lines.extend(
        ["", "gradient norm trajectories (sanity check, not a measurement):"]
    )
    for arm in result.arms:
        lines.append(
            f"  {arm:22s} {_trajectory(result.gradient_norms[arm], 'GRAD NORM')}"
        )

    lines.extend(
        [
            "",
            "Note: region rows time whole compiled forward/backward blocks; they",
            "are not measurements of an individual generated Inductor kernel.",
            "Span times include idle gaps where the GPU waited on the host, so",
            "they move with host speed; kernel times count only GPU execution.",
        ]
    )
    return "\n".join(lines)


def render_kernel_results(result: KernelScenarioResult) -> str:
    """Render one kernel scenario: per-mode timings and correctness gates."""
    shapes = "  ".join(
        f"{name}={value}" for name, value in result.shapes.items()
    )
    lines = [
        f"== kernel scenario: {result.scenario}   hardware: {result.hardware} ==",
        f"shapes: {shapes}",
        f"n={result.n} interleaved cycles, warmup={result.warmup}, "
        f"seed={result.seed}",
    ]
    lines.extend(f"WARNING: {warning}" for warning in result.warnings)

    comparisons = {
        (row["arm"], row["mode"]): row for row in result.comparisons
    }
    for mode in ("forward", "backward", "forward_backward"):
        arms = [
            name
            for name, arm in result.arms.items()
            if mode in arm.modes
        ]
        if not arms:
            continue
        lines.extend(
            [
                "",
                f"{mode}:",
                "  "
                + f"{'arm':22s} {'median us':>10s} {'sd':>8s} {'vs':>18s} "
                + f"{'ratio':>7s} {'GB/s':>8s} {'x floor':>8s} "
                + f"{'Welch p':>9s} {'MWU p':>9s} {'Wilcoxon p':>11s} {'d':>6s}",
            ]
        )
        for name in arms:
            mode_result = result.arms[name].modes[mode]
            row = comparisons.get((name, mode))
            derived = mode_result.derived
            lines.append(
                f"  {name:22s} "
                f"{mode_result.summary.median_us:10.2f} "
                f"{mode_result.summary.standard_deviation_us:8.2f} "
                f"{(row['opponent'] if row else '-'):>18s} "
                f"{_value(row['median_ratio'] if row else None, 7, 4)} "
                f"{_value(derived.get('gbps'), 8, 1)} "
                f"{_value(derived.get('x_floor'), 8, 2)} "
                f"{_pvalue(row['welch_p'] if row else None, 9)} "
                f"{_pvalue(row['mwu_p'] if row else None, 9)} "
                f"{_pvalue(row['wilcoxon_p'] if row else None, 11)} "
                f"{_value(row['cohens_d'] if row else None, 6, 2)}"
            )

    memory = {
        name: arm.peak_memory_gib
        for name, arm in result.arms.items()
        if arm.peak_memory_gib is not None
    }
    if memory:
        lines.extend(["", "peak memory (GiB, heaviest mode, isolated):"])
        for name, peak in memory.items():
            lines.append(f"  {name:22s} {peak:8.3f}")

    bursts = {
        name: arm.burst_us_per_call
        for name, arm in result.arms.items()
        if arm.burst_us_per_call
    }
    if bursts:
        sizes = sorted(next(iter(bursts.values())), key=int)
        lines.extend(
            [
                "",
                "burst dispatch diagnostic (us per call, forward):",
                "  " + f"{'arm':22s} " + " ".join(f"{size:>9s}" for size in sizes),
            ]
        )
        for name, values in bursts.items():
            lines.append(
                f"  {name:22s} "
                + " ".join(f"{values[size]:9.2f}" for size in sizes)
            )
        lines.append(
            "  Falling per-call time means single calls were dispatch-bound."
        )

    lines.extend(["", "correctness:"])
    if not result.correctness:
        lines.append("  (no checks declared)")
    for row in result.correctness:
        if row.passed is None:
            verdict = "INFO"
        else:
            verdict = "PASS" if row.passed else "FAIL"
        limit = "-" if row.threshold is None else f"{row.threshold:g}"
        lines.append(
            f"  {verdict:4s} {row.arm:22s} {row.output:14s} vs "
            f"{row.reference:16s} {row.metric}={row.value:.4g} (limit {limit})"
        )
    lines.append(
        "  ALL CORRECTNESS PASSED"
        if result.all_correctness_passed
        else "  CORRECTNESS FAILURES PRESENT"
    )

    lines.extend(
        [
            "",
            "Arms are timed round-robin within each cycle, so drift affects all",
            "arms equally and these p-values are inferential for this run. These",
            "are isolated-kernel numbers on synthetic inputs; never present them",
            "as end-to-end training results.",
        ]
    )
    return "\n".join(lines)
