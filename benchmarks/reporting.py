"""Human-readable rendering of benchmark evaluation results."""

from __future__ import annotations

import math

from benchmarks.metrics import EvaluationResult


def _value(value: float | None, width: int, precision: int = 1) -> str:
    if value is None:
        return f"{'n/a':>{width}s}"
    return f"{value:{width}.{precision}f}"


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
            + f"{'peak GiB':>9s} {'forward mean us':>16s} {'backward mean us':>17s}",
        ]
    )
    for arm in result.arms:
        training = result.training[arm]
        forward = result.regions[arm].get("forward_block")
        backward = result.regions[arm].get("backward_block")
        lines.append(
            f"  {arm:22s} "
            f"{_value(training.stable_tokens_per_second, 15)} "
            f"{training.stable_sample_count:4d} "
            f"{_value(training.baseline_ratio, 8, 4)} "
            f"{_value(training.peak_memory_gib, 9, 2)} "
            f"{_value(forward.mean_us if forward else None, 16)} "
            f"{_value(backward.mean_us if backward else None, 17)}"
        )

    for arm in result.arms:
        if arm == "baseline":
            continue
        lines.extend(
            [
                "",
                f"compiled-region invocation-distribution diagnostics, "
                f"baseline vs {arm} "
                f"(pooled over {result.trace_windows['baseline']}+"
                f"{result.trace_windows[arm]} windows):",
                "  "
                + f"{'region':16s} {'n':>4} | {'base mean':>10} {'median':>9} "
                + f"{'sd':>7} | {'arm mean':>10} {'median':>9} {'sd':>7} | "
                + f"{'delta':>8} {'ratio':>7} | {'diag Welch p':>12} "
                + f"{'diag MWU p':>10} "
                + f"{'d':>6}",
            ]
        )
        for row in result.comparisons[arm]:
            lines.append(
                f"  {row['region']:16s} {row['n_arm']:>4} | "
                f"{row['base_mean_us']:10.1f} {row['base_median_us']:9.1f} "
                f"{row['base_sd_us']:7.1f} | {row['arm_mean_us']:10.1f} "
                f"{row['arm_median_us']:9.1f} {row['arm_sd_us']:7.1f} | "
                f"{row['delta_us']:+8.1f} {row['ratio']:7.4f} | "
                f"{row['welch_p']:12.3g} {row['mwu_p']:10.3g} "
                f"{row['cohens_d']:6.2f}"
            )

    lines.extend(
        [
            "",
            "Significance limitation: pooled compiled-region invocations share",
            "training steps and layer structure. Welch/MWU p-values and Cohen's d",
            "describe invocation distributions; they are not inference from",
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
        ]
    )
    return "\n".join(lines)
