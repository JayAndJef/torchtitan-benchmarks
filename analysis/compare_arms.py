"""Compare benchmark arms by compiled forward/backward region GPU spans.

Usage:
    python compare_arms.py <out-dir>            # out/<ts>/<scenario>/<hardware>
    python compare_arms.py <out-dir> --arms baseline helion

For each arm it expects <out-dir>/<arm>/profiling/traces*/iteration_*/
rank0_trace.json.gz plus <out-dir>/manifest.json written by the runner.
The measurement is the GPU span of the scenario's declared compiled regions
(``## Call CompiledFxGraph`` GPU annotations), pooled over all profiler
windows, with each window's invocation count validated first. Reports per
region: count, mean, median, standard deviation, arm-vs-baseline delta and
ratio, Welch's t-test, Mann-Whitney U, and Cohen's d.

These are whole compiled forward/backward block timings, not timings of any
individual generated Inductor kernel: generated kernel names are unstable
across torch versions and arms, so per-kernel attribution is out of scope.
Stable-step throughput, peak memory, loss, and gradient-norm trajectories are
also reported from the training logs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy import stats as scipy_stats  # noqa: E402

from benchmarks.profile_regions import pooled_region_samples  # noqa: E402
from benchmarks.runner import trace_files  # noqa: E402
from benchmarks.scenarios import PIPER_1B_REGIONS, Region  # noqa: E402


STEP_METRICS = re.compile(
    r"step:\s*(\d+).*?memory:\s*([0-9.]+)GiB.*?tps:\s*([0-9,]+)"
)


def describe(values: list[float]) -> tuple[int, float, float, float]:
    """Return (n, mean, sd, median) of one region's pooled samples."""
    n = len(values)
    sd = statistics.stdev(values) if n > 1 else 0.0
    return n, statistics.mean(values), sd, statistics.median(values)


def region_comparison(
    base: dict[str, list[float]],
    arm: dict[str, list[float]],
    regions: tuple[Region, ...],
) -> list[dict[str, float | int | str]]:
    """Compare each declared region's pooled samples between two arms."""
    rows: list[dict[str, float | int | str]] = []
    for region in regions:
        base_values, arm_values = base[region.name], arm[region.name]
        n_base, mean_base, sd_base, median_base = describe(base_values)
        n_arm, mean_arm, sd_arm, median_arm = describe(arm_values)
        welch = scipy_stats.ttest_ind(base_values, arm_values, equal_var=False)
        mwu = scipy_stats.mannwhitneyu(base_values, arm_values, alternative="two-sided")
        pooled_sd = (
            (
                ((n_base - 1) * sd_base**2 + (n_arm - 1) * sd_arm**2)
                / (n_base + n_arm - 2)
            )
            ** 0.5
            if n_base + n_arm > 2
            else 0.0
        )
        rows.append(
            {
                "region": region.name,
                "n_base": n_base,
                "base_mean_us": mean_base,
                "base_median_us": median_base,
                "base_sd_us": sd_base,
                "n_arm": n_arm,
                "arm_mean_us": mean_arm,
                "arm_median_us": median_arm,
                "arm_sd_us": sd_arm,
                "delta_us": mean_arm - mean_base,
                "ratio": mean_arm / mean_base,
                "welch_p": float(welch.pvalue),
                "mwu_p": float(mwu.pvalue),
                "cohens_d": (mean_arm - mean_base) / pooled_sd if pooled_sd else 0.0,
            }
        )
    return rows


def losses(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    out = []
    pattern = re.compile(r"step:\s*(\d+).*?loss:\s*([0-9.eE+-]+|nan|inf)")
    with open(log_path, errors="replace") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                out.append((int(match.group(1)), float(match.group(2))))
    return out


def grad_norms(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    out = []
    pattern = re.compile(r"step:\s*(\d+).*?grad_norm:\s*([0-9.eE+-]+|nan|inf)")
    with open(log_path, errors="replace") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                out.append((int(match.group(1)), float(match.group(2))))
    return out


def training_metrics(log_path: Path) -> list[tuple[int, float, int]]:
    """Return (step, peak-memory-GiB, tokens/s) rows from a training log."""
    if not log_path.exists():
        return []
    rows = []
    with open(log_path, errors="replace") as log_file:
        for line in log_file:
            match = STEP_METRICS.search(line)
            if match:
                rows.append(
                    (
                        int(match.group(1)),
                        float(match.group(2)),
                        int(match.group(3).replace(",", "")),
                    )
                )
    return rows


def stable_tps(rows: list[tuple[int, float, int]], workload: dict) -> list[int]:
    """Select post-compile steps before each profiler warmup begins."""
    profile_freq = int(workload.get("profile_freq", 20))
    wait = profile_freq - int(workload.get("profiler_warmup", 5)) - int(
        workload.get("profiler_active", 5)
    )
    return [
        tps
        for step, _, tps in rows
        if 2 <= ((step - 1) % profile_freq) + 1 <= wait
    ]


def load_run(out_dir: Path, arms_override: list[str] | None) -> tuple[dict, list[str], tuple[Region, ...]]:
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not manifest:
        print(f"WARNING: no manifest.json under {out_dir}; assuming piper-1B regions")
    manifest_regions = manifest.get("regions", [])
    if manifest_regions and not all("phase" in region for region in manifest_regions):
        print(
            f"WARNING: {manifest_path} predates phase-based regions "
            f"(schema {manifest.get('schema_version')}); assuming piper-1B regions"
        )
        manifest_regions = []
    regions = tuple(Region(**region) for region in manifest_regions) or PIPER_1B_REGIONS
    arms = (
        arms_override
        or manifest.get("selected_arms")
        or [arm["name"] for arm in manifest.get("arms", [])]
        or ["baseline", "helion", "te"]
    )
    return manifest, arms, regions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare benchmark arms by compiled-region GPU spans."
    )
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--arms", nargs="+", help="Subset of arms (must include baseline).")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    manifest, arms, regions = load_run(out_dir, args.arms)
    if "baseline" not in arms:
        raise SystemExit("FATAL: comparison needs the baseline arm")

    print(f"== {out_dir} ==")
    print(f"scenario: {manifest.get('scenario', 'unknown')}   "
          f"hardware: {manifest.get('hardware', 'unknown')}")

    pooled: dict[str, dict[str, list[float]]] = {}
    windows: dict[str, int] = {}
    for arm in arms:
        paths = trace_files(out_dir / arm)
        if not paths:
            raise SystemExit(f"FATAL: no profiler traces under {out_dir / arm}")
        try:
            pooled[arm] = pooled_region_samples(paths, regions)
        except ValueError as error:
            raise SystemExit(f"FATAL: {arm}: {error}") from error
        windows[arm] = len(paths)

    for arm in arms:
        if arm == "baseline":
            continue
        print(f"\ncompiled-region GPU span us/call, baseline vs {arm} "
              f"(pooled over {windows['baseline']}+{windows[arm]} windows):")
        header = (f"  {'region':16s} {'n':>4} | {'base mean':>10} {'median':>9} {'sd':>7} | "
                  f"{'arm mean':>10} {'median':>9} {'sd':>7} | {'delta':>8} {'ratio':>7} | "
                  f"{'Welch p':>10} {'MWU p':>10} {'d':>6}")
        print(header)
        for row in region_comparison(pooled["baseline"], pooled[arm], regions):
            print(
                f"  {row['region']:16s} {row['n_arm']:>4} | "
                f"{row['base_mean_us']:10.1f} {row['base_median_us']:9.1f} "
                f"{row['base_sd_us']:7.1f} | "
                f"{row['arm_mean_us']:10.1f} {row['arm_median_us']:9.1f} "
                f"{row['arm_sd_us']:7.1f} | "
                f"{row['delta_us']:+8.1f} {row['ratio']:7.4f} | "
                f"{row['welch_p']:10.3g} {row['mwu_p']:10.3g} {row['cohens_d']:6.2f}"
            )

    print("\nend-to-end training metrics:")
    workload = manifest.get("workload", {})
    metric_rows = {
        arm: training_metrics(out_dir / f"{arm}.log") for arm in arms
    }
    base_tps = stable_tps(metric_rows["baseline"], workload)
    base_median = statistics.median(base_tps) if base_tps else math.nan
    for arm in arms:
        rows = metric_rows[arm]
        samples = stable_tps(rows, workload)
        median_tps = statistics.median(samples) if samples else math.nan
        peak_memory = max((memory for _, memory, _ in rows), default=math.nan)
        ratio = median_tps / base_median if base_median and samples else math.nan
        print(
            f"  {arm:22s} stable tps {median_tps:10.1f} "
            f"(n={len(samples):2d}, {ratio:7.4f}x)   peak {peak_memory:6.2f} GiB"
        )

    print("\nloss trajectories (sanity check, not a measurement):")
    for arm in arms:
        arm_losses = losses(out_dir / f"{arm}.log")
        if not arm_losses:
            print(f"  {arm:22s} (no log)")
            continue
        picks = [arm_losses[0]] + [
            arm_losses[i] for i in (9, 19, 29, 39) if i < len(arm_losses)
        ]
        line = "  ".join(f"s{step}:{value:.5f}" for step, value in picks)
        finite = all(math.isfinite(value) for _, value in arm_losses)
        print(f"  {arm:22s} {line}" + ("" if finite else "   NON-FINITE LOSS"))

    print("\ngradient norm trajectories (sanity check, not a measurement):")
    for arm in arms:
        values = grad_norms(out_dir / f"{arm}.log")
        if not values:
            print(f"  {arm:22s} (no log)")
            continue
        picks = [values[0]] + [values[i] for i in (9, 19, 29, 39) if i < len(values)]
        line = "  ".join(f"s{step}:{value:.5f}" for step, value in picks)
        finite = all(math.isfinite(value) for _, value in values)
        print(f"  {arm:22s} {line}" + ("" if finite else "   NON-FINITE GRAD NORM"))

    print("\nNote: rows above time whole compiled forward/backward blocks; they")
    print("are not measurements of any individual generated Inductor kernel.")


if __name__ == "__main__":
    main()
