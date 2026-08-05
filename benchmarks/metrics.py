"""Reusable extraction and comparison of benchmark run metrics."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scipy import stats as scipy_stats

from benchmarks.artifacts import atomic_write_json, trace_files
from benchmarks.profile_regions import pooled_region_samples
from benchmarks.scenarios import PIPER_1B_REGIONS, Region


STEP_METRICS = re.compile(
    r"step:\s*(\d+).*?memory:\s*([0-9.]+)GiB.*?tps:\s*([0-9,]+)"
)
LOSS_METRIC = re.compile(r"step:\s*(\d+).*?loss:\s*([0-9.eE+-]+|nan|inf)")
GRAD_NORM_METRIC = re.compile(
    r"step:\s*(\d+).*?grad_norm:\s*([0-9.eE+-]+|nan|inf)"
)
SIGNIFICANCE_METHODOLOGY = {
    "interpretation": "invocation_distribution_diagnostic",
    "sample_unit": "compiled_region_invocation",
    "independence_assumption_met": False,
    "limitation": (
        "Invocations share training steps and layer structure within one run. "
        "Welch and Mann-Whitney p-values are distribution diagnostics, not "
        "inferential evidence from independent benchmark repetitions."
    ),
}


@dataclass(frozen=True)
class SampleSummary:
    count: int
    mean_us: float
    median_us: float
    standard_deviation_us: float


@dataclass(frozen=True)
class TrainingSummary:
    stable_tokens_per_second: float | None
    stable_sample_count: int
    baseline_ratio: float | None
    peak_memory_gib: float | None


@dataclass(frozen=True)
class EvaluationResult:
    """Complete machine-readable result for one benchmark output directory."""

    output_dir: str
    scenario: str
    hardware: str
    arms: tuple[str, ...]
    trace_windows: dict[str, int]
    regions: dict[str, dict[str, SampleSummary]]
    comparisons: dict[str, list[dict[str, float | int | str]]]
    training: dict[str, TrainingSummary]
    losses: dict[str, list[tuple[int, float]]]
    gradient_norms: dict[str, list[tuple[int, float]]]
    significance_methodology: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": 2,
            "output_dir": self.output_dir,
            "scenario": self.scenario,
            "hardware": self.hardware,
            "arms": list(self.arms),
            "trace_windows": self.trace_windows,
            "regions": {
                arm: {name: asdict(summary) for name, summary in values.items()}
                for arm, values in self.regions.items()
            },
            "comparisons": self.comparisons,
            "training": {
                arm: asdict(summary) for arm, summary in self.training.items()
            },
            "losses": {
                arm: [{"step": step, "value": value} for step, value in values]
                for arm, values in self.losses.items()
            },
            "gradient_norms": {
                arm: [{"step": step, "value": value} for step, value in values]
                for arm, values in self.gradient_norms.items()
            },
            "significance_methodology": self.significance_methodology,
            "warnings": list(self.warnings),
        }
        return _json_safe(value)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with null so results are strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def describe(values: list[float]) -> tuple[int, float, float, float]:
    """Return (n, mean, sd, median) for one sample."""
    count = len(values)
    if not count:
        raise ValueError("cannot summarize an empty sample")
    standard_deviation = statistics.stdev(values) if count > 1 else 0.0
    return (
        count,
        statistics.mean(values),
        standard_deviation,
        statistics.median(values),
    )


def summarize(values: list[float]) -> SampleSummary:
    count, mean, standard_deviation, median = describe(values)
    return SampleSummary(count, mean, median, standard_deviation)


def region_comparison(
    base: dict[str, list[float]],
    arm: dict[str, list[float]],
    regions: tuple[Region, ...],
) -> list[dict[str, float | int | str]]:
    """Compare invocation distributions without claiming independent samples."""
    rows: list[dict[str, float | int | str]] = []
    for region in regions:
        base_values, arm_values = base[region.name], arm[region.name]
        n_base, mean_base, sd_base, median_base = describe(base_values)
        n_arm, mean_arm, sd_arm, median_arm = describe(arm_values)
        welch = scipy_stats.ttest_ind(base_values, arm_values, equal_var=False)
        mann_whitney = scipy_stats.mannwhitneyu(
            base_values, arm_values, alternative="two-sided"
        )
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
                "mwu_p": float(mann_whitney.pvalue),
                "cohens_d": (mean_arm - mean_base) / pooled_sd if pooled_sd else 0.0,
            }
        )
    return rows


def _trajectory(log_path: Path, pattern: re.Pattern[str]) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    values = []
    with log_path.open(errors="replace") as log_file:
        for line in log_file:
            match = pattern.search(line)
            if match:
                values.append((int(match.group(1)), float(match.group(2))))
    return values


def losses(log_path: Path) -> list[tuple[int, float]]:
    return _trajectory(log_path, LOSS_METRIC)


def grad_norms(log_path: Path) -> list[tuple[int, float]]:
    return _trajectory(log_path, GRAD_NORM_METRIC)


def training_metrics(log_path: Path) -> list[tuple[int, float, int]]:
    """Return (step, peak-memory-GiB, tokens/s) rows from a training log."""
    if not log_path.exists():
        return []
    rows = []
    with log_path.open(errors="replace") as log_file:
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


def stable_tps(
    rows: list[tuple[int, float, int]], workload: dict[str, Any]
) -> list[int]:
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


def load_run(
    out_dir: Path, arms_override: list[str] | tuple[str, ...] | None
) -> tuple[dict[str, Any], list[str], tuple[Region, ...], list[str]]:
    """Load current and legacy manifests without hiding compatibility warnings."""
    warnings: list[str] = []
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not manifest:
        warnings.append(f"no manifest.json under {out_dir}; assuming piper-1B regions")
    manifest_regions = manifest.get("regions", [])
    if manifest_regions and not all("phase" in region for region in manifest_regions):
        warnings.append(
            f"{manifest_path} predates phase-based regions "
            f"(schema {manifest.get('schema_version')}); assuming piper-1B regions"
        )
        manifest_regions = []
    regions = tuple(Region(**region) for region in manifest_regions) or PIPER_1B_REGIONS
    arms = list(
        arms_override
        or manifest.get("selected_arms")
        or [arm["name"] for arm in manifest.get("arms", [])]
        or ["baseline", "helion", "te"]
    )
    return manifest, arms, regions, warnings


def evaluate_run(
    out_dir: Path, arms_override: list[str] | tuple[str, ...] | None = None
) -> EvaluationResult:
    """Evaluate profiler regions and end-to-end metrics for selected arms."""
    out_dir = out_dir.resolve()
    manifest, arms, declared_regions, warnings = load_run(out_dir, arms_override)
    if "baseline" not in arms:
        raise ValueError("comparison needs the baseline arm")

    pooled: dict[str, dict[str, list[float]]] = {}
    trace_windows: dict[str, int] = {}
    for arm in arms:
        paths = trace_files(out_dir / arm)
        if not paths:
            raise ValueError(f"no profiler traces under {out_dir / arm}")
        try:
            pooled[arm] = pooled_region_samples(paths, declared_regions)
        except ValueError as error:
            raise ValueError(f"{arm}: {error}") from error
        trace_windows[arm] = len(paths)

    regions = {
        arm: {
            region.name: summarize(pooled[arm][region.name])
            for region in declared_regions
        }
        for arm in arms
    }
    comparisons = {
        arm: region_comparison(pooled["baseline"], pooled[arm], declared_regions)
        for arm in arms
        if arm != "baseline"
    }

    workload = manifest.get("workload", {})
    raw_training = {
        arm: training_metrics(out_dir / f"{arm}.log") for arm in arms
    }
    stable_samples = {
        arm: stable_tps(rows, workload) for arm, rows in raw_training.items()
    }
    baseline_samples = stable_samples["baseline"]
    baseline_median = (
        statistics.median(baseline_samples) if baseline_samples else None
    )
    training = {}
    for arm in arms:
        samples = stable_samples[arm]
        median_tps = statistics.median(samples) if samples else None
        peak_memory = max(
            (memory for _, memory, _ in raw_training[arm]), default=None
        )
        ratio = (
            median_tps / baseline_median
            if median_tps is not None
            and baseline_median is not None
            and baseline_median != 0
            else None
        )
        training[arm] = TrainingSummary(
            stable_tokens_per_second=median_tps,
            stable_sample_count=len(samples),
            baseline_ratio=ratio,
            peak_memory_gib=peak_memory,
        )

    return EvaluationResult(
        output_dir=str(out_dir),
        scenario=manifest.get("scenario", "unknown"),
        hardware=manifest.get("hardware", "unknown"),
        arms=tuple(arms),
        trace_windows=trace_windows,
        regions=regions,
        comparisons=comparisons,
        training=training,
        losses={arm: losses(out_dir / f"{arm}.log") for arm in arms},
        gradient_norms={arm: grad_norms(out_dir / f"{arm}.log") for arm in arms},
        significance_methodology=SIGNIFICANCE_METHODOLOGY.copy(),
        warnings=tuple(warnings),
    )


def write_results(result: EvaluationResult, path: Path | None = None) -> Path:
    destination = path or Path(result.output_dir) / "results.json"
    atomic_write_json(destination, result.to_dict())
    return destination
