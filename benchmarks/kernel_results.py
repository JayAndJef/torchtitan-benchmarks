"""Machine-readable results for kernel-isolation benchmark runs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.artifacts import atomic_write_json
from benchmarks.metrics import SampleSummary


KERNEL_RESULTS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModeResult:
    """One arm's timings for one mode; raw samples kept for re-analysis."""

    summary: SampleSummary
    samples_us: tuple[float, ...]
    derived: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ArmResult:
    name: str
    modes: dict[str, ModeResult]
    peak_memory_gib: float | None = None
    burst_us_per_call: dict[str, float] | None = None


@dataclass(frozen=True)
class CorrectnessResult:
    """Outcome of one CorrectnessCheck on one output tensor."""

    arm: str
    reference: str
    kind: str
    output: str
    metric: str
    value: float
    threshold: float | None
    passed: bool | None
    informational: bool


@dataclass(frozen=True)
class KernelScenarioResult:
    scenario: str
    hardware: str
    spec: dict[str, Any]
    shapes: dict[str, Any]
    n: int
    warmup: int
    seed: int
    arms: dict[str, ArmResult]
    comparisons: list[dict[str, Any]]
    correctness: list[CorrectnessResult]
    all_correctness_passed: bool
    methodology: dict[str, Any]
    environment: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": KERNEL_RESULTS_SCHEMA_VERSION,
            "kind": "kernel",
            "scenario": self.scenario,
            "hardware": self.hardware,
            "spec": self.spec,
            "shapes": self.shapes,
            "n": self.n,
            "warmup": self.warmup,
            "seed": self.seed,
            "arms": {
                name: {
                    "modes": {
                        mode: {
                            "summary": asdict(result.summary),
                            "samples_us": list(result.samples_us),
                            "derived": result.derived,
                        }
                        for mode, result in arm.modes.items()
                    },
                    "peak_memory_gib": arm.peak_memory_gib,
                    "burst_us_per_call": arm.burst_us_per_call,
                }
                for name, arm in self.arms.items()
            },
            "comparisons": self.comparisons,
            "correctness": [asdict(row) for row in self.correctness],
            "all_correctness_passed": self.all_correctness_passed,
            "methodology": self.methodology,
            "environment": self.environment,
            "warnings": list(self.warnings),
        }
        return _json_safe(value)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KernelScenarioResult":
        version = value.get("schema_version")
        if version != KERNEL_RESULTS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported kernel results schema {version!r}; "
                f"expected {KERNEL_RESULTS_SCHEMA_VERSION}"
            )
        arms = {
            name: ArmResult(
                name=name,
                modes={
                    mode: ModeResult(
                        summary=SampleSummary(**entry["summary"]),
                        samples_us=tuple(entry["samples_us"]),
                        derived=dict(entry.get("derived", {})),
                    )
                    for mode, entry in arm["modes"].items()
                },
                peak_memory_gib=arm.get("peak_memory_gib"),
                burst_us_per_call=arm.get("burst_us_per_call"),
            )
            for name, arm in value["arms"].items()
        }
        return cls(
            scenario=value["scenario"],
            hardware=value["hardware"],
            spec=value["spec"],
            shapes=value["shapes"],
            n=value["n"],
            warmup=value["warmup"],
            seed=value["seed"],
            arms=arms,
            comparisons=list(value["comparisons"]),
            correctness=[
                CorrectnessResult(**row) for row in value["correctness"]
            ],
            all_correctness_passed=value["all_correctness_passed"],
            methodology=value["methodology"],
            environment=value["environment"],
            warnings=tuple(value.get("warnings", ())),
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_kernel_results(result: KernelScenarioResult, path: Path) -> Path:
    atomic_write_json(path, result.to_dict())
    return path


def load_kernel_results(path: Path) -> KernelScenarioResult:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read kernel results {path}: {error}") from error
    return KernelScenarioResult.from_dict(value)
