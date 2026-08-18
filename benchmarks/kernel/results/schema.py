"""Machine-readable results for kernel-isolation benchmark runs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from benchmarks.artifacts.layout import atomic_write_json
from benchmarks.artifacts.summaries import SampleSummary


# 2: the single flat shape record was split into the model geometry
# (model_size / model_shape) and the workload run through it, so schema-1
# files carry no equivalent of either field and are rejected outright.
#
# 3: the measurand changed from a round-robin wall median to a burst-amortized
# per-call time. ``n`` and ``warmup`` counted interleaved *cycles* and
# were printed as such, so they are **replaced** by ``replicates`` /
# ``samples_per_replicate`` / ``burst_k`` / ``warmup_calls`` rather than
# reinterpreted -- a repurposed field would put a false statement in every
# table generated from the file. ``ModeResult.samples_us`` becomes
# ``replicates_us`` for the same reason: the replicate boundaries are the
# repetition unit the statistics are computed over, and a flat list cannot
# express them.
#
# 4: every declared arm now appears in ``arms``, carrying a ``status``. At
# schema 3 an arm this host could not run and an arm that was never declared
# looked identical -- both were simply absent -- so a reader could not tell a
# short roster from a complete one. An arm that did not measure has empty
# ``modes`` and a ``status_reason`` saying why.
KERNEL_RESULTS_SCHEMA_VERSION = 4

# ``ok`` measured. ``skipped`` was never launched, because this host cannot
# run it or because the gates failed first. ``failed`` was launched and did
# not produce a complete replicate set.
ARM_STATUSES = ("ok", "skipped", "failed")


@dataclass(frozen=True)
class ModeResult:
    """One arm's timings for one mode; raw samples kept for re-analysis.

    ``replicates_us`` is replicate-major: one inner tuple per replicate, each
    holding that replicate's burst-amortized per-call samples. ``summary``
    describes the pooled samples.
    """

    summary: SampleSummary
    replicates_us: tuple[tuple[float, ...], ...]
    derived: dict[str, float] = field(default_factory=dict)

    @property
    def samples_us(self) -> tuple[float, ...]:
        """The pooled samples, for callers that do not need the boundaries."""
        return tuple(
            value for replicate in self.replicates_us for value in replicate
        )


@dataclass(frozen=True)
class ArmResult:
    """One declared arm's outcome. ``modes`` is empty unless ``status`` is
    ``ok``: an arm that did not measure still appears, so the roster in the
    file matches the roster in the registry."""

    name: str
    modes: dict[str, ModeResult]
    peak_memory_gib: float | None = None
    # Keyed by mode, then by burst size. Schema 2 keyed by burst size alone,
    # because the pass only ever ran on "forward".
    burst_us_per_call: dict[str, dict[str, float]] | None = None
    status: str = "ok"
    status_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ARM_STATUSES:
            raise ValueError(
                f"{self.name}: unknown arm status {self.status!r}; "
                f"expected one of {', '.join(ARM_STATUSES)}"
            )


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
    model_size: str
    model_shape: dict[str, Any]
    workload: dict[str, Any]
    shapes: dict[str, Any]
    replicates: int
    samples_per_replicate: int
    burst_k: int
    warmup_calls: int
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
            "model_size": self.model_size,
            "model_shape": self.model_shape,
            "workload": self.workload,
            "shapes": self.shapes,
            "replicates": self.replicates,
            "samples_per_replicate": self.samples_per_replicate,
            "burst_k": self.burst_k,
            "warmup_calls": self.warmup_calls,
            "seed": self.seed,
            "arms": {
                name: {
                    "modes": {
                        mode: {
                            "summary": asdict(result.summary),
                            "replicates_us": [
                                list(replicate)
                                for replicate in result.replicates_us
                            ],
                            "derived": result.derived,
                        }
                        for mode, result in arm.modes.items()
                    },
                    "peak_memory_gib": arm.peak_memory_gib,
                    "burst_us_per_call": arm.burst_us_per_call,
                    "status": arm.status,
                    "status_reason": arm.status_reason,
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
                        replicates_us=tuple(
                            tuple(replicate)
                            for replicate in entry["replicates_us"]
                        ),
                        derived=dict(entry.get("derived", {})),
                    )
                    for mode, entry in arm["modes"].items()
                },
                peak_memory_gib=arm.get("peak_memory_gib"),
                burst_us_per_call=arm.get("burst_us_per_call"),
                status=arm["status"],
                status_reason=arm["status_reason"],
            )
            for name, arm in value["arms"].items()
        }
        return cls(
            scenario=value["scenario"],
            hardware=value["hardware"],
            model_size=value["model_size"],
            model_shape=value["model_shape"],
            workload=value["workload"],
            shapes=value["shapes"],
            replicates=value["replicates"],
            samples_per_replicate=value["samples_per_replicate"],
            burst_k=value["burst_k"],
            warmup_calls=value["warmup_calls"],
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
