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
#
# 5: a run at ``replicates_per_process > 1`` publishes every statistic taken
# over the per-replicate log-ratios under a ``within_process_`` name --
# ``within_process_ratio_ci_low``/``_high`` and
# ``within_process_replicate_ratio_spread`` -- instead of the honest ones, and
# ``methodology`` gains ``replicates_per_process``. Renamed rather than
# reinterpreted: those replicates are consecutive measurements of one build in
# one interpreter, so the interval is a lower bound and the spread is the
# round-to-round spread that sharing a process removes, and anything reading
# an honest name must find nothing rather than a narrower number. A schema-4
# comparison row always carried ``ratio_ci_low``; after this a row may not,
# which is a change to the shape of the file and therefore a bump rather than
# an addition.
#
# 7: spans arrive, and with them a second shape of kernel results file. A
# **span** is an implementation that fuses across a scenario cut, so it is
# declared over an ordered scenario range and its claim is the span against
# the **sum of the scenarios it replaces**. A span result therefore holds two
# totals where a scenario result holds one, and ``KernelSpanResult`` is a
# separate type rather than a ``KernelScenarioResult`` whose ``scenario``
# field secretly holds a span name.
#
# The **rename** that makes this a bump rather than an addition is the value
# of ``kind``. It was ``"kernel"`` on every file, and every file was a
# scenario -- so the word named the family *and* one member of it. That is
# one word with two meanings, which is how ``n`` and ``warmup`` went wrong at
# schema 3.
#
# The concrete wrong reading it prevents: a **recursive** walk of ``out/``
# -- ``rglob("results.json")``, ``find out -name results.json``, or any
# reader handed a path -- meets both shapes and has ``kind`` as its only way
# to tell them apart. Keyed on ``kind == "kernel"`` it would take a span
# total for a scenario total and sum it beside the scenarios the span
# replaces, double-counting the same work.
#
# **Not** because a span escapes a glob. A span writes one directory deeper,
# under ``out/<ts>/kernels/spans/<name>/<hw>/``, so the shallow
# ``out/*/kernels/*/*/results.json`` pattern cannot reach it -- that
# separation is what the directory is for
# (``benchmarks.kernel.runner``), and it is a second guard rather than the
# reason for this one. The rename is what protects every reader that does
# not use that pattern.
#
# Rejected: keep ``"kernel"`` for scenarios and add ``"kernel_span"``. That
# lets a reader testing ``kind == "kernel"`` keep working while silently
# skipping every span, which is the half-read the exact-equality loader
# exists to prevent. Retiring the value makes both sides fail loudly.
KERNEL_RESULTS_SCHEMA_VERSION = 7

# The two shapes a kernel results file takes, named once. ``load_kernel_
# results`` dispatches on the value, and each ``from_dict`` refuses the
# other's.
KERNEL_SCENARIO_RESULT_KIND = "kernel_scenario"
KERNEL_SPAN_RESULT_KIND = "kernel_span"

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
    compiled: bool = True
    eager_reason: str | None = None

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


def _arms_to_dict(arms: dict[str, ArmResult]) -> dict[str, Any]:
    """One arm roster, serialized. Shared by both result kinds.

    A span's arms are arms: same statuses, same modes, same declared compile
    treatment. Only what surrounds them differs.
    """
    return {
        name: {
            "modes": {
                mode: {
                    "summary": asdict(result.summary),
                    "replicates_us": [
                        list(replicate) for replicate in result.replicates_us
                    ],
                    "derived": result.derived,
                }
                for mode, result in arm.modes.items()
            },
            "peak_memory_gib": arm.peak_memory_gib,
            "burst_us_per_call": arm.burst_us_per_call,
            "status": arm.status,
            "status_reason": arm.status_reason,
            # The declared compile treatment, carried here from the
            # registry by the merge. A cross-engine ratio compares
            # two treatments and not two kernels, so a row that does
            # not name both sides cannot be read. A reader of this
            # file holds no registry.
            "compiled": arm.compiled,
            "eager_reason": arm.eager_reason,
        }
        for name, arm in arms.items()
    }


def _arms_from_dict(value: dict[str, Any]) -> dict[str, ArmResult]:
    return {
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
            compiled=arm["compiled"],
            eager_reason=arm["eager_reason"],
        )
        for name, arm in value.items()
    }


def _check_envelope(value: dict[str, Any], expected_kind: str) -> None:
    """Refuse a file of the wrong schema, and of the wrong kind.

    The version check has always been exact: an older file is rejected rather
    than half-read. The kind check is the schema-7 half of the same
    discipline. Two shapes of kernel results file now exist, and a span file
    read as a scenario file would publish the span's own number as a
    scenario's -- which is half of a two-total claim presented as a whole
    one.
    """
    version = value.get("schema_version")
    if version != KERNEL_RESULTS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported kernel results schema {version!r}; "
            f"expected {KERNEL_RESULTS_SCHEMA_VERSION}"
        )
    kind = value.get("kind")
    if kind != expected_kind:
        raise ValueError(
            f"expected a {expected_kind!r} results file, got {kind!r}"
        )


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
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": KERNEL_RESULTS_SCHEMA_VERSION,
            "kind": KERNEL_SCENARIO_RESULT_KIND,
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
            "arms": _arms_to_dict(self.arms),
            "comparisons": self.comparisons,
            "correctness": [asdict(row) for row in self.correctness],
            "all_correctness_passed": self.all_correctness_passed,
            "methodology": self.methodology,
            "environment": self.environment,
            "warnings": list(self.warnings),
            "description": self.description,
        }
        return _json_safe(value)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KernelScenarioResult":
        _check_envelope(value, KERNEL_SCENARIO_RESULT_KIND)
        arms = _arms_from_dict(value["arms"])
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
            description=value["description"],
        )


@dataclass(frozen=True)
class SpanPartResult:
    """One enclosed scenario's contribution to one span arm's parts total.

    ``replicate_medians_us`` maps a mode to that part arm's **per-replicate
    median**, one value per replicate, in replicate order. The parts total of
    a replicate is the sum of these across the range, so the sum is auditable
    from the file rather than asserted by it.

    Summed at the replicate level and never at the sample level. Sample ``i``
    of one scenario and sample ``i`` of the next are unrelated bursts taken in
    different sweeps, so adding them element-wise would invent a pairing that
    does not exist. Replicate ``r`` of both really is replicate ``r`` of one
    run, which is the pairing there is.

    ``results_path`` is where the part was measured. It is provenance a reader
    needs and cannot derive: the parts total is a number this file did not
    take.
    """

    scenario: str
    arm: str
    replicate_medians_us: dict[str, tuple[float, ...]]
    results_path: str


@dataclass(frozen=True)
class KernelSpanResult:
    """One span: its own measurement, and the sum of what it replaces.

    A span is an implementation that fuses across a scenario cut, so it
    belongs to no single scenario. It is declared over an **ordered scenario
    range** and its claim is the span against the **sum of the scenarios it
    replaces**. This file therefore holds two totals, and the whole of its
    shape is about keeping them apart:

    * ``arms`` holds the span's **own** measurement, in the field every
      kernel results file uses for a measured arm.
    * ``parts`` holds the other side, keyed by span arm: which scenario,
      which arm of it, that arm's per-replicate medians, and where it was
      measured.
    * ``comparisons`` keeps the meaning it has in a scenario file -- arm
      against arm, **both measured inside the span**.
    * ``parts_comparisons`` is the span-versus-parts claim, in a separate
      list under a separate name.

    A reader who opens one and finds the other has opened the wrong field.

    ``scenarios`` is the declared range, in order. ``shapes`` is keyed by
    enclosed scenario name and holds each one's ``shape_summary``: a span's
    inputs are the first cut's and its outputs are the last cut's, so no
    single entry describes it and the union is what a reader needs.
    """

    span: str
    scenarios: tuple[str, ...]
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
    parts: dict[str, tuple[SpanPartResult, ...]]
    comparisons: list[dict[str, Any]]
    parts_comparisons: list[dict[str, Any]]
    correctness: list[CorrectnessResult]
    all_correctness_passed: bool
    methodology: dict[str, Any]
    environment: dict[str, Any]
    warnings: tuple[str, ...] = ()
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": KERNEL_RESULTS_SCHEMA_VERSION,
            "kind": KERNEL_SPAN_RESULT_KIND,
            # Not "scenario". A span name is not a scenario name, and a
            # reader must not be able to look this up in KERNEL_SCENARIOS.
            "span": self.span,
            "scenarios": list(self.scenarios),
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
            "arms": _arms_to_dict(self.arms),
            "parts": {
                arm: [
                    {
                        "scenario": part.scenario,
                        "arm": part.arm,
                        "replicate_medians_us": {
                            mode: list(values)
                            for mode, values in
                            part.replicate_medians_us.items()
                        },
                        "results_path": part.results_path,
                    }
                    for part in parts
                ]
                for arm, parts in self.parts.items()
            },
            "comparisons": self.comparisons,
            "parts_comparisons": self.parts_comparisons,
            "correctness": [asdict(row) for row in self.correctness],
            "all_correctness_passed": self.all_correctness_passed,
            "methodology": self.methodology,
            "environment": self.environment,
            "warnings": list(self.warnings),
            "description": self.description,
        }
        return _json_safe(value)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KernelSpanResult":
        _check_envelope(value, KERNEL_SPAN_RESULT_KIND)
        return cls(
            span=value["span"],
            scenarios=tuple(value["scenarios"]),
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
            arms=_arms_from_dict(value["arms"]),
            parts={
                arm: tuple(
                    SpanPartResult(
                        scenario=part["scenario"],
                        arm=part["arm"],
                        replicate_medians_us={
                            mode: tuple(values)
                            for mode, values in
                            part["replicate_medians_us"].items()
                        },
                        results_path=part["results_path"],
                    )
                    for part in parts
                )
                for arm, parts in value["parts"].items()
            },
            comparisons=list(value["comparisons"]),
            parts_comparisons=list(value["parts_comparisons"]),
            correctness=[
                CorrectnessResult(**row) for row in value["correctness"]
            ],
            all_correctness_passed=value["all_correctness_passed"],
            methodology=value["methodology"],
            environment=value["environment"],
            warnings=tuple(value.get("warnings", ())),
            description=value["description"],
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_kernel_results(
    result: KernelScenarioResult | KernelSpanResult, path: Path
) -> Path:
    atomic_write_json(path, result.to_dict())
    return path


def load_kernel_results(
    path: Path,
) -> KernelScenarioResult | KernelSpanResult:
    """Read either kind of kernel results file, dispatching on ``kind``.

    A caller that wants one kind specifically calls that type's
    ``from_dict``, which refuses the other. This function is for a caller
    that reads whatever is on disk.
    """
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read kernel results {path}: {error}") from error
    if value.get("kind") == KERNEL_SPAN_RESULT_KIND:
        return KernelSpanResult.from_dict(value)
    return KernelScenarioResult.from_dict(value)
