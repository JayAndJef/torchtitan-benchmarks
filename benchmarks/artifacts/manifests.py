"""``manifest.json``: how the runner writes the record of one run, and how a reader and a resume read it back."""

from __future__ import annotations

import dataclasses
import json
import typing
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from benchmarks.artifacts.layout import atomic_write_json
from benchmarks.artifacts.manifest_v18 import V18_SCHEMA_VERSION, upgrade_v18
from benchmarks.e2e.engines.api import (
    Arm,
    DataSpec,
    EngineConfig,
    ProfileWindow,
    RunSpec,
)
from benchmarks.e2e.engines.registry import engine_for, engine_named
from benchmarks.e2e.overrides import field_types
from benchmarks.e2e.parallelism import ParallelismSpec, describe
from benchmarks.e2e.schema import Scenario
from benchmarks.models.piper_qwen3.shape import PiperShape


MANIFEST_SCHEMA_VERSION = 19

THROUGHPUT_DEFINITION = "tokens_per_second_per_device"
"""What the tokens/s of a step sample and ``stable_tokens_per_second`` count."""

HOST_KEYS = (
    "nvidia_smi",
    "cpu_pinning",
    "torchtitan_git_rev",
    "benchmarks_git_rev",
    "megatron_git_rev",
)
"""The ``hardware_metadata`` keys that a resume must find unchanged."""


@dataclass(frozen=True)
class ArmRecord:
    """One arm of a run, as its manifest records it."""

    arm: Arm
    command: tuple[str, ...]
    env_delta: Mapping[str, str] | None
    """The keys that the launcher and the engine set; a schema 18 manifest records none."""
    cpu_pinning: str
    """The pinning prefix text, the host's reason for none, or ``declined by engine``."""
    execution_model: str | None
    """How the arm holds the model state; a schema 18 manifest records it for TorchTitan arms alone."""


@dataclass(frozen=True)
class RunRecord:
    """One run, as its manifest records it."""

    scenario: str
    description: str
    hardware: str
    metadata: Mapping[str, str]
    run: RunSpec
    arms: tuple[ArmRecord, ...]
    """The selected arms, in run order."""

    def arm(self, name: str) -> ArmRecord:
        """The record of the arm ``name``."""
        for record in self.arms:
            if record.arm.name == name:
                return record
        raise ValueError(
            f"the run records no arm {name!r}. Available: "
            + ", ".join(record.arm.name for record in self.arms)
        )


def run_json(run: RunSpec) -> dict[str, Any]:
    """The ``run`` block of a manifest."""
    return {
        "shape": run.shape.describe(seq_len=run.data.seq_len),
        "data": asdict(run.data),
        "parallelism": describe(
            run.parallelism, local_batch_size=run.data.local_batch_size
        ),
        "ac_mode": run.ac_mode,
        "profile": run.profile,
        "window": asdict(run.window),
        "warmup_steps": run.warmup_steps,
        "seed": run.seed,
    }


def config_json(config: EngineConfig) -> dict[str, Any]:
    """The JSON form of an engine config, one key per field."""
    return json.loads(json.dumps(asdict(config)))


def arm_json(record: ArmRecord) -> dict[str, Any]:
    """The manifest record of one arm."""
    arm = record.arm
    return {
        "name": arm.name,
        "description": arm.description,
        "engine": engine_for(arm).name,
        "config_type": type(arm.config).__name__,
        "config": config_json(arm.config),
        "execution_model": record.execution_model,
        "command": list(record.command),
        "env_delta": None if record.env_delta is None else dict(record.env_delta),
        "cpu_pinning": record.cpu_pinning,
    }


def manifest_data(
    *,
    scenario: Scenario,
    hardware: str,
    metadata: Mapping[str, str],
    run: RunSpec,
    arms: tuple[ArmRecord, ...],
) -> dict[str, Any]:
    """The schema 19 manifest of one run."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scenario": scenario.name,
        "description": scenario.description,
        "hardware": hardware,
        "hardware_metadata": dict(metadata),
        "run": run_json(run),
        "arms": [arm_json(record) for record in arms],
        "throughput_definition": THROUGHPUT_DEFINITION,
    }


def write_manifest(out_dir: Path, **fields: Any) -> None:
    """Write ``manifest_data(**fields)`` to ``out_dir/manifest.json``."""
    atomic_write_json(out_dir / "manifest.json", manifest_data(**fields))


def _read(out_dir: Path) -> tuple[dict[str, Any], Path]:
    """The JSON of one manifest, of any schema, and its path."""
    path = out_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"manifest is missing: {path}")
    try:
        return json.loads(path.read_text()), path
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {path}: {error}") from error


def load_manifest(out_dir: Path) -> dict[str, Any]:
    """The schema 19 manifest that a resume continues; any other schema is refused by name."""
    manifest, path = _read(out_dir)
    found = manifest.get("schema_version")
    if found == V18_SCHEMA_VERSION:
        raise ValueError(
            f"{path} records manifest schema 18, which a resume cannot "
            f"continue; this code resumes schema {MANIFEST_SCHEMA_VERSION} "
            "only. Start a new run"
        )
    if found != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{path} records manifest schema {found!r}; this code resumes "
            f"schema {MANIFEST_SCHEMA_VERSION} only"
        )
    return manifest


def _typed(annotation: Any, value: Any, where: str) -> Any:
    """``value`` from JSON, as the field type ``annotation`` holds it."""
    origin = typing.get_origin(annotation)
    if origin is tuple:
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"{where} is not a list of strings: {value!r}")
        return tuple(value)
    if origin is Literal:
        if value not in typing.get_args(annotation):
            raise ValueError(
                f"{where} {value!r} is not one of "
                + ", ".join(str(choice) for choice in typing.get_args(annotation))
            )
        return value
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if annotation in (bool, int, str) and type(value) is annotation:
        return value
    raise ValueError(f"{where} is not a {annotation!r}: {value!r}")


def _config(
    config_type: type[EngineConfig], values: Mapping[str, Any]
) -> EngineConfig:
    """The engine config that ``values`` records."""
    types = field_types(config_type)
    name = config_type.__name__
    unknown = sorted(set(values) - set(types))
    missing = sorted(set(types) - set(values))
    if unknown or missing:
        raise ValueError(
            f"the {name} record has unknown fields {unknown} and lacks "
            f"fields {missing}"
        )
    return config_type(
        **{
            field: _typed(annotation, values[field], f"{name}.{field}")
            for field, annotation in types.items()
        }
    )


def _arm_record(record: Mapping[str, Any]) -> ArmRecord:
    """The arm that one manifest record describes."""
    engine = engine_named(record["engine"])
    if record["config_type"] != engine.config_type.__name__:
        raise ValueError(
            f"arm {record['name']!r} records config type "
            f"{record['config_type']!r}, and engine {engine.name!r} takes "
            f"{engine.config_type.__name__}"
        )
    env_delta = record["env_delta"]
    return ArmRecord(
        arm=Arm(
            name=record["name"],
            description=record["description"],
            config=_config(engine.config_type, record["config"]),
        ),
        command=tuple(record["command"]),
        env_delta=None if env_delta is None else dict(env_delta),
        cpu_pinning=record["cpu_pinning"],
        execution_model=record["execution_model"],
    )


def _shape(described: Mapping[str, Any]) -> PiperShape:
    """The shape that a manifest describes, built from the record and not from the registry."""
    shape = PiperShape(
        **{field.name: described[field.name] for field in dataclasses.fields(PiperShape)}
    )
    if shape.param_count != described["param_count"]:
        raise ValueError(
            f"shape {shape.name!r} records {described['param_count']:,} "
            f"parameters, and its fields give {shape.param_count:,}"
        )
    return shape


def _run(block: Mapping[str, Any]) -> RunSpec:
    """The run that a manifest's ``run`` block describes."""
    parallelism = block["parallelism"]
    return RunSpec(
        shape=_shape(block["shape"]),
        data=DataSpec(**block["data"]),
        parallelism=ParallelismSpec(
            **{
                field.name: parallelism[field.name]
                for field in dataclasses.fields(ParallelismSpec)
            }
        ),
        ac_mode=block["ac_mode"],
        profile=block["profile"],
        window=ProfileWindow(**block["window"]),
        warmup_steps=block["warmup_steps"],
        seed=block["seed"],
    )


def run_record(manifest: Mapping[str, Any], source: str) -> RunRecord:
    """The run that a schema 18 or schema 19 manifest records; ``source`` names the file."""
    found = manifest.get("schema_version")
    try:
        if found == V18_SCHEMA_VERSION:
            manifest = upgrade_v18(manifest)
        elif found != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"{source} records manifest schema {found!r}; this code reads "
                f"schemas {V18_SCHEMA_VERSION} and {MANIFEST_SCHEMA_VERSION}"
            )
        return RunRecord(
            scenario=manifest["scenario"],
            description=manifest["description"],
            hardware=manifest["hardware"],
            metadata=dict(manifest["hardware_metadata"]),
            run=_run(manifest["run"]),
            arms=tuple(_arm_record(record) for record in manifest["arms"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{source} does not record a readable run: "
            f"{type(error).__name__}: {error}"
        ) from error


def load_run_record(out_dir: Path) -> RunRecord:
    """The run that ``out_dir/manifest.json`` records, under schema 18 or 19."""
    manifest, path = _read(out_dir)
    return run_record(manifest, str(path))


def resume_mismatches(
    manifest: Mapping[str, Any], *, run: RunSpec, arms: tuple[Arm, ...]
) -> list[str]:
    """Each run key and each arm field where a resume request differs from the manifest."""
    mismatches = []
    recorded_run = manifest["run"]
    for key, value in run_json(run).items():
        if recorded_run.get(key) != value:
            mismatches.append(f"run.{key}")
    recorded_arms = {record["name"]: record for record in manifest["arms"]}
    if list(recorded_arms) != [arm.name for arm in arms]:
        mismatches.append("arms")
    for arm in arms:
        recorded = recorded_arms.get(arm.name)
        if recorded is None:
            continue
        if recorded["engine"] != engine_for(arm).name:
            mismatches.append(f"{arm.name}.engine")
            continue
        recorded_config = recorded["config"]
        for field, value in config_json(arm.config).items():
            if recorded_config.get(field) != value:
                mismatches.append(f"{arm.name}.config.{field}")
    return mismatches


def host_mismatches(
    manifest: Mapping[str, Any], *, hardware: str, metadata: Mapping[str, str]
) -> list[str]:
    """The hardware label and each ``HOST_KEYS`` value where this host differs from the manifest."""
    mismatches = []
    if manifest["hardware"] != hardware:
        mismatches.append("hardware")
    recorded = manifest["hardware_metadata"]
    for key in HOST_KEYS:
        if recorded.get(key) != metadata.get(key):
            mismatches.append(f"hardware_metadata.{key}")
    return mismatches
