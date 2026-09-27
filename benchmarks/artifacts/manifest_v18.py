"""Read a schema 18 manifest as the schema 19 manifest of the same run."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from benchmarks.e2e.engines.api import Engine
from benchmarks.e2e.engines.registry import engine_named


V18_SCHEMA_VERSION = 18

V18_DATASET = "c4_test"
"""The dataset of every schema 18 run."""

ARM_KEYS = (
    "compile",
    "override_imports",
    "overrides_per_block",
    "trace_kernel_markers",
    "requires_gcc_toolset",
)
"""The config fields that a schema 18 arm record holds under the same name."""

TOP_LEVEL_KEYS = {
    "p2p_sync": "megatron_p2p_sync",
    "nan_guard": "megatron_nan_guard",
    "precision": "megatron_precision",
}
"""The config fields that schema 18 held as top-level keys, by field name."""

EXTRA_FLAGS_KEYS = {
    "torchtitan": "extra_torchtitan_args",
    "megatron_stock": "extra_megatron_args",
}
"""The top-level passthrough list that reached each engine."""

EXECUTION_MODEL_ENGINES = ("torchtitan",)
"""The engines that the top-level ``execution_model`` of schema 18 describes."""

PARALLELISM_KEYS = (
    "dp",
    "pp",
    "ep",
    "pp_schedule",
    "pp_microbatch_size",
    "zero",
    "world_size",
    "n_microbatches",
)
"""The keys of the schema 18 ``parallelism`` block that schema 19 keeps."""


def _config(
    manifest: Mapping[str, Any], record: Mapping[str, Any], engine: Engine
) -> dict[str, Any]:
    """The schema 19 config of one schema 18 arm record."""
    values = {}
    for field in dataclasses.fields(engine.config_type):
        name = field.name
        if name == "extra_flags":
            values[name] = manifest[EXTRA_FLAGS_KEYS[engine.name]]
        elif name == "module":
            values[name] = manifest["workload"]["module"]
        elif name == "config":
            values[name] = record["config"] or manifest["workload"]["config"]
        elif name in TOP_LEVEL_KEYS:
            values[name] = manifest[TOP_LEVEL_KEYS[name]]
        elif name in ARM_KEYS:
            values[name] = record[name]
        else:
            raise ValueError(
                f"schema 18 records no value for "
                f"{engine.config_type.__name__}.{name}"
            )
    return values


def _arm(manifest: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    """The schema 19 record of one selected schema 18 arm."""
    engine = engine_named(record["engine"])
    return {
        "name": record["name"],
        "description": record["description"],
        "engine": engine.name,
        "config_type": engine.config_type.__name__,
        "config": _config(manifest, record, engine),
        "execution_model": manifest["execution_model"]
        if engine.name in EXECUTION_MODEL_ENGINES
        else None,
        "command": manifest["commands"][record["name"]],
        "env_delta": None,
        "cpu_pinning": manifest["hardware_metadata"]["cpu_pinning"],
    }


def upgrade_v18(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The schema 19 form of a schema 18 manifest; it holds the selected arms alone."""
    if manifest.get("schema_version") != V18_SCHEMA_VERSION:
        raise ValueError(
            f"the manifest records schema {manifest.get('schema_version')!r}, "
            f"not {V18_SCHEMA_VERSION}"
        )
    workload = manifest["workload"]
    parallelism = manifest["parallelism"]
    declared = {record["name"]: record for record in manifest["arms"]}
    return {
        "schema_version": 19,
        "scenario": manifest["scenario"],
        "description": manifest["description"],
        "hardware": manifest["hardware"],
        "hardware_metadata": manifest["hardware_metadata"],
        "run": {
            "shape": manifest["model_shape"],
            "data": {
                "dataset": V18_DATASET,
                "seq_len": workload["seq_len"],
                "local_batch_size": workload["local_batch_size"],
                "steps": workload["steps"],
            },
            "parallelism": {key: parallelism[key] for key in PARALLELISM_KEYS},
            "ac_mode": manifest["ac_mode"],
            "profile": manifest["profile"],
            "window": {
                "freq": workload["profile_freq"],
                "warmup": workload["profiler_warmup"],
                "active": workload["profiler_active"],
                "min_windows": workload["min_trace_windows"],
            },
            "warmup_steps": manifest["warmup_steps"],
            "seed": workload["seed"],
        },
        "arms": [_arm(manifest, declared[name]) for name in manifest["selected_arms"]],
        "throughput_definition": manifest["throughput_definition"],
    }
