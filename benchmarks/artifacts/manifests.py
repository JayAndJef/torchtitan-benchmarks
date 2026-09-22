"""``manifest.json``: what a run is, and whether it may be resumed.

Everything here serializes one run's *identity* -- scenario, arms, the
command line each arm was launched with, the eight global run axes the
``RunAxes`` record carries, the resolved model shape, the execution model,
and the provenance block -- reads one back, and decides whether a recorded
run is the same run the caller is now asking for.

Those last two are here rather than in modules of their own on purpose. A
manifest field and its resume rule are two halves of one invariant: a new
axis adds a field to ``RunAxes`` *and* a key to ``AXIS_KEYS``, which the
writer and ``_resume_mismatches`` both read, and a field recorded but not
gated is a comparability boundary that silently does not hold. Splitting
reader from writer would divide the same invariant the other way -- a schema
bump has to move both together, and neither half is meaningful alone.

**The axis keys stay flat, and their names stay fixed.** One nested
``axes`` object would read better and would break every external reader
that addresses ``manifest["parallelism"]`` or ``manifest["ac_mode"]`` by
name. ``RunAxes`` is the in-process grouping; the file keeps the shape it
had.

**The reader takes one schema.** Every field above is a comparability
boundary, so a manifest another schema wrote cannot be read as this one.
``load_manifest`` names the version it found and the version it wants, and
refuses. There are no per-version branches and no defaulted lookups: a run
recorded under an older schema is read by the code that wrote it.

What *is* split out is everything engine-neutral: output layout and the
atomic writer are ``layout.py``, the progress ledger is ``run_state.py``,
sample summarization is ``summaries.py``. This module is exactly the part
that is not, which is why the whole of the edge described below now lands in
one file.

**Structural edge, almost closed.** A manifest serializes a run's scenario
and arms, so the builders here need those types by construction. ``Arm``
and ``Scenario`` come from ``benchmarks.e2e.schema``, which imports nothing
first-party, and ``RunAxes`` and ``RunRequest`` from
``benchmarks.e2e.axes``. Neither needs a ``TYPE_CHECKING`` block. The
``RunRequest`` import is the sharpest case: it used to come from
``e2e/runner.py``, which imports this module, and only the
``TYPE_CHECKING`` guard kept that pair out of a module-level cycle. It sits
in ``axes.py`` now, below this module, so the cycle cannot come back.

No name crosses to ``benchmarks.e2e.registry`` any more. ``Workload``,
which ``_resume_workload`` reconstructs and revalidates from recorded JSON,
comes from ``schema`` as well.

At *package* granularity ``artifacts/`` and ``e2e/`` remain mutually
dependent, because ``e2e/runner.py`` and ``e2e/results.py`` import from this
module. The candidate resolution is to move this module into ``e2e/``
outright, leaving ``artifacts/`` engine-neutral; splitting ``layout.py`` and
``run_state.py`` off reduced it to a single-file move. The move itself is
deferred rather than made permanent.

"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from benchmarks.artifacts.layout import atomic_write_json
from benchmarks.e2e.axes import RunAxes, RunRequest
from benchmarks.e2e.parallelism import (
    ParallelismSpec,
    describe as describe_parallelism,
    execution_model,
)
from benchmarks.e2e.schema import Arm, Scenario, Workload
from benchmarks.models.piper_qwen3.shape import (
    canonical_size_name,
    shape_by_name,
)


MANIFEST_SCHEMA_VERSION = 18

THROUGHPUT_DEFINITION = "tokens_per_second_per_device"
"""What the ``tps`` step-log figure, and ``stable_tokens_per_second``, count.

Both engines divide one rank's own token count by ``cp * tp * pp``, so the
value is per device. The manifest records the definition because a later
tensor or context parallel degree would move the divisor again. It is not
resume-gated: one revision cannot write two definitions.
"""


def _parallelism_record(
    scenario: Scenario, parallelism: ParallelismSpec
) -> dict[str, Any]:
    """The ``parallelism`` block for one run, written and compared here.

    ``describe`` needs the local batch size, because the microbatch count is
    arithmetic over the batch and the microbatch size. Taking it from the
    scenario's own workload keeps the writer and the resume comparison
    reading one number: one call produces both.
    """
    return describe_parallelism(
        parallelism, local_batch_size=scenario.workload.local_batch_size
    )


AXIS_KEYS = (
    "ac_mode",
    "model_size",
    "parallelism",
    "megatron_p2p_sync",
    "megatron_nan_guard",
    "megatron_precision",
    "profile",
    "warmup_steps",
)
"""The manifest keys that record the run axes, one key per ``RunAxes`` field.

The names are flat and written out here rather than derived from the
dataclass. External readers address them by name, so a renamed field must
stay a deliberate schema change. ``tests/test_axes.py`` compares this tuple
against ``RunAxes``, so a new axis that nothing recorded fails there.
"""


def _axis_record(scenario: Scenario, axes: RunAxes) -> dict[str, Any]:
    """The flat axis keys of one manifest, written and compared here.

    Two of the eight are not the field value itself. ``model_size`` is
    canonicalized, so a fresh manifest never carries a retired alias.
    ``parallelism`` becomes the described block, which holds the derived
    degrees a reader needs beside the six spec fields.
    """
    record = {**asdict(axes)}
    record["model_size"] = canonical_size_name(axes.model_size)
    record["parallelism"] = _parallelism_record(scenario, axes.parallelism)
    return {key: record[key] for key in AXIS_KEYS}


def manifest_data(
    scenario: Scenario,
    selected_arms: tuple[Arm, ...],
    commands: dict[str, list[str]],
    hardware: str,
    metadata: dict[str, str],
    torchtitan_args: list[str] | tuple[str, ...],
    *,
    # No default, because the resume check compares every field: a writer
    # that defaults what the checker demands records the wrong run.
    megatron_args: list[str] | tuple[str, ...],
    axes: RunAxes,
) -> dict[str, Any]:
    shape = shape_by_name(canonical_size_name(axes.model_size))
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scenario": scenario.name,
        "description": scenario.description,
        "hardware": hardware,
        "hardware_metadata": metadata,
        "workload": asdict(scenario.workload),
        "arms": [asdict(arm) for arm in scenario.arms],
        "selected_arms": [arm.name for arm in selected_arms],
        "commands": commands,
        "extra_torchtitan_args": list(torchtitan_args),
        "extra_megatron_args": list(megatron_args),
        **_axis_record(scenario, axes),
        "model_shape": shape.describe(seq_len=scenario.workload.seq_len),
        "throughput_definition": THROUGHPUT_DEFINITION,
        "execution_model": execution_model(axes.parallelism),
    }


def write_manifest(
    out_dir: Path,
    scenario: Scenario,
    selected_arms: tuple[Arm, ...],
    commands: dict[str, list[str]],
    hardware: str,
    metadata: dict[str, str],
    torchtitan_args: list[str] | tuple[str, ...],
    *,
    megatron_args: list[str] | tuple[str, ...],
    axes: RunAxes,
) -> None:
    atomic_write_json(
        out_dir / "manifest.json",
        manifest_data(
            scenario,
            selected_arms,
            commands,
            hardware,
            metadata,
            torchtitan_args,
            megatron_args=megatron_args,
            axes=axes,
        ),
    )


def load_manifest(out_dir: Path) -> dict[str, Any]:
    """Read one manifest. Refuse anything another schema wrote."""
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {manifest_path}: {error}") from error
    found = manifest.get("schema_version")
    if found != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{manifest_path} records manifest schema {found!r}; this code "
            f"reads schema {MANIFEST_SCHEMA_VERSION} only"
        )
    return manifest


def _resume_mismatches(
    manifest: dict,
    scenario: Scenario,
    arms: tuple[Arm, ...],
    hardware: str,
    metadata: dict[str, str],
    torchtitan_args: tuple[str, ...],
    *,
    megatron_args: tuple[str, ...],
    axes: RunAxes,
) -> list[str]:
    axis_record = _axis_record(scenario, axes)
    # Compared on its own, because a recorded manifest may carry an alias.
    del axis_record["model_size"]
    expected = {
        "scenario": scenario.name,
        "workload": asdict(scenario.workload),
        "selected_arms": [arm.name for arm in arms],
        "hardware": hardware,
        "extra_torchtitan_args": list(torchtitan_args),
        "extra_megatron_args": list(megatron_args),
        **axis_record,
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    if canonical_size_name(
        str(manifest.get("model_size"))
    ) != canonical_size_name(axes.model_size):
        mismatches.append("model_size")
    existing_metadata = manifest.get("hardware_metadata", {})
    for key in (
        "nvidia_smi",
        "cpu_pinning",
        "torchtitan_git_rev",
        "benchmarks_git_rev",
        "megatron_git_rev",
    ):
        if existing_metadata.get(key) != metadata.get(key):
            mismatches.append(f"hardware_metadata.{key}")
    return mismatches


def _resume_workload(
    manifest: dict,
    request: RunRequest,
    environment: Mapping[str, str],
) -> Workload:
    """Hydrate unspecified settings and reject explicit resume conflicts."""
    try:
        workload = Workload(**manifest["workload"])
    except (KeyError, TypeError) as error:
        raise ValueError("resume manifest has an invalid workload") from error

    requested_values = {
        "seq_len": request.seq_len
        if request.seq_len is not None
        else environment.get("SEQ"),
        "steps": request.steps
        if request.steps is not None
        else environment.get("STEPS"),
        "local_batch_size": request.batch
        if request.batch is not None
        else environment.get("BATCH"),
    }
    conflicts = [
        name
        for name, value in requested_values.items()
        if value is not None and int(value) != getattr(workload, name)
    ]
    if conflicts:
        raise ValueError(
            "resume request conflicts with the recorded workload: "
            + ", ".join(conflicts)
        )
    return workload


def load_run(
    out_dir: Path, arms_override: list[str] | tuple[str, ...] | None
) -> tuple[dict[str, Any], list[str]]:
    """The manifest of one run, and the arms a reader asked for."""
    manifest = load_manifest(out_dir)
    arms = list(arms_override or manifest["selected_arms"])
    return manifest, arms
