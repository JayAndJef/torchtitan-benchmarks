"""``manifest.json``: what a run is, and whether it may be resumed.

Everything here serializes one run's *identity* -- scenario, arms, the
command line each arm was launched with, the four global axes
(``compile_mode`` / ``ac_mode`` / ``model_size`` / ``parallelism``), the
resolved model shape, the execution model, and the provenance block -- reads
one back including manifests written by older schemas, and decides whether a
recorded run is the same run the caller is now asking for.

Those last two are here rather than in modules of their own on purpose. A
manifest field and its resume rule are two halves of one invariant: adding
the ``--model-size`` axis (schema 9) added ``model_size`` to
``manifest_data`` *and* a ``model_size`` comparison to
``_resume_mismatches`` in a single commit, and a field recorded but not
gated is a comparability boundary that silently does not hold. Splitting
reader from writer would divide the same invariant the other way -- a schema
bump has to move both together, and neither half is meaningful alone. Schema
10 added ``parallelism`` the same way, in one commit and to both halves.

What *is* split out is everything engine-neutral: output layout and the
atomic writer are ``layout.py``, the progress ledger is ``run_state.py``,
sample summarization is ``summaries.py``. This module is exactly the part
that is not, which is why the whole of the edge described below now lands in
one file.

**Structural edge, pending resolution.** This module imports from
``benchmarks.e2e.registry`` at runtime, which inverts the intuitive layering
(``artifacts/`` looks lower-level than ``e2e/`` and is not). It is
*structural* rather than incidental: a manifest is a serialization of a run's
scenario and arms, so the builders here need those types by construction. Two
names cross at runtime, ``Workload`` (``_resume_workload`` reconstructs and
revalidates it from recorded JSON) and ``EXECUTION_MODEL`` (a manifest
self-description field); ``Scenario`` and ``Arm`` are annotation-only and are
imported under ``TYPE_CHECKING``, so they cost nothing at runtime.

Be precise about what does and does not cycle. At *module* granularity there
is no cycle: ``e2e/registry.py`` imports nothing from ``artifacts/``. At
*package* granularity there is one, because ``e2e/runner.py`` and
``e2e/results.py`` import from this module. (``e2e/validation.py`` used to as
well; it needed only ``trace_files`` and now takes it from ``layout.py``.) So
``artifacts/`` and ``e2e/`` are mutually dependent as packages and only the
module-level ordering keeps imports resolvable. The third name this module
takes from ``e2e`` makes the point sharpest: ``RunRequest`` comes from
``e2e/runner.py``, which imports this module at runtime, so that pair *would*
be a genuine module-level cycle -- it is legal only because it is confined to
``TYPE_CHECKING``. Anything moved out of that block must be re-checked. The
candidate resolution is to move this module into ``e2e/`` outright, leaving
``artifacts/`` engine-neutral; splitting ``layout.py`` and ``run_state.py``
off was the step that reduced it to a single-file move, since every symbol
that would have had to stay behind is already elsewhere. The cheapest
partial step remains relocating ``EXECUTION_MODEL`` into this module, whose
``manifest_data`` is its only consumer repo-wide, which deletes one of the
two runtime names outright and leaves ``Workload`` as the single genuinely
structural edge. Neither is done here: this commit splits by concern, and
the redesign is deferred rather than made permanent.

A **second and unrelated** ``e2e`` edge exists here: ``load_run`` imports
``PIPER_1B_REGIONS`` to infer regions for schema-<8 manifests. That
contradicts the rule that missing historical fields stay explicitly unknown
rather than being inferred from current declarations, and it is a known
defect. It is preserved byte-identical here because fixing it is a behavior
change; when it is fixed, this edge disappears on its own.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from benchmarks.artifacts.layout import atomic_write_json
from benchmarks.e2e.parallelism import (
    ParallelismSpec,
    TRIVIAL_SPEC,
    describe as describe_parallelism,
)
from benchmarks.e2e.registry import (
    DEFAULT_MODEL_SIZE,
    EXECUTION_MODEL,
    PIPER_1B_REGIONS,
    Workload,
)
from benchmarks.models.piper_qwen3.shape import (
    canonical_size_name,
    shape_by_name,
)
from benchmarks.traces.schema import Region

if TYPE_CHECKING:
    from benchmarks.e2e.registry import Arm, Scenario
    from benchmarks.e2e.runner import RunRequest


MANIFEST_SCHEMA_VERSION = 11

# What the ``tps`` figure in every step log line, and therefore
# ``stable_tokens_per_second`` in ``results.json``, counts.
#
# Both engines divide one rank's own token count by ``cp * tp * pp``: the
# ranks of one pipeline share a batch, and each data-parallel rank reads a
# batch of its own, so the data-parallel degree is absent from the divisor
# and the value is per device either way. Recorded rather than assumed
# because tensor and context parallelism would each move the divisor again,
# and a reader of an old directory cannot tell which definition produced its
# numbers.
#
# It is not resume-gated. The value follows from this code rather than from
# an operator's choice, so two directories written by one revision cannot
# disagree, and gating a constant would refuse nothing.
THROUGHPUT_DEFINITION = "tokens_per_second_per_device"


def _parallelism_record(
    scenario: Scenario, parallelism: ParallelismSpec
) -> dict[str, Any]:
    """The ``parallelism`` block for one run, written and compared here.

    ``describe`` needs the local batch size, because the microbatch count is
    arithmetic over the batch and the microbatch size. Taking it from the
    scenario's own workload keeps the writer and the resume comparison
    reading one number: the same call produces the recorded block and the
    default an older manifest is read through.
    """
    return describe_parallelism(
        parallelism, local_batch_size=scenario.workload.local_batch_size
    )


def manifest_data(
    scenario: Scenario,
    selected_arms: tuple[Arm, ...],
    commands: dict[str, list[str]],
    hardware: str,
    metadata: dict[str, str],
    extra_args: list[str] | tuple[str, ...],
    compile_mode: str,
    ac_mode: str,
    # No default. This value is what the manifest *claims* the run was, and
    # _resume_mismatches below already requires it explicitly; a writer that
    # defaults what the checker demands is the asymmetry that lets a huge run
    # be recorded, resumed and published as "1b".
    model_size: str,
    *,
    # No default either, and for the same reason one step further: an omitted
    # argument would record dp 1 x pp 1 for a run of any mesh, which is a
    # single-GPU claim about a job that was not one. Keyword-only because the
    # nine positional parameters above are the schema-9 signature.
    parallelism: ParallelismSpec,
) -> dict[str, Any]:
    # Recorded canonically, so a fresh manifest never carries a retired name.
    model_size = canonical_size_name(model_size)
    shape = shape_by_name(model_size)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "scenario": scenario.name,
        "description": scenario.description,
        "hardware": hardware,
        "hardware_metadata": metadata,
        "workload": asdict(scenario.workload),
        "regions": [asdict(region) for region in scenario.regions],
        "arms": [asdict(arm) for arm in scenario.arms],
        "selected_arms": [arm.name for arm in selected_arms],
        "commands": commands,
        "extra_torchtitan_args": list(extra_args),
        "compile_mode": compile_mode,
        "ac_mode": ac_mode,
        "model_size": model_size,
        "model_shape": shape.describe(seq_len=scenario.workload.seq_len),
        "parallelism": _parallelism_record(scenario, parallelism),
        "throughput_definition": THROUGHPUT_DEFINITION,
        "execution_model": EXECUTION_MODEL,
    }


def write_manifest(
    out_dir: Path,
    scenario: Scenario,
    selected_arms: tuple[Arm, ...],
    commands: dict[str, list[str]],
    hardware: str,
    metadata: dict[str, str],
    extra_args: list[str] | tuple[str, ...],
    compile_mode: str,
    ac_mode: str,
    model_size: str,
    *,
    parallelism: ParallelismSpec,
) -> None:
    atomic_write_json(
        out_dir / "manifest.json",
        manifest_data(
            scenario,
            selected_arms,
            commands,
            hardware,
            metadata,
            extra_args,
            compile_mode,
            ac_mode,
            model_size,
            parallelism=parallelism,
        ),
    )


def load_manifest(out_dir: Path) -> dict[str, Any]:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"manifest is missing: {manifest_path}")
    try:
        return json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {manifest_path}: {error}") from error


def _resume_mismatches(
    manifest: dict,
    scenario: Scenario,
    arms: tuple[Arm, ...],
    hardware: str,
    metadata: dict[str, str],
    extra_args: tuple[str, ...],
    compile_mode: str,
    ac_mode: str,
    model_size: str,
    *,
    parallelism: ParallelismSpec,
) -> list[str]:
    expected = {
        "scenario": scenario.name,
        "workload": asdict(scenario.workload),
        "selected_arms": [arm.name for arm in arms],
        "hardware": hardware,
        "extra_torchtitan_args": list(extra_args),
        "compile_mode": compile_mode,
        "ac_mode": ac_mode,
    }
    mismatches = [
        key for key, value in expected.items() if manifest.get(key) != value
    ]
    # Defaulted lookup rather than a generic entry: schema <= 8 output
    # directories predate the axis and are still resumable as the 1B shape.
    # Both sides go through canonical_size_name, because 42 e2e manifests on
    # disk record the retired name "normal" and 88 more record no size at
    # all, and all of them name the 1B shape. Without that, a resume of a
    # real run would be refused over a rename.
    recorded = canonical_size_name(
        str(manifest.get("model_size", DEFAULT_MODEL_SIZE))
    )
    if recorded != canonical_size_name(model_size):
        mismatches.append("model_size")
    # The same shape of defaulted lookup, one axis later: schema <= 9 output
    # directories predate the parallelism axis and every one of them ran on
    # one GPU, so they are read through the trivial spec's own record and
    # still resume. The default is computed rather than written out, so it
    # cannot drift from what a trivial-spec request produces for this
    # workload.
    recorded_parallelism = manifest.get(
        "parallelism", _parallelism_record(scenario, TRIVIAL_SPEC)
    )
    if recorded_parallelism != _parallelism_record(scenario, parallelism):
        mismatches.append("parallelism")
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
    regions = tuple(Region(**region) for region in manifest_regions)
    if not regions and int(manifest.get("schema_version") or 0) < 8:
        # Pre-schema-8 manifests never declared empty regions on purpose;
        # from schema 8 on, an empty list is an honest declaration (the
        # megatron scenario) and must not be second-guessed.
        regions = PIPER_1B_REGIONS
    arms = list(
        arms_override
        or manifest.get("selected_arms")
        or [arm["name"] for arm in manifest.get("arms", [])]
        or ["baseline", "helion", "te"]
    )
    return manifest, arms, regions, warnings
