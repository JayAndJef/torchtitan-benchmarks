"""Pin the compatibility contract for pre-restructure run artifacts.

A package restructure deletes ``piper1b/``, ``megatron_baseline/`` and every
current ``benchmarks/*.py`` module path. Roughly 200 historical run
directories under ``out/`` record those paths *as data*: ``workload.module``,
full argv containing ``-m megatron_baseline.train``, kernel builder strings
like ``benchmarks.kernel_arms:build_qkv_unfused``.

The contract those artifacts live under is narrow and deliberate:

* they stay **readable as data** -- a schema-aware decoder still parses them;
* reading them **never imports** the paths they name;
* they are **not resumable**, **not replayable**, and **not reconstructible**.

Nothing here adds decoding capability. Every test asserts a property the
current code already has, so the restructure cannot silently drop it.

Fixtures (checked in because ``out/`` is gitignored, copied byte-for-byte --
no recorded value is hand-edited):

* ``tests/fixtures/legacy/e2e_schema8/{manifest,run_state}.json``
  from ``out/20260807T175156Z/piper1b_megatron/nvidia-h200/`` -- an e2e
  manifest at schema 8 (current is 9), scenario ``piper1b_megatron``,
  a retired compile-mode value, ``ac_mode=none``, recorded on
  ``benchmarks_git_rev`` ``de8bc295f5``. Chosen because it is the richest
  single artifact for this purpose: it names ``piper1b`` as a workload
  module, ``megatron_baseline.train`` in argv, and
  ``piper1b.swiglu.combined_swiglu`` in ``override_imports``.
* ``tests/fixtures/legacy/kernel_schema1/{manifest,results}.json``
  from ``out/20260806T042025Z/kernels/qkv/nvidia-h200/`` -- a kernel
  results/manifest pair at schema 1 (current is 2), with the pre-split flat
  ``spec`` key and ``benchmarks.kernel_arms:build_qkv_*`` builder strings.
"""

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.manifests import (
    MANIFEST_SCHEMA_VERSION,
    _resume_mismatches,
    load_manifest,
    load_run,
)
from benchmarks.e2e.parallelism import TRIVIAL_SPEC
from benchmarks.e2e.registry import Arm, ENGINES
from benchmarks.e2e.runner import RunRequest, execute_run
from benchmarks.execution.affinity import CpuPinning
from benchmarks.kernel.results.schema import (
    KERNEL_RESULTS_SCHEMA_VERSION,
    load_kernel_results,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY = Path(__file__).resolve().parent / "fixtures" / "legacy"
E2E_SCHEMA_8 = LEGACY / "e2e_schema8"
KERNEL_SCHEMA_1 = LEGACY / "kernel_schema1"

# Module paths the fixtures record as data. Every one of them is deleted or
# moved by the restructure, and none may ever be imported as a side effect of
# reading an artifact.
RECORDED_MODULE_PATHS = (
    "megatron_baseline",
    "megatron_baseline.train",
    "piper1b.config_registry",
    "piper1b.pretokenized_data",
    "piper1b.swiglu",
    "piper1b.swiglu.combined_swiglu",
    "benchmarks.kernel_arms",
    "benchmarks.kernel_worker",
)

# Reads of the manifest's recorded ``commands``. ``execute_run`` rebuilds every
# command from current code, so this must stay a write-only provenance field.
COMMANDS_READ = re.compile(
    r"""\[\s*["']commands["']\s*\]|\.get\(\s*["']commands["']"""
)
SKIPPED_TREES = (".venv", "third_party", "out", "reports", ".git", "tests")


def _fixture_e2e_manifest() -> dict:
    return json.loads((E2E_SCHEMA_8 / "manifest.json").read_text())


def _legacy_copy(destination: Path) -> Path:
    """Copy the legacy run directory so no test can touch the fixture."""
    shutil.copytree(E2E_SCHEMA_8, destination)
    return destination


class LegacyFixtureTests(unittest.TestCase):
    """The fixtures are genuinely old, or the rest of this file proves nothing."""

    def test_the_e2e_fixture_predates_the_current_manifest_schema(self) -> None:
        manifest = _fixture_e2e_manifest()
        self.assertEqual(manifest["schema_version"], 8)
        self.assertLess(manifest["schema_version"], MANIFEST_SCHEMA_VERSION)
        state = json.loads((E2E_SCHEMA_8 / "run_state.json").read_text())
        self.assertEqual(state["status"], "completed")

    def test_the_e2e_fixture_records_the_doomed_module_paths(self) -> None:
        manifest = _fixture_e2e_manifest()
        self.assertEqual(manifest["workload"]["module"], "piper1b")
        self.assertIn("megatron_baseline.train", manifest["commands"]["baseline"])
        overrides = {
            entry
            for arm in manifest["arms"]
            for entry in arm["override_imports"]
        }
        self.assertTrue(
            any(entry.startswith("piper1b.swiglu.") for entry in overrides),
            overrides,
        )

    def test_the_kernel_fixture_predates_the_current_results_schema(self) -> None:
        results = json.loads((KERNEL_SCHEMA_1 / "results.json").read_text())
        self.assertEqual(results["schema_version"], 1)
        self.assertLess(results["schema_version"], KERNEL_RESULTS_SCHEMA_VERSION)
        # Schema 1 carried a flat ``spec``; schema 2 split it into
        # model_size / model_shape / workload.
        self.assertIn("spec", results)
        self.assertNotIn("model_shape", results)

    def test_the_kernel_fixture_records_builder_strings(self) -> None:
        manifest = json.loads((KERNEL_SCHEMA_1 / "manifest.json").read_text())
        builders = [arm["builder"] for arm in manifest["arms"]]
        self.assertTrue(builders)
        for builder in builders:
            self.assertTrue(
                builder.startswith("benchmarks.kernel_arms:"), builder
            )
        self.assertIn("benchmarks.kernel_worker", manifest["command"])


# Run the decode in a clean interpreter: the parent test process has already
# imported half the repo, so only a fresh process can tell "the decoder needed
# this" from "the artifact caused this".
IMPORT_PROBE = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {repo!r})

    from benchmarks.artifacts.manifests import load_run
    from pathlib import Path

    # Everything the decoder itself pulls in, before it has seen an artifact.
    baseline = set(sys.modules)

    manifest, arms, warnings = load_run(Path({e2e!r}), None)
    kernel_manifest = json.loads(
        (Path({kernel!r}) / "manifest.json").read_text()
    )
    builders = [arm["builder"] for arm in kernel_manifest["arms"]]

    delta = sorted(set(sys.modules) - baseline)
    print(json.dumps({{
        "schema_version": manifest.get("schema_version"),
        "scenario": manifest.get("scenario"),
        "arms": arms,
        "warnings": warnings,
        "workload_module": manifest["workload"]["module"],
        "workload_module_type": type(manifest["workload"]["module"]).__name__,
        "builders": builders,
        "builder_types": sorted({{type(b).__name__ for b in builders}}),
        "module_delta": delta,
        "baseline_had": sorted(
            name for name in {recorded!r} if name in baseline
        ),
        "recorded_now_imported": sorted(
            name for name in {recorded!r} if name in sys.modules
        ),
    }}))
    """
)


class LegacyDecodeTests(unittest.TestCase):
    """Property: legacy artifacts decode as data, and only as data."""

    def test_load_run_decodes_the_schema_8_manifest(self) -> None:
        manifest, arms, warnings = load_run(E2E_SCHEMA_8, None)
        self.assertEqual(manifest["schema_version"], 8)
        self.assertEqual(manifest["scenario"], "piper1b_megatron")
        self.assertEqual(
            arms,
            [
                "baseline",
                "titan_stock",
                "titan_swiglu",
                "titan_lm_head",
                "titan_swiglu_lm_head",
            ],
        )
        self.assertEqual(warnings, [])

    def test_decoding_never_imports_what_the_artifact_records(self) -> None:
        probe = IMPORT_PROBE.format(
            repo=str(REPO_ROOT),
            e2e=str(E2E_SCHEMA_8),
            kernel=str(KERNEL_SCHEMA_1),
            recorded=list(RECORDED_MODULE_PATHS),
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)

        # It really decoded.
        self.assertEqual(report["schema_version"], 8)
        self.assertEqual(report["scenario"], "piper1b_megatron")

        # Reading the artifact imported nothing at all. This is the strong
        # form: not "it did not import the bad ones", but "it did not import".
        self.assertEqual(report["module_delta"], [])

        # And none of the recorded paths is live in the process. (Guard
        # against a decoder that starts depending on them at import time:
        # anything already in ``baseline`` would slip past the delta check.)
        self.assertEqual(report["baseline_had"], [])
        self.assertEqual(report["recorded_now_imported"], [])

        # The recorded paths came back as inert strings, not resolved objects.
        self.assertEqual(report["workload_module"], "piper1b")
        self.assertEqual(report["workload_module_type"], "str")
        self.assertEqual(report["builder_types"], ["str"])
        self.assertIn("benchmarks.kernel_arms:build_qkv_fused", report["builders"])

    def test_decoding_does_not_touch_the_fixture(self) -> None:
        before = {
            path: path.read_bytes()
            for path in sorted(E2E_SCHEMA_8.iterdir())
        }
        load_run(E2E_SCHEMA_8, None)
        after = {
            path: path.read_bytes()
            for path in sorted(E2E_SCHEMA_8.iterdir())
        }
        self.assertEqual(before, after)


class LegacyKernelResultsTests(unittest.TestCase):
    """Property: the kernel results loader fails CLOSED on schema 1.

    Fail-closed is correct. A schema-1 file has a flat ``spec`` where schema 2
    wants ``model_size`` / ``model_shape`` / ``workload``; a lenient read would
    publish a number under the wrong model identity. Do not "fix" this.
    """

    def test_schema_1_results_are_rejected(self) -> None:
        with self.assertRaises(ValueError) as caught:
            load_kernel_results(KERNEL_SCHEMA_1 / "results.json")
        message = str(caught.exception)
        # Diagnosable: it names what it found and what it wanted, so the
        # operator can tell a stale artifact from a corrupt one.
        self.assertIn("unsupported kernel results schema", message)
        self.assertIn("1", message)
        self.assertIn(str(KERNEL_RESULTS_SCHEMA_VERSION), message)

    def test_rejection_is_not_a_crash_and_not_a_partial_read(self) -> None:
        raw = json.loads((KERNEL_SCHEMA_1 / "results.json").read_text())
        # The payload is well-formed and rich -- rejection is a schema
        # decision, not a parse failure or a missing-key accident.
        self.assertTrue(raw["arms"])
        self.assertTrue(raw["comparisons"])
        with self.assertRaises(ValueError):
            load_kernel_results(KERNEL_SCHEMA_1 / "results.json")
        # A ValueError, specifically: callers catch it. Not KeyError/TypeError
        # leaking out of a half-built object.
        try:
            load_kernel_results(KERNEL_SCHEMA_1 / "results.json")
        except ValueError:
            pass
        except Exception as error:  # pragma: no cover - documents the contract
            self.fail(f"schema-1 rejection raised {type(error).__name__}: {error}")


class LegacyResumeTests(unittest.TestCase):
    """Property: a pre-restructure run directory is not resumable.

    The move commit changes ``benchmarks_git_rev``, and ``_resume_mismatches``
    compares that field. So every existing output directory becomes
    unresumable the moment the restructure lands -- no new code required. This
    pins that the guard is really the git-rev field.
    """

    def setUp(self) -> None:
        self.manifest = _fixture_e2e_manifest()
        self.recorded_metadata = dict(self.manifest["hardware_metadata"])
        # The recorded scenario is retired, so the roster is rebuilt from
        # the manifest. Only the git rev may then differ.
        self.scenario = replace(
            ENGINES,
            name=self.manifest["scenario"],
            arms=tuple(
                Arm(name=name, description=name)
                for name in self.manifest["selected_arms"]
            ),
        )

    def _mismatches(self, benchmarks_git_rev: str) -> list[str]:
        metadata = {
            **self.recorded_metadata,
            "benchmarks_git_rev": benchmarks_git_rev,
        }
        return _resume_mismatches(
            self.manifest,
            self.scenario,
            self.scenario.arms,
            self.manifest["hardware"],
            metadata,
            tuple(self.manifest["extra_torchtitan_args"]),
            self.manifest["compile_mode"],
            self.manifest["ac_mode"],
            "normal",
            # A legacy fixture predates the parallelism axis, so it carries no
            # ``parallelism`` key and is read through the trivial spec's own
            # record. Requesting that spec is what a plain resume does.
            parallelism=TRIVIAL_SPEC,
            megatron_p2p_sync="on",
            megatron_nan_guard="on",
            megatron_precision="stock",
        )

    def test_the_git_rev_field_is_the_guard(self) -> None:
        recorded_rev = self.recorded_metadata["benchmarks_git_rev"]
        after_restructure = self._mismatches("restructure-commit-sha")
        unchanged_tree = self._mismatches(recorded_rev)

        self.assertIn("hardware_metadata.benchmarks_git_rev", after_restructure)
        self.assertNotIn(
            "hardware_metadata.benchmarks_git_rev", unchanged_tree
        )
        # Toggling *only* the git rev toggles *only* that mismatch: the
        # refusal is caused by the field, not by something incidental that
        # happens to co-vary with it.
        self.assertEqual(
            set(after_restructure) - set(unchanged_tree),
            {"hardware_metadata.benchmarks_git_rev"},
        )
        self.assertEqual(set(unchanged_tree) - set(after_restructure), set())

    def test_resuming_a_pre_restructure_run_is_refused(self) -> None:
        metadata = {
            **self.recorded_metadata,
            "benchmarks_git_rev": "restructure-commit-sha",
        }
        pinning = CpuPinning((), self.recorded_metadata["cpu_pinning"])
        never = mock.Mock(side_effect=AssertionError("training must not launch"))

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=(self.manifest["hardware"], metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning", return_value=pinning
        ):
            out_dir = _legacy_copy(Path(temporary) / "legacy")
            request = RunRequest(
                gpu=self.recorded_metadata["requested_gpu"],
                scenario_name=None,
                resume_dir=out_dir,
            )
            with self.assertRaises(ValueError) as caught:
                execute_run(
                    request,
                    process_runner=never,
                    environment={"PATH": os.environ["PATH"]},
                )

        # The recorded scenario is retired, so the refusal names it. The
        # git-rev guard below is the one that outlives the rename.
        message = str(caught.exception)
        self.assertIn("Unknown scenario", message)
        never.assert_not_called()


class LegacyCommandsAreWriteOnlyTests(unittest.TestCase):
    """Property: the manifest's recorded ``commands`` is never read back.

    ``execute_run`` rebuilds every arm's command from current code and only
    ever indexes its own freshly built ``commands`` dict. The manifest field is
    provenance: it records what *was* run, and is never a source for what *will*
    be run. That is what makes a legacy manifest non-replayable.
    """

    def _source_files(self) -> list[Path]:
        return [
            path
            for path in REPO_ROOT.rglob("*.py")
            if not any(part in SKIPPED_TREES for part in path.parts)
        ]

    def test_only_a_reporting_tool_reads_the_recorded_commands(self) -> None:
        readers = {
            path.relative_to(REPO_ROOT).as_posix()
            for path in self._source_files()
            if COMMANDS_READ.search(path.read_text(encoding="utf-8", errors="ignore"))
        }
        # ``collect_matrix.py`` (today at tools/collect_matrix.py) copies the
        # field verbatim into a matrix summary. That is a data pass-through --
        # it never executes it, and it is the *only* reader in the tree. Pinned
        # by file name rather than by path so the restructure may relocate it;
        # a *new* name here means something started consuming recorded argv,
        # and that is the regression this guards.
        self.assertEqual({Path(name).name for name in readers}, {"collect_matrix.py"})

    def test_the_execution_path_does_not_read_the_manifest_commands(self) -> None:
        # Locate the module through the function itself, so the assertion
        # follows ``execute_run`` wherever the restructure puts it.
        runner_path = Path(inspect.getsourcefile(execute_run) or "")
        self.assertTrue(runner_path.is_file(), runner_path)
        self.assertIsNone(COMMANDS_READ.search(runner_path.read_text()))
        # ``load_manifest`` hands the whole dict back; the write side is in
        # artifacts/manifests.py and stores what the runner just built.
        self.assertIn("commands", load_manifest(E2E_SCHEMA_8))

    def test_resume_never_gets_far_enough_to_read_commands(self) -> None:
        """Corrupt the recorded argv; the refusal is still about the git rev."""
        metadata = {
            **json.loads((E2E_SCHEMA_8 / "manifest.json").read_text())[
                "hardware_metadata"
            ],
            "benchmarks_git_rev": "restructure-commit-sha",
        }
        recorded = _fixture_e2e_manifest()["hardware_metadata"]
        pinning = CpuPinning((), recorded["cpu_pinning"])
        never = mock.Mock(side_effect=AssertionError("training must not launch"))

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("nvidia-h200", metadata),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning", return_value=pinning
        ):
            out_dir = _legacy_copy(Path(temporary) / "legacy")
            manifest_path = out_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["commands"] = {
                name: ["/nonexistent/python", "-m", "megatron_baseline.train"]
                for name in manifest["commands"]
            }
            manifest_path.write_text(json.dumps(manifest))

            request = RunRequest(
                gpu=recorded["requested_gpu"],
                scenario_name=None,
                resume_dir=out_dir,
            )
            with self.assertRaises(ValueError) as caught:
                execute_run(
                    request,
                    process_runner=never,
                    environment={"PATH": os.environ["PATH"]},
                )

        # The recorded scenario is retired, so the refusal now names it.
        # Either way the resume stops before it reads the recorded argv.
        message = str(caught.exception)
        self.assertIn("Unknown scenario", message)
        self.assertNotIn("commands", message)
        never.assert_not_called()


if __name__ == "__main__":
    unittest.main()
