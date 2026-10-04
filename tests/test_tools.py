"""Structural tests for the developer scripts outside the Python packages.

Three things are pinned here:

* the pre-push hook, which runs the CPU suite before a push leaves the
  machine, and the ``sync.sh`` step that installs it;
* the guards of the Slurm matrix driver that need no Slurm and no GPU; and
* the module list of the continuous-integration workflow, which must name
  test modules that exist and that a host without the machine-learning
  stack can import.

It also runs the cell slug of the matrix script, whose failure is a silent
refusal of every pass.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_import_boundaries import REPO_ROOT

PRE_PUSH = REPO_ROOT / "tools" / "pre-push.sh"
SYNC = REPO_ROOT / "sync.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"
RUN_MATRIX = REPO_ROOT / "tools" / "run_matrix.sh"
RUN_MATRIX_CELL = REPO_ROOT / "tools" / "run_matrix_cell.sh"

FORBIDDEN_IMPORTS = ("torch", "transformer_engine")
"""Module roots that a hosted runner cannot install."""

_MODULE_REFERENCE = re.compile(r"\btests\.(test_\w+)\b")


def workflow_modules() -> tuple[str, ...]:
    """The test modules the workflow runs, in the order it names them."""
    return tuple(_MODULE_REFERENCE.findall(WORKFLOW.read_text()))


def _module_path(name: str) -> Path | None:
    """The file of ``name`` if it is an in-repo module, else ``None``."""
    base = REPO_ROOT.joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def import_closure(start: str) -> frozenset[str]:
    """Every module name reachable from ``start`` through ``ast`` imports.

    The walk follows ``import`` and ``from`` statements anywhere in a file,
    not at module scope alone, and it stops at the first module that is not
    in this repository. The result therefore holds the in-repo modules plus
    the outside names any of them reaches.
    """
    reached: set[str] = set()
    pending = [start]
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        path = _module_path(name)
        if path is None:
            continue
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    root = ".".join(parts[: len(parts) - node.level + 1])
                    target = f"{root}.{node.module}" if node.module else root
                else:
                    target = node.module or ""
                pending.append(target)
                pending.extend(f"{target}.{alias.name}" for alias in node.names)
    return frozenset(reached)


class PrePushHookTests(unittest.TestCase):
    def test_the_hook_is_executable(self) -> None:
        self.assertTrue(PRE_PUSH.is_file(), f"{PRE_PUSH} is absent")
        self.assertTrue(
            os.access(PRE_PUSH, os.X_OK),
            "tools/pre-push.sh must be executable; git runs it directly",
        )

    def test_the_hook_starts_with_the_bash_shebang(self) -> None:
        first = PRE_PUSH.read_text().splitlines()[0]
        self.assertEqual(first, "#!/usr/bin/env bash")

    def test_the_hook_runs_the_suite_and_names_the_bypass(self) -> None:
        source = PRE_PUSH.read_text()
        self.assertIn("unittest discover", source)
        self.assertIn("--no-verify", source)

    def test_sync_installs_the_hook_in_the_common_directory(self) -> None:
        source = SYNC.read_text()
        self.assertIn("git rev-parse --git-common-dir", source)
        self.assertIn("pre-push", source)


class MatrixScriptTests(unittest.TestCase):
    """The refusals of the Slurm matrix driver that need no Slurm and no GPU."""

    def _driver(self, **overrides: str) -> subprocess.CompletedProcess:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SLURM_")
        }
        env.update(DRY_RUN="1", NGPUS="1", TIME="10")
        env.update(overrides)
        return subprocess.run(
            ["bash", str(RUN_MATRIX)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_the_scripts_are_executable_bash(self) -> None:
        for script in (RUN_MATRIX, RUN_MATRIX_CELL):
            with self.subTest(script=script.name):
                self.assertTrue(os.access(script, os.X_OK))
                first = script.read_text().splitlines()[0]
                self.assertEqual(first, "#!/usr/bin/env bash")
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_the_driver_refuses_the_reserved_partitions(self) -> None:
        for partition in ("placeholder", "exceptions", "admin"):
            with self.subTest(partition=partition):
                result = self._driver(PARTITION=partition)
                self.assertEqual(result.returncode, 1)
                self.assertIn("run-gpu-job", result.stderr)

    def test_the_driver_refuses_the_physical_gpu_variable(self) -> None:
        result = self._driver(GPU="4")
        self.assertEqual(result.returncode, 1)
        self.assertIn("NGPUS", result.stderr)

    def test_the_driver_needs_a_time_limit(self) -> None:
        result = self._driver(TIME="")
        self.assertEqual(result.returncode, 1)
        self.assertIn("TIME is required", result.stderr)

    def test_the_cell_runner_refuses_to_run_outside_a_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("SLURM_")
            }
            env.update(
                MATRIX_LOG=f"{tmp}/sweep.log",
                MATRIX_OUT=f"{tmp}/cell",
                MATRIX_REV="0" * 40,
                MATRIX_NGPUS="1",
                FOREIGN_MEM_MIB="2000",
                CONTENDED_LOAD="150",
                WATCH_INTERVAL="15",
            )
            result = subprocess.run(
                ["bash", str(RUN_MATRIX_CELL), "run", "0"],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("only inside a Slurm job", result.stdout)


class WorkflowModuleListTests(unittest.TestCase):
    def test_the_workflow_names_at_least_five_modules(self) -> None:
        self.assertGreaterEqual(len(workflow_modules()), 5)

    def test_every_named_module_exists(self) -> None:
        missing = [
            name
            for name in workflow_modules()
            if not (REPO_ROOT / "tests" / f"{name}.py").is_file()
        ]
        self.assertEqual(
            missing,
            [],
            "The workflow names test modules that do not exist: "
            + ", ".join(missing),
        )

    def test_no_named_module_reaches_the_machine_learning_stack(self) -> None:
        offenders = []
        for name in workflow_modules():
            closure = import_closure(f"tests.{name}")
            roots = {reached.partition(".")[0] for reached in closure}
            for forbidden in FORBIDDEN_IMPORTS:
                if forbidden in roots:
                    offenders.append(f"tests.{name} reaches {forbidden}")
        self.assertEqual(
            offenders,
            [],
            "A hosted runner installs neither of these. Remove the module "
            "from the workflow, or defer the import:\n" + "\n".join(offenders),
        )

    def test_the_workflow_does_not_fetch_the_submodules(self) -> None:
        self.assertIn("submodules: false", WORKFLOW.read_text())


def cell_slug(cell: str) -> str:
    """The directory name ``tools/run_matrix.sh`` gives ``cell``."""
    source = RUN_MATRIX.read_text()
    match = re.search(r"^cell_slug\(\) \{\n.*?^\}\n", source, re.M | re.S)
    assert match, "tools/run_matrix.sh has no cell_slug function"
    return subprocess.run(
        ["bash", "-c", match.group(0) + 'cell_slug "$1"', "bash", cell],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


class MatrixCellSlugTests(unittest.TestCase):
    PIPELINE_FLAGS_CELL = (
        "--scenario engines --model-size 30b-a3b-20l --dp 2 --pp 2 --ep 2 "
        "--zero 1 --pp-schedule 1F1B --pp-microbatch-size 4 --batch 8 "
        "--profile --arm megatron_stock "
        "--megatron-arg=--cross-entropy-loss-fusion "
        "--megatron-arg=--moe-permute-fusion "
        "--megatron-arg=--overlap-grad-reduce "
        "--megatron-arg=--overlap-param-gather"
    )

    def test_a_short_cell_keeps_its_whole_slug(self) -> None:
        self.assertEqual(
            cell_slug("--dp 2 --arm titan_eager"), "dp-2-arm-titan_eager"
        )

    def test_the_longest_derived_name_fits_name_max(self) -> None:
        """The 2026-09-30 pipeline cell slugged to 272 bytes, and every pass
        failed before it wrote a log."""
        slug = cell_slug(self.PIPELINE_FLAGS_CELL)
        self.assertEqual(len(slug), 200)
        self.assertLessEqual(
            len(slug + ".contaminated-20260930T064353Z.log"), 255
        )

    def test_two_long_cells_that_share_a_prefix_differ(self) -> None:
        other = self.PIPELINE_FLAGS_CELL.replace(
            "overlap-param-gather", "use-flash-attn"
        )
        self.assertNotEqual(
            cell_slug(self.PIPELINE_FLAGS_CELL), cell_slug(other)
        )



if __name__ == "__main__":
    unittest.main()
