"""Structural tests for the developer scripts outside the Python packages.

Two things are pinned here, because both fail silently:

* the pre-push hook, which runs the CPU suite before a push leaves the
  machine, and the ``sync.sh`` step that installs it; and
* the module list of the continuous-integration workflow, which must name
  test modules that exist and that a host without the machine-learning
  stack can import.
"""

import ast
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_import_boundaries import REPO_ROOT

PRE_PUSH = REPO_ROOT / "tools" / "pre-push.sh"
SYNC = REPO_ROOT / "sync.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"

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


if __name__ == "__main__":
    unittest.main()
