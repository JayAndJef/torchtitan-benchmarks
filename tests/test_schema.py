"""Structural tests for ``benchmarks/e2e/schema.py``.

The module holds the e2e type declarations and imports the standard library
alone. That property is what lets any layer read a type without paying for
the scenario table, the schedule table or the validator, so a test asserts
it on the source rather than trusting a convention.

The check reads the source with ``ast``. A runtime check would pass on a
first-party import inside a function body, and would also depend on what
the test process already imported.
"""

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "benchmarks" / "e2e" / "schema.py"

# The names the module must declare. Listed rather than derived: a moved
# type that lost its declaration would otherwise make every assertion here
# vacuous.
DECLARED_TYPES = (
    "Workload",
    "Arm",
    "Scenario",
    "PipelineSchedule",
    "ParallelismSpec",
    "ValidationProfile",
    "RunRequest",
)


def imported_modules(path: Path) -> tuple[tuple[str, int], ...]:
    """Every module name the source imports, at any depth, with its line."""
    tree = ast.parse(path.read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.append(("." * node.level + (node.module or ""), node.lineno))
            else:
                found.append((node.module or "", node.lineno))
    return tuple(found)


class SchemaImportsNothingFirstPartyTest(unittest.TestCase):
    def test_the_source_file_exists(self):
        """Negative control: a renamed module makes the rest vacuous."""
        self.assertTrue(SCHEMA_PATH.is_file(), f"{SCHEMA_PATH} is missing")

    def test_schema_imports_no_first_party_module(self):
        offenders = [
            f"{name} (line {lineno})"
            for name, lineno in imported_modules(SCHEMA_PATH)
            if name.startswith("benchmarks") or name.startswith(".")
        ]
        self.assertEqual(
            offenders,
            [],
            "benchmarks/e2e/schema.py imports a first-party module; the "
            "declarations must cost the standard library alone:\n  "
            + "\n  ".join(offenders),
        )

    def test_every_declared_type_is_present(self):
        tree = ast.parse(SCHEMA_PATH.read_text())
        declared = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
        }
        missing = [name for name in DECLARED_TYPES if name not in declared]
        self.assertEqual(
            missing,
            [],
            f"benchmarks/e2e/schema.py declares none of: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
