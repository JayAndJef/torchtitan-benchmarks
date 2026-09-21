"""Structural tests for ``benchmarks/e2e/schema.py`` and the e2e layering.

The module holds the e2e type declarations and imports the standard library
alone. That property is what lets any layer read a type without paying for
the scenario table, the schedule table or the validator, so a test asserts
it on the source rather than trusting a convention.

``LAYER_ORDER`` below states the one direction the e2e and artifact modules
may import in. It is a declaration and a test at once: a module may import
only modules that sit earlier in the list, so a cycle cannot be written and
a new edge that inverts the stack fails here rather than at import time.

Both checks read the source with ``ast``. A runtime check would pass on a
first-party import inside a function body, and would also depend on what
the test process already imported.
"""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "benchmarks" / "e2e" / "schema.py"

# The packages this layering covers. ``benchmarks.kernel`` is out of scope:
# it runs its own registry and imports none of these modules.
LAYERED_PACKAGES = ("benchmarks.e2e", "benchmarks.artifacts", "benchmarks.cli")

# The import direction of the e2e stack, lowest first. A module may import
# only modules that sit EARLIER in this list.
#
# The declaration answers two questions a reader otherwise has to rebuild by
# hand: which module owns a type, and which module may read it.
#
# ``benchmarks.artifacts.layout`` sits near the bottom because the validator,
# the manifest writer and the runner all read the output layout, and it reads
# only the declarations. ``benchmarks.artifacts.manifests`` sits above
# ``engines`` and below ``runner``, because the runner writes the manifest.
LAYER_ORDER = (
    "benchmarks.e2e.megatron_stock",
    "benchmarks.cli.rendering",
    "benchmarks.artifacts.summaries",
    "benchmarks.e2e.schema",
    "benchmarks.artifacts.layout",
    "benchmarks.artifacts.run_state",
    "benchmarks.e2e.parallelism",
    "benchmarks.e2e.axes",
    "benchmarks.e2e.registry",
    "benchmarks.e2e.megatron_stock.flags",
    "benchmarks.e2e.megatron_stock.markers",
    "benchmarks.e2e.megatron_stock.train",
    "benchmarks.e2e.megatron_stock.model_builder",
    "benchmarks.e2e.launch",
    "benchmarks.e2e.validation",
    "benchmarks.e2e.engines",
    "benchmarks.artifacts.manifests",
    "benchmarks.e2e.runner",
    "benchmarks.e2e.results",
    "benchmarks.cli.e2e",
    "benchmarks.cli.kernel",
    "benchmarks.cli.main",
    "benchmarks.cli.__main__",
)

# The names the module must declare. Listed rather than derived: a moved
# type that lost its declaration would otherwise make every assertion here
# vacuous.
DECLARED_TYPES = (
    "Workload",
    "Arm",
    "Scenario",
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


def module_name(path: str) -> str:
    """The dotted name of a tracked source file."""
    name = path[: -len(".py")].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def layered_sources() -> dict[str, Path]:
    """Every tracked module of the layered packages, by dotted name."""
    listed = subprocess.run(
        ["git", "ls-files", "benchmarks"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    found = {}
    for path in listed:
        if not path.endswith(".py"):
            continue
        name = module_name(path)
        if any(
            name == package or name.startswith(package + ".")
            for package in LAYERED_PACKAGES
        ):
            found[name] = REPO_ROOT / path
    return found


def module_scope_nodes(tree: ast.Module):
    """Every import statement at module scope, including a ``TYPE_CHECKING``
    block.

    A deferred import inside a function body is excluded on purpose. That is
    the one legal way to break a cycle, and one pair uses it: the stock
    megatron driver imports its own model builder inside a function, because
    the builder imports the driver at module scope.
    """
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, ast.If):
            for inner in ast.walk(node):
                if isinstance(inner, (ast.Import, ast.ImportFrom)):
                    yield inner


def layered_imports(path: Path) -> tuple[tuple[str, int], ...]:
    """The layered modules this source imports at module scope."""
    tree = ast.parse(path.read_text())
    found = []
    for node in module_scope_nodes(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module]
        else:
            continue
        for name in names:
            if any(
                name == package or name.startswith(package + ".")
                for package in LAYERED_PACKAGES
            ):
                found.append((name, node.lineno))
    return tuple(found)


class LayeringTest(unittest.TestCase):
    """The e2e stack imports in one direction, and the list states it."""

    def test_every_listed_module_exists(self):
        """Negative control: a stale name would check nothing."""
        sources = layered_sources()
        missing = [name for name in LAYER_ORDER if name not in sources]
        self.assertEqual(missing, [], f"LAYER_ORDER names no source: {missing}")

    def test_every_module_with_a_layered_import_is_listed(self):
        """A new module cannot join the stack without taking a position."""
        sources = layered_sources()
        unlisted = sorted(
            name
            for name, path in sources.items()
            if name not in LAYER_ORDER and layered_imports(path)
        )
        self.assertEqual(
            unlisted,
            [],
            "these modules import the e2e stack and hold no position in "
            "LAYER_ORDER, so nothing asserts their direction:\n  "
            + "\n  ".join(unlisted),
        )

    def test_every_import_points_down_the_stack(self):
        sources = layered_sources()
        rank = {name: index for index, name in enumerate(LAYER_ORDER)}
        violations = []
        for name in LAYER_ORDER:
            for imported, lineno in layered_imports(sources[name]):
                if imported not in rank:
                    violations.append(
                        f"{name}:{lineno}: {imported} holds no position"
                    )
                elif rank[imported] >= rank[name]:
                    violations.append(
                        f"{name}:{lineno}: {imported} sits at or above it"
                    )
        self.assertEqual(
            violations,
            [],
            "an import points up the e2e stack; move the declaration down "
            "rather than the layer up:\n  " + "\n  ".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
