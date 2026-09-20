"""Structural tests that pin how the e2e modules carry their prose.

Three rules, all of them mechanical:

* A docstring or a comment names no file outside this repository. An
  upstream file moves between revisions, so a path to one is wrong as soon
  as a submodule is bumped. State the fact the file established instead.
* A dataclass field carries no ``#`` comment. The field documentation lives
  in an ``Attributes:`` section of the class docstring, where a reader and
  ``help()`` both find it.
* A module-level constant carries no ``#`` comment above it. Its
  documentation is a string literal directly below the assignment.

The rules cover ``benchmarks/{e2e,artifacts,traces,cli,execution}`` alone.
``benchmarks/cli/kernel.py`` is excluded with the rest of kernel-bench.
"""

import ast
import io
import re
import sys
import tokenize
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_import_boundaries import REPO_ROOT, tracked_python_files

IN_SCOPE_PREFIXES = (
    "benchmarks/e2e/",
    "benchmarks/artifacts/",
    "benchmarks/traces/",
    "benchmarks/cli/",
    "benchmarks/execution/",
)

EXCLUDED = frozenset({"benchmarks/cli/kernel.py"})

OUT_OF_TREE_REFERENCE = re.compile(
    r"third_party/|megatron/core|torchtitan/|reports/|AGENTS\.md|CLAUDE\.md"
    r"|distributed/compile\.py"
)


def in_scope_files() -> tuple[str, ...]:
    """Every tracked ``.py`` file these rules apply to."""
    return tuple(
        path
        for path in tracked_python_files()
        if path.startswith(IN_SCOPE_PREFIXES) and path not in EXCLUDED
    )


def _is_dataclass(node: ast.ClassDef) -> bool:
    for decorator in node.decorator_list:
        target = (
            decorator.func if isinstance(decorator, ast.Call) else decorator
        )
        name = (
            target.attr
            if isinstance(target, ast.Attribute)
            else getattr(target, "id", "")
        )
        if name == "dataclass":
            return True
    return False


def _comments(source: str) -> tuple[frozenset[int], frozenset[int]]:
    """The own-line comment lines and the trailing-comment lines."""
    lines = source.splitlines()
    own_line: set[int] = set()
    trailing: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        line, column = token.start
        if lines[line - 1][:column].strip():
            trailing.add(line)
        else:
            own_line.add(line)
    return frozenset(own_line), frozenset(trailing)


def _previous_nonblank(lines: list[str], line: int) -> int:
    index = line - 1
    while index >= 1 and not lines[index - 1].strip():
        index -= 1
    return index


class OutOfTreeReferenceTests(unittest.TestCase):
    def test_no_module_names_a_file_outside_this_repository(self) -> None:
        offenders = []
        for path in in_scope_files():
            lines = (REPO_ROOT / path).read_text().splitlines()
            for number, line in enumerate(lines, 1):
                if OUT_OF_TREE_REFERENCE.search(line):
                    offenders.append(f"{path}:{number}: {line.strip()}")
        self.assertEqual(
            offenders,
            [],
            "These lines name a file outside this repository. State the "
            "fact the file established instead:\n" + "\n".join(offenders),
        )

    def test_the_scope_is_not_empty(self) -> None:
        paths = in_scope_files()
        self.assertGreater(len(paths), 20)
        self.assertNotIn("benchmarks/cli/kernel.py", paths)


class FieldCommentTests(unittest.TestCase):
    def test_no_dataclass_field_carries_a_comment(self) -> None:
        offenders = []
        for path in in_scope_files():
            source = (REPO_ROOT / path).read_text()
            lines = source.splitlines()
            own_line, trailing = _comments(source)
            for node in ast.walk(ast.parse(source)):
                if not isinstance(node, ast.ClassDef):
                    continue
                if not _is_dataclass(node):
                    continue
                for statement in node.body:
                    if not isinstance(statement, (ast.AnnAssign, ast.Assign)):
                        continue
                    previous = _previous_nonblank(lines, statement.lineno)
                    if previous in own_line:
                        offenders.append(
                            f"{path}:{statement.lineno}: comment above the "
                            f"field {lines[statement.lineno - 1].strip()!r}"
                        )
                    for number in range(
                        statement.lineno, statement.end_lineno + 1
                    ):
                        if number in trailing:
                            offenders.append(
                                f"{path}:{number}: trailing comment on a "
                                "field"
                            )
        self.assertEqual(
            offenders,
            [],
            "Move these into an Attributes: section of the class "
            "docstring:\n" + "\n".join(offenders),
        )


class ConstantCommentTests(unittest.TestCase):
    def test_no_module_constant_carries_a_comment_above_it(self) -> None:
        offenders = []
        for path in in_scope_files():
            source = (REPO_ROOT / path).read_text()
            lines = source.splitlines()
            own_line, _ = _comments(source)
            for statement in ast.parse(source).body:
                if not isinstance(statement, ast.Assign):
                    continue
                names = [
                    target.id
                    for target in statement.targets
                    if isinstance(target, ast.Name)
                ]
                if not names:
                    continue
                if not all(re.fullmatch(r"_?[A-Z0-9_]+", n) for n in names):
                    continue
                previous = _previous_nonblank(lines, statement.lineno)
                if previous in own_line:
                    offenders.append(
                        f"{path}:{statement.lineno}: comment above "
                        f"{', '.join(names)}"
                    )
        self.assertEqual(
            offenders,
            [],
            "Move these below the assignment, as a string literal:\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
