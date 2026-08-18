"""Structural tests that pin this repository's import boundaries.

The harness is two processes with very different import budgets:

* the **parent** -- the Click CLI, the scenario and kernel registries, the
  artifact/trace/results packages, the model-shape registry -- which only
  ever builds command lines, validates logs, and reads traces; and
* the **worker** -- the TorchTitan training subprocess, the megatron driver,
  and ``benchmarks.kernel.worker``'s post-argument-parsing body -- which is
  where torch, TransformerEngine, Megatron, FlashAttention, Triton and
  torchtitan legitimately live.

The parent's import graph is entirely free of that ML stack, which is also the
reason it cannot initialize CUDA: ``./run_bench.sh scenarios``, ``--help``,
resume validation and evaluation all run in well under a second and never
touch the GPU. Nothing enforced that until this file. These tests are
deliberately structural (subprocess imports plus ``ast`` inspection of the
tracked sources) so a package move cannot quietly reintroduce a heavy import
or a non-canonical import root.

Nothing here imports
``benchmarks.models.piper_qwen3.components.rope.te_rope_override``: that
module JIT-builds a CUDA extension at import time.
"""

import ast
import functools
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


REPO_ROOT = Path(__file__).resolve().parent.parent
TITAN_DIR = REPO_ROOT / "third_party" / "torchtitan"

# The import roots intra-repo imports are allowed to target. The package
# restructure folded the two former top-level packages -- the Piper config
# port and the Megatron baseline -- into ``benchmarks``, so there is now
# exactly one library root. (Their retired names are spelled out in exactly
# two places, both deliberate: CLAUDE.md's provenance-boundary paragraph and
# tests/test_legacy_artifacts.py. This is not one of them.) This tuple
# shrinking to a single entry *is* the assertion that the move happened: a
# reintroduced top-level package would fail
# ``test_module_scope_intra_repo_imports_use_canonical_roots`` the moment
# anything imported it.
CANONICAL_ROOTS = ("benchmarks",)

# Importing any of these means the process paid for the ML stack -- and, for
# torch, that CUDA initialization is one attribute access away.
HEAVY_MODULES = (
    "torch",
    "transformer_engine",
    "megatron",
    "flash_attn",
    "flash_attn_interface",
    "torchtitan",
    "triton",
    "helion",
)

# Every module the parent process may reach. Measured, not assumed: each one
# is imported in a subprocess below and its ``sys.modules`` checked.
PARENT_SIDE_MODULES = (
    "benchmarks.cli.main",
    "benchmarks.e2e.registry",
    "benchmarks.e2e.launch",
    "benchmarks.e2e.runner",
    "benchmarks.e2e.results",
    "benchmarks.e2e.validation",
    "benchmarks.traces.schema",
    "benchmarks.traces.extraction",
    "benchmarks.artifacts.manifests",
    "benchmarks.artifacts.summaries",
    "benchmarks.execution.environment",
    "benchmarks.kernel.registry",
    "benchmarks.kernel.runner",
    "benchmarks.kernel.worker",
    "benchmarks.kernel.engine.statistics",
    "benchmarks.kernel.results.schema",
    "benchmarks.kernel.results.reporting",
    "benchmarks.models.piper_qwen3.shape",
    "benchmarks.models.piper_qwen3.megatron_bootstrap",
    # The ``--module`` chain. TorchTitan resolves ``--module
    # benchmarks.models.piper_qwen3`` inside the *training* subprocess, which
    # executes ``benchmarks/__init__.py``, ``benchmarks/models/__init__.py``
    # and this package's ``__init__.py`` before anything else. They are
    # docstring-only by design; listing the package here is what asserts it.
    "benchmarks.models.piper_qwen3",
)

# Modules that import the ML stack at module scope. This is correct and
# expected -- they run inside the worker -- so the boundary is asserted from
# the other side: no parent-side module may import them at module scope.
WORKER_SIDE_MODULES = (
    "benchmarks.models.piper_qwen3.config_registry",
    "benchmarks.models.piper_qwen3.parallelize",
    "benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu",
    "benchmarks.models.piper_qwen3.components.lm_head.losses",
    "benchmarks.models.piper_qwen3.components.lm_head.te_cross_entropy",
    "benchmarks.models.piper_qwen3.components.lm_head.te_common_cross_entropy",
    "benchmarks.models.piper_qwen3.components.lm_head.te_triton_cross_entropy",
    "benchmarks.models.piper_qwen3.components.lm_head.piper_optimized_cross_entropy",
    "benchmarks.models.piper_qwen3.components.rope.te_rope_override",
    "benchmarks.e2e.data.piper_qwen3",
    "benchmarks.e2e.megatron.data",
    "benchmarks.kernel.operations.arms",
    "benchmarks.kernel.engine.run",
)

# The third category, and the reason two lists were never enough. These run in
# a worker -- so the parent must not import them -- but they are ML-free at
# module scope, because each defers its heavy imports into a function body.
# That deferral is load-bearing rather than stylistic:
#
# * ``e2e.megatron.train`` is a ``python -m`` entry point, so ``--help`` must
#   not pay for torch, exactly as ``kernel.worker`` does not; and
# * ``models.piper_qwen3.megatron_model`` *cannot* import Megatron at module
#   scope, because ``megatron_bootstrap`` has to put Megatron on ``sys.path``
#   first. Hoisting its imports would not be a style regression, it would
#   break the module outright.
#
# They fail WORKER_SIDE_MODULES' justification test by construction (they
# import nothing heavy to find), which is what makes them a category and not
# an oversight. They get the same parent-must-not-import assertion, plus the
# dynamic ML-free probe that the parent-side modules get -- the deferral is
# the property worth locking.
WORKER_SIDE_DEFERRED_MODULES = (
    "benchmarks.e2e.megatron.train",
    "benchmarks.models.piper_qwen3.megatron_model",
)

# Every module the parent must not reach at module scope, whichever reason.
ALL_WORKER_SIDE_MODULES = WORKER_SIDE_MODULES + WORKER_SIDE_DEFERRED_MODULES


# --------------------------------------------------------------------------
# Subprocess import probe
# --------------------------------------------------------------------------

# Imports the requested modules one at a time in a fresh interpreter and
# attributes each newly-appearing forbidden module to the import that pulled
# it in. A subprocess is mandatory: the rest of this suite imports torch, so
# an in-process ``sys.modules`` check would always see it and assert nothing.
_IMPORT_PROBE = """
import importlib
import json
import sys

payload = json.loads(sys.argv[1])
forbidden = payload["forbidden"]
offenders = {}
seen = set()
for name in payload["modules"]:
    importlib.import_module(name)
    fresh = sorted(m for m in forbidden if m in sys.modules and m not in seen)
    if fresh:
        offenders[name] = fresh
        seen.update(fresh)
print(json.dumps({"offenders": offenders}))
"""


def _subprocess_env(**overrides: str) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep + pythonpath if pythonpath else ''}"
    env.update(overrides)
    return env


def run_import_probe(
    modules: tuple[str, ...], forbidden: tuple[str, ...] = HEAVY_MODULES
) -> dict[str, list[str]]:
    """Import ``modules`` in a fresh interpreter; return offender attribution."""
    payload = json.dumps({"modules": list(modules), "forbidden": list(forbidden)})
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE, payload],
        cwd=REPO_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "import probe failed for "
            f"{', '.join(modules)}\n--- stdout ---\n{completed.stdout}"
            f"\n--- stderr ---\n{completed.stderr}"
        )
    return json.loads(completed.stdout.splitlines()[-1])["offenders"]


# --------------------------------------------------------------------------
# Static source inspection
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def tracked_python_files() -> tuple[str, ...]:
    """Repo-relative paths of every in-repo ``.py`` file, via ``git ls-files``.

    Git is the enumerator on purpose: ``out/``, ``reports/`` and the stale
    worktree under ``.claude/`` all hold gitignored copies of these sources,
    and a filesystem walk would parse them as if they were the real thing.
    Submodule contents are gitlinks here, so ``third_party/`` contributes
    nothing -- which is also why ``import torchtitan`` is not intra-repo.

    ``--others --exclude-standard`` adds files that exist but are not staged
    yet, so a module written in this working tree is checked before it is
    committed rather than after; ``--exclude-standard`` keeps the gitignore
    protection above intact. The ``is_file`` filter drops the mirror case --
    a path git still has in its index whose file has been deleted from the
    working tree. That is a pending deletion, not an import to check.
    """
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return tuple(
        sorted(
            {
                p
                for p in completed.stdout.split("\0")
                if p and (REPO_ROOT / p).is_file()
            }
        )
    )


@functools.lru_cache(maxsize=1)
def repo_import_roots() -> frozenset[str]:
    """Top-level names that resolve to something inside this repository."""
    roots = set()
    for path in tracked_python_files():
        head, _, tail = path.partition("/")
        roots.add(head[: -len(".py")] if not tail else head)
    return frozenset(roots)


def own_root(path: str) -> str:
    """The import root a source file itself lives under."""
    head, _, tail = path.partition("/")
    return head[: -len(".py")] if not tail else head


def module_scope_imports(path: str) -> tuple[tuple[str, int], ...]:
    """Absolute dotted names imported at module scope by ``path``.

    Walks everything except function bodies, so a module-level ``try:`` or
    ``if TYPE_CHECKING:`` block still counts (it runs at import time), while
    the sanctioned lazy pattern -- an import inside a function -- does not.
    Relative imports are resolved against the file's own package.
    """
    tree = ast.parse((REPO_ROOT / path).read_text(), filename=path)
    package = path[: -len(".py")].replace("/", ".")
    package = package.rsplit(".", 1)[0] if "." in package else ""
    if package.endswith(".__init__"):
        package = package[: -len(".__init__")]

    found: list[tuple[str, int]] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(child, ast.Import):
                found.extend((alias.name, child.lineno) for alias in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.level:
                    parts = package.split(".") if package else []
                    base = ".".join(parts[: len(parts) - (child.level - 1)])
                    module = f"{base}.{child.module}" if child.module else base
                else:
                    module = child.module or ""
                if module:
                    found.append((module, child.lineno))
            visit(child)

    visit(tree)
    return tuple(found)


def source_path(module: str) -> str:
    """Repo-relative source file for a dotted module name.

    Resolves a package to its ``__init__.py``: after the restructure the
    lists above name packages as well as modules (``benchmarks.models.
    piper_qwen3`` is the ``--module`` token), and a package's ``__init__``
    is exactly the file whose import cost is being asserted.
    """
    stem = module.replace(".", "/")
    for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
        if (REPO_ROOT / candidate).is_file():
            return candidate
    raise AssertionError(f"no source file for module {module!r} at {stem}[.py]")


def targets(imported: str, module: str) -> bool:
    """True when ``imported`` is ``module`` or one of its submodules."""
    return imported == module or imported.startswith(module + ".")


# --------------------------------------------------------------------------
# 1. The parent import graph is Torch-free
# --------------------------------------------------------------------------


class TorchFreeParentGraphTest(unittest.TestCase):
    """The parent process must never pay for -- or link against -- the ML stack."""

    def test_probe_attributes_transitive_imports_to_the_right_module(self):
        """Negative control: the probe must actually catch what it looks for.

        Uses stdlib stand-ins so the control costs nothing. ``csv`` pulls in
        the C module ``_csv``, so a passing result proves the probe sees both
        direct and transitive imports and blames the right module for them.
        """
        offenders = run_import_probe(("json", "csv"), forbidden=("json", "_csv"))
        self.assertEqual(offenders, {"json": ["json"], "csv": ["_csv"]})

    def test_parent_modules_import_without_the_ml_stack(self):
        """Every parent-side module imports without torch et al. appearing.

        One subprocess imports the whole list in order and reports which
        import first introduced each forbidden module, so a regression names
        the culprit rather than just the set.
        """
        offenders = run_import_probe(PARENT_SIDE_MODULES)
        self.assertEqual(
            offenders,
            {},
            "parent-side modules pulled in the ML stack: "
            + "; ".join(f"{k} -> {', '.join(v)}" for k, v in sorted(offenders.items())),
        )

    def test_parent_side_modules_all_exist(self):
        for module in PARENT_SIDE_MODULES:
            with self.subTest(module=module):
                source_path(module)


# --------------------------------------------------------------------------
# 2. Worker-side modules are correctly classified
# --------------------------------------------------------------------------


class WorkerSideBoundaryTest(unittest.TestCase):
    """The heavy modules are heavy, and the parent does not reach them."""

    def test_worker_side_modules_import_the_ml_stack_at_module_scope(self):
        """Justifies each module's place on the worker side of the boundary.

        Static, not dynamic: importing these would cost the suite a torch
        import per module for a fact the source already states.
        """
        for module in WORKER_SIDE_MODULES:
            with self.subTest(module=module):
                roots = {
                    imported.split(".")[0]
                    for imported, _ in module_scope_imports(source_path(module))
                }
                self.assertTrue(
                    roots & set(HEAVY_MODULES),
                    f"{module} is listed as worker-side but imports none of "
                    f"{HEAVY_MODULES} at module scope -- it may belong on the "
                    "parent side now",
                )

    def test_deferred_worker_modules_stay_ml_free_at_import(self):
        """The deferral in WORKER_SIDE_DEFERRED_MODULES actually holds.

        Dynamic, unlike the static check above, because that is the whole
        claim: these have no module-scope heavy import to inspect, so the
        only way to know the imports really are deferred is to import them
        and look. For ``megatron_model`` this is also a correctness gate --
        Megatron is not importable until ``megatron_bootstrap`` has placed it
        on ``sys.path``, so a hoisted import would break the module, not just
        slow it down.
        """
        offenders = run_import_probe(WORKER_SIDE_DEFERRED_MODULES)
        self.assertEqual(
            offenders,
            {},
            "deferred worker modules pulled in the ML stack at import: "
            + "; ".join(f"{k} -> {', '.join(v)}" for k, v in sorted(offenders.items())),
        )

    def test_parent_side_modules_do_not_import_worker_side_modules(self):
        """The boundary itself: no module-scope path from parent to worker.

        Function-body imports are the sanctioned lazy pattern (see
        ``benchmarks/kernel/worker.py``'s ``main``) and are not flagged.
        """
        for module in PARENT_SIDE_MODULES:
            path = source_path(module)
            for imported, lineno in module_scope_imports(path):
                for worker in ALL_WORKER_SIDE_MODULES:
                    with self.subTest(module=module, imported=imported):
                        self.assertFalse(
                            targets(imported, worker),
                            f"{path}:{lineno} imports worker-side {imported!r} at "
                            "module scope; move it inside the function that "
                            "needs it",
                        )


def tracked_benchmarks_modules() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Every module under ``benchmarks/``, split into leaves and packages.

    Derived from ``git ls-files`` rather than listed, because the point is to
    catch the module nobody remembered to classify.
    """
    leaves, packages = [], []
    for path in tracked_python_files():
        parts = Path(path).with_suffix("").parts
        if parts[0] != "benchmarks":
            continue
        if parts[-1] == "__main__":
            continue  # executed, never imported by name
        if parts[-1] == "__init__":
            packages.append(".".join(parts[:-1]))
        else:
            leaves.append(".".join(parts))
    return tuple(sorted(leaves)), tuple(sorted(packages))


class ClassificationCompletenessTest(unittest.TestCase):
    """The lists above describe the whole tree, not a remembered subset.

    Without this, every boundary assertion is silently scoped to whatever
    was listed: a new module -- or one that a package move renamed -- simply
    goes unchecked, and the suite still reports green. The restructure
    produced exactly that gap, which is why this exists.
    """

    def test_every_module_is_classified_exactly_once(self):
        leaves, _ = tracked_benchmarks_modules()
        classified = list(PARENT_SIDE_MODULES) + list(ALL_WORKER_SIDE_MODULES)
        # Packages may be listed (benchmarks.models.piper_qwen3 is the
        # --module token); leaves must be.
        unclassified = [m for m in leaves if m not in set(classified)]
        self.assertEqual(
            unclassified,
            [],
            "modules under benchmarks/ belong to no import-boundary class, so "
            "nothing asserts where they may be imported from:\n  "
            + "\n  ".join(unclassified),
        )
        duplicated = sorted({m for m in classified if classified.count(m) > 1})
        self.assertEqual(
            duplicated,
            [],
            f"modules classified more than once: {duplicated}",
        )

    def test_package_inits_are_docstring_only(self):
        """``__init__.py`` files declare ownership and import nothing.

        A package whose ``__init__`` imports its own submodules would drag
        the ML stack in through any import of the package -- and the
        ``--module benchmarks.models.piper_qwen3`` chain executes three of
        these before torchtitan reads a single config.
        """
        _, packages = tracked_benchmarks_modules()
        offenders = []
        for package in packages:
            path = source_path(package)
            for imported, lineno in module_scope_imports(path):
                offenders.append(f"{path}:{lineno}: {imported}")
        self.assertEqual(
            offenders,
            [],
            "package __init__ files must stay docstring-only:\n  "
            + "\n  ".join(offenders),
        )


# --------------------------------------------------------------------------
# 3. Canonical import roots
# --------------------------------------------------------------------------


class CanonicalImportRootsTest(unittest.TestCase):
    """Intra-repo imports go through the declared roots and nowhere else."""

    def test_canonical_roots_exist_in_the_repository(self):
        self.assertTrue(
            set(CANONICAL_ROOTS) <= repo_import_roots(),
            f"CANONICAL_ROOTS {CANONICAL_ROOTS} names something absent from the "
            f"repo; tracked roots are {sorted(repo_import_roots())}",
        )

    def test_module_scope_intra_repo_imports_use_canonical_roots(self):
        """Every intra-repo import targets ``CANONICAL_ROOTS``.

        A file may additionally import its own root, which is what lets the
        ``tools/`` scripts and this test package refer to their own siblings
        without being library code.
        """
        violations = []
        for path in tracked_python_files():
            allowed = set(CANONICAL_ROOTS) | {own_root(path)}
            for imported, lineno in module_scope_imports(path):
                root = imported.split(".")[0]
                if root in repo_import_roots() and root not in allowed:
                    violations.append(f"{path}:{lineno}: {imported}")
        self.assertEqual(
            violations,
            [],
            "intra-repo imports outside CANONICAL_ROOTS "
            f"{CANONICAL_ROOTS}:\n" + "\n".join(violations),
        )


# --------------------------------------------------------------------------
# 4. The ``benchmarks`` name-shadowing hazard
# --------------------------------------------------------------------------


@unittest.skipUnless(TITAN_DIR.is_dir(), "torchtitan submodule not checked out")
class BenchmarksNameShadowingTest(unittest.TestCase):
    """Guards a name collision that is invisible until it silently isn't.

    The **training** subprocess -- and only that one -- is exposed. It runs
    with ``cwd=third_party/torchtitan`` (``benchmarks/e2e/runner.py``, the
    ``process_runner(..., cwd=paths.titan_dir)`` call) and
    ``PYTHONPATH=<repo root>`` (``benchmarks/execution/environment.py``,
    ``runtime_environment``). ``python -m`` puts the cwd at ``sys.path[0]``,
    *ahead* of everything ``PYTHONPATH`` contributes -- and the torchtitan
    submodule ships a ``benchmarks/`` directory of its own. The kernel
    worker is *not* exposed: it runs with ``cwd=paths.bench_dir``
    (``benchmarks/kernel/runner.py``), so the repo root is already
    ``sys.path[0]`` there and our package wins outright. Do not widen this
    docstring to claim otherwise.

    The exposure is new, and the restructure created it. The training
    subprocess used to resolve ``--module piper1b`` -- a name with no
    collision. It now resolves ``--module benchmarks.models.piper_qwen3``,
    and ``benchmarks`` is precisely the colliding name, so a shadowed import
    would no longer cost one module: it would cost the whole library.

    Our package wins today for exactly one reason: that directory holds no
    ``__init__.py``, so it registers only as a namespace portion, and a
    regular package found later on the path beats it. Should a submodule bump
    ever add an ``__init__.py`` there, ``import benchmarks`` inside every
    training subprocess would resolve to torchtitan's directory instead of
    ours, and the failure would surface as an unrelated-looking
    ``AttributeError`` or ``ModuleNotFoundError`` deep inside a run. This is
    the regression test CLAUDE.md's submodule-bump checklist refers to; do
    not delete it to make a bump green.
    """

    def test_torchtitan_benchmarks_directory_is_not_a_package(self):
        shadow = TITAN_DIR / "benchmarks"
        if not shadow.is_dir():
            self.skipTest("torchtitan no longer ships a benchmarks/ directory")
        self.assertFalse(
            (shadow / "__init__.py").exists(),
            f"{shadow} has become a regular package and now shadows this "
            "repo's 'benchmarks' inside every training subprocess (cwd is "
            "sys.path[0], ahead of PYTHONPATH)",
        )

    def test_our_benchmarks_package_wins_the_training_subprocess_sys_path(self):
        """Reproduce the training subprocess's sys.path and resolve the name."""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import benchmarks, json, sys; "
                "print(json.dumps({'path0': sys.path[0], "
                "'file': benchmarks.__file__, "
                "'origin': benchmarks.__spec__.origin}))",
            ],
            cwd=TITAN_DIR,
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"importing 'benchmarks' from {TITAN_DIR} failed:\n{completed.stderr}",
        )
        info = json.loads(completed.stdout.splitlines()[-1])

        self.assertIsNotNone(
            info["origin"],
            "'benchmarks' resolved to a namespace package, not this repo's "
            f"package: {info}",
        )
        resolved = Path(info["file"]).resolve()
        self.assertEqual(
            resolved.parent.parent,
            REPO_ROOT,
            f"'benchmarks' resolved outside this repository: {resolved}",
        )


# --------------------------------------------------------------------------
# 5. ``benchmarks.kernel.worker`` stays cheap to import
# --------------------------------------------------------------------------


class KernelWorkerImportCostTest(unittest.TestCase):
    """The kernel worker's heavy imports live inside ``main()``, deliberately.

    Its module scope holds only stdlib, so ``python -m
    benchmarks.kernel.worker --help`` and an argument error (exit 2) both
    return without paying for torch. Assertion 1 covers this incidentally by
    including the module in the parent-side list; it is restated here so the
    intent survives if that list is ever rearranged.
    """

    def test_kernel_worker_imports_without_the_ml_stack(self):
        offenders = run_import_probe(("benchmarks.kernel.worker",))
        self.assertEqual(
            offenders,
            {},
            "benchmarks.kernel.worker pulled in "
            f"{offenders.get('benchmarks.kernel.worker')} at import time; its "
            "heavy imports belong inside main()",
        )

    def test_kernel_worker_module_scope_is_stdlib_only(self):
        path = source_path("benchmarks.kernel.worker")
        roots = {imported.split(".")[0] for imported, _ in module_scope_imports(path)}
        self.assertEqual(
            roots - {"__future__"},
            {"argparse", "sys", "traceback", "pathlib"},
            f"{path} grew a module-scope import outside the stdlib argument-"
            "parsing set; keep it inside main()",
        )


if __name__ == "__main__":
    unittest.main()
