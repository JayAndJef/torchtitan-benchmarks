"""Audit: no retired module path survives anywhere in the tracked tree.

The ``benchmarks`` restructure folded three top-level packages into one and
split six flat modules into subpackages. Almost every reference it had to
update is a *string* -- an import line, a ``mock.patch`` target, a ``-m``
subprocess argument, a builder path handed to ``importlib``. None of those
are type-checked and most fail only at run time, so a single missed
reference can survive a green test suite and die 40 minutes into a GPU run.

This file is the sweep that says "and nothing else is left".

Two design rules, both learned the hard way, both load-bearing:

1. **Enumerate from ``git ls-files``, never from a filesystem walk.**
   ``out/`` holds ~200 historical run directories, ``reports/`` holds notes,
   and ``.claude/worktrees/`` can hold a whole stale copy of the repo. All
   three are gitignored, all three legitimately contain the retired strings,
   and a ``grep -r`` or ``Path.rglob`` would report every one of them as a
   violation. Git is the only enumerator that sees the tree this repo
   actually ships.

2. **Key on dotted-path and path forms, never on bare tokens.** Several
   retired *names* survive as frozen identifiers that must never change:

   ===========================  ==========================================
   still legal                  because
   ===========================  ==========================================
   ``piper1b_rope`` et al.      the six scenario ids; they name directories
                                under ``out/`` and keys in every manifest
   ``qwen3_piper_1b*``          the ``--config`` names
   ``parallelize_piper1b``      a function name
   ``model_shape``              a manifest JSON key (``model_size`` /
                                ``model_shape``), named in the kernel
                                schema 1->2 history comments -- rewriting
                                those would falsify the schema record
   ``kernel_bench.log``         the kernel run's stdout artifact filename,
                                documented in CLAUDE.md's output layout
   ``scenarios``, ``kernels``   domain nouns ("benchmark scenarios", "GPU
                                kernels") far more often than they are the
                                retired ``benchmarks/scenarios.py`` and
                                ``benchmarks/kernels.py``
   ===========================  ==========================================

   Only the dotted and path forms are retired: ``piper1b.`` and ``piper1b/``,
   not ``piper1b``; ``benchmarks.kernel_bench`` and
   ``benchmarks/kernel_bench.py``, not ``kernel_bench``. A bare-token rule
   would go red on all six rows above, and the cheapest way to green it
   would be to edit the schema comments, rename the log file, or reword the
   docs -- i.e. to damage real data to satisfy an audit. That failure mode
   is exactly what §I.4 of the rename map exists to prevent, and it applies
   here for the same reason.

Markdown *is* in scope, with one scoped exemption. Documentation is where a
stale path is most likely to survive unnoticed, so the sweep reads it. The
exception is AGENTS.md's "Provenance boundary" section, which legitimately
names the retired paths as *history* -- it is what tells a future reader why
~197 manifests under ``out/`` still decode. Every retired name in that file
was deliberately consolidated into that one section so a single scoped
exclusion suffices, and the exclusion keys on the *heading*, not the
filename: a stale path introduced anywhere else in AGENTS.md still fails.
"""

import ast
import hashlib
import re
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Section-scoped exemptions.
# --------------------------------------------------------------------------

# Preferred over a whole-file allowlist wherever a file has exactly one
# legitimate region: the rest of the file stays guarded.
#
# AGENTS.md's "Provenance boundary" section states, and must keep stating,
# that ~197 manifests under out/ record `--module piper1b`,
# `benchmarks.kernel_arms:<builder>` and `python -m megatron_baseline.train`
# as inert data. Naming those strings is the section's entire job. Run
# without this exemption the audit goes red on it, and the cheapest way to
# green it would be to delete the explanation -- the same evidence-destroying
# fix §I.4 of the rename map forbids for the JSON fixtures, for the same
# reason. Delete this entry only if the section itself goes away.
SECTION_EXEMPTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "AGENTS.md",
        "### Provenance boundary: artifacts written before the restructure",
        "documents the retired names as history so pre-flag-day artifacts "
        "stay interpretable; deleting the hit deletes the explanation",
    ),
)

# Markdown headings at this level or above end an exempt section.
_HEADING = re.compile(r"^#{1,6} ")


def _mask_exempt_sections(path: str, text: str) -> str:
    """Blank the exempt sections of ``path``, preserving line numbering.

    Blanking rather than slicing keeps reported line numbers true, so a
    violation elsewhere in the file still points at its real line.
    """
    headings = [h for file_path, h, _ in SECTION_EXEMPTIONS if file_path == path]
    if not headings:
        return text
    lines = text.splitlines(keepends=True)
    masked, skipping = [], False
    for line in lines:
        if line.rstrip("\n") in headings:
            skipping = True
        elif skipping and _HEADING.match(line):
            skipping = False
        masked.append("\n" if skipping and line.endswith("\n") else
                      ("" if skipping else line))
    return "".join(masked)


# --------------------------------------------------------------------------
# The allowlist: three entries, three different reasons.
# --------------------------------------------------------------------------

# Each is exempt because it is *supposed* to contain retired paths, and in
# each case the obvious way to make this test green would destroy the thing
# the exemption protects. The reasons are recorded here, in the guard
# itself, rather than applied ad hoc at a grep command line, so that whoever
# next sees this test go red does not "fix" it by editing the evidence.
#
# See §I.4 of the frozen rename map (/tmp/rename-map.md), which states the
# directive in the imperative: do not edit, reformat, re-indent, re-key,
# pretty-print, lint, or search-and-replace anything under
# ``tests/fixtures/legacy/``.
ALLOWLIST: tuple[tuple[str, str], ...] = (
    (
        "tests/fixtures/legacy/",
        # Byte-for-byte recordings of runs that actually happened, checked in
        # because out/ is gitignored. The retired strings they contain are
        # the payload under test: tests/test_legacy_artifacts.py proves an
        # artifact written before the restructure still decodes as inert
        # data. "Correcting" them to the new paths inverts that test into a
        # tautology -- it would then assert that the *current* modules are
        # not imported, which proves nothing about old artifacts -- and the
        # loss is unrecoverable, because the code that produced them no
        # longer exists. See §I.4.
        "recorded pre-restructure artifacts; the retired strings are the "
        "payload under test and the files are byte-frozen (§I.4)",
    ),
    (
        "tests/test_legacy_artifacts.py",
        # Its RECORDED_MODULE_PATHS constant names the retired paths in
        # source on purpose: the test asserts that reading an artifact never
        # imports them. They are pre-move paths by design, and rewriting
        # them would make the assertion vacuous. See §I.4.
        "RECORDED_MODULE_PATHS names the retired paths in source, "
        "deliberately (§I.4)",
    ),
    (
        "tests/test_retired_paths.py",
        # This file. A sweep must spell out what it forbids: the docstring
        # tabulates every retired form against its still-legal twin, and
        # test_the_patterns_actually_match_the_retired_forms is a negative
        # control built entirely from retired strings. Without it, a rule
        # that silently stopped matching anything would pass.
        #
        # The exemption is bounded rather than trusted:
        # test_the_audit_itself_imports_nothing_first_party below proves this
        # module has no first-party imports at all, so there is no stale
        # import here for the sweep to have caught.
        "the audit spells out the forms it forbids, including a negative "
        "control built from them; bounded by the no-first-party-import test",
    ),
)


# --------------------------------------------------------------------------
# Retired forms.
# --------------------------------------------------------------------------

# The six flat modules the restructure split, plus the seven kernel modules
# it repackaged. None of these names is now a module or a package, so the
# dotted form is unambiguous. ``benchmarks.artifacts`` and ``benchmarks.cli``
# are deliberately absent: both are now real packages and the bare dotted
# prefix is legal. Their retired *symbol* imports are caught by AST instead,
# in PackageSymbolImportTests below.
RETIRED_BENCHMARKS_MODULES = (
    "scenarios",
    "runner",
    "runtime",
    "metrics",
    "reporting",
    "profile_regions",
    "kernels",
    "kernel_arms",
    "kernel_bench",
    "kernel_runner",
    "kernel_stats",
    "kernel_results",
    "kernel_worker",
)

# Packages that replaced a module of the same name. ``benchmarks.artifacts``
# used to be artifacts.py and is now artifacts/; likewise cli.py -> cli/. A
# stale ``from benchmarks.artifacts import trace_files`` is therefore a
# *symbol* error, not a module error: the dotted prefix still resolves, so no
# regex can tell it from the legal ``benchmarks.artifacts.layout``.
MODULES_THAT_BECAME_PACKAGES = ("benchmarks.artifacts", "benchmarks.cli")

# ``(?<![A-Za-z0-9_])`` is the left word boundary and the trailing ``[./]``
# is the right one. Together they are what distinguishes the retired dotted
# and path forms from the frozen identifiers tabulated in the docstring.
_LEFT = r"(?<![A-Za-z0-9_])"
_RIGHT = r"(?![A-Za-z0-9_])"

RETIRED_FORMS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "piper1b module or directory",
        # Requires a trailing "." or "/", so piper1b_rope (scenario id),
        # qwen3_piper_1b (config name) and parallelize_piper1b (function)
        # are all untouched. Nothing legal spells "piper1b." or "piper1b/".
        re.compile(_LEFT + r"piper1b[./]"),
        "benchmarks/models/piper_qwen3/ (or benchmarks/e2e/data/piper_qwen3)",
    ),
    (
        "megatron_baseline package",
        # The one place a bare token is safe: megatron_baseline is a
        # snake_case package name with no frozen-identifier twin anywhere in
        # the repo. Prose says "the Megatron baseline", with a space.
        re.compile(_LEFT + r"megatron_baseline" + _RIGHT),
        "benchmarks/e2e/megatron/ and benchmarks/models/piper_qwen3/",
    ),
    (
        "analysis/ scripts",
        # analysis/components.py was removed outright (CLAUDE.md explains
        # why); analyze.py and per_block.py moved to tools/. Pinned to the
        # three filenames rather than to the directory, because "analysis"
        # on its own is a domain noun.
        re.compile(_LEFT + r"analysis[./](?:components|analyze|per_block)" + _RIGHT),
        "tools/analyze.py and tools/per_block.py (components.py was removed)",
    ),
    (
        "retired benchmarks module",
        re.compile(
            _LEFT
            + r"benchmarks[./](?:"
            + "|".join(RETIRED_BENCHMARKS_MODULES)
            + r")"
            + _RIGHT
        ),
        "the benchmarks.{e2e,kernel,traces,artifacts,execution} subpackages",
    ),
)


# --------------------------------------------------------------------------
# Enumeration.
# --------------------------------------------------------------------------


def tracked_files() -> tuple[str, ...]:
    """Every path git would ship. See design rule 1 in the docstring.

    ``--cached --others --exclude-standard`` is tracked files *plus*
    untracked ones that are not ignored. The ``--others`` half matters: a
    file that has been written but not yet ``git add``ed is exactly the file
    most likely to carry a stale path, and a ``--cached``-only sweep would
    declare the tree clean until the moment it was staged. ``.gitignore``
    still applies, so ``out/``, ``reports/``, ``.venv/`` and ``.claude/``
    stay out -- which is the whole point of enumerating with git.
    """
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return tuple(sorted({p for p in completed.stdout.split("\0") if p}))


def is_allowlisted(path: str) -> bool:
    return any(path.startswith(prefix) for prefix, _ in ALLOWLIST)


def cached_files() -> frozenset[str]:
    """Only the paths git actually tracks (no ``--others``)."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return frozenset(p for p in completed.stdout.split("\0") if p)


def is_draft(path: str, cached: frozenset[str]) -> bool:
    """An untracked Markdown file is a draft, and drafts are out of scope.

    The sweep covers untracked *code* on purpose (see ``tracked_files``): a
    module written but not yet staged is where a stale path is most likely
    to be hiding. Untracked *prose* is the opposite -- a planning document
    that quotes the pre-restructure layout because describing it is the
    document's job. ``BENCHMARK_IMPROVEMENT_SPEC.md`` is the live example,
    and §G.6 of the rename map flags it as untracked and therefore outside
    the map's authority. Scoping by "untracked + Markdown" keeps that out
    without an allowlist entry that would rot the moment the draft is
    deleted or committed.
    """
    return path.endswith(".md") and path not in cached


def scanned_files() -> tuple[str, ...]:
    """Every path the sweep reads, and each file exactly once.

    git lists a symlink as a path of its own, so an alias of a file already
    in the candidate set would be read twice. ``CLAUDE.md`` is the live
    case: it points at ``AGENTS.md``, and Claude Code reads only the former
    name while the repository documents the latter. A second read reports
    every hit twice, and the second copy arrives under a path no section
    exemption keys on -- so the "Provenance boundary" section would fail
    the audit through its alias while passing under its own name.

    A symlink whose target is *not* a candidate is still read. Nothing else
    here delivers that content, so dropping it would lose coverage rather
    than remove a duplicate.
    """
    cached = cached_files()
    candidates = tuple(
        path
        for path in tracked_files()
        # third_party/ entries are gitlinks: git lists the submodule path
        # itself and none of its contents, so there is nothing to read.
        if not path.startswith("third_party/")
        and not is_allowlisted(path)
        and not is_draft(path, cached)
        and (REPO_ROOT / path).is_file()
    )
    # Real files only: a symlink must not put its own target in the set it
    # is tested against, or every symlink would look like a duplicate.
    real = {
        (REPO_ROOT / path).resolve()
        for path in candidates
        if not (REPO_ROOT / path).is_symlink()
    }
    return tuple(
        path
        for path in candidates
        if not (REPO_ROOT / path).is_symlink()
        or (REPO_ROOT / path).resolve() not in real
    )


def violations_in(path: str) -> list[str]:
    text = _mask_exempt_sections(path, (REPO_ROOT / path).read_text(errors="ignore"))
    lines = text.splitlines()
    found = []
    for label, pattern, replacement in RETIRED_FORMS:
        for match in pattern.finditer(text):
            lineno = text[: match.start()].count("\n") + 1
            source = lines[lineno - 1].strip() if lineno <= len(lines) else ""
            found.append(
                f"{path}:{lineno}: {label} {match.group(0)!r} -> use "
                f"{replacement}\n    {source[:120]}"
            )
    return found


class RetiredPathAuditTests(unittest.TestCase):
    def test_git_is_the_enumerator_not_the_filesystem(self) -> None:
        """The gitignored trees must not be in the candidate set.

        This is the assertion behind design rule 1. ``out/`` alone holds
        roughly 200 manifests recording the retired paths; if any of it
        reached the scan, the audit below would be unsatisfiable except by
        deleting evidence.

        ``.claude/`` is deliberately absent from the refused prefixes.
        ``.gitignore`` ignores ``/.claude/*`` and then un-ignores
        ``/.claude/skills/``, so the skills are tracked first-party
        documentation and the audit must read them. Every other part of
        ``.claude/`` stays ignored, and ``git ls-files`` cannot return an
        ignored path, so git remains the enumerator either way.
        """
        scanned = scanned_files()
        self.assertTrue(scanned, "git ls-files returned nothing to scan")
        for path in scanned:
            self.assertFalse(
                path.startswith(("out/", "reports/", ".venv/")),
                f"{path} is gitignored output and must not be scanned",
            )
        # Sanity: the scan really does reach real source, config and docs.
        for anchor in (
            "benchmarks/__init__.py",
            "tests/test_runner.py",
            "run_bench.sh",
            "pyproject.toml",
            "AGENTS.md",
            ".claude/skills/piper-comparison/SKILL.md",
        ):
            self.assertIn(anchor, scanned)

    def test_an_alias_is_not_scanned_beside_its_target(self) -> None:
        """One file, one read, however many names point at it.

        ``CLAUDE.md`` is a symlink to ``AGENTS.md`` and git tracks both, so
        the enumerator returns both. See ``scanned_files`` for why reading
        the second one would fail the audit on an exempt section.
        """
        scanned = scanned_files()
        real = {
            (REPO_ROOT / path).resolve()
            for path in scanned
            if not (REPO_ROOT / path).is_symlink()
        }
        for path in scanned:
            candidate = REPO_ROOT / path
            if not candidate.is_symlink():
                continue
            with self.subTest(path=path):
                self.assertNotIn(
                    candidate.resolve(),
                    real,
                    f"{path} aliases a file the sweep already reads under "
                    "its own name",
                )

    def test_no_retired_dotted_or_path_form_survives(self) -> None:
        found: list[str] = []
        for path in scanned_files():
            found.extend(violations_in(path))
        self.assertEqual(
            found,
            [],
            "retired module paths survive in tracked sources:\n"
            + "\n".join(found),
        )

    def test_the_frozen_identifiers_are_not_flagged(self) -> None:
        """Regression for design rule 2, one case per docstring table row.

        If someone widens a pattern to a bare token, this fails before the
        sweep above starts reporting the frozen identifiers as violations.
        """
        legal = (
            'scenario_by_name("piper1b_rope")',
            'SCENARIOS["piper1b_megatron"]',
            "qwen3_piper_1b_piper_optimized_te_ce_pretokenized",
            "from x import parallelize_piper1b",
            'manifest["model_shape"], manifest["model_size"]',
            "kernel_bench.log",
            "the five titan scenarios share PIPER_1B_REGIONS",
            "competing kernels are timed head-to-head",
            "benchmarks/models/piper_qwen3/shape.py",
            "benchmarks.kernel.registry, benchmarks.kernel.runner",
            "benchmarks.artifacts.manifests, benchmarks.cli.main",
            "python -m benchmarks.cli",
        )
        for line in legal:
            for label, pattern, _ in RETIRED_FORMS:
                with self.subTest(line=line, rule=label):
                    self.assertIsNone(
                        pattern.search(line),
                        f"{label} flagged the frozen identifier {line!r}",
                    )

    def test_the_patterns_actually_match_the_retired_forms(self) -> None:
        """Negative control: an audit that matches nothing proves nothing."""
        retired = (
            "from piper1b.model_shape import NORMAL",
            "piper1b/lm_head/losses.py",
            'python -m megatron_baseline.train',
            "from megatron_baseline.data import thd_batches",
            "analysis/components.py attributed per-component GPU time",
            "from benchmarks.scenarios import SCENARIOS",
            "from benchmarks.runner import execute_run",
            "benchmarks/profile_regions.py",
            '"benchmarks.kernel_arms:build_qkv_fused"',
            "mock.patch('benchmarks.runtime.run_text')",
        )
        for line in retired:
            with self.subTest(line=line):
                self.assertTrue(
                    any(pattern.search(line) for _, pattern, _ in RETIRED_FORMS),
                    f"no rule caught the retired form {line!r}",
                )


class SectionExemptionTests(unittest.TestCase):
    """The section exclusion must be narrow, live, and still needed."""

    def test_every_exempt_section_exists_verbatim(self) -> None:
        for path, heading, reason in SECTION_EXEMPTIONS:
            with self.subTest(path=path, heading=heading):
                self.assertTrue(reason.strip())
                lines = (REPO_ROOT / path).read_text().splitlines()
                self.assertEqual(
                    lines.count(heading),
                    1,
                    f"{path} does not contain exactly one {heading!r}; the "
                    "exemption is keyed on that heading and is now either "
                    "dead or ambiguous",
                )

    def test_the_exemption_is_still_carrying_its_weight(self) -> None:
        """Without it the file is dirty; with it, clean.

        Both halves matter. If the unmasked scan were clean the exemption
        would be dead weight hiding future regressions; if the masked scan
        were dirty the retired paths have escaped the one section that is
        allowed to name them.
        """
        for path, _, _ in SECTION_EXEMPTIONS:
            raw = (REPO_ROOT / path).read_text(errors="ignore")
            with self.subTest(path=path):
                self.assertTrue(
                    any(p.search(raw) for _, p, _ in RETIRED_FORMS),
                    f"{path} no longer names any retired path; drop its "
                    "section exemption rather than leaving it dead",
                )
                self.assertEqual(violations_in(path), [])

    def test_masking_does_not_shift_line_numbers(self) -> None:
        for path, _, _ in SECTION_EXEMPTIONS:
            raw = (REPO_ROOT / path).read_text(errors="ignore")
            with self.subTest(path=path):
                self.assertEqual(
                    _mask_exempt_sections(path, raw).count("\n"),
                    raw.count("\n"),
                )

    def test_masking_stops_at_the_next_heading(self) -> None:
        """A section exemption must not swallow the rest of the file."""
        sample = (
            "## Keep\n"
            "### Provenance boundary: artifacts written before the restructure\n"
            "records `python -m megatron_baseline.train`\n"
            "## After\n"
            "from benchmarks.scenarios import SCENARIOS\n"
        )
        masked = _mask_exempt_sections("AGENTS.md", sample)
        self.assertNotIn("megatron_baseline", masked)
        self.assertIn("benchmarks.scenarios", masked)
        self.assertIn("## Keep", masked)


class AllowlistTests(unittest.TestCase):
    """The allowlist must stay honest: exempt, present, and still guilty."""

    def test_every_allowlisted_path_exists_and_is_tracked(self) -> None:
        tracked = tracked_files()
        for prefix, reason in ALLOWLIST:
            with self.subTest(path=prefix):
                self.assertTrue(reason.strip(), f"{prefix} has no reason")
                self.assertTrue(
                    any(path.startswith(prefix) for path in tracked),
                    f"{prefix} is allowlisted but git tracks nothing there; "
                    "delete the entry rather than leaving a dead exemption",
                )

    def test_the_audit_itself_imports_nothing_first_party(self) -> None:
        """Bounds this file's own exemption.

        Allowlisting a file normally costs coverage. It costs none here,
        because this module imports only the standard library -- so there is
        no first-party import inside it that could have gone stale. If that
        ever changes, the exemption must be narrowed instead of widened.
        """
        tree = ast.parse(Path(__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(
            imported & {"benchmarks", "tests", "tools"},
            set(),
            "the retired-path audit grew a first-party import; its "
            "self-exemption no longer costs nothing",
        )

    def test_every_allowlisted_path_still_needs_its_exemption(self) -> None:
        """A dead exemption is worse than none: it hides the next regression.

        If one of these stops containing a retired path, the entry has
        outlived its purpose -- or, far more likely, someone has just
        rewritten the evidence. Either way it should be looked at.
        """
        tracked = tracked_files()
        for prefix, _ in ALLOWLIST:
            paths = [p for p in tracked if p.startswith(prefix)]
            with self.subTest(path=prefix):
                self.assertTrue(
                    any(violations_in(path) for path in paths),
                    f"{prefix} no longer contains any retired path. If its "
                    "recorded strings were 'corrected', revert that: they are "
                    "historical evidence, not code (see §I.4).",
                )


class LegacyFixtureIntegrityTests(unittest.TestCase):
    """Pin the fixtures' bytes.

    Belt and braces for the allowlist above. The allowlist stops the audit
    from *ordering* an edit; these digests turn an edit that happens anyway
    -- a stray formatter, an over-eager search-and-replace, a JSON
    pretty-printer -- from a silent loss of evidence into a loud failure.

    If this goes red: do not regenerate the digests. Restore the files with
    ``git checkout -- tests/fixtures/legacy/``.

    Six of the eight files are verbatim copies of what a run wrote. The two
    ``e2e_results3`` traces are the exception and are labelled below: they are
    a lossless projection of a run's traces, kept small enough to check in.
    ``tests/test_parallel_traces.py`` states the projection rule and proves it
    lossless against the verbatim ``results.json`` beside them, so they are as
    unre-derivable as the rest -- the full traces they came from live only in
    a gitignored ``out/``.
    """

    DIGESTS = {
        "tests/fixtures/legacy/e2e_schema8/manifest.json": (
            "ae283d7c9cb614319ff853ce92eb357e0e0d7f8e5355031ebfe6d90262b97ef1"
        ),
        "tests/fixtures/legacy/e2e_schema8/run_state.json": (
            "f43c1a3b52f2fdb679f91fd5c3c11e0c4e5657c6c8f6eba221f10fa79181db35"
        ),
        "tests/fixtures/legacy/kernel_schema1/manifest.json": (
            "4033d69c3331261183de77d2ac46c940f1afadc528a7018208085e0769aa9c9a"
        ),
        "tests/fixtures/legacy/kernel_schema1/results.json": (
            "5f6fa7082f78b22bc21748aefa692ba8ed189fa74a583c33418adfe141ac4e8c"
        ),
        # Verbatim: the manifest and results.json of
        # out/20260807T175156Z/piper1b_qkv/nvidia-h200, at manifest schema 8
        # and results schema 3 -- the results schema this commit replaces.
        "tests/fixtures/legacy/e2e_results3/manifest.json": (
            "ae4decd1617b1baf99e68ae744d59fdb8e981e1d5633d0fdbef60dda9a973eda"
        ),
        "tests/fixtures/legacy/e2e_results3/results.json": (
            "c83803823a609ea9e2883d79fa2c30b5eb8d7147f765d6d1712031558c65ab18"
        ),
        # Projected, not verbatim: that run's two baseline trace windows with
        # every event no extraction branch reads removed.
        (
            "tests/fixtures/legacy/e2e_results3/baseline/profiling/traces/"
            "iteration_20/rank0_trace.json.gz"
        ): (
            "8b7d393a3fbd8a5ecdcb411e5b67c7f911ce1feecbd81ae7ec3dbf8c66538e77"
        ),
        (
            "tests/fixtures/legacy/e2e_results3/baseline/profiling/traces/"
            "iteration_40/rank0_trace.json.gz"
        ): (
            "d3f2b6f60a90bdba8b57d076b1b62fe9aa4f92a7026de96014afdda9e2c7427b"
        ),
    }

    def test_the_fixtures_are_byte_identical_to_their_recording(self) -> None:
        for path, expected in self.DIGESTS.items():
            with self.subTest(fixture=path):
                actual = hashlib.sha256(
                    (REPO_ROOT / path).read_bytes()
                ).hexdigest()
                self.assertEqual(
                    actual,
                    expected,
                    f"{path} changed. These are byte-for-byte recordings of "
                    "runs that happened; they cannot be re-recorded, because "
                    "the code that produced them no longer exists. Restore "
                    "with `git checkout -- tests/fixtures/legacy/`, do not "
                    "update this digest.",
                )

    def test_the_digest_set_covers_every_tracked_fixture(self) -> None:
        tracked = {
            path
            for path in tracked_files()
            if path.startswith("tests/fixtures/legacy/")
        }
        self.assertEqual(tracked, set(self.DIGESTS))


# --------------------------------------------------------------------------
# The two ambiguous names, handled precisely rather than allowlisted.
# --------------------------------------------------------------------------


class PackageSymbolImportTests(unittest.TestCase):
    """``from benchmarks.artifacts import trace_files`` must not survive.

    ``artifacts.py`` became ``artifacts/`` and ``cli.py`` became ``cli/``, so
    the dotted prefix still resolves and the text sweep above cannot see the
    difference between the retired ``from benchmarks.artifacts import
    trace_files`` and the correct ``from benchmarks.artifacts.layout
    import trace_files``. Rather than weaken the sweep or allowlist the two
    names -- which would leave the most likely stale import unguarded -- this
    resolves them exactly, with ``ast``: an import of a name that is neither
    a submodule of the package nor defined in its ``__init__.py`` is an
    ImportError waiting for the first caller.

    The packages' ``__init__`` files are docstring-only by design (they run
    inside the training subprocess; see tests/test_import_boundaries.py), so
    in practice the legal set is exactly their submodules.
    """

    def _package_members(self, package: str) -> set[str]:
        directory = REPO_ROOT / package.replace(".", "/")
        members = {
            entry.stem if entry.suffix == ".py" else entry.name
            for entry in directory.iterdir()
            if (entry.suffix == ".py" and entry.stem != "__init__")
            or (entry.is_dir() and (entry / "__init__.py").is_file())
        }
        tree = ast.parse((directory / "__init__.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                members.update(
                    t.id for t in node.targets if isinstance(t, ast.Name)
                )
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                members.add(node.name)
            elif isinstance(node, ast.ImportFrom):
                members.update(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                members.update(
                    (alias.asname or alias.name).split(".")[0]
                    for alias in node.names
                )
        return members

    def test_no_module_imports_a_missing_name_from_those_packages(self) -> None:
        legal = {
            package: self._package_members(package)
            for package in MODULES_THAT_BECAME_PACKAGES
        }
        violations = []
        for path in scanned_files():
            if Path(path).suffix != ".py":
                continue
            tree = ast.parse(
                (REPO_ROOT / path).read_text(errors="ignore"), filename=path
            )
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level:
                    continue
                members = legal.get(node.module or "")
                if members is None:
                    continue
                for alias in node.names:
                    if alias.name not in members:
                        violations.append(
                            f"{path}:{node.lineno}: from {node.module} import "
                            f"{alias.name} -- {node.module} is a package now; "
                            f"{alias.name} lives in one of its submodules "
                            f"({', '.join(sorted(members))})"
                        )
        self.assertEqual(
            violations,
            [],
            "symbol imported from a package that replaced a module:\n"
            + "\n".join(violations),
        )

    def test_the_check_has_something_to_check(self) -> None:
        # Guards against the members set silently going empty (a moved
        # package, a renamed directory), which would make the test above
        # vacuous rather than failing.
        for package in MODULES_THAT_BECAME_PACKAGES:
            with self.subTest(package=package):
                members = self._package_members(package)
                self.assertTrue(members, f"{package} exposes no submodules")
                self.assertTrue(
                    (REPO_ROOT / package.replace(".", "/") / "__init__.py").is_file(),
                    f"{package} is not a package",
                )


if __name__ == "__main__":
    unittest.main()
