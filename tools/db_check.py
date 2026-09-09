"""Check the engine knowledge base under ``database/``.

    .venv/bin/python tools/db_check.py [--strict] [--window N]

Three checks, and nothing else:

1. Every code ref resolves. A ref is a code span ``root path:line`` (or
   ``:lo-hi``), optionally followed by a second code span that names a
   symbol. The roots and their pins are the table in
   ``database/00-overview.md``. The file is read at the pin
   (``git show pin:path``; a root under ``.venv/`` is read from the tree),
   and the symbol must appear within ``--window`` lines of the cited line.
2. Every ``## Terms`` entry matches ``database/GLOSSARY.md`` byte for byte.
3. Every content file appears in ``database/INDEX.md`` exactly once.

Findings print as ``database/<file>:<line>: <CODE> <message>``. Exit 0 when
clean, 1 on any error, 2 when the scaffold itself is missing. A warning
never changes the exit code unless ``--strict`` promotes it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATABASE = REPO_ROOT / "database"
OVERVIEW = "00-overview.md"
GLOSSARY = "GLOSSARY.md"
INDEX = "INDEX.md"
NOT_CONTENT = {INDEX, GLOSSARY, "STYLE.md"}

PIN_RE = re.compile(
    r"^\|\s*`(?P<root>[a-z]+)`\s*\|\s*`(?P<path>[^`]+)`\s*\|\s*`(?P<pin>[^`]+)`\s*\|"
)
TERM_RE = re.compile(r"^- \*\*(?P<term>[^*]+)\*\*: (?P<sentence>.+)$")
LINK_RE = re.compile(r"\]\((?P<target>[^)#\s]+\.md)(?:#[^)]*)?\)")
IDENT_RE = re.compile(r"^[\w.]+$")
VERSION_RE = re.compile(r"__version__\s*=\s*['\"]([^'\"]+)")


@dataclass(frozen=True)
class Root:
    name: str
    path: Path
    pin: str

    @property
    def from_tree(self) -> bool:
        return ".venv" in self.path.parts

    @property
    def external(self) -> bool:
        return self.path != REPO_ROOT and REPO_ROOT not in self.path.parents


@dataclass
class Finding:
    file: Path
    line: int
    code: str
    message: str

    def render(self) -> str:
        return f"{self.file.relative_to(REPO_ROOT)}:{self.line}: {self.code} {self.message}"


def read_roots(text: str) -> dict[str, Root]:
    roots: dict[str, Root] = {}
    for line in text.splitlines():
        match = PIN_RE.match(line)
        if match:
            path = Path(match["path"])
            if not path.is_absolute():
                path = (REPO_ROOT / path).resolve()
            roots[match["root"]] = Root(match["root"], path, match["pin"])
    return roots


def read_glossary(text: str) -> tuple[dict[str, str], list[tuple[int, str]]]:
    glossary: dict[str, str] = {}
    duplicates: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        match = TERM_RE.match(line)
        if match:
            if match["term"] in glossary:
                duplicates.append((lineno, match["term"]))
            glossary[match["term"]] = match["sentence"]
    return glossary, duplicates


def content_files() -> list[Path]:
    files = []
    for path in sorted(DATABASE.rglob("*.md")):
        rel = path.relative_to(DATABASE)
        if any(part.startswith("_") for part in rel.parts):
            continue
        if len(rel.parts) == 1 and rel.name in NOT_CONTENT:
            continue
        files.append(path)
    return files


class Source:
    """Reads a cited file at its root's pin, once per (root, path)."""

    def __init__(self) -> None:
        self.cache: dict[tuple[str, str], list[str] | None] = {}
        self.skipped: set[str] = set()

    def lines(self, root: Root, path: str) -> list[str] | None:
        key = (root.name, path)
        if key not in self.cache:
            self.cache[key] = self._read(root, path)
        return self.cache[key]

    @staticmethod
    def _read(root: Root, path: str) -> list[str] | None:
        if root.from_tree:
            target = root.path / path
            if not target.is_file():
                return None
            return target.read_text(errors="replace").splitlines()
        done = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(root.path), "show", f"{root.pin}:{path}"],
            capture_output=True,
            text=True,
        )
        return done.stdout.splitlines() if done.returncode == 0 else None


def symbol_matcher(symbol: str):
    if IDENT_RE.match(symbol):
        pattern = re.compile(r"(?<![\w])" + re.escape(symbol) + r"(?![\w])")
        return lambda line: bool(pattern.search(line))
    return lambda line: symbol in line


def check_refs(
    file: Path, text: str, roots: dict[str, Root], source: Source, window: int, findings: list[Finding]
) -> int:
    ref_re = re.compile(
        r"`(?P<root>" + "|".join(re.escape(name) for name in roots) + r") "
        r"(?P<path>[^`\s:]+):(?P<lo>\d+)(?:-(?P<hi>\d+))?`(?:[ ]*`(?P<symbol>[^`\n]+)`)?"
    )
    count = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in ref_re.finditer(line):
            count += 1
            root = roots[match["root"]]
            cited = match["path"]
            if root.external and not root.path.is_dir():
                source.skipped.add(root.name)
                continue
            lines = source.lines(root, cited)
            if lines is None:
                findings.append(Finding(file, lineno, "E-REF-MISSING", f"{root.name}@{root.pin} has no {cited}"))
                continue
            lo = int(match["lo"])
            hi = int(match["hi"] or lo)
            if hi > len(lines):
                findings.append(Finding(file, lineno, "E-REF-LINE", f"{cited} has {len(lines)} lines; ref cites :{lo}"))
                continue
            symbol = match["symbol"]
            if symbol is None:
                findings.append(Finding(file, lineno, "W-REF-NOSYMBOL", f"{cited}:{lo} names no symbol"))
                continue
            matches = symbol_matcher(symbol)
            span = range(max(1, lo - window), min(len(lines), hi + window) + 1)
            if any(matches(lines[i - 1]) for i in span):
                continue
            hits = [i for i, text_line in enumerate(lines, 1) if matches(text_line)]
            hint = f"found at :{min(hits, key=lambda i: abs(i - lo))}" if hits else "not in file"
            findings.append(
                Finding(file, lineno, "E-REF-SYMBOL", f"`{symbol}` not within {window} lines of {cited}:{lo}; {hint}")
            )
    return count


def terms_block(lines: list[str]) -> tuple[int, int] | None:
    """The 0-based [start, end) line range of the ``## Terms`` section."""
    for start, line in enumerate(lines):
        if line.strip() == "## Terms":
            end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
            return start, end
    return None


def check_terms(file: Path, text: str, glossary: dict[str, str], findings: list[Finding]) -> None:
    lines = text.splitlines()
    block = terms_block(lines)
    if block is None:
        findings.append(Finding(file, 1, "E-TERMS-MISSING", "no `## Terms` section"))
        return
    start, end = block
    declared: set[str] = set()
    for offset in range(start + 1, end):
        match = TERM_RE.match(lines[offset])
        if not match:
            continue
        term = match["term"]
        declared.add(term)
        if term not in glossary:
            findings.append(Finding(file, offset + 1, "W-TERM-UNKNOWN", f"`{term}` is not in GLOSSARY.md"))
        elif match["sentence"] != glossary[term]:
            findings.append(Finding(file, offset + 1, "E-TERM-DRIFT", f"`{term}` differs from GLOSSARY.md"))
    body = "\n".join(lines[:start] + lines[end:])
    for term in glossary:
        if term in declared:
            continue
        if re.search(r"(?<![\w])" + re.escape(term) + r"(?![\w])", body):
            findings.append(Finding(file, 1, "W-TERM-UNDECLARED", f"`{term}` is used but not declared in Terms"))


def check_index(files: list[Path], index_text: str, findings: list[Finding]) -> None:
    index = DATABASE / INDEX
    listed = Counter((DATABASE / target).resolve() for target in LINK_RE.findall(index_text))
    expected = set(files) | {DATABASE / GLOSSARY}
    for path in sorted(expected):
        if listed.get(path, 0) != 1:
            findings.append(Finding(index, 1, "E-INDEX", f"{path.relative_to(DATABASE)} is listed {listed.get(path, 0)} times"))
    for path in listed:
        if path not in expected:
            findings.append(Finding(index, 1, "E-INDEX", f"{path} is listed but is not a content file"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--window", type=int, default=3, help="lines around a cited line to search for the symbol")
    parser.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = parser.parse_args(argv)

    for name in (OVERVIEW, GLOSSARY, INDEX):
        if not (DATABASE / name).is_file():
            print(f"scaffold error: database/{name} is missing")
            return 2
    roots = read_roots((DATABASE / OVERVIEW).read_text())
    if not roots:
        print(f"scaffold error: no pinned revisions table in database/{OVERVIEW}")
        return 2

    findings: list[Finding] = []
    glossary, duplicates = read_glossary((DATABASE / GLOSSARY).read_text())
    for lineno, term in duplicates:
        findings.append(Finding(DATABASE / GLOSSARY, lineno, "E-TERM-DUP", f"`{term}` is defined twice"))
    for root in roots.values():
        if root.from_tree:
            version_file = root.path / "version.py"
            found = VERSION_RE.search(version_file.read_text()) if version_file.is_file() else None
            if found is None or found.group(1) != root.pin:
                seen = found.group(1) if found else "absent"
                findings.append(Finding(DATABASE / OVERVIEW, 1, "E-ROOT-PIN", f"{root.name} tree is {seen}; pin is {root.pin}"))

    source = Source()
    files = content_files()
    refs = 0
    for file in files:
        text = file.read_text()
        refs += check_refs(file, text, roots, source, args.window, findings)
        check_terms(file, text, glossary, findings)
    check_index(files, (DATABASE / INDEX).read_text(), findings)

    for finding in findings:
        print(finding.render())
    errors = sum(1 for f in findings if f.code.startswith("E-") or (args.strict and f.code.startswith("W-")))
    warnings = len(findings) - errors
    skipped = ", ".join(sorted(source.skipped)) or "none"
    print(f"refs {refs}, files {len(files)}, errors {errors}, warnings {warnings}, roots skipped: {skipped}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
