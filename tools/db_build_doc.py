"""Build one HTML document from ``database/`` in ``INDEX.md`` order.

    .venv/bin/python tools/db_build_doc.py [--out PATH]

Each ``##`` heading of ``INDEX.md`` becomes a top-level section; each listed
file follows under it with its headings promoted one level, so Google Docs
imports the outline as section -> file -> topic. Relative links between
files become in-document anchors. The CSS is inline and there are no
images, so the file uploads as one ``text/html`` payload.

Writes ``database/_build/engine-kb-<YYYYMMDD>-<bench rev>.html`` by default
and prints the path and its size.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import os
import re
import subprocess
from pathlib import Path

from markdown_it import MarkdownIt

REPO_ROOT = Path(__file__).resolve().parent.parent
DATABASE = REPO_ROOT / "database"
INDEX = DATABASE / "INDEX.md"

SECTION_RE = re.compile(r"^## (?P<title>.+)$")
ENTRY_RE = re.compile(r"^- \[(?P<title>[^\]]+)\]\((?P<path>[^)]+\.md)\)")
DB_COMMENT_RE = re.compile(r"<!--\s*db:.*?-->\n?")

CSS = """
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; max-width: 52em; margin: 2em auto; line-height: 1.4; }
code, pre { font-family: Menlo, Consolas, monospace; font-size: 0.92em; }
table { border-collapse: collapse; margin: 0.6em 0; }
th, td { border: 1px solid #999; padding: 0.25em 0.5em; vertical-align: top; }
h1 { page-break-before: always; border-bottom: 2px solid #444; }
h1:first-of-type { page-break-before: auto; }
nav ul { list-style: none; padding-left: 1em; }
"""


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def file_slug(path: Path) -> str:
    return slugify(str(path.relative_to(DATABASE).with_suffix("")))


def read_index() -> list[tuple[str, list[tuple[str, Path]]]]:
    sections: list[tuple[str, list[tuple[str, Path]]]] = []
    for line in INDEX.read_text().splitlines():
        section = SECTION_RE.match(line)
        if section:
            sections.append((section["title"], []))
            continue
        entry = ENTRY_RE.match(line)
        if entry:
            if not sections:
                raise SystemExit(f"INDEX.md lists {entry['path']} before any section heading")
            target = (DATABASE / entry["path"]).resolve()
            if not target.is_file():
                raise SystemExit(f"INDEX.md lists {entry['path']}, which does not exist")
            sections[-1][1].append((entry["title"], target))
    return sections


def rewrite_href(href: str, current: Path) -> str | None:
    if re.match(r"^[a-z]+:", href) or not href:
        return None
    path, _, anchor = href.partition("#")
    if path:
        target = (current.parent / path).resolve()
        if DATABASE not in target.parents or target.suffix != ".md":
            return None
        slug = file_slug(target)
    else:
        slug = file_slug(current)
    return f"#{slug}--{anchor}" if anchor else f"#{slug}"


def render_file(md: MarkdownIt, path: Path) -> str:
    text = DB_COMMENT_RE.sub("", path.read_text())
    tokens = md.parse(text)
    slug = file_slug(path)
    for i, token in enumerate(tokens):
        if token.type in ("heading_open", "heading_close"):
            level = min(int(token.tag[1]) + 1, 6)
            token.tag = f"h{level}"
            if token.type == "heading_open":
                heading = tokens[i + 1].content
                token.attrSet("id", slug if level == 2 else f"{slug}--{slugify(heading)}")
        elif token.type == "inline" and token.children:
            for child in token.children:
                if child.type == "link_open":
                    new = rewrite_href(child.attrGet("href") or "", path)
                    if new:
                        child.attrSet("href", new)
    return md.renderer.render(tokens, md.options, {})


def build(out: Path) -> Path:
    rev = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    today = dt.date.today()
    title = f"Engine Knowledge Base {today.isoformat()} (bench {rev})"
    md = MarkdownIt("commonmark").enable("table")
    sections = read_index()

    toc = ["<nav><h2>Contents</h2><ul>"]
    body: list[str] = []
    for section, entries in sections:
        section_id = f"sec-{slugify(section)}"
        toc.append(f'<li><a href="#{section_id}">{html.escape(section)}</a><ul>')
        body.append(f'<h1 id="{section_id}">{html.escape(section)}</h1>')
        for entry_title, path in entries:
            toc.append(f'<li><a href="#{file_slug(path)}">{html.escape(entry_title)}</a></li>')
            body.append(render_file(md, path))
        toc.append("</ul></li>")
    toc.append("</ul></nav>")

    document = (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>\n"
        f"<h1>{html.escape(title)}</h1>\n" + "\n".join(toc) + "\n" + "\n".join(body) + "\n</body></html>\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(document)
    os.replace(tmp, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="output path; default database/_build/engine-kb-<date>-<rev>.html")
    args = parser.parse_args(argv)
    if args.out is None:
        rev = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        args.out = DATABASE / "_build" / f"engine-kb-{dt.date.today():%Y%m%d}-{rev}.html"
    out = build(args.out)
    print(f"{out.relative_to(REPO_ROOT) if REPO_ROOT in out.parents else out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
