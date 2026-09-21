"""Verify every path, flag and identifier the two guides name.

``AGENTS.md`` and ``README.md`` are read by people and by agents, and a
stale path or a retired flag in either one sends the reader to a file that
moved or to an option that no longer exists. Five mechanical rules close
that gap:

* every backticked token that looks like a repository path exists;
* every backticked option in a ``run`` section is a parameter of
  ``run_command``, and every one in a kernel section is a parameter of
  ``kernel_bench_command``;
* every backticked ``module:name`` identifier imports and resolves;
* the flag table of ``AGENTS.md`` names every ``run`` option exactly once;
  and
* each default in that table equals the default the code holds.

A token counts as a path when it holds no whitespace, no ``<`` placeholder
and no URL scheme, and it either holds a ``/`` or ends in one of
``PATH_EXTENSIONS``. Four prefixes name directories a run writes and git
ignores, so a token under one of them is documented rather than checked.
``ARTIFACT_FILE_NAMES`` is the other exemption: those are file names a run
writes, and they end in ``.json`` without ever naming a tracked file.
``NOT_PATHS`` holds the tokens that carry a slash and name something else;
it is one entry long and every addition needs a reason.
"""

import re
import sys
import unittest
from importlib import import_module
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.cli.e2e import run_command
from benchmarks.cli.kernel import kernel_bench_command
from benchmarks.e2e.registry import (
    DEFAULT_AC_MODE,
    DEFAULT_MEGATRON_NAN_GUARD,
    DEFAULT_MEGATRON_P2P_SYNC,
    DEFAULT_MEGATRON_PRECISION,
    DEFAULT_MODEL_SIZE,
    DEFAULT_PROFILE,
    DEFAULT_WARMUP_STEPS,
    ENGINES,
)
from benchmarks.e2e.parallelism import DEFAULT_ZERO, ParallelismSpec

DOCS = ("AGENTS.md", "README.md")

BACKTICKED = re.compile(r"`([^`\n]+)`")
OPTION = re.compile(r"\A--[a-z0-9][a-z0-9-]*\Z")
IDENTIFIER = re.compile(r"\Abenchmarks(?:\.[A-Za-z_][A-Za-z_0-9]*)+:[A-Za-z_][A-Za-z_0-9]*\Z")

PATH_EXTENSIONS = (".py", ".sh", ".md", ".yml", ".toml", ".json")
IGNORED_PREFIXES = ("out/", "reports/", ".cuda-compat/", ".venv/")
ARTIFACT_FILE_NAMES = frozenset(
    {"manifest.json", "results.json", "run_state.json"}
)
NOT_PATHS = frozenset(
    {
        # The pinned branch of the TorchTitan fork, not a directory.
        "bench/torchtitan-benchmarks",
        # A kernel arm name. A cross-engine arm is named engine/profile.
        "mcore/base",
    }
)

ALWAYS_VALID_OPTIONS = frozenset({"--help"})
"""Click adds this one to every command after the declared parameters."""

UNSET = click.Option(["--unset-probe"]).default
"""What Click holds on an option that declares no default.

Click 8.4 holds a sentinel object there rather than ``None``, so the value
is read from a probe option instead of being spelled out.
"""


def _has_no_default(option: click.Option) -> bool:
    """Whether the operator gets no value from the option itself."""
    return option.default is None or option.default is UNSET


def _sections(text: str) -> dict[str, str]:
    """The document split at its level-two headings, heading to body."""
    parts = re.split(r"^## ", text, flags=re.MULTILINE)
    sections = {}
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        sections[heading.strip()] = body
    return sections


def _tokens(text: str) -> list[str]:
    return BACKTICKED.findall(text)


def _options_of(command: click.Command) -> list[click.Option]:
    return [
        parameter
        for parameter in command.params
        if isinstance(parameter, click.Option)
    ]


def _flags_of(command: click.Command) -> set[str]:
    flags: set[str] = set(ALWAYS_VALID_OPTIONS)
    for option in _options_of(command):
        flags.update(option.opts)
        flags.update(option.secondary_opts)
    return flags


def _looks_like_a_path(token: str) -> bool:
    if any(character.isspace() for character in token):
        return False
    if "<" in token or token.startswith("http"):
        return False
    if token in ARTIFACT_FILE_NAMES or token in NOT_PATHS:
        return False
    return "/" in token or token.endswith(PATH_EXTENSIONS)


class DocumentedPathTests(unittest.TestCase):
    def test_every_documented_path_exists(self) -> None:
        for name in DOCS:
            text = (REPO_ROOT / name).read_text()
            for token in _tokens(text):
                if not _looks_like_a_path(token):
                    continue
                if token.startswith(IGNORED_PREFIXES):
                    continue
                with self.subTest(document=name, path=token):
                    self.assertTrue(
                        (REPO_ROOT / token).exists(),
                        f"{name} names {token!r}, which does not exist",
                    )


class DocumentedOptionTests(unittest.TestCase):
    def _assert_section_options(
        self, name: str, heading_part: str, command: click.Command
    ) -> None:
        text = (REPO_ROOT / name).read_text()
        matched = [
            body
            for heading, body in _sections(text).items()
            if heading_part in heading
        ]
        self.assertTrue(matched, f"{name} has no {heading_part!r} section")
        flags = _flags_of(command)
        for body in matched:
            for token in _tokens(body):
                if not OPTION.match(token):
                    continue
                with self.subTest(document=name, option=token):
                    self.assertIn(
                        token,
                        flags,
                        f"{name} documents {token!r}, which "
                        f"{command.name!r} does not take",
                    )

    def test_run_sections_name_run_options(self) -> None:
        self._assert_section_options("AGENTS.md", "`run` command", run_command)
        self._assert_section_options("README.md", "Run", run_command)

    def test_kernel_sections_name_kernel_options(self) -> None:
        for name in DOCS:
            self._assert_section_options(
                name, "Kernel-isolation benchmarks", kernel_bench_command
            )


class DocumentedIdentifierTests(unittest.TestCase):
    def test_every_documented_identifier_resolves(self) -> None:
        for name in DOCS:
            text = (REPO_ROOT / name).read_text()
            for token in _tokens(text):
                if not IDENTIFIER.match(token):
                    continue
                module_name, _, attribute = token.partition(":")
                with self.subTest(document=name, identifier=token):
                    module = import_module(module_name)
                    self.assertTrue(
                        hasattr(module, attribute),
                        f"{name} names {token!r}, and {module_name} has no "
                        f"{attribute!r}",
                    )


def _documented_defaults() -> dict[str, str]:
    """The ``run`` flag table of ``AGENTS.md``, as flag to default cell."""
    text = (REPO_ROOT / "AGENTS.md").read_text()
    body = next(
        body
        for heading, body in _sections(text).items()
        if "`run` command" in heading
    )
    documented: dict[str, str] = {}
    for line in body.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4:
            continue
        flag = BACKTICKED.findall(cells[0])
        if not flag or not OPTION.match(flag[0]):
            continue
        default = BACKTICKED.findall(cells[2])
        documented[flag[0]] = default[0] if default else cells[2]
    return documented


def _expected_defaults() -> dict[str, str]:
    """The default of every ``run`` option, read from the code."""
    workload = ENGINES.workload
    trivial = ParallelismSpec()
    resolved = {
        "--seq-len": str(workload.seq_len),
        "--steps": str(workload.steps),
        "--batch": str(workload.local_batch_size),
        "--ac": DEFAULT_AC_MODE,
        "--model-size": DEFAULT_MODEL_SIZE,
        "--dp": str(trivial.dp),
        "--pp": str(trivial.pp),
        "--ep": str(trivial.ep),
        "--pp-schedule": "--" if trivial.pp_schedule is None else trivial.pp_schedule,
        "--pp-microbatch-size": str(trivial.pp_microbatch_size),
        "--zero": str(DEFAULT_ZERO),
        "--megatron-p2p-sync": DEFAULT_MEGATRON_P2P_SYNC,
        "--megatron-nan-guard": DEFAULT_MEGATRON_NAN_GUARD,
        "--megatron-precision": DEFAULT_MEGATRON_PRECISION,
        "--profile": "on" if DEFAULT_PROFILE else "off",
        "--warmup-steps": str(DEFAULT_WARMUP_STEPS),
    }
    expected = {}
    for option in _options_of(run_command):
        flag = option.opts[0]
        if flag in resolved:
            expected[flag] = resolved[flag]
        elif _has_no_default(option):
            expected[flag] = "--"
        else:
            expected[flag] = str(option.default)
    return expected


class FlagTableTests(unittest.TestCase):
    def test_the_table_names_every_run_option(self) -> None:
        self.assertEqual(
            sorted(_documented_defaults()),
            sorted(_expected_defaults()),
            "the AGENTS.md flag table and run_command disagree about the "
            "option roster",
        )

    def test_every_documented_default_matches_the_code(self) -> None:
        documented = _documented_defaults()
        for flag, expected in _expected_defaults().items():
            with self.subTest(option=flag):
                self.assertEqual(
                    documented.get(flag),
                    expected,
                    f"AGENTS.md documents {flag} as "
                    f"{documented.get(flag)!r}, and the code holds "
                    f"{expected!r}",
                )


if __name__ == "__main__":
    unittest.main()
