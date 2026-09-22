"""Where a run's files go on disk, and the one way they are written.

The engine-neutral half of ``artifacts/``: nothing here knows what a scenario
or an arm *is*, only where its output lands and what a trace file is called.
That is an import fact as much as a tidiness one. ``benchmarks.kernel.
results.schema`` and ``benchmarks.kernel.runner`` need ``atomic_write_json``
and ``run_timestamp`` and nothing else, and taking them from ``manifests.py``
dragged ``benchmarks.e2e.registry`` -- every scenario, arm and workload
declaration in the repository -- into the kernel system's import graph for a
four-line JSON writer. ``benchmarks.e2e.validation`` and
``tools/collect_matrix.py`` are in the same position for ``trace_files``.
(This module owns the trace file-name grammar it globs. Both engines write
one trace per rank per profiler window, named ``rank<n>_trace.json.gz``.
``benchmarks.traces.extraction`` reads the same token to refuse a pooling
call that mixes two ranks.)

``trace_files`` returns **every rank's** traces. That is not the unit any
measurement is taken over: pooling two ranks' windows into one figure gives
an arithmetic mean across ranks, which is neither one rank's cost nor the
step's. ``trace_files_by_rank`` is the grouping every measurement path must
use, and ``benchmarks.traces.extraction.pooled_window_metrics`` refuses a
mixed-rank call outright rather than trusting each caller to remember.

``logs_by_rank`` is the same grouping for the other artifact a rank writes.
One ``<arm>.log`` holds every rank's output, because the runner gives one
subprocess one file, and torchrun prefixes each tee'd line with the rank
that wrote it. A rule read against the whole file therefore asks "did some
rank do this", which is the weaker question: a kernel that degraded on rank
1 alone passes it. The grammar is a launcher fact rather than a scenario
one, which is why the reader sits here beside the trace-file grammar.

``atomic_write_json`` is the only place anything under ``benchmarks/``
writes JSON -- manifests, run state, and both systems' ``results.json`` all
go through it. It
writes a sibling ``.tmp`` and renames, so a run killed mid-write leaves the
previous file intact rather than a truncated one -- which matters most for
``run_state.json``, rewritten on every arm transition -- and it passes
``allow_nan=False``, so a NaN metric fails loudly at write time instead of
producing a file no strict parser will read back.

``archive_incomplete_arm`` is the counterpart on the destructive side. A
resume never overwrites a failed arm's artifacts; it moves the arm directory
and its log under ``attempts/<timestamp>/<arm>/`` first, so the evidence of
why an arm failed survives the retry that replaces it. The numbered-suffix
loop is not paranoia: an arm that dies during launch fails inside the same
second as its predecessor.

``run_timestamp`` is public because ``benchmarks.cli.e2e`` generates one
stamp and hands it to every scenario of a multi-scenario ``run``,
which is what groups them under a single ``out/<timestamp>/``. A run may
name one scenario twice, so the second and later occurrences of a name take
a ``-run<n>`` suffix; without it they would share one directory and the
later one would die at ``mkdir`` after the earlier one had finished.
``_default_output_dir`` takes a ``Scenario`` only to read its ``.name``, so
the annotation lives under ``TYPE_CHECKING`` and this module keeps no
runtime dependency on ``benchmarks.e2e`` at all.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from benchmarks.execution.paths import BENCH_DIR

if TYPE_CHECKING:
    from benchmarks.e2e.schema import Scenario


_LOG_LINE_RANK = re.compile(r"^\[rank(\d+)\]:")
"""The rank prefix torchrun puts on every tee'd line.

Two spellings reach a log here and this pattern matches both. A multi-rank
run sets ``TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE`` to name the global rank,
because the default local-rank spelling agrees with it only on one node.
"""

TRACE_FILE_GLOB = "rank*_trace.json.gz"
"""What a profiler window is called, on every rank and both engines."""

_TRACE_FILE_NAME = re.compile(r"\Arank(\d+)_trace\.json\.gz\Z")


def rank_of_trace(path: Path) -> int | None:
    """The rank that wrote this trace, or ``None`` when the name does not say.

    ``None`` is a real answer, not an error: the synthetic traces the tests
    write carry whatever name the test chose, and a caller that only wants to
    know whether two paths came from *different* ranks must be able to say
    "this one does not claim a rank" without failing. A caller that needs the
    rank -- ``trace_files_by_rank`` -- raises on ``None`` itself, because
    there the name matched ``TRACE_FILE_GLOB`` and must therefore parse.
    """
    match = _TRACE_FILE_NAME.match(Path(path).name)
    return int(match.group(1)) if match else None


def trace_files(arm_dir: Path) -> list[Path]:
    """Every profiler window this arm wrote, from every rank.

    Use it to answer "does this arm have traces at all" and "does any trace
    contain this marker". Do **not** feed the result to a pooling call: see
    ``trace_files_by_rank``.
    """
    return sorted(
        arm_dir.glob(f"profiling/traces*/iteration_*/{TRACE_FILE_GLOB}")
    )


def trace_files_by_rank(arm_dir: Path) -> dict[int, list[Path]]:
    """The same files, grouped by the rank that wrote them, rank order.

    One rank is one process on one GPU, and its windows are the only set a
    per-step figure may be pooled over. A single-GPU run yields exactly one
    entry, keyed 0, holding what ``trace_files`` returns.
    """
    by_rank: dict[int, list[Path]] = {}
    for path in trace_files(arm_dir):
        rank = rank_of_trace(path)
        if rank is None:
            raise ValueError(
                f"{path}: matched the trace glob {TRACE_FILE_GLOB!r} but does "
                "not name a rank; a trace file is 'rank<n>_trace.json.gz'"
            )
        by_rank.setdefault(rank, []).append(path)
    return {rank: by_rank[rank] for rank in sorted(by_rank)}


def logs_by_rank(text: str) -> dict[int, str]:
    """One arm log, split into the text each rank wrote.

    A log with **fewer than two** ranks in it comes back whole, under the one
    rank it names. That is not a convenience: it is what makes this reader
    inert on every log written so far. A single-rank log holds unprefixed
    lines too -- the runner's own header, the ``nvidia-smi`` block, and
    ``run_train.sh``'s shell trace -- and those belong to no rank, so a split
    would drop them and change what every rule reads today. Two ranks make
    the split necessary and the unprefixed lines the launcher's.

    **The key is the rank the file names, never 0 by assumption.** A log
    written only by rank 1, because rank 0 died before it wrote anything,
    comes back as ``{1: ...}``. Returning it under 0 would be a repair, and
    a caller that does not validate -- ``evaluate`` on its own does not --
    would then publish rank 1's rows under rank 0's name. A file that names
    no rank at all is a single-rank log from before the prefix existed, and
    0 is what it was.

    A rank that wrote nothing therefore does not appear, and a two-rank run
    where one rank died silently comes back as one entry. The caller checks
    the returned ranks against the world size it asked for; this function
    reports what is in the file and repairs nothing.

    **A NUL run is dropped before the split.** Every rank writes to one
    shared descriptor, and a concurrent write can leave a hole that reads
    back as NUL bytes. One eight-rank run put 232 of them in front of rank
    3's completion line, which broke the prefix anchor and failed arm rule 1
    on a rank that had trained every step. A NUL is never log content, so
    removing it recovers the prefix and weakens no rule: the rank the line
    names and the message it carries both survive.

    **Known limit.** The split reads a line prefix, so it inherits whatever
    the launcher wrote. A torn write could still put two prefixes on one
    line, and this reader would credit the first. Nobody has produced one.
    It is recorded rather than guarded.
    """
    text = text.replace("\x00", "")
    by_rank: dict[int, list[str]] = {}
    for line in text.splitlines(keepends=True):
        match = _LOG_LINE_RANK.match(line)
        if match:
            by_rank.setdefault(int(match.group(1)), []).append(
                line[match.end() :]
            )
    if len(by_rank) < 2:
        return {next(iter(by_rank), 0): text}
    return {rank: "".join(by_rank[rank]) for rank in sorted(by_rank)}


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def archive_incomplete_arm(out_dir: Path, arm_name: str) -> Path | None:
    """Move incomplete artifacts aside so retrying never destroys evidence."""
    arm_dir = out_dir / arm_name
    log_path = out_dir / f"{arm_name}.log"
    if not arm_dir.exists() and not log_path.exists():
        return None

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = out_dir / "attempts" / timestamp / arm_name
    suffix = 1
    while archive.exists():
        archive = out_dir / "attempts" / f"{timestamp}-{suffix}" / arm_name
        suffix += 1
    archive.mkdir(parents=True)
    if arm_dir.exists():
        shutil.move(str(arm_dir), str(archive / "artifacts"))
    if log_path.exists():
        shutil.move(str(log_path), str(archive / log_path.name))
    return archive


def run_timestamp() -> str:
    """Directory-safe UTC stamp; shared by every scenario of one run."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_output_dir(
    scenario: Scenario,
    hardware: str,
    requested: Path | None,
    environment: Mapping[str, str],
    timestamp: str | None = None,
    occurrence: int = 1,
) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    if env_out := environment.get("OUT"):
        return Path(env_out).expanduser().resolve()
    directory = scenario.name if occurrence == 1 else f"{scenario.name}-run{occurrence}"
    return BENCH_DIR / "out" / (timestamp or run_timestamp()) / directory / hardware
