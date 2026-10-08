"""The arm log: torchrun tees the lines of every rank into it, and each line arrives whole."""

import io
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from torch.distributed.elastic.multiprocessing import tail_log

from benchmarks.e2e.engines.api import Launch
from benchmarks.execution import torchrun
from benchmarks.execution.affinity import CpuPinning
from benchmarks.execution.launcher import LaunchedCommand, build_command


RANKS = 2
LINES = 10000

EMITTER = f"""
import os

rank = os.environ["RANK"]
for index in range({LINES}):
    print(f"rank={{rank}} line={{index}} " + "x" * (index % 300))
"""
"""A rank that prints many lines of different lengths, so that the tee threads of torchrun write at the same time."""


def _expected(rank: int, index: int) -> bytes:
    return f"[rank{rank}]:rank={rank} line={index} ".encode() + b"x" * (index % 300)


def _run(launched: LaunchedCommand, argv: tuple[str, ...], log_path: Path) -> bytes:
    """Run ``argv`` with stdout and stderr in one log, as the runner does, and return the log."""
    with log_path.open("w") as log:
        subprocess.run(
            argv,
            cwd=launched.cwd,
            env=dict(launched.env),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return log_path.read_bytes()


def _lost_lines(log: bytes) -> int:
    """The emitted lines that the log does not hold exactly once."""
    counts = Counter(log.split(b"\n"))
    return sum(
        counts[_expected(rank, index)] != 1
        for rank in range(RANKS)
        for index in range(LINES)
    )


class TeedLogTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "rank_log_emitter.py").write_text(EMITTER)
        self.launched = build_command(
            Launch(
                target=("-m", "rank_log_emitter"),
                processes="per_rank",
                pin=False,
                cwd=self.root,
            ),
            world_size=RANKS,
            gpu=",".join(str(rank) for rank in range(RANKS)),
            pinning=CpuPinning((), "unpinned"),
            base_env={
                **{key: value for key, value in os.environ.items() if key != "PYTHONUNBUFFERED"},
                "PYTHONPATH": str(REPO_ROOT),
                "TMPDIR": str(self.root),
            },
        )
        self.assertIn("-u", self.launched.argv)

    def test_buffered_torchrun_loses_lines(self) -> None:
        """Without ``-u`` the tee threads share one buffered stream, and CPython 3.10 drops a line there and writes its length in NUL or stale bytes."""
        argv = tuple(token for token in self.launched.argv if token != "-u")
        log = _run(self.launched, argv, self.root / "buffered.log")
        self.assertGreater(_lost_lines(log), 0)

    def test_the_launcher_torchrun_loses_no_line(self) -> None:
        log = _run(self.launched, self.launched.argv, self.root / "arm.log")
        self.assertEqual(log.count(b"\x00"), 0)
        self.assertEqual(_lost_lines(log), 0)


class _SecondWrite(threading.Event):
    """A finished event that appends the rest of a line the first time a tail finds the end of the file."""

    def __init__(self, path: Path, rest: str) -> None:
        super().__init__()
        self.path = path
        self.rest = rest

    def is_set(self) -> bool:
        if self.rest:
            with self.path.open("a") as file:
                file.write(self.rest)
            self.rest = ""
            return False
        return True


class TailTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "stdout.log"

    def _tail(self, tail, first: str, rest: str) -> str:
        """The text that ``tail`` writes for a worker that writes ``first`` and then ``rest``."""
        self.path.write_text(first)
        dst = io.StringIO()
        tail(
            header="[rank0]:",
            file=str(self.path),
            dst=dst,
            finished=_SecondWrite(self.path, rest),
            interval_sec=0,
            log_line_filter=lambda _: True,
        )
        return dst.getvalue()

    def test_torch_tail_writes_a_partial_line(self) -> None:
        """The cause: a worker writes the text and the newline of one line in two calls, and torch writes the text alone."""
        self.assertEqual(
            self._tail(tail_log.tail_logfile, "step 1", "\n"), "[rank0]:step 1[rank0]:\n"
        )

    def test_the_launcher_tail_writes_a_whole_line(self) -> None:
        self.assertEqual(
            self._tail(torchrun.tail_whole_lines, "a\nstep 1", "\nb\n"),
            "[rank0]:a\n[rank0]:step 1\n[rank0]:b\n",
        )

    def test_the_launcher_tail_ends_the_last_partial_line(self) -> None:
        self.assertEqual(
            self._tail(torchrun.tail_whole_lines, "a\nstep", " 1"),
            "[rank0]:a\n[rank0]:step 1\n",
        )

    def test_install_refuses_a_changed_torch_tail(self) -> None:
        def other(header, file, dst, finished):
            pass

        original = tail_log.tail_logfile
        tail_log.tail_logfile = other
        self.addCleanup(setattr, tail_log, "tail_logfile", original)
        with self.assertRaisesRegex(RuntimeError, "port tail_whole_lines"):
            torchrun.install()


if __name__ == "__main__":
    unittest.main()
