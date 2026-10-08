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
from unittest import mock

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
        self.assertIn(torchrun.__name__, self.launched.argv)

    def test_the_launcher_torchrun_loses_no_line(self) -> None:
        log = _run(self.launched, self.launched.argv, self.root / "arm.log")
        self.assertEqual(log.count(b"\x00"), 0)
        self.assertEqual(_lost_lines(log), 0)


class _SwitchInWrite(io.BytesIO):
    """A buffer whose first write runs the write of another thread on the same wrapper, as a thread switch inside the write does."""

    def __init__(self, other: str) -> None:
        super().__init__()
        self.other = other
        self.wrapper: io.TextIOWrapper | None = None

    def write(self, data) -> int:
        if self.other:
            other, self.other = self.other, ""
            self.wrapper.write(other)
        return super().write(data)


OTHER_LINE = "rank=1 line=0\n"
"""The line that the second thread writes."""

TEE_LINES = tuple(f"rank=0 line={index} " + "x" * 100 + "\n" for index in range(200))
"""The lines that the tee thread writes, more than one buffer of them."""


def _wrapper_write(write_through: bool) -> bytes:
    """The bytes that the tee thread and the second thread give a wrapper over ``_SwitchInWrite``."""
    buffer = _SwitchInWrite(OTHER_LINE)
    wrapper = io.TextIOWrapper(buffer, encoding="utf-8", write_through=write_through)
    buffer.wrapper = wrapper
    for line in TEE_LINES:
        wrapper.write(line)
    wrapper.flush()
    return buffer.getvalue()


class StdoutTests(unittest.TestCase):
    def test_a_buffered_wrapper_loses_the_line_of_another_thread(self) -> None:
        """The CPython 3.10 race that ``-u`` stops: a write that flushes a full buffer drops the line that another thread wrote meanwhile, and NUL or stale bytes take its place."""
        written = _wrapper_write(write_through=False)
        self.assertNotIn(OTHER_LINE.encode(), written)
        self.assertEqual(len(written), len(OTHER_LINE) + sum(map(len, TEE_LINES)))

    def test_a_write_through_wrapper_keeps_every_line(self) -> None:
        written = Counter(_wrapper_write(write_through=True).decode().splitlines(keepends=True))
        self.assertEqual(written, Counter((OTHER_LINE, *TEE_LINES)))

    def test_u_makes_the_standard_streams_write_through(self) -> None:
        printed = subprocess.run(
            [sys.executable, "-u", "-c", "import sys; print(sys.stdout.write_through, sys.stderr.write_through)"],
            env={key: value for key, value in os.environ.items() if key != "PYTHONUNBUFFERED"},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertEqual(printed, "True True\n")


class _Worker(threading.Event):
    """A worker that writes its next part each time a tail sleeps, and that has exited when it has no part left."""

    def __init__(self, path: Path, parts: tuple[str, ...]) -> None:
        super().__init__()
        self.path = path
        self.parts = list(parts)

    def write_next(self, _: float) -> None:
        if self.parts:
            with self.path.open("a") as file:
                file.write(self.parts.pop(0))

    def is_set(self) -> bool:
        return not self.parts


class _ExitAtCheck(threading.Event):
    """A worker that writes its last line and exits just before a tail first reads the event."""

    def __init__(self, path: Path, last: str) -> None:
        super().__init__()
        self.path = path
        self.last = last

    def is_set(self) -> bool:
        if self.last:
            with self.path.open("a") as file:
                file.write(self.last)
            self.last = ""
        return True


class TailTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "stdout.log"

    def _tail(self, tail, first: str, finished: threading.Event) -> str:
        """The text that ``tail`` writes for a worker that wrote ``first`` and then acts as ``finished`` says."""
        self.path.write_text(first)
        dst = io.StringIO()
        tail(
            header="[rank0]:",
            file=str(self.path),
            dst=dst,
            finished=finished,
            interval_sec=0,
            log_line_filter=lambda _: True,
        )
        return dst.getvalue()

    def _tail_while_writing(self, tail, first: str, *parts: str) -> str:
        """The text that ``tail`` writes for a worker that writes ``first``, then one of ``parts`` at each sleep of the tail."""
        worker = _Worker(self.path, parts)
        with mock.patch("time.sleep", worker.write_next):
            return self._tail(tail, first, worker)

    def test_torch_tail_writes_a_partial_line(self) -> None:
        """The cause: a worker writes the text and the newline of one line in two calls, and torch writes the text alone."""
        self.assertEqual(
            self._tail_while_writing(tail_log.tail_logfile, "step 1", "\n"),
            "[rank0]:step 1[rank0]:\n",
        )

    def test_the_launcher_tail_writes_a_whole_line(self) -> None:
        self.assertEqual(
            self._tail_while_writing(torchrun.tail_whole_lines, "a\nstep", " 1", "\nb\n"),
            "[rank0]:a\n[rank0]:step 1\n[rank0]:b\n",
        )

    def test_the_launcher_tail_ends_the_last_partial_line(self) -> None:
        self.assertEqual(
            self._tail_while_writing(torchrun.tail_whole_lines, "a\nstep", " 1"),
            "[rank0]:a\n[rank0]:step 1\n",
        )

    def test_torch_tail_loses_a_line_written_just_before_the_exit(self) -> None:
        self.assertEqual(
            self._tail(tail_log.tail_logfile, "step 1\n", _ExitAtCheck(self.path, "done\n")),
            "[rank0]:step 1\n",
        )

    def test_the_launcher_tail_keeps_a_line_written_just_before_the_exit(self) -> None:
        self.assertEqual(
            self._tail(torchrun.tail_whole_lines, "step 1\n", _ExitAtCheck(self.path, "done\n")),
            "[rank0]:step 1\n[rank0]:done\n",
        )


class InstallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(setattr, tail_log, "tail_logfile", tail_log.tail_logfile)
        self.addCleanup(setattr, tail_log.TailLog, "start", tail_log.TailLog.start)

    def test_install_replaces_the_torch_tail(self) -> None:
        torchrun.install()
        self.assertIs(tail_log.tail_logfile, torchrun.tail_whole_lines)

    def test_install_refuses_a_changed_torch_tail(self) -> None:
        def other(header, file, dst, finished, interval_sec, log_line_filter=None):
            pass

        tail_log.tail_logfile = other
        with self.assertRaisesRegex(RuntimeError, "source SHA-256"):
            torchrun.install()

    def test_install_refuses_a_start_that_calls_no_torch_tail(self) -> None:
        def start(self):
            return self

        tail_log.TailLog.start = start
        with self.assertRaisesRegex(RuntimeError, "no longer calls tail_logfile"):
            torchrun.install()


if __name__ == "__main__":
    unittest.main()
