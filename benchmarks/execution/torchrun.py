"""Torchrun, with a tee that holds a partial worker line until its newline arrives."""

from __future__ import annotations

import hashlib
import inspect
import os
import time
from collections.abc import Callable
from threading import Event
from typing import TextIO

from torch.distributed.elastic.multiprocessing import tail_log
from torch.distributed.run import main


TAIL_SOURCE_SHA256 = "283496f2a8c65f833d6d57ecf37652a506f95f618330a89f289fd34fc8cda2b6"
"""The SHA-256 of the source of torch's ``tail_logfile``, which ``tail_whole_lines`` replaces."""


def tail_whole_lines(
    header: str,
    file: str,
    dst: TextIO,
    finished: Event,
    interval_sec: float,
    log_line_filter: Callable[[str], bool] | None = None,
) -> None:
    """Torch's ``tail_logfile``, but it writes a line only when its newline has arrived."""
    while not os.path.exists(file):
        if finished.is_set():
            return
        time.sleep(interval_sec)

    with open(file, errors="replace") as fp:
        line = ""
        while True:
            # Read the event first, so that the loop also reads a line that the worker wrote before it exited.
            done = finished.is_set()
            part = fp.readline()
            if part:
                line += part
                if line.endswith("\n"):
                    if log_line_filter and log_line_filter(line):
                        dst.write(f"{header}{line}")
                    line = ""
            elif done:
                # The worker exited inside a line, so end that line here.
                if line and log_line_filter and log_line_filter(line):
                    dst.write(f"{header}{line}\n")
                break
            else:
                time.sleep(interval_sec)


def install() -> None:
    """Make ``TailLog`` call ``tail_whole_lines``; raise when torch's tail differs from the one it replaces."""
    found = hashlib.sha256(inspect.getsource(tail_log.tail_logfile).encode()).hexdigest()
    if found != TAIL_SOURCE_SHA256:
        raise RuntimeError(
            f"torch's tail_logfile has the source SHA-256 {found}, not {TAIL_SOURCE_SHA256}; "
            "port tail_whole_lines to the new torch"
        )
    if "tail_logfile" not in tail_log.TailLog.start.__code__.co_names:
        raise RuntimeError(
            "torch's TailLog.start no longer calls tail_logfile; "
            "port tail_whole_lines to the new torch"
        )
    tail_log.tail_logfile = tail_whole_lines


if __name__ == "__main__":
    install()
    main()
