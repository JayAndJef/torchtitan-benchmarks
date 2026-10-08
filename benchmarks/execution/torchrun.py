"""Torchrun, with a tee that holds a partial worker line until its newline arrives."""

from __future__ import annotations

import inspect
import os
import time
from collections.abc import Callable
from threading import Event
from typing import TextIO

from torch.distributed.elastic.multiprocessing import tail_log
from torch.distributed.run import main


TAIL_PARAMETERS = ("header", "file", "dst", "finished", "interval_sec", "log_line_filter")
"""The parameters of torch's ``tail_logfile``, which ``TailLog.start`` passes by keyword."""


def tail_whole_lines(
    header: str,
    file: str,
    dst: TextIO,
    finished: Event,
    interval_sec: float,
    log_line_filter: Callable[[str], bool] | None = None,
) -> None:
    """Torch's ``tail_logfile``, which writes a line only when it is whole.

    A worker can write one line in two calls, and torch's version then
    writes the first part with no newline, so another rank's line joins it.
    """
    while not os.path.exists(file):
        if finished.is_set():
            return
        time.sleep(interval_sec)

    with open(file, errors="replace") as fp:
        line = ""
        while True:
            part = fp.readline()
            if part:
                line += part
                if line.endswith("\n"):
                    if log_line_filter and log_line_filter(line):
                        dst.write(f"{header}{line}")
                    line = ""
            elif finished.is_set():
                # The worker exited inside a line, so end that line here.
                if line and log_line_filter and log_line_filter(line):
                    dst.write(f"{header}{line}\n")
                break
            else:
                time.sleep(interval_sec)


def install() -> None:
    """Make ``TailLog`` call ``tail_whole_lines``; raise when torch's tail no longer matches it."""
    found = tuple(inspect.signature(tail_log.tail_logfile).parameters)
    if found != TAIL_PARAMETERS:
        raise RuntimeError(
            f"torch's tail_logfile takes {found}, not {TAIL_PARAMETERS}; "
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
