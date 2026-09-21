"""GPU worker entry point for one kernel-benchmark pass.

The parent launches this module in the prepared environment. One invocation
runs one pass over one unit, which ``--scenario`` or ``--span`` names, and
the parent merges the fragments into ``results.json``. A span composes a
``KernelScenario``, and that composed scenario is what this worker measures.

One arm per process keeps an arm's dependencies out of every other arm's
interpreter. ``--replicate-count`` adds replicates to that process and never
a second arm, so it costs no isolation.

Exit codes: 0 success; 3 correctness gates failed, with the fragment still
written; 2 bad arguments; 1 build or environment failure.

Module scope stays stdlib-only, so ``--help`` and an argument error return
without paying for torch. ``tests/test_import_boundaries.py`` pins that.

Every fragment also carries a phase table of named wall-clock spans. It is
provenance, and the merge reads no phase.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="benchmarks.kernel.worker")
    parser.add_argument("--scenario", default=None)
    parser.add_argument(
        "--span",
        default=None,
        help=(
            "a kernel span instead of a scenario. The worker measures the "
            "span's own head-to-head, which is a KernelScenario the span "
            "composes; the parent assembles the second total. Exactly one of "
            "--scenario and --span is required."
        ),
    )
    parser.add_argument("--mode", required=True, choices=("correctness", "timing"))
    parser.add_argument(
        "--fragment", default=None, type=Path, help="correctness mode only"
    )
    parser.add_argument(
        "--fragments-dir",
        default=None,
        type=Path,
        help=(
            "timing mode only: the directory the run's fragments live in. A "
            "batched worker writes one file per replicate, so it is handed "
            "the directory rather than a path."
        ),
    )
    parser.add_argument("--arm", default=None, help="timing mode only")
    parser.add_argument(
        "--replicate", type=int, default=None, help="timing mode only"
    )
    parser.add_argument(
        "--replicate-count",
        type=int,
        default=1,
        metavar="N",
        help=(
            "timing mode only: how many consecutive replicates this process "
            "measures, starting at --replicate. The arm is built once and "
            "reused, which is the whole saving."
        ),
    )
    parser.add_argument(
        "--skip-arm",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "correctness mode only: an arm this host cannot run, so it is "
            "neither built nor gated. Repeatable."
        ),
    )
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--samples-per-replicate", type=int, default=40)
    parser.add_argument("--burst-k", type=int, default=16)
    parser.add_argument("--warmup-calls", type=int, default=30)
    parser.add_argument("--burst", action="store_true")
    parser.add_argument("--model-size", default="30b-a3b")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if (args.scenario is None) == (args.span is None):
        # Neither would leave the worker with nothing to build; both would
        # leave it choosing, and the two rosters are disjoint on purpose.
        parser.error("give exactly one of --scenario and --span")
    if args.mode == "timing":
        if args.arm is None or args.replicate is None:
            parser.error("--mode timing requires --arm and --replicate")
        if args.fragments_dir is None:
            parser.error("--mode timing requires --fragments-dir")
        if args.replicate_count < 1:
            parser.error("--replicate-count must be >= 1")
    elif args.fragment is None:
        parser.error("--mode correctness requires --fragment")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from benchmarks.kernel.engine import phases

    startup = phases.process_start_offset()
    if startup is not None:
        # Interpreter startup and argument parsing, which begin before us.
        phases.record("process_startup", startup)

    with phases.phase("import_registry"):
        from benchmarks.kernel.registry import kernel_scenario_by_name
        from benchmarks.kernel.schema import resolve_shape_and_workload
        from benchmarks.kernel.spans import kernel_span_by_name

    try:
        # A span composes a KernelScenario, and that is what the two passes
        # below measure. The engine therefore never learns that spans exist.
        scenario = (
            kernel_scenario_by_name(args.scenario)
            if args.scenario is not None
            else kernel_span_by_name(args.span).measurement
        )
        shape, workload = resolve_shape_and_workload(
            model_size=args.model_size,
            batch=args.batch,
            seq_len=args.seq_len,
            max_seq_len=args.max_seq_len,
        )
        if args.mode == "timing":
            scenario.arm(args.arm)
        for name in args.skip_arm:
            scenario.arm(name)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    with phases.phase("import_torch"):
        from torch._functorch import config as functorch_config

    # Backward timing re-runs the graph, and a donated buffer forbids that.
    # This changes backward buffer reuse alone, for every arm alike.
    functorch_config.donated_buffer = False

    with phases.phase("import_engine"):
        from benchmarks.artifacts.layout import atomic_write_json
        from benchmarks.kernel.engine.run import (
            RunOptions,
            run_correctness_pass,
            time_replicate_block,
        )
        from benchmarks.kernel.schema import timing_fragment_path

    options = RunOptions(
        replicates=args.replicates,
        samples_per_replicate=args.samples_per_replicate,
        burst_k=args.burst_k,
        warmup_calls=args.warmup_calls,
        burst=args.burst,
        seed=args.seed,
    )
    written: list[tuple[Path, dict]] = []
    try:
        if args.mode == "correctness":
            fragment = run_correctness_pass(
                scenario, shape, workload, options, frozenset(args.skip_arm)
            )
            written.append((args.fragment, fragment))
        else:
            # The engine owns the build, the loop and the rule that the
            # per-arm extras attach to replicate 0.
            fragments = time_replicate_block(
                scenario,
                args.arm,
                args.replicate,
                args.replicate_count,
                shape,
                workload,
                options,
            )
            written.extend(
                (
                    timing_fragment_path(
                        args.fragments_dir, args.arm, replicate
                    ),
                    payload,
                )
                for replicate, payload in fragments.items()
            )
    except Exception:
        traceback.print_exc()
        return 1

    # The table describes the process, so every fragment carries the same one.
    table = phases.phases()
    for path, payload in written:
        payload["phases"] = table
        atomic_write_json(path, payload)
        print(f"fragment: {path}")
    for entry in table:
        print(f"phase {entry['phase']}: {entry['seconds']:.3f}s")

    if args.mode == "correctness" and not fragment["all_passed"]:
        failed = [
            f"{row['arm']}.{row['output']} {row['metric']}={row['value']:.4g}"
            f" (limit {row['threshold']})"
            for row in fragment["rows"]
            if row["passed"] is False
        ]
        print("correctness FAILED: " + "; ".join(failed), file=sys.stderr)
        return 3
    return 0


def _exit_now(code: int) -> None:
    """End the process at ``code`` without unwinding the interpreter.

    ``main`` writes the fragment before this runs, so a graceful shutdown
    only releases state the kernel reclaims anyway. What it costs is the
    ``atexit`` join of Inductor's 32 compile workers: measured on an H200 at
    rope/baseline, the worker reaches the end of ``main`` at 11.0 s and
    exits at 18.3 s. Those seconds were read at load average 28 to 81, so
    they order the two paths and may not be cited.

    The flushes below keep the diagnostics, because ``os._exit`` flushes
    neither Python's buffers nor libc's. The alternative,
    ``torch._inductor.config.compile_threads = 1``, saves the same seconds
    but also takes those 32 subprocesses off a host-bound measurement, so it
    needs a re-baseline and this exit does not.

    Two limits. The exit skips ``atexit`` unconditionally, so re-check the
    compile-cache fact after a torch bump: a cold-cache pass wrote the same
    13 Inductor and 65 Triton files either way. Do not copy this into the e2e
    training driver, because nobody checked this argument against the
    machinery that writes a profiler trace.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    # libc's own streams, which the two flushes above and ``os._exit`` miss.
    # The import is deferred, and a failed flush must not change the code.
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError):
        pass
    os._exit(code)


if __name__ == "__main__":
    _exit_now(main())
