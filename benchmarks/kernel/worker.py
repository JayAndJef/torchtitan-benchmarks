"""GPU worker entry point for one kernel-benchmark pass.

Launched by the parent as ``numactl ... python -m benchmarks.kernel.worker``
inside the prepared environment (CUDA_VISIBLE_DEVICES, cache dirs, compiler
env). One invocation runs **one** pass and writes **one** JSON fragment; the
parent merges the fragments into ``results.json``.

Two modes, matching the two passes:

* ``--mode correctness`` builds every arm and gates them. Once per scenario,
  and first -- a failed gate means no timing is worth taking. ``--skip-arm``
  removes an arm this host cannot run, so a missing compiler costs the TE arm
  and not the whole scenario.
* ``--mode timing --arm NAME --replicate N`` builds that one arm and times it
  for replicates ``N`` through ``N + --replicate-count - 1``, writing one
  fragment per replicate under ``--fragments-dir``.

The timing mode is why this file exists in this shape. **One arm per process**
is what keeps an arm's dependencies out of every other arm's interpreter: a
build failure, a leaked CUDA context or a JIT-built CUDA extension in one arm
cannot reach another. ``--replicate-count`` moves replicates into
that process and never a second arm, so the isolation the split exists for is
untouched by it. What it does cost is stated where the parent chooses the
value: ``benchmarks.kernel.runner``.

The batch is written at the end rather than replicate by replicate. A worker
that dies mid-batch costs the arm either way -- the merge requires a complete
replicate set and fails the arm without one -- so there is nothing for a
partial write to save.

Exit codes: 0 success; 3 correctness gates failed (the fragment is still
written); 2 bad arguments; 1 build or environment failure.

Module scope stays stdlib-only, so ``--help`` and an argument error return
without paying for torch. ``tests/test_import_boundaries.py`` pins that.

**The process is ended, not unwound** -- see ``_exit_now``. A worker is one
pass and its whole product is the fragment; a graceful interpreter shutdown
after that costs several seconds and buys nothing.

Every fragment carries a **phase table** -- named wall-clock spans from
process exec to the end of the pass. It is provenance, not a result: the
merge reads no phase and ``results.json`` carries none. It rides in the
fragment because the fragment is the one artifact a worker already writes,
and a phase table is worth nothing unless it survives the process that
produced it.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="benchmarks.kernel.worker")
    parser.add_argument("--scenario", required=True)
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
    parser.add_argument("--model-size", default="normal")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
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
        # Interpreter startup plus this module's stdlib imports plus argument
        # parsing. Not timed from inside, because it begins before any of our
        # Python runs.
        phases.record("process_startup", startup)

    with phases.phase("import_registry"):
        from benchmarks.kernel.registry import kernel_scenario_by_name
        from benchmarks.kernel.schema import resolve_shape_and_workload

    try:
        scenario = kernel_scenario_by_name(args.scenario)
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

    # Backward-mode timing re-runs a compiled backward graph with
    # retain_graph=True; AOT autograd's donated-buffer optimization forbids
    # re-execution and raises on arms that save intermediates
    # (expert_mlp's gate_up). Disabling it changes backward buffer reuse, not the
    # generated kernels, and applies to every arm alike.
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
            # The build, the replicate loop and the rule that the per-arm
            # extras attach to replicate 0 all live in the engine, in one
            # function. At --replicate-count 1 it is the order the
            # single-replicate pass has always used, because
            # run_timing_pass is now the same function's count=1 case.
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

    # Provenance, not a result: the merge reads no phase and results.json
    # carries none. See this module's docstring. The table describes the
    # *process*, so every fragment a batched worker writes carries the same
    # one.
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

    A worker's whole product is the fragment, and ``main`` has written it
    before this runs. What a graceful shutdown does after that is release
    state the kernel is about to reclaim anyway: a compiled graph, its Triton
    modules, the device allocations, the CUDA context, and above all
    Inductor's compile-worker pool. The first ``torch.compile`` in the
    process starts 32 ``compile_worker`` subprocesses, each of which imports
    torch, and an ``atexit`` handler joins them. Measured on an H200,
    rope/baseline: the worker reaches the end of ``main`` at 11.0 s, finishes
    its ``atexit`` handlers at 16.6 s and exits at 18.3 s.

    Ending the process instead is safe here for four separate reasons, and
    each was checked rather than assumed:

    * **It cannot move a number.** Every sample is taken, every gate is run
      and the fragment is on disk before this line. There is no measurement
      left to disturb.
    * **The fragment survives.** ``atomic_write_json`` writes a temporary
      file, closes it and renames it, all before ``main`` returns. Page-cache
      data outlives ``_exit``; only unflushed *process* buffers do not, which
      is what the three flushes below are for. There are three because
      Python owns two of the process's buffers and libc owns the rest: a
      normal ``exit()`` flushes libc's streams and ``os._exit`` does not, and
      the parent redirects this process's stdout to a file, so libc's
      ``stdout`` is block-buffered rather than line-buffered. Any C or C++
      extension that prints through ``printf`` or ``std::cout`` -- CUDA,
      cuDNN and TransformerEngine all can -- would otherwise lose its output.
      That output is never a measurement, but it is diagnostic, and the exit
      codes it matters most for are 1 and 3, whose whole report to the
      operator is a tail of this log.
    * **The compile caches survive.** Both Inductor and Triton write their
      artifacts when the kernel is compiled, not at exit. A cold-cache pass
      run both ways left 13 inductor files and 65 triton files either way,
      1,168,648 against 1,168,632 bytes.
    * **The compile pool is not orphaned.** Each of those subprocesses is
      given ``--parent`` and exits when it is reparented. Counted on the
      hardware: the ``compile_worker`` processes of one worktree rise to 15
      during a helion build and are back to the pre-run count within two
      seconds of the worker ending.

    ``main`` itself only *returns* the code, so a caller that imports this
    module decides its own exit and is unaffected.

    The alternative -- ``torch._inductor.config.compile_threads = 1``, so
    there is no pool to join -- saves the same seconds and is *not*
    equivalent: it also removes those 32 subprocesses from the host during
    the timed region, and this workload is host-dispatch bound. Measured at
    n=3 it cut the per-run standard deviation 3-11x on the dispatch-bound
    arms while the medians moved in both directions. That is a change to the
    measurement and needs a re-baseline. This one does not.

    The saving: 4.5-6 s of a 12-18 s worker across seven arms A/B'd back to
    back. **Every one of those numbers was measured on a box carrying load
    average 28-81 from concurrent agents, so each is uncitable and pending
    re-measurement on an idle box.** The reason to do this is the causal
    argument above, not the size of the number.

    **Two limits on the four facts, so the next reader does not inherit them
    as unconditional.**

    * The facts are scoped to today's torch. The exit skips ``atexit``
      *unconditionally*, so anything a later torch registers there is skipped
      too. Re-check the compile-cache fact after a torch bump.
    * **Do not copy this into the e2e training driver.** That process writes
      profiler traces, and a trace is written by machinery this argument has
      not been checked against. The cache check above was run for Inductor
      and Triton only.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    # libc's own streams, which the two flushes above do not reach and
    # ``os._exit`` does not flush. ``fflush(NULL)`` flushes every open output
    # stream. Deferred rather than imported at module scope, so an argument
    # error still returns without paying for it, and guarded because a
    # failure to flush a diagnostic must never change the exit code the
    # parent reads.
    try:
        import ctypes

        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError):
        pass
    os._exit(code)


if __name__ == "__main__":
    _exit_now(main())
