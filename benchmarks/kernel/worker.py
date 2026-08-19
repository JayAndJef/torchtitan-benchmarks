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
  for that one replicate.

The timing mode is why this file exists in this shape. One arm per process is
what keeps an arm's dependencies out of every other arm's interpreter: FA3
and TransformerEngine cannot share a process at all (a cuDNN soname
collision, see CLAUDE.md), and a build failure or a leaked CUDA context in
one arm cannot reach another.

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
    parser.add_argument("--fragment", required=True, type=Path)
    parser.add_argument("--arm", default=None, help="timing mode only")
    parser.add_argument(
        "--replicate", type=int, default=None, help="timing mode only"
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
    if args.mode == "timing" and (args.arm is None or args.replicate is None):
        parser.error("--mode timing requires --arm and --replicate")
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
    # re-execution and raises on arms that save intermediates (swiglu's
    # gate_up). Disabling it changes backward buffer reuse, not the
    # generated kernels, and applies to every arm alike.
    functorch_config.donated_buffer = False

    with phases.phase("import_engine"):
        from benchmarks.artifacts.layout import atomic_write_json
        from benchmarks.kernel.engine.run import (
            RunOptions,
            run_correctness_pass,
            run_timing_pass,
        )

    options = RunOptions(
        replicates=args.replicates,
        samples_per_replicate=args.samples_per_replicate,
        burst_k=args.burst_k,
        warmup_calls=args.warmup_calls,
        burst=args.burst,
        seed=args.seed,
    )
    try:
        if args.mode == "correctness":
            fragment = run_correctness_pass(
                scenario, shape, workload, options, frozenset(args.skip_arm)
            )
        else:
            fragment = run_timing_pass(
                scenario, args.arm, args.replicate, shape, workload, options
            )
    except Exception:
        traceback.print_exc()
        return 1

    # Provenance, not a result: the merge reads no phase and results.json
    # carries none. See this module's docstring.
    fragment["phases"] = phases.phases()
    atomic_write_json(args.fragment, fragment)
    print(f"fragment: {args.fragment}")
    for entry in fragment["phases"]:
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
      is what the two flushes below are for.
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
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    _exit_now(main())
