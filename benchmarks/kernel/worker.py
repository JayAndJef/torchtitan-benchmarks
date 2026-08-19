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

Every fragment carries a **phase table** -- named wall-clock spans from
process exec to the end of the pass. It is provenance, not a result: the
merge reads no phase and ``results.json`` carries none. It rides in the
fragment because the fragment is the one artifact a worker already writes,
and a phase table is worth nothing unless it survives the process that
produced it.
"""

from __future__ import annotations

import argparse
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


if __name__ == "__main__":
    sys.exit(main())
