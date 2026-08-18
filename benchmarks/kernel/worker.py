"""GPU worker entry point for one kernel-benchmark scenario.

Launched by the CLI parent as ``numactl ... python -m benchmarks.kernel.worker``
inside the prepared environment (CUDA_VISIBLE_DEVICES, cache dirs, compiler
env). Writes results.json into the prepared output directory.

Exit codes: 0 success; 3 correctness gates failed (results.json still
written); 2 bad arguments; 1 build or environment failure.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="benchmarks.kernel.worker")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--burst", action="store_true")
    parser.add_argument("--model-size", default="normal")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    from torch._functorch import config as functorch_config

    # Backward-mode timing re-runs a compiled backward graph with
    # retain_graph=True; AOT autograd's donated-buffer optimization forbids
    # re-execution and raises on arms that save intermediates (swiglu's
    # gate_up). Disabling it changes backward buffer reuse, not the
    # generated kernels, and applies to every arm alike.
    functorch_config.donated_buffer = False

    from benchmarks.kernel.engine.run import RunOptions, run_kernel_scenario
    from benchmarks.kernel.results.schema import write_kernel_results

    options = RunOptions(
        n=args.n, warmup=args.warmup, burst=args.burst, seed=args.seed
    )
    try:
        result = run_kernel_scenario(
            scenario, shape, workload, options, args.hardware
        )
    except Exception:
        traceback.print_exc()
        return 1

    destination = write_kernel_results(result, args.out_dir / "results.json")
    print(f"results: {destination}")
    if not result.all_correctness_passed:
        failed = [
            f"{row.arm}.{row.output} {row.metric}={row.value:.4g}"
            f" (limit {row.threshold})"
            for row in result.correctness
            if row.passed is False
        ]
        print("correctness FAILED: " + "; ".join(failed), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
