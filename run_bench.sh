#!/usr/bin/env bash
# Usage: ./run_bench.sh run-all <gpu-index> --scenario <name>

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$BENCH_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "run_bench.sh: no environment at $PYTHON; run 'uv sync'." >&2
    exit 1
fi

# cu130 wheels on a pre-13.0 driver need the forward-compat userspace
# libcuda; this is a no-op on drivers that already report CUDA 13.0+.
source "$BENCH_DIR/cuda_compat.sh"

cd "$BENCH_DIR"
exec "$PYTHON" -u -m benchmarks.cli "$@"
