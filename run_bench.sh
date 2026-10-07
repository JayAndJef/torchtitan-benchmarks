#!/usr/bin/env bash
# Usage: ./run_bench.sh run <gpu-index> [--scenario <name>]

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$BENCH_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "run_bench.sh: no environment at $PYTHON; run 'uv sync'." >&2
    exit 1
fi

# cu130 wheels on a pre-r580 driver need the forward-compat userspace
# libcuda; this is a no-op on drivers that are already r580+.
source "$BENCH_DIR/cuda_compat.sh"

# A stable, writable datasets cache. HF_HOME can point at a directory another
# user owns, and every arm then dies on a builder.lock PermissionError before
# it trains one step. tools/run_matrix.sh has set this since it was written;
# a single run needs it for the same reason. An explicit HF_DATASETS_CACHE
# still wins, so an operator who has a writable shared cache keeps it.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/hf-datasets}"
mkdir -p "$HF_DATASETS_CACHE"

cd "$BENCH_DIR"
exec "$PYTHON" -u -m benchmarks.cli "$@"
