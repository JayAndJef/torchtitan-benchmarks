#!/usr/bin/env bash
# Usage: ./run_bench.sh run-all <gpu-index> --scenario <name>

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$BENCH_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "run_bench.sh: no environment at $PYTHON; run 'uv sync'." >&2
    exit 1
fi

cd "$BENCH_DIR"
exec "$PYTHON" -u -m benchmarks.cli "$@"
