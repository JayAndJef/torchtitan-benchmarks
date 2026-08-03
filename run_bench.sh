#!/usr/bin/env bash
# Compatibility entry point for declarative end-to-end benchmark scenarios.
#
# Usage: ./run_bench.sh run-all <gpu-index> --scenario <name>
# Legacy usage remains supported: ./run_bench.sh <gpu-index> --scenario <name>

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TITAN_DIR="${TITAN_DIR:-$BENCH_DIR/../torchtitan}"
TITAN_PYTHON="${TITAN_PYTHON:-$TITAN_DIR/.venv/bin/python}"
cd "$BENCH_DIR"
exec "$TITAN_PYTHON" -u -m benchmarks.cli "$@"
