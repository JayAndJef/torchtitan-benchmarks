#!/usr/bin/env bash
# Compatibility entry point for declarative end-to-end benchmark scenarios.
#
# Usage:
#   ./run_bench.sh <gpu-index> [--scenario piper1b_rope|piper1b_swiglu] \
#       [--arm <name>] [--hardware <label>] [-- extra torchtitan args...]

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BENCH_DIR"
exec python3 -u -m benchmarks.runner "$@"
