#!/usr/bin/env bash
# Three-arm RoPE benchmark: stock inductor vs Helion override vs TE override.
#
# This encodes the torchtitan profiling command used for the original
# qwen3_debugmodel RoPE traces (2026-07-30), defaulting to the piper-1B port:
#
#   NGPU=1 ./run_train.sh --module <M> --config <C> \
#       --training.seq-len <SEQ> --training.steps <STEPS> \
#       --compile.enable \
#       --profiler.enable_profiling --profiler.profile_freq 20 \
#       --profiler.profiler_active 5 --profiler.profiler_warmup 5 \
#       [--override.imports <target>]
#
# Usage:
#   ./run_bench.sh <gpu-index> [--arm baseline|helion|te] [extra torchtitan args...]
#
#   <gpu-index> is REQUIRED and is a PCI-order index (CUDA_DEVICE_ORDER is
#   forced to PCI_BUS_ID; the default CUDA ordering on this box is NOT PCI
#   order and silently lands on the wrong device). Check `nvidia-smi` first
#   and pick an IDLE GPU.
#
# Env overrides:
#   MODULE=piper1b CONFIG=qwen3_piper_1b   # or MODULE=qwen3 CONFIG=qwen3_debugmodel
#   SEQ=1024 STEPS=40 BATCH=4              # BATCH=1 reproduces piper's
#                                          #   per-microbatch RoPE shape (1,1024)
#   OUT=<dir>                              # default torchtitan-benchmarks/out/<UTC timestamp>
#
# Each arm dumps to $OUT/<arm>/ (torchtitan --dump-folder), so traces land in
# $OUT/<arm>/profiling/traces/iteration_{20,40}/rank0_trace.json.gz.
# Analyze with: python analysis/compare_arms.py $OUT

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TITAN_DIR="/data/zejiaqi/torchtitan"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <gpu-index> [--arm baseline|helion|te] [extra args...]" >&2
    exit 1
fi
GPU="$1"; shift

ARMS=(baseline helion te)
if [[ "${1:-}" == "--arm" ]]; then
    ARMS=("$2"); shift 2
fi

MODULE="${MODULE:-piper1b}"
CONFIG="${CONFIG:-qwen3_piper_1b}"
SEQ="${SEQ:-1024}"
STEPS="${STEPS:-40}"
BATCH="${BATCH:-4}"
OUT="${OUT:-$BENCH_DIR/out/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"

# Host compiler for the TE CUDA extension (C++20 headers need gcc >= 10; the
# system gcc is 8.5). Harmless for the other arms.
if [[ -f /opt/rh/gcc-toolset-13/enable ]]; then
    # shellcheck disable=SC1091
    source /opt/rh/gcc-toolset-13/enable
fi

override_for_arm() {
    case "$1" in
        baseline) echo "" ;;
        helion)   echo "torchtitan.overrides.helion_rope.helion_cos_sin_rope" ;;
        te)       echo "te_rope_override.te_rope" ;;
        *)        echo "unknown arm: $1" >&2; exit 1 ;;
    esac
}

echo "GPU (PCI index): $GPU"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
echo "arms: ${ARMS[*]}   module: $MODULE/$CONFIG   seq=$SEQ steps=$STEPS batch=$BATCH"
echo "output: $OUT"

for arm in "${ARMS[@]}"; do
    ovr="$(override_for_arm "$arm")"
    log="$OUT/$arm.log"
    echo
    echo "=== arm: $arm ==="
    args=(
        --module "$MODULE" --config "$CONFIG"
        --training.seq-len "$SEQ" --training.steps "$STEPS"
        --training.local-batch-size "$BATCH"
        --compile.enable
        --profiler.enable_profiling --profiler.profile_freq 20
        --profiler.profiler_active 5 --profiler.profiler_warmup 5
        --dump-folder "$OUT/$arm"
    )
    if [[ -n "$ovr" ]]; then
        args+=(--override.imports "$ovr")
    fi
    echo "run_train.sh ${args[*]} $*"

    # Full output goes to $log (no tee pipeline: it would swallow the training
    # exit code under pipefail). Follow along with: tail -f $log
    echo "[$arm] streaming log: $log"
    # Record GPU provenance in the preserved log, not just the console.
    {
        echo "# torchtitan-benchmarks arm=$arm gpu_pci_index=$GPU $(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
    } >"$log"
    rc=0
    (
        cd "$TITAN_DIR"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
        PYTHONPATH="$BENCH_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        PATH="$TITAN_DIR/.venv/bin:$PATH" \
        TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/data/zejiaqi/tmp/torch_extensions}" \
        TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/data/zejiaqi/tmp/inductor_cache}" \
        TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/zejiaqi/tmp/triton_cache}" \
        NGPU=1 ./run_train.sh "${args[@]}" "$@"
    ) >>"$log" 2>&1 || rc=$?
    grep -E "step: |Training completed" "$log" | tail -5 || true

    # Post-run gates: crashed, partial, or silently-fallen-back runs must fail
    # loudly, never be analyzed as a valid arm.
    if [[ "$rc" -ne 0 ]]; then
        echo "[$arm] FATAL: training exited with code $rc (see $log); aborting." >&2
        exit 1
    fi
    if ! grep -q "Training completed" "$log"; then
        echo "[$arm] FATAL: no 'Training completed' in log; aborting." >&2
        exit 1
    fi
    if [[ -n "$ovr" ]]; then
        n_ovr=$(grep -c "\[Override\]" "$log" || true)
        echo "[$arm] override applications in log: $n_ovr (expect one per layer)"
        if [[ "$n_ovr" -eq 0 ]]; then
            echo "[$arm] FATAL: override did not apply; aborting." >&2
            exit 1
        fi
    fi
    if grep -q "falling back to the PyTorch" "$log"; then
        echo "[$arm] FATAL: override fell back to the PyTorch path; the arm" \
             "did not measure its kernel. See $log." >&2
        exit 1
    fi
    # Both profiler windows must exist (freq 20, STEPS>=40 -> iteration_20+40),
    # and override arms must actually contain their marker kernel in the trace.
    for it in iteration_20 iteration_40; do
        tr=$(ls "$OUT/$arm"/profiling/traces*/$it/rank0_trace.json.gz 2>/dev/null | head -1)
        if [[ -z "$tr" ]]; then
            echo "[$arm] FATAL: missing $it trace; aborting." >&2
            exit 1
        fi
    done
    marker=""
    case "$arm" in
        helion) marker="_helion__rope_cos_sin_fwd" ;;
        te)     marker="fused_rope_forward_kernel" ;;
    esac
    if [[ -n "$marker" ]]; then
        tr=$(ls "$OUT/$arm"/profiling/traces*/iteration_20/rank0_trace.json.gz | head -1)
        if ! zgrep -q "$marker" "$tr"; then
            echo "[$arm] FATAL: marker kernel '$marker' absent from $tr;" \
                 "the override kernel did not run. Aborting." >&2
            exit 1
        fi
    fi
done

echo
echo "All arms done. Analyze with:"
echo "  $TITAN_DIR/.venv/bin/python $BENCH_DIR/analysis/compare_arms.py $OUT"
