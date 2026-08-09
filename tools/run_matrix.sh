#!/usr/bin/env bash
# Supervisor for a multi-cell benchmark matrix on a SHARED box.
#
# Why it exists
# -------------
# `run-all --all-scenarios` is fail-fast and refuses --resume, so one foreign
# job landing mid-sweep throws away every scenario that already succeeded.
# This runs one (size, ac, mode, scenario) cell per invocation under a shared
# --out root, so a failure costs one cell and --resume picks it up.
#
# More importantly, this box is shared with jobs that CYCLE: they appear,
# take ~100 GB/GPU and drive the 1-minute load average past 400, then leave.
# A post-hoc "is the GPU busy now?" check misses them entirely, and a
# contaminated cell still produces a results.json, so the bad numbers become
# permanent. Every cell therefore runs under a watchdog that samples during
# the run; anything it flags moves the output aside so the next pass redoes
# the cell from scratch rather than keeping it.
#
# The workload is host-bound at these sizes, so host contention corrupts
# tokens/s even when we have the GPU to ourselves. Load is a first-class
# contamination signal, not a nicety.
#
# Usage
# -----
#   GPU=4 nohup ./tools/run_matrix.sh > /dev/null 2>&1 &
#   tail -f out/matrix-<utc>/sweep.log
#
# Environment (all optional except GPU):
#   GPU              PCI index; one GPU for the whole run (required)
#   ROOT             output root (default out/matrix-<utc>)
#   PASSES           retry passes over failed/contaminated cells (default 3)
#   STEPS            training steps per arm (default 80)
#   CELLS            "huge", "normal", or "all" (default all; huge runs first)
#   IDLE_MEM_MIB     GPU memory below which the card counts as idle (2000)
#   IDLE_LOAD        1-min loadavg below which the host counts as idle (60)
#   CONTENDED_LOAD   1-min loadavg during a cell that means contention (150)
#   WAIT_TIMEOUT     seconds to wait for idle before skipping a cell (43200)
#   WATCH_INTERVAL   watchdog sample period in seconds (30)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

GPU="${GPU:-}"
ROOT="${ROOT:-out/matrix-$(date -u +%Y%m%dT%H%M%SZ)}"
PASSES="${PASSES:-3}"
STEPS="${STEPS:-80}"
CELLS="${CELLS:-all}"
IDLE_MEM_MIB="${IDLE_MEM_MIB:-2000}"
IDLE_LOAD="${IDLE_LOAD:-60}"
CONTENDED_LOAD="${CONTENDED_LOAD:-150}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-43200}"
WATCH_INTERVAL="${WATCH_INTERVAL:-30}"

# A stable, writable datasets cache. The shared HF_HOME is owned by another
# user and every arm dies on a builder.lock PermissionError without this.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/hf-datasets}"
mkdir -p "$HF_DATASETS_CACHE"

# --out is passed explicitly per cell; an inherited OUT would silently
# redirect every one of them into the same directory.
unset OUT

mkdir -p "$ROOT"
LOG="$ROOT/sweep.log"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- preflight
fail() { echo "run_matrix.sh: $*" >&2; exit 1; }

[ -n "$GPU" ] || fail "GPU is required (one PCI index for the whole matrix)"
[ -x "$REPO/.venv/bin/python" ] || fail "no environment at $REPO/.venv/bin/python"
# hardware_metadata records `git rev-parse HEAD`, which silently ignores
# uncommitted edits, so a dirty tree would mislabel the whole matrix.
[ -z "$(git -C "$REPO" status --porcelain)" ] || fail "working tree is dirty; commit first"
command -v flock >/dev/null || fail "flock is required (single-instance lock)"

exec 9>"$ROOT/.lock"
flock -n 9 || fail "another run_matrix.sh already holds $ROOT/.lock"

SID=$(ps -o sid= -p $$ | tr -d ' ')

say "=== run_matrix.sh ==="
say "repo:        $REPO"
say "git rev:     $(git -C "$REPO" rev-parse HEAD)"
say "gpu:         $GPU -> $(nvidia-smi --id="$GPU" --query-gpu=index,name,uuid,driver_version --format=csv,noheader 2>&1)"
say "numactl:     $(command -v numactl || echo 'NOT AVAILABLE (runs will be unpinned)')"
say "steps:       $STEPS"
say "passes:      $PASSES"
say "root:        $ROOT"
say "session id:  $SID (foreign = any compute PID outside it)"
say "hf cache:    $HF_DATASETS_CACHE"

# ------------------------------------------------------------- the cell list
# "size|ac|mode|scenario". Huge first: it is the new measurement and the one
# the report is built around.
CELL_LIST=()
if [ "$CELLS" = all ] || [ "$CELLS" = huge ]; then
    for mode in default cuda-graph; do
        for scenario in piper1b_megatron piper1b_attention; do
            CELL_LIST+=("huge|none|$mode|$scenario")
        done
    done
fi
if [ "$CELLS" = all ] || [ "$CELLS" = normal ]; then
    for ac in sac none; do
        for mode in default cuda-graph; do
            for scenario in piper1b_rope piper1b_swiglu piper1b_qkv \
                            piper1b_lm_head piper1b_attention piper1b_megatron; do
                # megatron declares supported_ac_modes=("none",); the CLI
                # errors rather than skipping on a direct --scenario request.
                [ "$scenario" = piper1b_megatron ] && [ "$ac" != none ] && continue
                CELL_LIST+=("normal|$ac|$mode|$scenario")
            done
        done
    done
fi
say "cells:       ${#CELL_LIST[@]}"
for cell in "${CELL_LIST[@]}"; do say "  $cell"; done

# --------------------------------------------------------------- idle gating
gpu_mem() {
    local value
    value=$(nvidia-smi --id="$GPU" --query-gpu=memory.used \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    # An unparseable answer is a broken query, not a busy GPU: returning a
    # huge number here would stall the sweep for the whole timeout.
    case "$value" in
        ''|*[!0-9]*) echo "-1" ;;
        *) echo "$value" ;;
    esac
}
load1() { awk '{printf "%.0f", $1}' /proc/loadavg; }

wait_for_idle() {   # -> 0 idle, 1 timed out
    local waited=0 mem load
    while :; do
        mem=$(gpu_mem); load=$(load1)
        if [ "$mem" = "-1" ]; then
            say "  nvidia-smi query failed; proceeding (cannot distinguish busy)"
            return 0
        fi
        [ "$mem" -le "$IDLE_MEM_MIB" ] && [ "$load" -le "$IDLE_LOAD" ] && return 0
        [ $((waited % 300)) -eq 0 ] && \
            say "  waiting for idle: gpu${GPU}=${mem}MiB load=${load}"
        sleep 60; waited=$((waited + 60))
        if [ "$waited" -ge "$WAIT_TIMEOUT" ]; then return 1; fi
    done
}

# ----------------------------------------------------------------- watchdog
# Writes one line per suspicious sample to $1. A non-empty file condemns the
# cell. Two independent foreign-usage signals, because nvidia-smi does not
# always expose other users' PIDs:
#   FOREIGN_PID  a compute PID outside our session id
#   FOREIGN_MEM  GPU memory that no PID of ours accounts for
watchdog() {
    local watch_file="$1"
    local mem load pid used ours residual line
    while :; do
        mem=$(gpu_mem); load=$(load1)
        ours=0
        while IFS=',' read -r pid used; do
            pid=$(echo "$pid" | tr -d ' ')
            used=$(echo "$used" | tr -d ' MiB')
            [ -z "$pid" ] && continue
            case "$used" in ''|*[!0-9]*) used=0 ;; esac
            local sid
            sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')
            if [ -n "$sid" ] && [ "$sid" = "$SID" ]; then
                ours=$((ours + used))
            else
                echo "FOREIGN_PID pid=$pid sid=${sid:-unknown} mem=${used}MiB $(date -u +%T)" \
                    >>"$watch_file"
            fi
        done < <(nvidia-smi --id="$GPU" --query-compute-apps=pid,used_memory \
                 --format=csv,noheader 2>/dev/null)
        if [ "$mem" != "-1" ]; then
            residual=$((mem - ours))
            [ "$residual" -gt "$IDLE_MEM_MIB" ] && \
                echo "FOREIGN_MEM used=${mem}MiB ours=${ours}MiB residual=${residual}MiB $(date -u +%T)" \
                    >>"$watch_file"
        fi
        [ "$load" -gt "$CONTENDED_LOAD" ] && \
            echo "CONTENDED load1=${load} (limit ${CONTENDED_LOAD}) $(date -u +%T)" \
                >>"$watch_file"
        sleep "$WATCH_INTERVAL"
    done
}

# --------------------------------------------------------------- the sweep
declare -A STATUS
for cell in "${CELL_LIST[@]}"; do STATUS["$cell"]="PENDING"; done

for pass in $(seq 1 "$PASSES"); do
    say ""
    say "########## pass $pass/$PASSES ##########"
    remaining=0
    for cell in "${CELL_LIST[@]}"; do
        IFS='|' read -r size ac mode scenario <<<"$cell"
        out="$ROOT/$size/ac-$ac/$mode/$scenario"
        # The marker lives NEXT TO the directory, so it survives an early
        # failure that never created the directory at all.
        marker="$out.CONTAMINATED"

        if [ -f "$out/results.json" ] && [ ! -f "$marker" ]; then
            [ "${STATUS[$cell]}" = "PENDING" ] && STATUS["$cell"]="OK(pre-existing)"
            [ "$pass" -eq 1 ] && say "SKIP $cell (results.json present)"
            continue
        fi
        remaining=$((remaining + 1))

        mkdir -p "$(dirname "$out")"
        if ! wait_for_idle; then
            say "GAVE-UP $cell (no idle GPU within ${WAIT_TIMEOUT}s)"
            STATUS["$cell"]="SKIPPED-IDLE-TIMEOUT"
            continue   # never abort the whole sweep on one timeout
        fi

        say "RUN  $cell  (gpu=$(gpu_mem)MiB load=$(load1))  -> $out"
        rm -f "$marker"
        watch_file="$out.watch"
        : >"$watch_file"

        args=(run-all "$GPU" --scenario "$scenario" --ac "$ac"
              --compile-mode "$mode" --model-size "$size" --steps "$STEPS")
        if [ -f "$out/manifest.json" ]; then
            # Gate on the manifest, not the directory: a crash between mkdir
            # and write_manifest leaves a directory --resume cannot use.
            args+=(--resume "$out")
        else
            if [ -e "$out" ]; then
                stamp=$(date -u +%Y%m%dT%H%M%SZ)
                say "  moving manifest-less $out aside -> $out.nomanifest-$stamp"
                mv "$out" "$out.nomanifest-$stamp"
            fi
            args+=(--out "$out")
        fi

        watchdog "$watch_file" &
        watch_pid=$!
        ./run_bench.sh "${args[@]}" >>"$out.log" 2>&1
        rc=$?
        kill "$watch_pid" 2>/dev/null; wait "$watch_pid" 2>/dev/null

        after=$(gpu_mem)
        if [ -s "$watch_file" ] || { [ "$after" != "-1" ] && [ "$after" -gt "$IDLE_MEM_MIB" ]; }; then
            say "CONTAMINATED $cell rc=$rc (gpu after=${after}MiB)"
            head -3 "$watch_file" | sed 's/^/    /' | tee -a "$LOG"
            stamp=$(date -u +%Y%m%dT%H%M%SZ)
            {
                echo "contaminated at $stamp; rc=$rc; gpu after=${after}MiB"
                cat "$watch_file"
            } >"$marker"
            [ -e "$out" ] && mv "$out" "$out.contaminated-$stamp"
            [ -e "$out.log" ] && mv "$out.log" "$out.contaminated-$stamp.log"
            STATUS["$cell"]="CONTAMINATED"
        elif [ $rc -ne 0 ]; then
            say "FAIL $cell rc=$rc (retried next pass; see $out.log)"
            tail -5 "$out.log" | sed 's/^/    /' | tee -a "$LOG"
            STATUS["$cell"]="FAIL(rc=$rc)"
        else
            say "OK   $cell"
            STATUS["$cell"]="OK"
            # The launch-latency-spread warning is an in-band contamination
            # signal; surface it rather than leaving it in results.json.
            if [ -f "$out/results.json" ]; then
                warn=$(.venv/bin/python -c "
import json,sys
w=json.load(open(sys.argv[1])).get('warnings') or []
print('; '.join(w))" "$out/results.json" 2>/dev/null)
                [ -n "$warn" ] && say "  results.json warnings: $warn"
            fi
        fi
    done
    [ "$remaining" -eq 0 ] && { say "nothing left to run"; break; }
done

# ---------------------------------------------------------------- summary
say ""
say "########## summary ##########"
bad=0
for cell in "${CELL_LIST[@]}"; do
    IFS='|' read -r size ac mode scenario <<<"$cell"
    out="$ROOT/$size/ac-$ac/$mode/$scenario"
    say "$(printf '%-24s' "${STATUS[$cell]}") $cell  $out"
    case "${STATUS[$cell]}" in OK|OK\(pre-existing\)) ;; *) bad=$((bad + 1)) ;; esac
done
say "cells not OK: $bad / ${#CELL_LIST[@]}"
say "SWEEP COMPLETE -> $ROOT"
exit $(( bad > 0 ? 1 : 0 ))
