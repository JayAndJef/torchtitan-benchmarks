#!/usr/bin/env bash
# One cell of a benchmark matrix, inside the Slurm job of the matrix.
#
# tools/matrix_job.sbatch calls this file once per cell. This file runs the
# checks, the load gate, the watchdog and the `run` command inside the job,
# because only there does nvidia-smi see the cards of the job.
#
# Usage:
#   tools/run_matrix_cell.sh <root> <name> run <devices> <run flags>
#
# <root> is the output root of the matrix, and <name> names the cell. The
# cell runs `./run_bench.sh run <devices> <run flags> --out <root>/<name>`.
# The script adds --out itself, so it refuses --out and --resume in the
# flags. Inside a job, Slurm numbers the cards of the job from 0, so the
# device list is job-local, for example 0,1,2,3.
#
# A cell is done when <root>/<name>/results.json exists and <name>.status
# holds an OK state. The script skips a done cell. So a resubmit of the same
# job file runs only the cells that are not OK. A rerun always starts with
# --out, never with --resume.
#
# Environment (MATRIX_REV is required; the others have defaults):
#   MATRIX_REV       the commit that the job must run, `git rev-parse HEAD`
#                    at submission
#   IDLE_LOAD        the 1-minute load average that the load gate waits for (80)
#   IDLE_SETTLE      consecutive idle samples that open the gate (3)
#   IDLE_POLL        seconds between two load gate samples (20)
#   IDLE_MAX_WAIT    seconds after which the cell runs anyway, flagged (1200)
#   CONTENDED_LOAD   the 1-minute load average that condemns the cell (150)
#   FOREIGN_MEM_MIB  GPU memory that no PID of ours explains, which condemns
#                    the cell (2000)
#   WATCH_INTERVAL   the watchdog sample period in seconds (15)
#   MATRIX_LOADAVG   the file that gives the load average (/proc/loadavg)
#
# Outputs, in <root>:
#   sweep.log      one log for every cell; each outcome writes one line
#                  `STATUS <name> <state>`
#   <name>.status  the state of the last attempt
#   <name>.log     the output of `run_bench.sh`
#   <name>.watch   one line per suspicious sample; a non-empty file condemns the cell
#   <name>.rc      the exit code of `run_bench.sh`; absent when the job ended early
#
# The states:
#   OK               the run passed, and the watchdog flagged nothing
#   OK(existing:S)   the cell was done with the state S; nothing ran. This
#                    state goes to sweep.log alone, and <name>.status keeps S
#   OK(load-flagged) as OK, but the load gate timed out before the run;
#                    check the step times by hand before you report the cell
#   FAIL(rc=N)       the run exited with N; the script renames the cell as failed
#   FAIL(no-results) the run exited with 0 but wrote no results.json; the
#                    script renames the cell as failed
#   CONTAMINATED     the watchdog flagged the cell, or it could not read the
#                    cards; the script renames the cell as contaminated
#   PLACEMENT        the job holds no CPU on the NUMA node of some card, or
#                    the node of some card is unknown; nothing ran
#   ERROR            a check failed; nothing ran
#
# The script renames a cell to <name>.failed-<stamp>* or
# <name>.contaminated-<stamp>*: the directory, the log, the .watch and the
# .rc. The stamp is the UTC time and the job id, and the script refuses a
# rename onto a name that exists. A partial cell of an earlier attempt, or a
# results.json without an OK status, is renamed as failed before the run.
# The script exits 0 for an OK state alone.
set -uo pipefail

die() { echo "run_matrix_cell.sh: $*" >&2; exit 2; }

[ "$#" -ge 4 ] || die "usage: run_matrix_cell.sh <root> <name> run <devices> <run flags>"
[ -d "$1" ] || die "the output root '$1' is not a directory"
ROOT="$(cd "$1" && pwd)"
NAME="$2"
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || die "the cell name '$NAME' must match [A-Za-z0-9_-]+"
shift 2

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2

LOG="$ROOT/sweep.log"
OUT="$ROOT/$NAME"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
# Writes the one STATUS line of this attempt and the .status file.
finish() { echo "$1" >"$OUT.status"; say "STATUS $NAME $1"; }
fail() { say "  $NAME: CELL-ERROR $*"; finish ERROR; exit 2; }

[ "$1" = run ] || fail "the arguments after the name must start with: run <devices>"
DEVICES="$2"
[[ "$DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "the device list '$DEVICES' must be indices and commas"
for word in "$@"; do
    case "$word" in
        --out|--out=*|--resume|--resume=*)
            fail "the flags hold $word; the script adds --out <root>/<name> itself" ;;
    esac
done

[ -n "${MATRIX_REV:-}" ] || fail "MATRIX_REV is not set; submit with MATRIX_REV=\$(git rev-parse HEAD)"
IDLE_LOAD="${IDLE_LOAD:-80}"
IDLE_SETTLE="${IDLE_SETTLE:-3}"
IDLE_POLL="${IDLE_POLL:-20}"
IDLE_MAX_WAIT="${IDLE_MAX_WAIT:-1200}"
CONTENDED_LOAD="${CONTENDED_LOAD:-150}"
FOREIGN_MEM_MIB="${FOREIGN_MEM_MIB:-2000}"
WATCH_INTERVAL="${WATCH_INTERVAL:-15}"
MATRIX_LOADAVG="${MATRIX_LOADAVG:-/proc/loadavg}"
for setting in IDLE_LOAD IDLE_SETTLE IDLE_POLL IDLE_MAX_WAIT CONTENDED_LOAD \
                FOREIGN_MEM_MIB WATCH_INTERVAL; do
    [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] \
        || fail "$setting must be a positive integer, not '${!setting}'"
done
[ -r "$MATRIX_LOADAVG" ] || fail "MATRIX_LOADAVG '$MATRIX_LOADAVG' is not readable"

# The .status file keeps the state of the run, so OK(load-flagged) stays.
if [ -f "$OUT/results.json" ]; then
    previous="$(cat "$OUT.status" 2>/dev/null)"
    case "$previous" in
        OK*) say "STATUS $NAME OK(existing:$previous)"; exit 0 ;;
    esac
fi

[ -n "${SLURM_JOB_ID:-}" ] || fail "this file runs only inside a Slurm job"

# hardware_metadata records `git rev-parse HEAD`. The queue can hold a job
# for hours, so the job checks again that the tree is the one at submission.
rev="$(git -C "$REPO" rev-parse HEAD)"
[ "$rev" = "$MATRIX_REV" ] || fail "HEAD is $rev, but the job was submitted at $MATRIX_REV"
[ -z "$(git -C "$REPO" status --porcelain)" ] || fail "the working tree is dirty"

# The job-local indices are correct only when Slurm hides the other cards.
command -v nvidia-smi >/dev/null || fail "nvidia-smi is not on PATH"
listed="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>&1)" \
    || fail "nvidia-smi --query-gpu=index failed: $listed"
mapfile -t GPU_IDS < <(tr -d ' ' <<<"$listed" | grep .)
IFS=',' read -r -a wanted <<<"$DEVICES"
[ "${#GPU_IDS[@]}" = "${#wanted[@]}" ] \
    || fail "nvidia-smi sees ${#GPU_IDS[@]} cards (${GPU_IDS[*]}), but the device list $DEVICES names ${#wanted[@]}"
declare -A named=()
for device in "${wanted[@]}"; do
    [ -z "${named[$device]:-}" ] || fail "the device list $DEVICES names card $device twice"
    named[$device]=1
    printf '%s\n' "${GPU_IDS[@]}" | grep -qx "$device" \
        || fail "nvidia-smi sees no card $device; it sees ${GPU_IDS[*]}"
done
buses="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>&1)" \
    || fail "nvidia-smi --query-gpu=pci.bus_id failed: $buses"

# The harness binds each rank to the NUMA node of its card. That node can
# hold few or none of the CPUs of the job, so the log records the overlap.
expand_cpulist() {   # "0-3,8" -> one CPU number per line
    local part
    local -a parts
    IFS=',' read -r -a parts <<<"$1"
    for part in "${parts[@]}"; do
        case "$part" in
            *-*) seq "${part%-*}" "${part#*-}" ;;
            ?*) echo "$part" ;;
        esac
    done
}
cpu_report() {
    local allowed bus node node_cpus overlap
    allowed="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)"
    echo "allowed=$allowed"
    while IFS=',' read -r bus; do
        bus="$(echo "$bus" | tr -d ' ' | tr '[:upper:]' '[:lower:]')"
        node="$(cat "/sys/bus/pci/devices/${bus:4}/numa_node" 2>/dev/null || echo '?')"
        if [ -r "/sys/devices/system/node/node$node/cpulist" ]; then
            node_cpus="$(cat "/sys/devices/system/node/node$node/cpulist")"
            overlap="$(comm -12 <(expand_cpulist "$allowed" | sort) \
                                <(expand_cpulist "$node_cpus" | sort) | grep -c .)"
            echo "${bus:4}->node$node:${overlap}cpus"
        else
            echo "${bus:4}->node?"
        fi
    done <<<"$buses"
}

SUPERVISOR_PID=$$
ME=$(id -un)

cpus="$(cpu_report)"
say "  $NAME: job $SLURM_JOB_ID on $(hostname -s), partition ${SLURM_JOB_PARTITION:-?}"
say "  $NAME: gpu $(nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv,noheader 2>&1 | tr '\n' ';')"
say "  $NAME: cpu $(echo "$cpus" | tr '\n' ' ')"
say "  $NAME: numactl $(command -v numactl || echo 'NOT AVAILABLE (runs will be unpinned)')"
say "  $NAME: hf cache ${HF_DATASETS_CACHE:-unset; run_bench.sh sets its default}"

# A card whose node holds none of the CPUs of the job runs unpinned, and an
# unpinned rank measures scheduler placement. So does a card whose node is
# unknown. A partial overlap runs.
if grep -Eq -- '->node([0-9]+:0cpus|\?)$' <<<"$cpus"; then
    say "  $NAME: the job holds no CPU on the NUMA node of some card, or the node is unknown"
    finish PLACEMENT
    exit 3
fi

load1() { awk '{printf "%.0f", $1}' "$MATRIX_LOADAVG"; }

# The workload is host-bound, so the cell waits for an idle host. After
# IDLE_MAX_WAIT seconds it runs anyway, and an OK becomes OK(load-flagged).
load_flagged=0
idle=0
load_max=0
gate_start=$SECONDS
while :; do
    load=$(load1)
    [ "$load" -gt "$load_max" ] && load_max=$load
    if [ "$load" -le "$IDLE_LOAD" ]; then
        idle=$((idle + 1))
        [ "$idle" -ge "$IDLE_SETTLE" ] && break
    else
        idle=0
    fi
    if [ $((SECONDS - gate_start)) -ge "$IDLE_MAX_WAIT" ]; then
        say "  $NAME: LOAD-GATE-TIMEOUT load1=$load max=$load_max (limit $IDLE_LOAD, waited $((SECONDS - gate_start))s)"
        load_flagged=1
        break
    fi
    sleep "$IDLE_POLL"
done
[ "$load_flagged" -eq 1 ] \
    || say "  $NAME: load gate open after $((SECONDS - gate_start))s, load1=$load"

# Renames the directory, the log, the .watch and the .rc of the cell. A
# rename onto a name that exists would nest a directory or replace a log.
rename_cell() {   # $1 is failed or contaminated
    local kind="$1" stamp suffix
    stamp="$(date -u +%Y%m%dT%H%M%SZ)-j$SLURM_JOB_ID"
    for suffix in "" .log .watch .rc; do
        if [ -e "$OUT$suffix" ] && [ -e "$OUT.$kind-$stamp$suffix" ]; then
            fail "$OUT.$kind-$stamp$suffix exists, so $OUT$suffix keeps its name." \
                 "This collision is a coincidence: a resubmit clears it, because the next job id gives a new name"
        fi
    done
    for suffix in "" .log .watch .rc; do
        if [ -e "$OUT$suffix" ]; then
            mv -T "$OUT$suffix" "$OUT.$kind-$stamp$suffix" \
                || fail "mv could not rename $OUT$suffix"
        fi
    done
    say "  $NAME: renamed the cell to $OUT.$kind-$stamp"
}

# A job that ended inside this cell left a partial directory, and `run`
# refuses an --out that exists. A results.json without an OK status was
# never sorted. Both are renamed as a failed attempt.
if [ -e "$OUT" ] || [ -e "$OUT.log" ]; then
    say "  $NAME: an earlier attempt left $OUT without an OK status"
    rename_cell failed
fi

say "  $NAME: ./run_bench.sh $* --out $OUT"

# Tells whether a compute PID is ours. The test follows the parent PIDs up
# to the batch script. It does not use the session id. Torch elastic starts
# each worker with start_new_session=True, so each worker is the leader of
# its own session. A session test therefore flagged every arm as foreign.
# setsid changes the session of a process, but it never changes the parent.
is_ours() {
    local pid="$1" hops=0
    while [ -n "$pid" ] && [ "$pid" != "1" ] && [ "$pid" != "0" ] \
          && [ "$hops" -lt 32 ]; do
        [ "$pid" = "$SUPERVISOR_PID" ] && return 0
        pid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
        hops=$((hops + 1))
    done
    return 1
}

gpu_mem() {   # summed over every card of the job; -1 when unreadable
    local total=0 value id
    for id in "${GPU_IDS[@]}"; do
        value=$(nvidia-smi --id="$id" --query-gpu=memory.used \
                --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        case "$value" in
            ''|*[!0-9]*) echo "-1"; return ;;
        esac
        total=$((total + value))
    done
    echo "$total"
}

# Writes one line per suspicious sample to $1. A non-empty file condemns the
# cell. nvidia-smi does not always show the PIDs of other users, so the
# watchdog uses two independent signals:
#   FOREIGN_PID  a compute PID that is not a descendant of the batch script
#   FOREIGN_MEM  GPU memory that no PID of ours explains
# A failed query writes WATCH-BLIND, because a blind watchdog proves nothing.
# The load average is host-wide, so it also includes the jobs of other users.
# The workload is host-bound, so host load corrupts tokens/s.
watchdog() {
    local watch_file="$1"
    local mem load pid used ours residual id mem_streak=0 noted=0
    while :; do
        mem=$(gpu_mem); load=$(load1)
        [ "$mem" = "-1" ] && \
            echo "WATCH-BLIND memory.used is unreadable $(date -u +%T)" >>"$watch_file"
        ours=0
        for id in "${GPU_IDS[@]}"; do
            if ! apps="$(nvidia-smi --id="$id" --query-compute-apps=pid,used_memory \
                         --format=csv,noheader 2>&1)"; then
                echo "WATCH-BLIND compute-apps gpu=$id: $(head -n 1 <<<"$apps") $(date -u +%T)" \
                    >>"$watch_file"
                continue
            fi
            while IFS=',' read -r pid used; do
                pid=$(echo "$pid" | tr -d ' ')
                used=$(echo "$used" | tr -d ' MiB')
                [ -z "$pid" ] && continue
                case "$used" in ''|*[!0-9]*) used=0 ;; esac
                if is_ours "$pid"; then
                    ours=$((ours + used))
                elif ps -o pid= -p "$pid" >/dev/null 2>&1; then
                    # A live PID outside our tree is foreign. The test skips
                    # a PID that has exited, because that PID is usually one
                    # of our arms at shutdown. FOREIGN_MEM still finds real
                    # foreign use.
                    echo "FOREIGN_PID gpu=$id pid=$pid user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')" \
                         "sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ') mem=${used}MiB $(date -u +%T)" \
                        >>"$watch_file"
                fi
            done <<<"$apps"
        done
        if [ "$noted" -eq 0 ] && [ "$ours" -gt 0 ]; then
            # This line is information, not a flag. It shows that the
            # attribution works for this cell. It goes to the sweep log only.
            say "  $NAME: watchdog attributes ${ours}MiB to our own arms (user $ME)"
            noted=1
        fi
        if [ "$mem" != "-1" ]; then
            residual=$((mem - ours))
            if [ "$residual" -gt "$FOREIGN_MEM_MIB" ]; then
                # The flag needs two consecutive samples. One sample can
                # fall after one of our arms exits and before nvidia-smi
                # releases its memory.
                if [ "$mem_streak" -ge 1 ]; then
                    echo "FOREIGN_MEM used=${mem}MiB ours=${ours}MiB residual=${residual}MiB $(date -u +%T)" \
                        >>"$watch_file"
                fi
                mem_streak=$((mem_streak + 1))
            else
                mem_streak=0
            fi
        fi
        [ "$load" -gt "$CONTENDED_LOAD" ] && \
            echo "CONTENDED load1=${load} (limit ${CONTENDED_LOAD}) $(date -u +%T)" \
                >>"$watch_file"
        sleep "$WATCH_INTERVAL"
    done
}

watch_file="$OUT.watch"
: >"$watch_file"
watchdog "$watch_file" &
watch_pid=$!
trap 'kill "$watch_pid" 2>/dev/null' EXIT

./run_bench.sh "$@" --out "$OUT" >>"$OUT.log" 2>&1
rc=$?
kill "$watch_pid" 2>/dev/null; wait "$watch_pid" 2>/dev/null

# Memory that stays on the cards after our processes end is foreign.
after=$(gpu_mem)
if [ "$after" = "-1" ]; then
    echo "WATCH-BLIND memory.used is unreadable after the run $(date -u +%T)" >>"$watch_file"
elif [ "$after" -gt "$FOREIGN_MEM_MIB" ]; then
    echo "RESIDUAL_MEM after=${after}MiB (limit ${FOREIGN_MEM_MIB}) $(date -u +%T)" >>"$watch_file"
fi

echo "$rc" >"$OUT.rc"
say "  $NAME: end rc=$rc gpu after=${after}MiB"

# A contaminated cell still writes a results.json, so the bad numbers stay
# unless the script renames the cell. The next submission runs it again.
if [ -s "$watch_file" ]; then
    head -3 "$watch_file" | sed 's/^/    /' | tee -a "$LOG"
    rename_cell contaminated
    finish CONTAMINATED
    exit 1
fi
if [ "$rc" -ne 0 ]; then
    tail -5 "$OUT.log" | sed 's/^/    /' | tee -a "$LOG"
    rename_cell failed
    finish "FAIL(rc=$rc)"
    exit "$rc"
fi
if [ ! -f "$OUT/results.json" ]; then
    rename_cell failed
    finish "FAIL(no-results)"
    exit 1
fi

# An evaluation warning can show contamination. So the sweep log repeats
# each warning of results.json.
warn=$(.venv/bin/python -c "
import json,sys
w=json.load(open(sys.argv[1])).get('warnings') or []
print('; '.join(w))" "$OUT/results.json") \
    || { rename_cell failed; fail "the warnings of results.json are not readable"; }
[ -n "$warn" ] && say "  $NAME: results.json warnings: $warn"
if [ "$load_flagged" -eq 1 ]; then
    finish "OK(load-flagged)"
else
    finish OK
fi
exit 0
