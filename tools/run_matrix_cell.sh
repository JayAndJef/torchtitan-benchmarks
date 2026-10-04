#!/usr/bin/env bash
# One cell of tools/run_matrix.sh, inside the Slurm job of that cell.
#
# The driver submits a short batch script that sets the environment below
# and execs this file. This file runs the watchdog and the `run` command
# inside the job, because only there does nvidia-smi see the cards of the
# job. Do not start it by hand.
#
# The arguments are the `run_bench.sh` arguments of the cell. They start
# with `run` and the job-local device list, for example `run 0,1 ...`.
#
# Environment (all required; the batch script sets each one):
#   MATRIX_LOG       the sweep log of the matrix
#   MATRIX_OUT       the output directory of the cell
#   MATRIX_REV       the commit that the driver started from
#   MATRIX_NGPUS     the number of cards that the job holds
#   FOREIGN_MEM_MIB  GPU memory that no PID of ours explains, which flags the cell
#   CONTENDED_LOAD   the 1-minute load average that flags the cell
#   WATCH_INTERVAL   the watchdog sample period in seconds
#
# Outputs, next to the cell directory:
#   <cell>.log    the output of `run_bench.sh`
#   <cell>.watch  one line per suspicious sample; a non-empty file condemns the cell
#   <cell>.rc     the exit code of `run_bench.sh`; absent when the job ended early
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 2

for name in MATRIX_LOG MATRIX_OUT MATRIX_REV MATRIX_NGPUS FOREIGN_MEM_MIB \
            CONTENDED_LOAD WATCH_INTERVAL; do
    if [ -z "${!name:-}" ]; then
        echo "run_matrix_cell.sh: $name is not set; tools/run_matrix.sh sets it" >&2
        exit 2
    fi
done

LOG="$MATRIX_LOG"
OUT="$MATRIX_OUT"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
fail() { say "  job ${SLURM_JOB_ID:-?}: CELL-ERROR $*"; exit 2; }

[ -n "${SLURM_JOB_ID:-}" ] || fail "this file runs only inside a Slurm job"
[ "$#" -ge 2 ] && [ "$1" = run ] || fail "the arguments must start with: run <devices>"

# hardware_metadata records `git rev-parse HEAD`. The queue can hold a job
# for hours, so the job checks again that the tree is the one the driver saw.
rev="$(git -C "$REPO" rev-parse HEAD)"
[ "$rev" = "$MATRIX_REV" ] || fail "HEAD is $rev, but the driver started from $MATRIX_REV"
[ -z "$(git -C "$REPO" status --porcelain)" ] || fail "the working tree is dirty"

# The job-local indices are correct only when Slurm hides the other cards.
seen="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c .)"
[ "$seen" = "$MATRIX_NGPUS" ] \
    || fail "nvidia-smi sees $seen cards, but the job holds $MATRIX_NGPUS"
mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')

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
    done < <(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader)
}

SUPERVISOR_PID=$$
ME=$(id -un)

say "  job $SLURM_JOB_ID: start on $(hostname -s), partition ${SLURM_JOB_PARTITION:-?}"
say "  job $SLURM_JOB_ID: gpu $(nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv,noheader 2>&1 | tr '\n' ';')"
say "  job $SLURM_JOB_ID: cpu $(cpu_report | tr '\n' ' ')"
say "  job $SLURM_JOB_ID: numactl $(command -v numactl || echo 'NOT AVAILABLE (runs will be unpinned)')"
say "  job $SLURM_JOB_ID: hf cache $HF_DATASETS_CACHE"
say "  job $SLURM_JOB_ID: ./run_bench.sh $*"

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
load1() { awk '{printf "%.0f", $1}' /proc/loadavg; }

# Writes one line per suspicious sample to $1. A non-empty file condemns the
# cell. nvidia-smi does not always show the PIDs of other users, so the
# watchdog uses two independent signals:
#   FOREIGN_PID  a compute PID that is not a descendant of the batch script
#   FOREIGN_MEM  GPU memory that no PID of ours explains
# The load average is host-wide, so it also includes the jobs of other users.
# The workload is host-bound, so host load corrupts tokens/s.
watchdog() {
    local watch_file="$1"
    local mem load pid used ours residual id mem_streak=0 noted=0
    while :; do
        mem=$(gpu_mem); load=$(load1)
        ours=0
        for id in "${GPU_IDS[@]}"; do
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
            done < <(nvidia-smi --id="$id" --query-compute-apps=pid,used_memory \
                     --format=csv,noheader 2>/dev/null)
        done
        if [ "$noted" -eq 0 ] && [ "$ours" -gt 0 ]; then
            # This line is information, not a flag. It shows that the
            # attribution works for this cell. It goes to the sweep log only.
            say "  job $SLURM_JOB_ID: watchdog attributes ${ours}MiB to our own arms (user $ME)"
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
touch "$watch_file"
watchdog "$watch_file" &
watch_pid=$!
trap 'kill "$watch_pid" 2>/dev/null' EXIT

./run_bench.sh "$@" >>"$OUT.log" 2>&1
rc=$?
kill "$watch_pid" 2>/dev/null; wait "$watch_pid" 2>/dev/null

# Memory that stays on the cards after our processes end is foreign.
after=$(gpu_mem)
if [ "$after" != "-1" ] && [ "$after" -gt "$FOREIGN_MEM_MIB" ]; then
    echo "RESIDUAL_MEM after=${after}MiB (limit ${FOREIGN_MEM_MIB}) $(date -u +%T)" >>"$watch_file"
fi

echo "$rc" >"$OUT.rc"
say "  job $SLURM_JOB_ID: end rc=$rc gpu after=${after}MiB"
exit "$rc"
