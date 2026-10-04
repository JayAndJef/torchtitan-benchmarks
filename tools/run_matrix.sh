#!/usr/bin/env bash
# Run a multi-cell benchmark matrix as a series of Slurm jobs.
#
# Why it exists
# -------------
# A `run` over every scenario stops at the first failure, and --resume
# refuses more than one scenario. So one bad cell can throw away every
# good cell. This script runs each cell as its own `run` under one shared
# output root. A failure costs one cell, and --resume picks it up.
#
# Slurm allocates the cards. The script submits one job per cell, waits for
# that job to end, and only then submits the next cell. So our own cells do
# not contend with each other.
#
# Other users share this host. Their jobs can take GPU memory and drive the
# load average high. A contaminated cell still writes a results.json, so the
# bad numbers stay unless something removes them. A watchdog therefore runs
# inside each job and samples during the run. When it flags a cell, this
# script moves the output aside, and the next pass runs the cell again.
# The workload is host-bound, so host load corrupts tokens/s too.
#
# Cells
# -----
# A cell is a quoted string of `run` flags, one per array entry. The script
# word-splits it, so a cell string must contain no quotes and no space
# inside a value. Each cell becomes one job that runs:
#
#   ./run_bench.sh run <0,...,NGPUS-1> <cell flags> --steps <STEPS> --out <dir>
#
# Inside a job, Slurm numbers the cards of the job from 0. So the device
# list is always job-local, and never a physical index.
#
# The output directory is $ROOT/<slug of the cell>. The slug drops the `--`
# of each flag and joins the remaining words with `-`. The script refuses
# two cells with one slug, because they would share one directory.
# A slug above 200 bytes is cut to 191 bytes and given a hash of the whole slug.
#
# A job holds one time limit. A cell that runs every arm of a large shape
# can take longer than the 2 h cap of `main`. Give such a cell one --arm.
#
# Usage
# -----
#   NGPUS=1 TIME=1:50:00 nohup ./tools/run_matrix.sh > /dev/null 2>&1 &
#   tail -f out/matrix-<utc>/sweep.log
#
#   DRY_RUN=1 NGPUS=1 TIME=1:50:00 ./tools/run_matrix.sh   # print, submit nothing
#
# Environment (NGPUS and TIME are required; the others are optional):
#   NGPUS            the number of cards of each job. dp x pp of each cell
#                    must equal it.
#   TIME             the Slurm time limit of each job, for example 1:50:00.
#                    Give a realistic estimate, not the partition cap.
#   PARTITION        the Slurm partition (default main). The script refuses
#                    placeholder, exceptions and admin.
#   GPU_IDX          one specific card, as gpu:idx<GPU_IDX>:1. It needs NGPUS=1.
#   CPUS_PER_GPU     CPUs per card (default 32)
#   MEM_PER_GPU      host memory per card in MB (default 180000)
#   WAIT_TIMEOUT     seconds that a job can stay pending before the script
#                    cancels it and reports GAVE-UP (default 43200)
#   POLL_INTERVAL    seconds between two queue samples (default 60)
#   ROOT             output root (default out/matrix-<utc>)
#   PASSES           retry passes over the cells that are not OK (default 3)
#   STEPS            training steps per arm (default 80)
#   MATRIX_CELLS     newline-separated cell strings that replace the built-in
#                    list. The script refuses a dirty tree, so an operator
#                    uses this to select cells without a commit.
#   DRY_RUN          1 prints each sbatch command and batch script, and
#                    submits nothing
#   FOREIGN_MEM_MIB  GPU memory that no process of ours explains, above
#                    which the watchdog flags the cell (default 2000)
#   CONTENDED_LOAD   1-minute load average that flags the cell (default 150)
#   WATCH_INTERVAL   watchdog sample period in seconds (default 15)
#
# The partitions and their limits come from the admin skill run-gpu-job.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

fail() { echo "run_matrix.sh: $*" >&2; exit 1; }

# The old variables named physical cards and an idle gate. Slurm now owns
# both, so the script refuses each old name instead of ignoring it.
[ -z "${GPU:-}" ] || fail "GPU is gone: Slurm allocates the cards. Set NGPUS, and GPU_IDX for one specific card."
for old in IDLE_LOAD IDLE_SETTLE IDLE_POLL; do
    [ -z "${!old:-}" ] || fail "$old is gone: Slurm allocates the cards, so the idle gate is gone."
done
[ -z "${IDLE_MEM_MIB:-}" ] || fail "IDLE_MEM_MIB is gone: set FOREIGN_MEM_MIB."

NGPUS="${NGPUS:-}"
TIME="${TIME:-}"
PARTITION="${PARTITION:-main}"
GPU_IDX="${GPU_IDX:-}"
CPUS_PER_GPU="${CPUS_PER_GPU:-32}"
MEM_PER_GPU="${MEM_PER_GPU:-180000}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-43200}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"
ROOT="${ROOT:-out/matrix-$(date -u +%Y%m%dT%H%M%SZ)}"
PASSES="${PASSES:-3}"
STEPS="${STEPS:-80}"
DRY_RUN="${DRY_RUN:-0}"
FOREIGN_MEM_MIB="${FOREIGN_MEM_MIB:-2000}"
CONTENDED_LOAD="${CONTENDED_LOAD:-150}"
WATCH_INTERVAL="${WATCH_INTERVAL:-15}"

# ------------------------------------------------------------- the cell list
# The huge cell comes first, because a report starts from that measurement.
# The engines scenario declares supported_ac_modes=("none",), so each size
# has one legal cell.
CELLS=(
    "--scenario engines --model-size huge --ac none"
    "--scenario engines --model-size 1b --ac none"
)
if [ -n "${MATRIX_CELLS:-}" ]; then
    CELLS=()
    while IFS= read -r line; do
        case "$line" in ''|'#'*) continue ;; esac
        CELLS+=("$line")
    done <<<"$MATRIX_CELLS"
fi

# ---------------------------------------------------------------- preflight
is_count() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

# A Slurm time string in seconds; -1 for no limit; status 1 when malformed.
slurm_seconds() {
    local t="$1"
    case "$t" in infinite|UNLIMITED) echo -1; return 0 ;; esac
    if [[ "$t" =~ ^([0-9]+)-([0-9]+)(:([0-9]+)(:([0-9]+))?)?$ ]]; then
        echo $(( 10#${BASH_REMATCH[1]} * 86400 + 10#${BASH_REMATCH[2]} * 3600 \
                 + 10#${BASH_REMATCH[4]:-0} * 60 + 10#${BASH_REMATCH[6]:-0} ))
    elif [[ "$t" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        echo $(( 10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 \
                 + 10#${BASH_REMATCH[3]} ))
    elif [[ "$t" =~ ^([0-9]+):([0-9]+)$ ]]; then
        echo $(( 10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]} ))
    elif [[ "$t" =~ ^([0-9]+)$ ]]; then
        echo $(( 10#${BASH_REMATCH[1]} * 60 ))
    else
        return 1
    fi
}

[ -n "$NGPUS" ] || fail "NGPUS is required: the number of cards of each job"
is_count "$NGPUS" || fail "NGPUS must be a positive integer, not '$NGPUS'"
[ -n "$TIME" ] || fail "TIME is required: a realistic Slurm time limit per cell, for example 1:50:00"
time_seconds="$(slurm_seconds "$TIME")" || fail "TIME '$TIME' is not a Slurm time, for example 1:50:00"
[ "$time_seconds" -gt 0 ] || fail "TIME must be a finite limit above zero, not '$TIME'"
case "$PARTITION" in
    placeholder|exceptions|admin)
        fail "PARTITION=$PARTITION is refused. Read the admin skill run-gpu-job: placeholder is never for users, exceptions needs explicit permission, and admin needs a grant." ;;
    ''|*[!A-Za-z0-9_-]*) fail "PARTITION '$PARTITION' is not a partition name" ;;
esac
if [ -n "$GPU_IDX" ]; then
    [[ "$GPU_IDX" =~ ^[0-9]+$ ]] || fail "GPU_IDX must be a card index, not '$GPU_IDX'"
    [ "$NGPUS" = 1 ] || fail "GPU_IDX selects one card, so NGPUS must be 1, not $NGPUS"
    GRES="gpu:idx$GPU_IDX:1"
else
    GRES="gpu:$NGPUS"
fi
is_count "$CPUS_PER_GPU" || fail "CPUS_PER_GPU must be a positive integer, not '$CPUS_PER_GPU'"
is_count "$MEM_PER_GPU" || fail "MEM_PER_GPU must be a positive integer of MB, not '$MEM_PER_GPU'"
for name in WAIT_TIMEOUT POLL_INTERVAL PASSES STEPS FOREIGN_MEM_MIB CONTENDED_LOAD WATCH_INTERVAL; do
    is_count "${!name}" || fail "$name must be a positive integer, not '${!name}'"
done
case "$DRY_RUN" in 0|1) ;; *) fail "DRY_RUN must be 0 or 1, not '$DRY_RUN'" ;; esac
CPUS=$((NGPUS * CPUS_PER_GPU))
MEM_MB=$((NGPUS * MEM_PER_GPU))

[ -x "$REPO/.venv/bin/python" ] || fail "no environment at $REPO/.venv/bin/python"
# hardware_metadata records `git rev-parse HEAD`, and HEAD does not show
# uncommitted edits. So a dirty tree gives the whole matrix a false label.
# Prints HEAD when the tree is clean; status 1 when git fails or the tree is dirty.
clean_rev() {
    local porcelain
    porcelain="$(git -C "$REPO" status --porcelain)" || return 1
    [ -z "$porcelain" ] || return 1
    git -C "$REPO" rev-parse HEAD
}
START_REV="$(clean_rev)" || fail "working tree is dirty, or git status failed; commit first"
command -v flock >/dev/null || fail "flock is required (single-instance lock)"
[ "${#CELLS[@]}" -gt 0 ] || fail "the cell list is empty"
# The driver waits for hours. Inside a job, it holds cards that it never uses.
[ -z "${SLURM_JOB_ID:-}" ] || fail "run the driver outside a Slurm job; it submits its own jobs"

if [ "$DRY_RUN" != 1 ]; then
    for tool in sbatch squeue sacct scancel sinfo; do
        command -v "$tool" >/dev/null || fail "$tool is required; this script submits Slurm jobs"
    done
fi

# The partition limits are cheap to read. So the script refuses a request
# that cannot fit before the request waits in the queue. Without sinfo,
# Slurm refuses it at submission.
if command -v sinfo >/dev/null; then
    read -r cap node_cpus node_mem node_gres < <(sinfo -h -p "$PARTITION" -o '%l %c %m %G' | head -n 1)
    [ -n "${cap:-}" ] || fail "sinfo knows no partition '$PARTITION'"
    if cap_seconds="$(slurm_seconds "$cap")" && [ "$cap_seconds" -ge 0 ] \
            && [ "$time_seconds" -gt "$cap_seconds" ]; then
        fail "TIME=$TIME is above the $cap cap of partition $PARTITION"
    fi
    [[ "$node_cpus" =~ ^[0-9]+$ ]] && [ "$CPUS" -gt "$node_cpus" ] \
        && fail "$NGPUS x CPUS_PER_GPU=$CPUS_PER_GPU is $CPUS CPUs; the node has $node_cpus"
    [[ "$node_mem" =~ ^[0-9]+$ ]] && [ "$MEM_MB" -gt "$node_mem" ] \
        && fail "$NGPUS x MEM_PER_GPU=$MEM_PER_GPU is $MEM_MB MB; the node has $node_mem MB"
    if [ -n "$GPU_IDX" ]; then
        case ",$node_gres," in
            *",gpu:idx$GPU_IDX:"*) ;;
            *) fail "partition $PARTITION has no card gpu:idx$GPU_IDX (it has $node_gres)" ;;
        esac
    fi
fi

# The job-local device list: 0,1,...,NGPUS-1.
DEVICES="$(seq -s, 0 $((NGPUS - 1)))"

# The slug names the directory of the cell. A later pass finds the
# directory again by this name, and resumes it.
# A slug above 200 bytes keeps its first 191 and a hash of the whole, so the
# longest derived name, "<slug>.contaminated-<stamp>.log", fits NAME_MAX 255.
cell_slug() {
    local slug
    slug=$(printf '%s' "$1" \
        | sed -e 's/--//g' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
              -e 's/[[:space:]][[:space:]]*/-/g' -e 's/--*/-/g')
    if [ "${#slug}" -gt 200 ]; then
        slug="${slug:0:191}-$(printf '%s' "$slug" | sha1sum | cut -c1-8)"
    fi
    printf '%s' "$slug"
}

# The value of one integer flag of a cell, or 1 when the cell omits it.
cell_degree() {
    local flag="$1" previous="" word
    local -a words
    read -r -a words <<<"$2"
    for word in "${words[@]}"; do
        case "$word" in "$flag="*) echo "${word#*=}"; return ;; esac
        [ "$previous" = "$flag" ] && { echo "$word"; return; }
        previous="$word"
    done
    echo 1
}

declare -A SLUG_OF
declare -A SEEN_SLUG
for cell in "${CELLS[@]}"; do
    case "$cell" in
        *\"*|*\'*) fail "cell has a quote, which word splitting cannot honor: $cell" ;;
    esac
    slug="$(cell_slug "$cell")"
    [ -n "$slug" ] || fail "cell slugs to an empty directory name: $cell"
    [ -z "${SEEN_SLUG[$slug]:-}" ] || fail "two cells share the directory $slug"
    SEEN_SLUG["$slug"]=1
    SLUG_OF["$cell"]="$slug"
    # A wrong world size waits in the queue, and then fails at once.
    dp="$(cell_degree --dp "$cell")"; pp="$(cell_degree --pp "$cell")"
    if ! is_count "$dp" || ! is_count "$pp"; then
        fail "cell has a malformed --dp or --pp: $cell"
    fi
    [ $((dp * pp)) -eq "$NGPUS" ] \
        || fail "cell needs dp x pp = $((dp * pp)) cards, but NGPUS=$NGPUS: $cell"
done

# A stable, writable datasets cache. Another user owns the shared HF_HOME.
# Without this cache, every arm stops on a builder.lock PermissionError.
# A batch job reads no dotfiles, so the batch script sets the cache itself.
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/hf-datasets}"
mkdir -p "$HF_DATASETS_CACHE"

# Each cell gets its own --out. An inherited OUT sends every cell into one
# directory, and nothing reports it.
unset OUT

mkdir -p "$ROOT"
ROOT="$(cd "$ROOT" && pwd)"
LOG="$ROOT/sweep.log"
JOBS_DIR="$ROOT/slurm"
mkdir -p "$JOBS_DIR"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

exec 9>"$ROOT/.lock"
flock -n 9 || fail "another run_matrix.sh already holds $ROOT/.lock"

say "=== run_matrix.sh ==="
say "repo:        $REPO"
say "git rev:     $START_REV"
say "slurm:       partition=$PARTITION gres=$GRES time=$TIME cpus=$CPUS mem=${MEM_MB}M"
say "devices:     $DEVICES (job-local)"
say "steps:       $STEPS"
say "passes:      $PASSES"
say "queue:       poll every ${POLL_INTERVAL}s; give up after ${WAIT_TIMEOUT}s pending"
say "watchdog:    every ${WATCH_INTERVAL}s in the job; flags ${FOREIGN_MEM_MIB}MiB foreign or load above ${CONTENDED_LOAD}"
say "root:        $ROOT"
say "hf cache:    $HF_DATASETS_CACHE"
[ "$DRY_RUN" = 1 ] && say "DRY RUN:     printing commands only"
say "cells:       ${#CELLS[@]}"
for cell in "${CELLS[@]}"; do say "  ${SLUG_OF[$cell]}  <- $cell"; done

# ------------------------------------------------------------ the batch script
# The body of the batch script of one cell. Slurm copies it at submission,
# and it execs the committed cell runner, which the job checks against
# START_REV. $1 is the cell directory; the rest are the run_bench.sh words.
batch_script() {
    local out="$1"
    shift
    printf '#!/usr/bin/env bash\n'
    printf '# run_matrix.sh cell %s\n' "$(basename "$out")"
    printf 'set -uo pipefail\n'
    printf 'cd %q || exit 2\n' "$REPO"
    printf 'unset OUT\n'
    printf 'export HF_DATASETS_CACHE=%q\n' "$HF_DATASETS_CACHE"
    printf 'export MATRIX_LOG=%q\n' "$LOG"
    printf 'export MATRIX_OUT=%q\n' "$out"
    printf 'export MATRIX_REV=%q\n' "$START_REV"
    printf 'export MATRIX_NGPUS=%q\n' "$NGPUS"
    printf 'export FOREIGN_MEM_MIB=%q\n' "$FOREIGN_MEM_MIB"
    printf 'export CONTENDED_LOAD=%q\n' "$CONTENDED_LOAD"
    printf 'export WATCH_INTERVAL=%q\n' "$WATCH_INTERVAL"
    printf 'exec %q' "$REPO/tools/run_matrix_cell.sh"
    printf ' %q' "$@"
    printf '\n'
}

# ------------------------------------------------------------ job tracking
CURRENT_JOB=""

# The state of a job: squeue while Slurm still lists it, then sacct.
job_state() {
    local state
    state="$(squeue -h -j "$1" -o %T 2>/dev/null | head -n 1)"
    if [ -z "$state" ]; then
        state="$(sacct -j "$1" -X -n -P -o State 2>/dev/null | head -n 1)"
    fi
    # sacct writes "CANCELLED by <uid>"; the first word is the state.
    echo "${state%% *}"
}

is_terminal() {
    case "$1" in
        COMPLETED|FAILED|TIMEOUT|CANCELLED|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)
            return 0 ;;
    esac
    return 1
}

# A plain sleep defers a trap until it returns; a waited sleep does not.
pause() { sleep "$1" & wait $!; }

# Waits until job $1 ends, and sets JOB_STATE to its final state. The state
# is GAVE-UP when the job stays pending longer than WAIT_TIMEOUT, and UNKNOWN
# when neither squeue nor sacct knows the job for 10 minutes. The function
# runs in the driver shell, not in a subshell, so a signal stops it at once.
JOB_STATE=""
wait_for_job() {
    local id="$1" start=$SECONDS last_note=$SECONDS unknown_since="" state reason
    while :; do
        state="$(job_state "$id")"
        if is_terminal "$state"; then
            JOB_STATE="$state"; return
        fi
        if [ -z "$state" ]; then
            [ -n "$unknown_since" ] || unknown_since=$SECONDS
            if [ $((SECONDS - unknown_since)) -ge 600 ]; then
                JOB_STATE=UNKNOWN; return
            fi
        else
            unknown_since=""
        fi
        if [ "$state" = PENDING ] && [ $((SECONDS - start)) -ge "$WAIT_TIMEOUT" ]; then
            say "  job $id pending for more than ${WAIT_TIMEOUT}s; scancel $id"
            scancel "$id"
            # Wait for the cancel, so the job cannot start beside the next cell.
            for _ in $(seq 1 60); do
                is_terminal "$(job_state "$id")" && break
                pause 10
            done
            JOB_STATE=GAVE-UP; return
        fi
        if [ $((SECONDS - last_note)) -ge 600 ]; then
            reason="$(squeue -h -j "$id" -o %R 2>/dev/null | head -n 1)"
            say "  job $id ${state:-UNKNOWN} ${reason:+($reason) }after $(( (SECONDS - start) / 60 ))m"
            last_note=$SECONDS
        fi
        pause "$POLL_INTERVAL"
    done
}

# The driver owns its job: when the driver stops, the job stops too. Nobody
# else would sort the cell into OK, FAIL or CONTAMINATED.
# shellcheck disable=SC2329  # the trap below calls it
on_signal() {
    if [ -n "$CURRENT_JOB" ]; then
        say "driver stopped by a signal; scancel $CURRENT_JOB"
        scancel "$CURRENT_JOB"
    else
        say "driver stopped by a signal"
    fi
    exit 130
}
trap on_signal INT TERM HUP

# --------------------------------------------------------------- the sweep
declare -A STATUS
for cell in "${CELLS[@]}"; do STATUS["$cell"]="PENDING"; done
stop=0

for pass in $(seq 1 "$PASSES"); do
    say ""
    say "########## pass $pass/$PASSES ##########"
    remaining=0
    for cell in "${CELLS[@]}"; do
        out="$ROOT/${SLUG_OF[$cell]}"
        # The marker is NEXT TO the directory. So the marker stays after an
        # early failure that made no directory.
        marker="$out.CONTAMINATED"

        if [ -f "$out/results.json" ] && [ ! -f "$marker" ]; then
            [ "${STATUS[$cell]}" = "PENDING" ] && STATUS["$cell"]="OK(pre-existing)"
            [ "$pass" -eq 1 ] && say "SKIP $cell (results.json present)"
            continue
        fi
        remaining=$((remaining + 1))

        # The queue can hold the matrix for hours. A commit in that time
        # would mislabel every later cell, so the sweep stops.
        if [ "$(clean_rev)" != "$START_REV" ]; then
            say "STOP the tree is no longer clean at $START_REV; the sweep ends"
            stop=1
            break
        fi

        # The cell is unquoted on purpose. The cell string is a flag list,
        # and the word split turns it into arguments.
        # shellcheck disable=SC2206
        args=(run "$DEVICES" $cell --steps "$STEPS")
        if [ -f "$out/manifest.json" ]; then
            # The test reads the manifest, not the directory. A crash between
            # mkdir and write_manifest leaves a directory that --resume cannot use.
            args+=(--resume "$out")
        else
            args+=(--out "$out")
        fi

        script="$JOBS_DIR/${SLUG_OF[$cell]}.sbatch"
        sbatch_args=(
            --parsable
            --job-name="matrix-${SLUG_OF[$cell]}"
            --partition="$PARTITION"
            --time="$TIME"
            --gres="$GRES"
            --nodes=1
            --ntasks=1
            --cpus-per-task="$CPUS"
            --mem="${MEM_MB}M"
            --no-requeue
            --chdir="$REPO"
            --output="$JOBS_DIR/${SLUG_OF[$cell]}-%j.out"
            "$script"
        )

        if [ "$DRY_RUN" = 1 ]; then
            say "DRY  $cell"
            say "     sbatch $(printf '%q ' "${sbatch_args[@]}")"
            say "     --- $script ---"
            while IFS= read -r line; do say "     $line"; done < <(batch_script "$out" "${args[@]}")
            STATUS["$cell"]="DRY-RUN"
            continue
        fi

        if [ ! -f "$out/manifest.json" ] && [ -e "$out" ]; then
            stamp=$(date -u +%Y%m%dT%H%M%SZ)
            say "  moving manifest-less $out aside -> $out.nomanifest-$stamp"
            mv "$out" "$out.nomanifest-$stamp"
        fi
        rm -f "$marker" "$out.rc"
        : >"$out.watch"
        batch_script "$out" "${args[@]}" >"$script"

        submit_err="$JOBS_DIR/.sbatch.err"
        if ! submitted="$(sbatch "${sbatch_args[@]}" 2>"$submit_err")"; then
            say "SUBMIT-FAILED $cell; the sweep ends"
            sed 's/^/    /' "$submit_err" | tee -a "$LOG"
            STATUS["$cell"]="SUBMIT-FAILED"
            stop=1
            break
        fi
        CURRENT_JOB="${submitted%%;*}"
        sed 's/^/    /' "$submit_err" | tee -a "$LOG"
        say "SUBMIT $cell -> job $CURRENT_JOB ($JOBS_DIR/${SLUG_OF[$cell]}-$CURRENT_JOB.out)"

        wait_for_job "$CURRENT_JOB"
        state="$JOB_STATE"
        job="$CURRENT_JOB"
        CURRENT_JOB=""
        rc="$(cat "$out.rc" 2>/dev/null || true)"

        if [ "$state" = GAVE-UP ]; then
            say "GAVE-UP $cell (job $job pending more than ${WAIT_TIMEOUT}s)"
            STATUS["$cell"]="GAVE-UP"
        elif [ -s "$out.watch" ]; then
            say "CONTAMINATED $cell job=$job state=$state rc=${rc:-none}"
            head -3 "$out.watch" | sed 's/^/    /' | tee -a "$LOG"
            stamp=$(date -u +%Y%m%dT%H%M%SZ)
            {
                echo "contaminated at $stamp; job=$job state=$state rc=${rc:-none}"
                cat "$out.watch"
            } >"$marker"
            [ -e "$out" ] && mv "$out" "$out.contaminated-$stamp"
            [ -e "$out.log" ] && mv "$out.log" "$out.contaminated-$stamp.log"
            STATUS["$cell"]="CONTAMINATED"
        elif [ "$state" != COMPLETED ] && [ "$state" != FAILED ]; then
            # Slurm ended the job, or its state is unknown: the run did not
            # finish, so the next pass resumes the cell.
            say "SLURM-$state $cell job=$job (retried next pass; see $JOBS_DIR/${SLUG_OF[$cell]}-$job.out)"
            STATUS["$cell"]="SLURM-$state"
        elif [ -z "$rc" ]; then
            say "FAIL $cell job=$job state=$state, and the job wrote no exit code (see $JOBS_DIR/${SLUG_OF[$cell]}-$job.out)"
            tail -5 "$JOBS_DIR/${SLUG_OF[$cell]}-$job.out" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG"
            STATUS["$cell"]="FAIL(no-rc)"
        elif [ "$rc" -ne 0 ]; then
            say "FAIL $cell job=$job rc=$rc (retried next pass; see $out.log)"
            tail -5 "$out.log" | sed 's/^/    /' | tee -a "$LOG"
            STATUS["$cell"]="FAIL(rc=$rc)"
        else
            say "OK   $cell job=$job"
            STATUS["$cell"]="OK"
            # An evaluation warning can show contamination. So the sweep
            # log repeats each warning of results.json.
            if [ -f "$out/results.json" ]; then
                warn=$(.venv/bin/python -c "
import json,sys
w=json.load(open(sys.argv[1])).get('warnings') or []
print('; '.join(w))" "$out/results.json" 2>/dev/null)
                [ -n "$warn" ] && say "  results.json warnings: $warn"
            fi
        fi
    done
    [ "$stop" -eq 1 ] && break
    [ "$remaining" -eq 0 ] && { say "nothing left to run"; break; }
    [ "$DRY_RUN" = 1 ] && break
done

# ---------------------------------------------------------------- summary
say ""
say "########## summary ##########"
bad=0
for cell in "${CELLS[@]}"; do
    out="$ROOT/${SLUG_OF[$cell]}"
    say "$(printf '%-24s' "${STATUS[$cell]}") $cell  $out"
    case "${STATUS[$cell]}" in OK|OK\(pre-existing\)|DRY-RUN) ;; *) bad=$((bad + 1)) ;; esac
done
say "cells not OK: $bad / ${#CELLS[@]}"
say "SWEEP COMPLETE -> $ROOT"
exit $(( bad > 0 ? 1 : 0 ))
