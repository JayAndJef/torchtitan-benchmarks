#!/usr/bin/env bash
# Prove is_ours() identifies a descendant that has called setsid (which is
# exactly what defeated the session-id discriminator).
SUPERVISOR_PID=$$
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

echo "supervisor pid=$SUPERVISOR_PID sid=$(ps -o sid= -p $$ | tr -d ' ')"

# A descendant chain that mimics run_bench.sh -> python -> setsid'd worker,
# i.e. the torch-elastic start_new_session=True case.
python3 -c "
import subprocess, os, sys, time
worker = subprocess.Popen(['sleep', '8'], start_new_session=True)
print(worker.pid, flush=True)
time.sleep(7)
" > /tmp/_worker_pid &
sleep 2
worker=$(cat /tmp/_worker_pid)

for pid in $worker; do
    sid=$(ps -o sid= -p "$pid" | tr -d ' ')
    ppid=$(ps -o ppid= -p "$pid" | tr -d ' ')
    if is_ours "$pid"; then verdict=OURS; else verdict=FOREIGN; fi
    echo "  setsid'd descendant pid=$pid ppid=$ppid sid=$sid -> $verdict"
    [ "$sid" = "$pid" ] && echo "    (it IS its own session leader: this is the case that broke the old check)"
done

# A genuinely unrelated process must still read FOREIGN.
if is_ours 1; then echo "  init -> OURS (WRONG)"; else echo "  init(1) -> FOREIGN (correct)"; fi
wait
rm -f /tmp/_worker_pid
