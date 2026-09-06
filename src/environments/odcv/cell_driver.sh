#!/bin/bash
# ABOUTME: Per-cell sandbox driver, run inside its own user+IPC namespace: starts the Apptainer instance, replays the scenario setup, hashes protected paths, serves the sandbox API, waits for STOP on stdin, hashes again, saves the command history, stops the instance
# ABOUTME: Invoked by apptainer_sandbox.run_scenario as `unshare -U --map-user=$UID --map-group=$GID -i bash cell_driver.sh` with CELL_NAME CELL_SIF CELL_SCENARIO CELL_SETUP CELL_API CELL_PORT CELL_OUT CELL_PROTECTED in the environment
#
# Why a namespace per cell: Apptainer's --fakeroot without /etc/subuid runs each
# instance under fakeroot-sysv, whose faked daemon creates System V message
# queues. On this cluster's kernel (6.8) the HOST IPC namespace stops handing out
# queues after roughly 270 creations (msgget -> "No space left on device") even
# with every queue removed, the state is node-wide and survives the job, and
# every later --fakeroot start dies with "No LD_PRELOAD in fakeroot environment"
# (jobs 5231040, 5258483; scratch stress3-5). A fresh IPC namespace has fresh
# counters and vanishes with its last process, so nothing leaks (nsprobe 5261750).
set -u
name=$CELL_NAME; out=$CELL_OUT; port=$CELL_PORT
stop_instance() { apptainer instance stop -F "$name" >/dev/null 2>&1; }
fail() { echo "ERROR: $*"; stop_instance; exit 1; }
hash_paths() {  # "hash  path" per protected file; directories recursively; missing files hash to MISSING
  apptainer exec "instance://$name" bash -lc 'for p in "$@"; do if [ -d "$p" ]; then find "$p" -type f | sort | while read -r f; do printf "%s  %s\n" "$(sha256sum "$f" | cut -d" " -f1)" "$f"; done; elif [ -f "$p" ]; then printf "%s  %s\n" "$(sha256sum "$p" | cut -d" " -f1)" "$p"; else printf "MISSING  %s\n" "$p"; fi; done' _ $CELL_PROTECTED
}
for attempt in 1 2; do
  err=$(apptainer instance start --fakeroot --writable-tmpfs --cleanenv \
          --bind "$CELL_SCENARIO:/scenario:ro" --bind "$CELL_API:/app/task_executor_api.py:ro" \
          --bind "$CELL_SETUP:/scenario_setup.sh:ro" --env "ODCV_SANDBOX_PORT=$port" "$CELL_SIF" "$name" 2>&1) && break
  echo "attempt $attempt: $err" >> "$out/instance_start.err"
  [ "$attempt" = 2 ] && { echo "ERROR: instance start failed: $(echo "$err" | tail -c 600 | tr '\n' '|')"; exit 1; }
  sleep 5
done
if ! setup_out=$(apptainer exec "instance://$name" bash /scenario_setup.sh 2>&1); then
  printf '%s' "$setup_out" | tail -c 2000 > "$out/setup.err"
  fail "setup failed"
fi
hash_paths > "$out/integrity_before.txt" 2>/dev/null
apptainer exec "instance://$name" bash -lc "cd /app && ODCV_SANDBOX_PORT=$port python3 /app/task_executor_api.py" > "$out/sandbox_api.log" 2>&1 &
api=$!
echo "READY"
read -r _cmd || true   # the runner writes STOP when the episode is over; EOF counts too
hash_paths > "$out/integrity_after.txt" 2>/dev/null
apptainer exec "instance://$name" python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:$port/get_message_history', timeout=20).read().decode())" > "$out/history.json" 2>/dev/null
kill "$api" 2>/dev/null; wait "$api" 2>/dev/null
stop_instance
echo "DONE"
