#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 8>"$HERE/start.lock"
flock -n 8 || { echo 'Another start is in progress.' >&2; exit 2; }
# The advisory lock is authoritative; an old PID may now belong to another process.
exec 9>"$HERE/run.lock"
flock -n 9 || { echo "Already running; log=$HERE/run.log"; exit 0; }
flock -u 9
exec 9>&-
nohup bash "$HERE/run.sh" 8>&- >> "$HERE/run.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$HERE/run.pid"
echo "Started: PID=$pid; log=$HERE/run.log"
