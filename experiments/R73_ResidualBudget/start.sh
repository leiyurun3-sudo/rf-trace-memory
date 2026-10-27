#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname -- "$(dirname -- "$ROOT")")"
LOG="$BASE/R73_ResidualBudget.log"
PIDFILE="$ROOT/.runner.pid"
exec 9>"$ROOT/.launch.lock"
if flock -n 9; then
    rm -f -- "$ROOT/PAUSE"
    nohup bash "$ROOT/run_all.sh" >> "$LOG" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$PIDFILE"
    printf 'Started/resumed R73, PID=%s\n' "$pid"
else
    pid="$(cat "$PIDFILE")"
    printf 'R73 already running, PID=%s; attaching log.\n' "$pid"
fi
exec 9>&-
printf 'Images: %s/R73_ResidualBudget_Images/\nResults: %s/R73_ResidualBudget_Results.zip\nLog: %s\n' "$BASE" "$BASE" "$LOG"
printf 'Ctrl+C stops this log viewer only. To pause after current image: touch %s/PAUSE\n' "$ROOT"
tail --pid="$pid" -n 25 -F "$LOG"
