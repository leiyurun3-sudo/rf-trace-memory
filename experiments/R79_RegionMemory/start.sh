#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname -- "$(dirname -- "$ROOT")")"
OUTPUT_NAME="${OUTPUT_NAME:-R79_RegionMemory}"
export OUTPUT_NAME
LOG="$BASE/$OUTPUT_NAME.log"
PIDFILE="$ROOT/.runner.pid"
exec 9>"$ROOT/.launch.lock"
if flock -n 9; then
    rm -f -- "$ROOT/PAUSE"
    nohup bash "$ROOT/run_all.sh" >> "$LOG" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$PIDFILE"
    printf 'Started/resumed R79, PID=%s\n' "$pid"
else
    pid="$(cat "$PIDFILE")"
    printf 'R79 already running, PID=%s; attaching log.\n' "$pid"
fi
exec 9>&-
printf 'Images: %s/%s/\nResults: %s/%s.zip\nLog: %s\n' "$BASE" "$OUTPUT_NAME" "$BASE" "$OUTPUT_NAME" "$LOG"
printf 'Ctrl+C stops this log viewer only. To pause after current image: touch %s/PAUSE\n' "$ROOT"
tail --pid="$pid" -n 25 -F "$LOG"
