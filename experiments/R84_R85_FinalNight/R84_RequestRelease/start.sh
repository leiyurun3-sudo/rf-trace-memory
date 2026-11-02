#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname -- "$(dirname -- "$ROOT")")"
export OUTPUT_NAME="${OUTPUT_NAME:-R84_RequestRelease}"
LOG="$BASE/$OUTPUT_NAME.log"
exec 9>"$ROOT/.launch.lock"
if flock -n 9; then
    nohup bash "$ROOT/run_all.sh" "$@" >> "$LOG" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$ROOT/.runner.pid"
    printf 'Started %s, PID=%s\n' "$OUTPUT_NAME" "$pid"
else
    pid="$(cat "$ROOT/.runner.pid" 2>/dev/null || true)"
    printf '%s already running, PID=%s\n' "$OUTPUT_NAME" "$pid"
fi
exec 9>&-
printf 'Results: %s/%s.zip; log: %s\n' "$BASE" "$OUTPUT_NAME" "$LOG"
if [[ -n "$pid" ]]; then tail --pid="$pid" -n 25 -F "$LOG"; else tail -n 25 "$LOG"; fi
