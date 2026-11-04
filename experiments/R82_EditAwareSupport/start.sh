#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname -- "$(dirname -- "$ROOT")")"
export OUTPUT_NAME="${OUTPUT_NAME:-R82_EditAwareSupport}"
LOG="$BASE/$OUTPUT_NAME.log"
exec 9>"$ROOT/.launch.lock"
if flock -n 9; then
    nohup bash "$ROOT/run_all.sh" "$@" >> "$LOG" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$ROOT/.runner.pid"
    printf 'Started R82, PID=%s\n' "$pid"
else
    pid="$(cat "$ROOT/.runner.pid" 2>/dev/null || true)"
    printf 'R82 already running, PID=%s\n' "$pid"
fi
exec 9>&-
printf 'Results: %s/%s.zip; log: %s\n' "$BASE" "$OUTPUT_NAME" "$LOG"
if [[ -n "$pid" ]]; then tail --pid="$pid" -n 25 -F "$LOG"; else tail -n 25 "$LOG"; fi
