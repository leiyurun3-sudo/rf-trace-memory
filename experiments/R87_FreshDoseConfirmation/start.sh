#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
command -v flock >/dev/null || { echo 'Missing flock (util-linux).'; exit 2; }
exec 8>"$ROOT/.launch.lock"
flock -n 8 || { echo "Already running. Log: $ROOT/run.log"; exit 0; }
if [[ -e "$ROOT/run.log" ]]; then
  mv -- "$ROOT/run.log" "$ROOT/run.$(date +%Y%m%d_%H%M%S).log"
fi
nohup bash "$ROOT/run_all.sh" >"$ROOT/run.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$ROOT/run.pid"
printf 'Started PID=%s\nLog: %s/run.log\n' "$!" "$ROOT"

