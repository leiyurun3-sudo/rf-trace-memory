#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
R85="${R85:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready/R84_R85_FinalNight/R85_IndependentComparison}"
PYTHON_BIN="${R86_PYTHON:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready/.venv/bin/python}"
COHORT="${1:-pilot}"
if [[ "$COHORT" != pilot && "$COHORT" != full ]] || (( $# > 1 )); then echo 'Usage: bash start.sh [pilot|full]' >&2; exit 2; fi
if [[ ! -x "$PYTHON_BIN" ]]; then echo "Python unavailable: $PYTHON_BIN (set R86_PYTHON to the existing R85 environment)" >&2; exit 1; fi
if ! command -v flock >/dev/null; then echo 'flock is required for duplicate-launch protection.' >&2; exit 1; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_DATASETS_OFFLINE=1
export DOWNLOAD_EVAL_MODELS=0 DOWNLOAD_LPIPS_MODELS=0 PREPARE_EVAL_MODELS=0 EVAL_INSTALL_LPIPS_NO_DEPS=0 PYTHONUNBUFFERED=1
LOG="$ROOT/run_v2.log"
exec 9>>"$ROOT/.launch.lock"
if ! flock -n 9; then echo "R86 already running or preflighting; log: $LOG" >&2; exit 1; fi
printf '\n%s Preflight cohort=%s R85=%s python=%s\n' "$(date -Is)" "$COHORT" "$R85" "$PYTHON_BIN" | tee -a "$LOG"
if ! "$PYTHON_BIN" -u "$ROOT/run.py" --r85 "$R85" --cohort "$COHORT" --preflight-only 2>&1 | tee -a "$LOG"; then
  echo "Preflight failed; no background run started. See $LOG" >&2; exit 1
fi
nohup "$PYTHON_BIN" -u "$ROOT/run.py" --r85 "$R85" --cohort "$COHORT" >> "$LOG" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$ROOT/run.pid"
printf 'Started R86 %s PID=%s; log=%s\n' "$COHORT" "$pid" "$LOG"
