#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname -- "$ROOT")"
PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}"
cd -- "$ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OUTPUT_NAME="${OUTPUT_NAME:-R83_RequestMemory}"
set +e
"$PYTHON" -u "$ROOT/runner.py" "$@"
generation_rc=$?
set -e
if [[ $# -eq 0 && $generation_rc -eq 0 ]]; then
    mkdir -p "$ROOT/results"
    # Close the inherited launch lock. CPU evaluation never owns GPU queue locks.
    nohup env PREPARE_EVAL_MODELS=1 bash "$ROOT/evaluate_all.sh" >> "$ROOT/results/evaluation.log" 2>&1 < /dev/null 9>&- &
    printf '%s\n' "$!" > "$ROOT/.evaluator.pid"
    printf 'Generation exited (%s); CPU evaluator PID=%s; log=%s/results/evaluation.log\n' "$generation_rc" "$!" "$ROOT"
fi
exit "$generation_rc"
