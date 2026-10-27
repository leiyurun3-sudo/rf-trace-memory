#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname -- "$ROOT")"
PYTHON="$PROJECT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing existing Python environment: %s\n' "$PYTHON" >&2
  exit 1
fi
cd -- "$ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# Foreground by default. Optional: nohup bash run_all.sh > run.log 2>&1 &
# Then: tail -f run.log
exec "$PYTHON" -u "$ROOT/runner.py" "$@"
