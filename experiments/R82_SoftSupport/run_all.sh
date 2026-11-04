#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname -- "$ROOT")"
PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}"
cd -- "$ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OUTPUT_NAME="${OUTPUT_NAME:-R82_SoftSupport}"
exec "$PYTHON" -u "$ROOT/runner.py" "$@"
