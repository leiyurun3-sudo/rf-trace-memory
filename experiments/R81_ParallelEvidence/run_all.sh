#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname -- "$ROOT")"
PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}"
cd -- "$ROOT"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
# CPU-only weight preparation overlaps GPU generation/queue waiting.
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$PYTHON" "$ROOT/prepare_eval_models.py" &
eval_prepare_pid=$!
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON" -u "$ROOT/runner.py" "$@"
if wait "$eval_prepare_pid"; then
    source "$PROJECT/evaluation_models/R81_paths.env"
    bash "$ROOT/evaluate_all.sh"
else
    printf 'Generation results saved. Evaluation model download failed; set local CLIP_PATH/DINO_PATH and run evaluate_all.sh.
' >&2
    exit 2
fi
