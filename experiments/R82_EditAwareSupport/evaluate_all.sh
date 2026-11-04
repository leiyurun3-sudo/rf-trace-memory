#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname -- "$ROOT")"
PYTHON="${EVAL_PYTHON:-${PYTHON:-$PROJECT/.venv/bin/python}}"
# Existing offline paths prepared earlier; environment variables take precedence.
old_clip="${CLIP_PATH:-}"
old_dino="${DINO_PATH:-}"
if [[ -f "$PROJECT/evaluation_models/R82_paths.env" ]]; then
    source "$PROJECT/evaluation_models/R82_paths.env"
elif [[ -f "$PROJECT/evaluation_models/R81_paths.env" ]]; then
    source "$PROJECT/evaluation_models/R81_paths.env"
fi
if [[ -n "$old_clip" ]]; then export CLIP_PATH="$old_clip"; fi
if [[ -n "$old_dino" ]]; then export DINO_PATH="$old_dino"; fi
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
exec "$PYTHON" -u "$ROOT/eval_pipeline.py" "$@"
