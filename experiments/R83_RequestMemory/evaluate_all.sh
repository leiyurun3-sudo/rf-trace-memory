#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname -- "$ROOT")"
PYTHON="${EVAL_PYTHON:-${PYTHON:-$PROJECT/.venv/bin/python}}"
# Existing offline paths prepared earlier; environment variables take precedence.
old_clip="${CLIP_PATH:-}"
old_dino="${DINO_PATH:-}"
if [[ -f "$PROJECT/evaluation_models/R83_paths.env" ]]; then
    source "$PROJECT/evaluation_models/R83_paths.env"
elif [[ -f "$PROJECT/evaluation_models/R82_paths.env" ]]; then
    source "$PROJECT/evaluation_models/R82_paths.env"
elif [[ -f "$PROJECT/evaluation_models/R81_paths.env" ]]; then
    source "$PROJECT/evaluation_models/R81_paths.env"
fi
if [[ -n "$old_clip" ]]; then export CLIP_PATH="$old_clip"; fi
if [[ -n "$old_dino" ]]; then export DINO_PATH="$old_dino"; fi
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
if [[ "${PREPARE_EVAL_MODELS:-0}" == 1 ]]; then
    printf 'EVAL: preparing local models; DOWNLOAD_EVAL_MODELS=%s\n' "${DOWNLOAD_EVAL_MODELS:-1}"
    prepare_args=()
    if [[ "${DOWNLOAD_EVAL_MODELS:-1}" != 0 ]]; then prepare_args+=(--download); fi
    if ! "$PYTHON" -u "$ROOT/prepare_eval_models.py" "${prepare_args[@]}"; then
        printf 'ERROR: evaluation weights preparation failed; evaluation was not run. See evaluation_models/R83_prepare_status.json and this log.\n' >&2
        exit 2
    fi
    source "$PROJECT/evaluation_models/R83_paths.env"
fi
printf 'EVAL: starting native-PNG CPU evaluation; no GPU queue lock\n'
exec "$PYTHON" -u "$ROOT/eval_pipeline.py" "$@"
