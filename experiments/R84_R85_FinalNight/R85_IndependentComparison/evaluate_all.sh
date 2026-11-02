#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname -- "$ROOT")"
ROUND="$(basename -- "$ROOT")"
ROUND="${ROUND%%_*}"
PYTHON="${EVAL_PYTHON:-${PYTHON:-$PROJECT/.venv/bin/python}}"
old_clip="${CLIP_PATH:-}"
old_dino="${DINO_PATH:-}"
old_lpips="${LPIPS_PATH:-}"
old_trunk="${LPIPS_TRUNK_PATH:-}"
old_lpips_status="${LPIPS_STATUS:-}"
old_lpips_error="${LPIPS_ERROR:-}"
for name in "${ROUND}_paths.env" paths.env R85_paths.env R84_paths.env R83_paths.env R82_paths.env R81_paths.env; do
    if [[ -f "$PROJECT/evaluation_models/$name" ]]; then
        source "$PROJECT/evaluation_models/$name"
        break
    fi
done
if [[ -n "$old_clip" ]]; then export CLIP_PATH="$old_clip"; fi
if [[ -n "$old_dino" ]]; then export DINO_PATH="$old_dino"; fi
if [[ -n "$old_lpips" ]]; then export LPIPS_PATH="$old_lpips"; fi
if [[ -n "$old_trunk" ]]; then export LPIPS_TRUNK_PATH="$old_trunk"; fi
if [[ -n "$old_lpips_status" ]]; then export LPIPS_STATUS="$old_lpips_status" LPIPS_ERROR="$old_lpips_error"; fi
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
prepare_only=0
if [[ "${1:-}" == --prepare-only ]]; then prepare_only=1; shift; fi
if [[ "$prepare_only" == 1 || "${PREPARE_EVAL_MODELS:-1}" == 1 ]]; then
    mkdir -p "$PROJECT/evaluation_models"
    args=()
    if [[ "${DOWNLOAD_EVAL_MODELS:-1}" != 0 ]]; then args+=(--download); fi
    printf 'EVAL: preparing and CPU-loading CLIP/DINO/LPIPS; download=%s; install LPIPS only with EVAL_INSTALL_LPIPS_NO_DEPS=1; shared CPU cache lock only\n' "${DOWNLOAD_EVAL_MODELS:-1}"
    set +e
    flock "$PROJECT/evaluation_models/.prepare.lock" "$PYTHON" -u "$ROOT/prepare_eval_models.py" "${args[@]}"
    prepare_rc=$?
    set -e
    if [[ $prepare_rc -ne 0 ]]; then
        printf 'ERROR: evaluator preparation failed (%s); see evaluation_prepare_status.json and preparation log\n' "$prepare_rc" >&2
        if [[ "$prepare_only" == 1 ]]; then exit "$prepare_rc"; fi
        exec "$PYTHON" -u "$ROOT/eval_pipeline.py" "$@" --preparation-error-file "$ROOT/evaluation_prepare_status.json"
    fi
    source "$PROJECT/evaluation_models/${ROUND}_paths.env"
fi
if [[ "$prepare_only" == 1 ]]; then exit 0; fi
printf 'EVAL: starting blocking native-PNG CPU evaluation; no GPU queue lock\n'
exec "$PYTHON" -u "$ROOT/eval_pipeline.py" "$@"
