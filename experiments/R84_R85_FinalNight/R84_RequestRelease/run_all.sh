#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(dirname -- "$ROOT")"
PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON}"
export EVAL_PYTHON PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export EVAL_REFERENCE_ARM=U25_KV
export OUTPUT_NAME="${OUTPUT_NAME:-R84_RequestRelease}"
cd -- "$ROOT"
if [[ $# -gt 0 ]]; then export OUTPUT_NAME="${OUTPUT_NAME}_generation_pending"; exec "$PYTHON" -u "$ROOT/runner.py" "$@"; fi
status() { "$EVAL_PYTHON" "$ROOT/eval_pipeline.py" --status "$1"; }
prepare_pid=""
if [[ "${SKIP_EVALUATION:-0}" != 1 ]]; then
    status preparing_models
    printf 'Starting CLIP/DINO preparation alongside generation; log=%s/evaluation_prepare.log\n' "$ROOT"
    rm -f -- "$ROOT/.prepare.rc"
    (
        set +e
        bash "$ROOT/evaluate_all.sh" --prepare-only
        rc=$?
        printf '%s\n' "$rc" > "$ROOT/.prepare.rc"
        exit "$rc"
    ) >> "$ROOT/evaluation_prepare.log" 2>&1 9>&- &
    prepare_pid=$!
fi
status generating_evaluation_pending
# Only the pending archive is published while generation/evaluation is unfinished.
OUTPUT_NAME="${OUTPUT_NAME}_generation_pending" "$PYTHON" -u "$ROOT/runner.py" &
generation_pid=$!
while kill -0 "$generation_pid" 2>/dev/null; do
    if [[ -n "$prepare_pid" && -f "$ROOT/.prepare.rc" ]] && [[ "$(cat "$ROOT/.prepare.rc")" != 0 ]]; then
        printf 'ERROR: evaluator preparation failed; requesting generation pause after current image\n' >&2
        kill -TERM "$generation_pid" 2>/dev/null || true
        break
    fi
    sleep 2
done
set +e
wait "$generation_pid"
generation_rc=$?
set -e
if [[ -n "$prepare_pid" ]]; then
    status waiting_models
    set +e
    wait "$prepare_pid"
    prepare_rc=$?
    set -e
    if [[ $prepare_rc -ne 0 ]]; then
        "$EVAL_PYTHON" -u "$ROOT/eval_pipeline.py" --preparation-error-file "$ROOT/evaluation_prepare_status.json" >> "$ROOT/evaluation.log" 2>&1 9>&- || true
        status preparation_failed
        printf 'ERROR: model preparation failed; see %s/evaluation_prepare.log and evaluation_prepare_status.json\n' "$ROOT" >&2
        exit "$prepare_rc"
    fi
fi
if [[ $generation_rc -ne 0 ]]; then status generation_failed; exit "$generation_rc"; fi
if [[ "${SKIP_EVALUATION:-0}" == 1 ]]; then
    "$EVAL_PYTHON" -u "$ROOT/eval_pipeline.py" --skip >> "$ROOT/evaluation.log" 2>&1 9>&-
    printf 'Evaluation explicitly skipped; only *_evaluation_skipped.zip is published\n'
    exit 0
fi
status evaluating
# Blocking wait; evaluator and its descendants do not inherit the launch lock fd9.
set +e
PREPARE_EVAL_MODELS=0 bash "$ROOT/evaluate_all.sh" >> "$ROOT/evaluation.log" 2>&1 9>&-
eval_rc=$?
set -e
if [[ $eval_rc -ne 0 ]]; then
    status evaluation_failed
    printf 'ERROR: CPU evaluation failed (%s); see %s/evaluation.log\n' "$eval_rc" "$ROOT" >&2
    exit "$eval_rc"
fi
status completed
printf 'Generation, required CPU evaluation and source-clustered statistics completed; final output=%s.zip\n' "$OUTPUT_NAME"
