#!/usr/bin/env bash
set -euo pipefail
NIGHT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT="$(cd -- "$NIGHT/.." && pwd -P)"
LOG="$NIGHT/final_night.log"
export PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}" EVAL_PYTHON="${EVAL_PYTHON:-${PYTHON:-$PROJECT/.venv/bin/python}}"
export DOWNLOAD_EVAL_MODELS="${DOWNLOAD_EVAL_MODELS:-1}" DOWNLOAD_LPIPS_MODELS="${DOWNLOAD_LPIPS_MODELS:-1}"
export EVAL_CPU_THREADS="${EVAL_CPU_THREADS:-2}" OMP_NUM_THREADS="${EVAL_CPU_THREADS:-2}" MKL_NUM_THREADS="${EVAL_CPU_THREADS:-2}" OPENBLAS_NUM_THREADS="${EVAL_CPU_THREADS:-2}"
export EVAL_INSTALL_LPIPS_NO_DEPS="${EVAL_INSTALL_LPIPS_NO_DEPS:-1}"
mode="${1:-run}"
if [[ "$mode" != run && "$mode" != --eval-only ]]; then echo 'Usage: bash start_final_night.sh [--eval-only]' >&2; exit 2; fi
# Share the exact locks used by the old overnight launcher and both old round launchers.
exec 9>>"$PROJECT/.r84_r85_overnight.lock"
if ! flock -n 9; then
    echo "Overnight lock busy; no new process started. Log: $LOG"
    if [[ -s "$NIGHT/final_night.pid" ]]; then pid="$(cat "$NIGHT/final_night.pid")"; if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then exec tail --pid="$pid" -n 30 -F "$LOG"; fi; fi
    exit 1
fi
mkdir -p "$PROJECT/R84_RequestRelease" "$PROJECT/R85_IndependentComparison"
exec 8>>"$PROJECT/R84_RequestRelease/.launch.lock"
if ! flock -n 8; then echo 'R84 old round launch lock busy; no new process started.' >&2; exit 1; fi
exec 7>>"$PROJECT/R85_IndependentComparison/.launch.lock"
if ! flock -n 7; then echo 'R85 old round launch lock busy; no new process started.' >&2; exit 1; fi
# Lock path indirection makes the unmodified nested runners use PROJECT's real GPU queue.
for name in .ab_queue.lock .mint_runner.lock; do
    touch "$PROJECT/$name"
    if [[ ! -e "$NIGHT/$name" && ! -L "$NIGHT/$name" ]]; then ln -s "../$name" "$NIGHT/$name"; fi
    if [[ "$(readlink -f "$NIGHT/$name")" != "$(readlink -f "$PROJECT/$name")" ]]; then echo "Incorrect shared lock: $NIGHT/$name" >&2; exit 1; fi
done
mkdir -p "$PROJECT/evaluation_models"
if [[ ! -e "$NIGHT/evaluation_models" && ! -L "$NIGHT/evaluation_models" ]]; then ln -s ../evaluation_models "$NIGHT/evaluation_models"; fi
if [[ "$(readlink -f "$NIGHT/evaluation_models")" != "$(readlink -f "$PROJECT/evaluation_models")" ]]; then echo 'Evaluation cache is not the shared PROJECT cache.' >&2; exit 1; fi
printf 'Preparing evaluation models in foreground; CLIP, DINO and LPIPS must actually load.\n' | tee -a "$LOG"
if ! bash "$NIGHT/R85_IndependentComparison/evaluate_all.sh" --prepare-only 2>&1 | tee -a "$LOG"; then
    printf 'preparation_failed: see final_night.log\n' | tee "$NIGHT/final_night.status"; exit 1
fi
source "$PROJECT/evaluation_models/R85_paths.env"
printf "DONE: CLIP, DINO, processors, and LPIPS actually loaded on CPU\n"
export FINAL_NIGHT_LOCKS_HELD=1
nohup bash "$NIGHT/run_final_night.sh" "$mode" >> "$LOG" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$NIGHT/final_night.pid"
printf 'Started final night PID=%s; log=%s\n' "$pid" "$LOG"
exec 7>&- 8>&- 9>&-
printf 'Ctrl-C stops log viewing only.\n'
exec tail --pid="$pid" -n 40 -F "$LOG"
