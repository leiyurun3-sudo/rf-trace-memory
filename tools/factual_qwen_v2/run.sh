#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 9>"$HERE/run.lock"
flock -n 9 || { echo 'Factual judge already running.' >&2; exit 2; }
PY="${JUDGE_PYTHON:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready/.venv/bin/python}"
ROOT="${R85_ROOT:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready/R84_R85_FinalNight/R85_IndependentComparison}"
MODEL="${QWEN_MODEL:-/root/autodl-tmp/local-models/Qwen2.5-VL-7B-Instruct}"
OUT="${FACTUAL_JUDGE_OUT:-/root/autodl-tmp/A_Qwen_FactualJudge_v2_results}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
[[ -x "$PY" ]] || { echo "Python missing: $PY" >&2; exit 1; }
args=(--root "$ROOT" --model "$MODEL" --out "$OUT")
if [[ "${RUN_PREFLIGHT:-0}" == 1 ]]; then args+=(--preflight); fi
if [[ "${CONTROLS_ONLY:-0}" == 1 ]]; then args+=(--controls-only); fi
exec "$PY" -u "$HERE/judge.py" "${args[@]}"
