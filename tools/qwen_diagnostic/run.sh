#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${JUDGE_PYTHON:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready/.venv/bin/python}"
ROOT="${R85_ROOT:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready/R84_R85_FinalNight/R85_IndependentComparison}"
MODEL="${QWEN_MODEL:-/root/autodl-tmp/local-models/Qwen2.5-VL-7B-Instruct}"
OUT="${PROBE_OUT:-/root/autodl-tmp/A_Qwen_Diagnostic_results_$(date +%Y%m%d_%H%M%S)}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
exec 9>"$HERE/diagnostic.lock"
flock -n 9 || { echo 'Diagnostic already running.'; exit 2; }
"$PY" "$HERE/probe.py" --root "$ROOT" --model "$MODEL" --out "$OUT"
