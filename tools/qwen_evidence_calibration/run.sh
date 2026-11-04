#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 9>"$HERE/calibration.lock"
flock -n 9 || { echo "This calibration is already running."; exit 1; }
PY="${JUDGE_PYTHON:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready/.venv/bin/python}"
ROOT="${R85_ROOT:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready/R84_R85_FinalNight/R85_IndependentComparison}"
MODEL="${QWEN_MODEL:-/root/autodl-tmp/local-models/Qwen2.5-VL-7B-Instruct}"
OUT="${CALIBRATION_OUT:-/root/autodl-tmp/A_Qwen_EvidenceCalibration_results_$(date +%Y%m%d_%H%M%S)}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
test -x "$PY" || { echo "Python missing: $PY"; exit 1; }
test -d "$ROOT/data" || { echo "R85 data missing: $ROOT/data"; exit 1; }
test -f "$MODEL/model.safetensors.index.json" || { echo "Local model missing: $MODEL"; exit 1; }
printf 'Input: %s\nOutput: %s\n' "$ROOT" "$OUT"
"$PY" -u "$HERE/calibrate.py" --root "$ROOT" --model "$MODEL" --out "$OUT"
