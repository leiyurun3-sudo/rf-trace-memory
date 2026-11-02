#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=/root/autodl-tmp/MINT_Flow_Jupyter_Ready
PYTHON="${QWEN_PYTHON:-$BASE/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || command -v python)"; fi
ROOT="${R85_ROOT:-}"
if [[ -z "$ROOT" ]]; then
 for candidate in "$BASE/R84_R85_FinalNight/R85_IndependentComparison" "$BASE/R85_IndependentComparison" /root/autodl-tmp/R85_IndependentComparison; do
  if [[ -f "$candidate/plan.json" || -f "$candidate/results/plan.json" || -f "$candidate/results/R85/plan.json" ]]; then ROOT="$candidate"; break; fi
 done
fi
if [[ -z "$ROOT" || ! -d "$ROOT/data" ]]; then echo 'R85 package root not found. Set R85_ROOT to the directory containing data/ and results/.' >&2; exit 1; fi
MODEL="${QWEN_MODEL_DIR:-/root/autodl-tmp/local-models/Qwen2.5-VL-7B-Instruct}"
OUT="${QWEN_JUDGE_OUT:-/root/autodl-tmp/A_Qwen_LocalJudge_results}"
REPEATS="${QWEN_REPEAT_COUNT:-20}"
echo "R85 root: $ROOT"; echo "Local model: $MODEL"; echo "Output: $OUT"
"$PYTHON" -c 'import torch,transformers,accelerate,PIL,huggingface_hub; from transformers import Qwen2_5_VLForConditionalGeneration; print("torch",torch.__version__,"transformers",transformers.__version__)'
if [[ ! -f "$MODEL/judge_model_revision.json" ]]; then
 args=(); if [[ -n "${HF_ENDPOINT:-}" ]]; then args+=(--endpoint "$HF_ENDPOINT"); fi
 "$PYTHON" "$HERE/download_model.py" --model-dir "$MODEL" "${args[@]}"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# run() first checks all native inputs, then actually loads the model and processor on GPU before any evaluation.
"$PYTHON" "$HERE/judge.py" run --root "$ROOT" --model "$MODEL" --out "$OUT" --repeat-count "$REPEATS"
"$PYTHON" "$HERE/judge.py" summarize --out "$OUT"
"$PYTHON" - "$OUT" <<'PY'
from pathlib import Path
import sys,zipfile
p=Path(sys.argv[1]); target=p.with_suffix(".zip")
with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as z:
    for f in sorted(p.rglob("*")):
        if f.is_file(): z.write(f,Path(p.name)/f.relative_to(p))
print("Result package:",target)
PY
echo "Complete: $OUT"
