#!/usr/bin/env bash
set -uo pipefail
NIGHT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT="$(cd -- "$NIGHT/.." && pwd -P)"
if [[ "${FINAL_NIGHT_LOCKS_HELD:-}" != 1 ]]; then echo 'Use start_final_night.sh so preparation and shared locks are acquired.' >&2; exit 2; fi
export PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}" EVAL_PYTHON="${EVAL_PYTHON:-${PYTHON:-$PROJECT/.venv/bin/python}}"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 EVAL_REFERENCE_ARM=U25_KV
export EVAL_CPU_THREADS="${EVAL_CPU_THREADS:-2}" OMP_NUM_THREADS="${EVAL_CPU_THREADS:-2}" MKL_NUM_THREADS="${EVAL_CPU_THREADS:-2}" OPENBLAS_NUM_THREADS="${EVAL_CPU_THREADS:-2}"
mode="${1:-run}"
STATUS="$NIGHT/final_night.status"
status() { printf '%s %s\n' "$(date -Is)" "$*" | tee "$STATUS"; }
status "running mode=$mode"
declare -a eval_pids=() eval_rounds=() gen_failures=() eval_failures=()
for round in R84_RequestRelease R85_IndependentComparison; do
    root="$NIGHT/$round"
    name="${round%%_*}_FinalNight"
    if [[ "$mode" != --eval-only ]]; then
        status "$round generation"
        if OUTPUT_NAME="${name}_generation_pending" "$PYTHON" -u "$root/runner.py"; then
            # runner can return zero after per-job errors or an intentional pause.
            if ! "$PYTHON" - "$root/results/progress.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
print('Generation progress:',p.get('status'),p.get('completed'),'/',p.get('planned'),'failed',p.get('failed'),flush=True)
if p.get('completed') != p.get('planned') or p.get('failed',0): raise SystemExit(1)
PY
            then gen_failures+=("$round"); fi
        else gen_failures+=("$round"); fi
    fi
    if [[ "$mode" == --eval-only ]]; then
        if ! "$PYTHON" - "$root/results/progress.json" <<'PY'
import json,sys
try:
 p=json.load(open(sys.argv[1],encoding="utf-8"))
 if p.get("completed") != p.get("planned") or p.get("failed",0): raise ValueError("generation incomplete")
except Exception as e:
 print("Generation incomplete:",e,flush=True); raise SystemExit(1)
PY
        then gen_failures+=("$round"); fi
    fi
    status "$round evaluation launched on CPU"
    (
        export OUTPUT_NAME="$name" PREPARE_EVAL_MODELS=0 CUDA_VISIBLE_DEVICES=''
        exec bash "$root/evaluate_all.sh"
    ) >> "$root/evaluation.log" 2>&1 &
    eval_pids+=("$!"); eval_rounds+=("$round")
done
for i in "${!eval_pids[@]}"; do
    if wait "${eval_pids[$i]}"; then echo "${eval_rounds[$i]} evaluation finished"; else eval_failures+=("${eval_rounds[$i]}"); echo "${eval_rounds[$i]} evaluation failed or incomplete"; fi
done
if ((${#gen_failures[@]} || ${#eval_failures[@]})); then
    status "partial generation_failed=${gen_failures[*]:-none} evaluation_failed=${eval_failures[*]:-none}; retry with start_final_night.sh or --eval-only"
else
    status 'completed: both generation and evaluation finished'
fi
# Assemble only compact per-round exports and small orchestration status/log.
"$PYTHON" - "$NIGHT" "$PROJECT" <<'PY'
from pathlib import Path
import sys,zipfile,os
night,project=map(Path,sys.argv[1:])
out=project/'R84_R85_FinalNight_results.zip'; tmp=out.with_suffix('.zip.tmp')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in (night/'final_night.status',night/'final_night.log'):
        if path.is_file(): z.write(path,path.name)
    for label in ('R84_FinalNight','R85_FinalNight'):
        candidates=[project/(label+s+'.zip') for s in ('','_evaluation_incomplete','_evaluation_failed','_generation_pending')]
        existing=[p for p in candidates if p.is_file()]
        if existing:
            path=max(existing,key=lambda p:p.stat().st_mtime_ns)
            z.write(path,path.name)
os.replace(tmp,out)
print('FINAL BUNDLE',out,flush=True)
PY
if ((${#gen_failures[@]} || ${#eval_failures[@]})); then exit 2; fi
