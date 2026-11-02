#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 3 ]]; then echo "Usage: bash run_local_judge.sh {preflight|run|summarize} R85_ROOT MODEL_DIR [OUTPUT_DIR] [--repeat-count N]"; exit 2; fi
MODE="$1"; ROOT="$2"; MODEL="$3"; OUT="${4:-$HERE/judge_output}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "$MODE" in
 preflight) python "$HERE/judge.py" preflight --model "$MODEL" ;;
 run) shift 3; if [[ $# -gt 0 && "$1" != --* ]]; then shift; fi; python "$HERE/judge.py" run --root "$ROOT" --model "$MODEL" --out "$OUT" "$@" ;;
 summarize) python "$HERE/judge.py" summarize --out "$OUT" ;;
 *) echo "Unknown mode: $MODE"; exit 2 ;;
esac
