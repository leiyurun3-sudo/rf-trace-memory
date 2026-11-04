#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
export R87_SERVER_PROJECT="${R87_SERVER_PROJECT:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready}"
PYTHON="${PYTHON:-$R87_SERVER_PROJECT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "Python missing: $PYTHON"; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
STAGE=initializing
trap 'code=$?; if ((code)); then printf "Stopped in %s, code=%s. Existing outputs kept; rerun start.sh to resume.\n" "$STAGE" "$code"; fi' EXIT
STAGE=generation
"$PYTHON" -u run.py --generate
STAGE=CPU_evaluation
"$PYTHON" -u run.py --evaluate
STAGE=source_statistics
"$PYTHON" -u analyze.py
STAGE=final_package
"$PYTHON" -u run.py --package
DELIVERY="${R87_DELIVERY_DIR:-/root/autodl-tmp}"
mkdir -p -- "$DELIVERY"
ARCHIVE="$ROOT/R87_FreshDoseConfirmation_results.zip"
[[ -s "$ARCHIVE" ]] || { echo "Missing final package: $ARCHIVE"; exit 2; }
if [[ "$ARCHIVE" != "$DELIVERY/R87_FreshDoseConfirmation_results.zip" ]]; then
  cp -- "$ARCHIVE" "$DELIVERY/R87_FreshDoseConfirmation_results.zip.tmp"
  mv -- "$DELIVERY/R87_FreshDoseConfirmation_results.zip.tmp" "$DELIVERY/R87_FreshDoseConfirmation_results.zip"
fi
printf 'GENERATION + METRICS COMPLETE: %s/R87_FreshDoseConfirmation_results.zip\n' "$DELIVERY"
if [[ "${RUN_IDENTITY:-1}" == 1 ]]; then
  STAGE=Qwen_identity_control
  bash "$ROOT/identity.sh"
fi
STAGE=complete
date -Is > "$ROOT/complete.txt"
printf 'COMPLETE. Results are in %s\n' "$DELIVERY"

