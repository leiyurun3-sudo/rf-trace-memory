#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$(dirname -- "$ROOT")/.venv/bin/python}"
: "${CLIP_PATH:?Set CLIP_PATH to an existing local Hugging Face CLIP model directory}"
: "${DINO_PATH:?Set DINO_PATH to an existing local Hugging Face DINOv2 model directory}"
args=(--from-results "$ROOT/results" --clip-path "$CLIP_PATH" --dino-path "$DINO_PATH" --device "${EVAL_DEVICE:-cpu}")
if [[ -n "${LPIPS_PATH:-}" ]]; then
  : "${LPIPS_TRUNK_PATH:?Set local AlexNet weights with LPIPS_PATH}"
  args+=(--lpips-path "$LPIPS_PATH" --lpips-trunk-path "$LPIPS_TRUNK_PATH")
fi
"$PYTHON" "$ROOT/evaluate.py" "${args[@]}"
"$PYTHON" "$ROOT/analyze.py" --metrics "$ROOT/results/metrics.csv" --evaluation "$ROOT/results/evaluation.csv" --reference AUTO_DETAIL > "$ROOT/results/statistics.json"
"$PYTHON" "$ROOT/runner.py" --pack
