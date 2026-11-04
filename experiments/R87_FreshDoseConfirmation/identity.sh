#!/usr/bin/env bash
# Independent R85 40-source identity calibration; default 80 new observations.
# No download, installation, R87 generation, preservation/artifact judging, or 520-score writes.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${1:-}" == --help ]]; then
  echo 'Usage: bash identity.sh [--strict-model-hash] [--reuse-source-dir DIR]'
  echo 'Optional existing-local-path overrides: R87_SERVER_PROJECT R87_IDENTITY_PYTHON R85_ROOT QWEN_MODEL IDENTITY_OUT IDENTITY_JUDGE_DIR IDENTITY_TASKS IDENTITY_PLAN'
  echo 'Default: 40 R85 sources, 80 fresh observations plus bounded retries; independent ZIP /root/autodl-tmp/R87_IdentityControl_results.zip'
  exit 0
fi
PROJECT="${R87_SERVER_PROJECT:-/root/autodl-tmp/MINT_Flow_Jupyter_Ready}"
[[ -d "$PROJECT" ]] || { echo "Shared GPU-lock project missing: $PROJECT (set R87_SERVER_PROJECT)." >&2; exit 1; }
command -v flock >/dev/null || { echo 'flock is required for shared GPU exclusion.' >&2; exit 1; }

# Explicit overrides are authoritative; never silently replace an invalid override.
PY="${R87_IDENTITY_PYTHON:-${JUDGE_PYTHON:-}}"
if [[ -z "$PY" ]]; then
  for candidate in "$PROJECT/.venv/bin/python" /root/autodl-tmp/.venv/bin/python; do
    if [[ -x "$candidate" ]]; then PY="$candidate"; break; fi
  done
fi
[[ -n "$PY" && -x "$PY" ]] || { echo 'Existing Python environment missing; set R87_IDENTITY_PYTHON.' >&2; exit 1; }
ROOT="${R85_ROOT:-}"
if [[ -z "$ROOT" ]]; then
  for candidate in "$PROJECT/R84_R85_FinalNight/R85_IndependentComparison" \
                   "$PROJECT/R85_IndependentComparison" \
                   /root/autodl-tmp/R84_R85_FinalNight/R85_IndependentComparison \
                   /root/autodl-tmp/R85_IndependentComparison; do
    if [[ -f "$candidate/data/000000000100/source.png" ]]; then ROOT="$candidate"; break; fi
  done
fi
[[ -n "$ROOT" && -d "$ROOT/data" ]] || { echo 'Native R85 source root missing; set R85_ROOT.' >&2; exit 1; }
MODEL="${QWEN_MODEL:-}"
if [[ -z "$MODEL" ]]; then
  for candidate in /root/autodl-tmp/local-models/Qwen2.5-VL-7B-Instruct \
                   "$PROJECT/local-models/Qwen2.5-VL-7B-Instruct" \
                   /root/autodl-tmp/Qwen2.5-VL-7B-Instruct; do
    if [[ -f "$candidate/model.safetensors.index.json" ]]; then MODEL="$candidate"; break; fi
  done
fi
[[ -n "$MODEL" && -f "$MODEL/model.safetensors.index.json" ]] || { echo 'Complete local Qwen model missing; set QWEN_MODEL.' >&2; exit 1; }
SUPPORT="${IDENTITY_JUDGE_DIR:-$HERE/identity_support}"
TASKS="${IDENTITY_TASKS:-$SUPPORT/tasks.json}"
PLAN="${IDENTITY_PLAN:-$SUPPORT/r85_plan.json}"
for item in "$SUPPORT/judge.py" "$SUPPORT/aggregate.py" "$SUPPORT/calibration_lib.py" "$TASKS" "$PLAN"; do
  [[ -f "$item" ]] || { echo "Identity support file missing: $item" >&2; exit 1; }
done
OUT="${IDENTITY_OUT:-/root/autodl-tmp/R87_IdentityControl_results}"
ZIP=/root/autodl-tmp/R87_IdentityControl_results.zip
# Prevent ZIP-under-output recursion and accidental archive packaging of unrelated data.
OUT_REAL="$(realpath -m -- "$OUT")"
ZIP_REAL="$(realpath -m -- "$ZIP")"
[[ "$ZIP_REAL" != "$OUT_REAL" && "$ZIP_REAL" != "$OUT_REAL/"* ]] || { echo 'Identity ZIP must be outside its output directory.' >&2; exit 1; }

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Dedicated launch lock plus the SAME project locks/order used by the R87 runner.
# FDs remain owned by this shell while the foreground GPU child is running.
exec 7>>"$HERE/.identity_launch.lock"
flock -n 7 || { echo 'Identity calibration already running or queued.' >&2; exit 2; }
exec 8>>"$PROJECT/.ab_queue.lock"
printf 'Waiting for shared GPU queue: %s\n' "$PROJECT/.ab_queue.lock"
flock 8
exec 9>>"$PROJECT/.mint_runner.lock"
printf 'Waiting for shared GPU runner: %s\n' "$PROJECT/.mint_runner.lock"
flock 9

args=(--judge-dir "$SUPPORT" --tasks "$TASKS" --plan "$PLAN" --root "$ROOT" --model "$MODEL" --out "$OUT")
if [[ -n "${IDENTITY_REUSE_SOURCE_DIR:-}" ]]; then args+=(--reuse-source-dir "$IDENTITY_REUSE_SOURCE_DIR"); fi
if [[ "${IDENTITY_STRICT_MODEL_HASH:-0}" == 1 ]]; then args+=(--strict-model-hash); fi
# Only safe driver options may be forwarded; output/root overrides use the
# environment above so the packaging path cannot diverge from the run path.
while (( $# )); do
  case "$1" in
    --strict-model-hash) args+=("$1"); shift ;;
    --reuse-source-dir)
      (( $# >= 2 )) || { echo '--reuse-source-dir requires DIR' >&2; exit 2; }
      args+=("$1" "$2"); shift 2 ;;
    *) echo "Unknown identity option: $1 (use --help)" >&2; exit 2 ;;
  esac
done
RUN_STARTED=0
finish() {
  rc=$?
  trap - EXIT
  if [[ "$RUN_STARTED" == 1 && -f "$OUT/identity_control_protocol.json" ]]; then
    if ! "$PY" - "$OUT" "$ZIP" <<'PYZIP'
import json, os, sys, zipfile
from pathlib import Path
root, target = (Path(v).resolve() for v in sys.argv[1:])
protocol = json.loads((root / 'identity_control_protocol.json').read_text(encoding='utf-8'))
if protocol.get('version') != 'r87-source-as-output-edit-only-v2':
    raise SystemExit('Refusing non-identity output archive')
target.parent.mkdir(parents=True, exist_ok=True)
temp = target.with_name(target.name + '.tmp')
with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob('*')):
        if path.is_file() and not path.name.endswith('.tmp'):
            archive.write(path, root.name + '/' + path.relative_to(root).as_posix())
os.replace(temp, target)
print('Independent identity result ZIP:', target, flush=True)
PYZIP
    then
      echo 'Identity packaging failed; raw output directory is preserved.' >&2
      if [[ "$rc" == 0 ]]; then rc=1; fi
    fi
  fi
  exit "$rc"
}
trap finish EXIT
printf 'Identity only: 40 R85 sources; default 80 observations plus bounded schema retries.\nPython=%s\nR85=%s\nQwen=%s\nOutput=%s\n' "$PY" "$ROOT" "$MODEL" "$OUT"
RUN_STARTED=1
"$PY" -u "$HERE/identity_control.py" "${args[@]}"
