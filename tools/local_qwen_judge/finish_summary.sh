#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=/root/autodl-tmp/MINT_Flow_Jupyter_Ready
PYTHON="${QWEN_PYTHON:-$BASE/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3 || command -v python)"; fi
OUT="${QWEN_JUDGE_OUT:-/root/autodl-tmp/A_Qwen_LocalJudge_results}"
exec 9>"$HERE/.judge.lock"
echo 'Waiting for the active judge to exit; downloads and scoring continue unchanged.'
flock 9
if [[ ! -f "$OUT/jobs_private.json" || ! -f "$OUT/protocol.json" ]]; then
 echo "No saved judgments manifest yet at $OUT; finish model download and scoring first." >&2
 exit 1
fi
"$PYTHON" "$HERE/judge.py" summarize --out "$OUT"
"$PYTHON" - "$OUT" <<'PY'
import json, pathlib, sys, zipfile
p=pathlib.Path(sys.argv[1]); target=p.with_suffix('.zip'); tmp=target.with_suffix('.zip.tmp')
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(p.rglob('*')):
        if f.is_file() and not f.name.endswith('.tmp'):
            z.write(f,pathlib.Path(p.name)/f.relative_to(p))
tmp.replace(target)
summary=json.loads((p/'summary.json').read_text(encoding='utf-8'))
for arm,report in summary['arms'].items():
    d=report['joint'];print(arm,'joint determined:',d['rated_outputs'],'/',d['eligible_outputs'],'unresolved:',d['unresolved_outputs'],'bounds:',d['output_rate_lower'],d['output_rate_upper'])
print('Updated result package:',target)
PY
