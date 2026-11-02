"""Download the fixed local judge snapshot with cache-preserving retries."""
import argparse
import json
import os
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--model-dir', required=True)
p.add_argument('--revision', default=os.environ.get('QWEN_REVISION', 'cc594898137f460bfe9f0759e9844b3ce807cfb5'))
p.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT'))
a = p.parse_args()
# The Hub reads these at import time. Apply them before importing it.
if a.endpoint:
    os.environ['HF_ENDPOINT'] = a.endpoint
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN', '1')
os.environ['HF_HUB_OFFLINE'] = '0'
from huggingface_hub import HfApi, hf_hub_download

model = 'Qwen/Qwen2.5-VL-7B-Instruct'
target = Path(a.model_dir).resolve()
target.mkdir(parents=True, exist_ok=True)
attempts = max(1, int(os.environ.get('QWEN_DOWNLOAD_ATTEMPTS', '6')))

def retry(label, fn):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            print(f'{label}: attempt {attempt}/{attempts} failed ({type(exc).__name__}). Cached files retained.', flush=True)
            if attempt == attempts:
                raise
            delay = min(30, 5 * attempt)
            print(f'Retrying in {delay}s...', flush=True)
            time.sleep(delay)

manifest = target / 'judge_download_manifest.json'
if manifest.is_file():
    m = json.loads(manifest.read_text(encoding='utf-8'))
    if m['model'] != model or m['requested_revision'] != a.revision:
        raise RuntimeError('Existing download manifest differs; select another model directory for another revision.')
else:
    info = retry('Repository metadata', lambda: HfApi(endpoint=a.endpoint).model_info(model, revision=a.revision, timeout=120))
    # Inference configuration, tokenizer/processor files and one weight format only.
    names = [s.rfilename for s in info.siblings if '/' not in s.rfilename and s.rfilename.endswith(('.json', '.txt', '.model', '.tiktoken', '.safetensors'))]
    names.sort(key=lambda name: (name.endswith('.safetensors'), name))
    if 'config.json' not in names or not any(n.endswith('.safetensors') for n in names):
        raise RuntimeError('Repository lacks required inference files.')
    m = {'model': model, 'requested_revision': a.revision, 'revision': info.sha, 'files': names}
    temp = manifest.with_suffix('.tmp')
    temp.write_text(json.dumps(m, indent=2), encoding='utf-8')
    temp.replace(manifest)
print('Snapshot:', m['revision'], 'Files:', len(m['files']), 'Destination:', target, flush=True)
for index, name in enumerate(m['files'], 1):
    print(f'[{index}/{len(m["files"])}] {name}', flush=True)
    retry(name, lambda name=name: hf_hub_download(repo_id=model, filename=name, revision=m['revision'], local_dir=str(target), endpoint=a.endpoint, etag_timeout=120))
    print('Ready:', name, flush=True)
metadata = target / 'judge_model_revision.json'
temp = metadata.with_suffix('.tmp')
temp.write_text(json.dumps({'model': model, 'revision': m['revision']}, indent=2), encoding='utf-8')
temp.replace(metadata)
print('Download complete. Starting local GPU model preflight next.', flush=True)
