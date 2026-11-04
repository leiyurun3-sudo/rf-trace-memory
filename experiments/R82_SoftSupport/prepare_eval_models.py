"""Download/cache evaluation weights on CPU, independently of generation."""
import os
from pathlib import Path
import shlex
from huggingface_hub import snapshot_download

project = Path(__file__).resolve().parent.parent
cache = project / 'evaluation_models'
cache.mkdir(exist_ok=True)
models = [('CLIP_PATH', 'openai/clip-vit-large-patch14', ['*.json', '*.txt', 'pytorch_model.bin']),
          ('DINO_PATH', 'facebook/dinov2-base', ['*.json', 'model.safetensors'])]
lines = []
for variable, repo, patterns in models:
    path = os.environ.get(variable)
    if not path:
        print('Preparing evaluation weights: '+repo, flush=True)
        path = snapshot_download(repo, cache_dir=str(cache), allow_patterns=patterns)
    lines.append('export '+variable+'='+shlex.quote(path))
    print(variable+'='+path, flush=True)
(cache/'R81_paths.env').write_text('\n'.join(lines)+'\n', encoding='utf-8')
