"""Record existing local CLIP_PATH/DINO_PATH; no installation or downloads."""
import json
from pathlib import Path
import shlex
from eval_pipeline import readiness

project = Path(__file__).resolve().parent.parent
config = readiness(project)
cache = project / 'evaluation_models'
cache.mkdir(exist_ok=True)
lines = ['export '+variable+'='+shlex.quote(path) for variable,path in config['paths'].items()]
(cache/'R82_paths.env').write_text('\n'.join(lines)+'\n', encoding='utf-8')
print(json.dumps(config, ensure_ascii=False, indent=2))
print('Saved', cache/'R82_paths.env')
