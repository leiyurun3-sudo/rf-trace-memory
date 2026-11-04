"""Prepare local CLIP/DINO paths; only --download permits fetching missing files."""
import argparse
import json
import os
from pathlib import Path
import shlex
import time
import traceback
from eval_pipeline import local_model, readiness, validate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch missing official weights using existing huggingface_hub; no installation')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    cache = project/'evaluation_models'
    cache.mkdir(exist_ok=True)
    status_path = cache/'R83_prepare_status.json'
    def save_status(status):
        status.update(pid=os.getpid(), updated=time.time())
        temp_status = status_path.with_suffix('.json.tmp')
        temp_status.write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        os.replace(temp_status, status_path)
    save_status(dict(status='preparing', download_allowed=args.download))
    try:
        for variable, repo in (('CLIP_PATH','openai/clip-vit-large-patch14'), ('DINO_PATH','facebook/dinov2-base')):
            try:
                path = local_model(variable, project, repo)
                validate_model(variable, path)
            except RuntimeError:
                if not args.download:
                    raise
                print('Downloading missing evaluation model: '+repo, flush=True)
                save_status(dict(status='downloading', repo=repo, variable=variable))
                # Clear offline flags only here, before importing huggingface_hub.
                flags = ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE')
                previous = {key:os.environ.pop(key, None) for key in flags}
                try:
                    from huggingface_hub import snapshot_download
                    path = Path(snapshot_download(repo_id=repo, cache_dir=str(cache),
                        allow_patterns=['*.json','*.txt',
                                        'pytorch_model*.bin' if variable == 'CLIP_PATH' else '*.safetensors'],
                        ignore_patterns=['tf_model*','flax_model*','*.h5','*.msgpack','onnx/*','openvino/*']))
                finally:
                    for key, value in previous.items():
                        if value is not None:
                            os.environ[key] = value
                validate_model(variable, path)
            os.environ[variable] = str(path.resolve())
            save_status(dict(status='model_ready', variable=variable, path=str(path.resolve())))
        config = readiness(project)
        lines = ['export '+variable+'='+shlex.quote(path) for variable,path in config['paths'].items()]
        target = cache/'R83_paths.env'
        temp = target.with_suffix('.env.tmp')
        temp.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        os.replace(temp, target)
        status = dict(status='prepared', **config)
        save_status(status)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        print('Saved', target, flush=True)
        return 0
    except Exception as exc:
        save_status(dict(status='failed', error=str(exc)))
        print('EVALUATION MODEL PREPARATION FAILED: '+str(exc), flush=True)
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
