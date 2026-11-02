"""Prepare and CPU-load evaluators. Install LPIPS only on explicit EVAL_INSTALL_LPIPS_NO_DEPS=1."""
import argparse
import importlib.util
import subprocess
import sys
import gc
import json
import os
from pathlib import Path
import shlex
import shutil
import time
import traceback
from eval_pipeline import local_model, readiness, validate_model

ROOT = Path(__file__).resolve().parent
ROUND = ROOT.name.split('_')[0]
PROJECT = ROOT.parent
CACHE = PROJECT/'evaluation_models'


def save_status(**values):
    values.update(pid=os.getpid(), updated=time.time())
    for path in (ROOT/'evaluation_prepare_status.json', CACHE/(ROUND+'_prepare_status.json')):
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(values, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        os.replace(temp, path)
    print(json.dumps(values, ensure_ascii=False), flush=True)


def optional_lpips(download):
    result = dict(LPIPS_STATUS='unavailable', LPIPS_ERROR='')
    try:
        spec = importlib.util.find_spec('lpips')
        if spec is None and os.environ.get('EVAL_INSTALL_LPIPS_NO_DEPS') == '1':
            # Explicit opt-in: never let pip modify the existing torch stack.
            for dependency in ('torch', 'torchvision', 'numpy', 'scipy', 'tqdm'):
                if importlib.util.find_spec(dependency) is None:
                    raise RuntimeError('Missing existing dependency '+dependency+'; install it separately before LPIPS')
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', 'lpips==0.1.4'], check=True)
            importlib.invalidate_caches()
            spec = importlib.util.find_spec('lpips')
        if spec is None:
            result.update(LPIPS_STATUS='package_unavailable', LPIPS_ERROR='lpips is not installed; no pip attempted')
            return result
        folder = CACHE/'lpips'
        folder.mkdir(exist_ok=True)
        calibration = Path(os.environ.get('LPIPS_PATH') or folder/'alex_v0.1.pth').expanduser()
        if not calibration.is_file() and not os.environ.get('LPIPS_PATH'):
            bundled = Path(spec.origin).parent/'weights/v0.1/alex.pth'
            if bundled.is_file():
                shutil.copy2(bundled, calibration)
        trunk = Path(os.environ.get('LPIPS_TRUNK_PATH') or folder/'alexnet-owt-7be5be79.pth').expanduser()
        if not trunk.is_file() and not os.environ.get('LPIPS_TRUNK_PATH'):
            old = Path(os.environ.get('TORCH_HOME', Path.home()/'.cache/torch'))/'hub/checkpoints/alexnet-owt-7be5be79.pth'
            if old.is_file():
                shutil.copy2(old, trunk)
            elif download and os.environ.get('DOWNLOAD_LPIPS_MODELS', '0') == '1':
                from urllib.request import urlretrieve
                print('Downloading required AlexNet trunk from download.pytorch.org', flush=True)
                temp = trunk.with_suffix('.pth.part')
                urlretrieve('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth', temp)
                os.replace(temp, trunk)
        if calibration.is_file() and trunk.is_file():
            result.update(LPIPS_STATUS='ready', LPIPS_PATH=str(calibration.resolve()), LPIPS_TRUNK_PATH=str(trunk.resolve()))
        else:
            result.update(LPIPS_STATUS='weights_unavailable', LPIPS_ERROR='Need local LPIPS alex calibration and AlexNet trunk; optional DOWNLOAD_LPIPS_MODELS=1 allows trunk download')
    except Exception as exc:
        result.update(LPIPS_STATUS='preparation_failed', LPIPS_ERROR=str(exc))
    return result


def load_preflight(config, lpips):
    """Load each real evaluator on CPU, one at a time, before any generation."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoImageProcessor
    from evaluate import load_lpips
    clip_path, dino_path = config['paths']['CLIP_PATH'], config['paths']['DINO_PATH']
    CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
    model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to('cpu').eval()
    del model
    gc.collect()
    AutoImageProcessor.from_pretrained(dino_path, local_files_only=True)
    model = AutoModel.from_pretrained(dino_path, local_files_only=True).to('cpu').eval()
    del model
    gc.collect()
    model = load_lpips(Path(lpips['LPIPS_PATH']), Path(lpips['LPIPS_TRUNK_PATH']), 'cpu')
    del model
    gc.collect()
    return 'CLIP processor/model, DINO processor/model and LPIPS weights loaded on CPU; no inference'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)
    save_status(status='preparing', download_allowed=args.download)
    try:
        # Fail early on missing dependencies, before a large download or long generation.
        import importlib.metadata
        for package in ('torch','transformers','numpy','Pillow'):
            importlib.metadata.version(package)
        for variable, repo in (('CLIP_PATH','openai/clip-vit-large-patch14'),('DINO_PATH','facebook/dinov2-base')):
            try:
                path = local_model(variable, PROJECT, repo)
                validate_model(variable, path)
            except RuntimeError:
                if not args.download:
                    raise
                save_status(status='downloading', repo=repo)
                # Only this preparation process enables network, before importing the hub.
                for flag in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE'):
                    os.environ.pop(flag, None)
                from huggingface_hub import snapshot_download
                path = Path(snapshot_download(repo_id=repo, cache_dir=str(CACHE),
                    allow_patterns=['*.json','*.txt','pytorch_model*.bin' if variable=='CLIP_PATH' else '*.safetensors'],
                    ignore_patterns=['tf_model*','flax_model*','*.h5','*.msgpack','onnx/*','openvino/*']))
                validate_model(variable, path)
            os.environ[variable] = str(path.resolve())
            save_status(status='model_ready', model=variable, path=str(path.resolve()))
        config = readiness(PROJECT)
        lpips = optional_lpips(args.download)
        if lpips['LPIPS_STATUS'] != 'ready':
            raise RuntimeError('LPIPS required: '+lpips.get('LPIPS_ERROR', lpips['LPIPS_STATUS']))
        save_status(status='loading_preflight', device='cpu')
        config['actual_load_preflight'] = load_preflight(config, lpips)
        exports = dict(config['paths'], **lpips)
        text = ''.join('export '+key+'='+shlex.quote(value)+'\n' for key,value in exports.items())
        for name in (ROUND+'_paths.env', 'paths.env'):
            target = CACHE/name
            temp = target.with_suffix('.env.tmp')
            temp.write_text(text, encoding='utf-8')
            os.replace(temp, target)
        save_status(status='prepared', **config, lpips=lpips)
        return 0
    except Exception as exc:
        save_status(status='failed', error=str(exc))
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
