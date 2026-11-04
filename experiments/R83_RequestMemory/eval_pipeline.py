"""Offline CPU evaluation and export; callable after generation or for R81 server results.

No installation, download, generation, or shared GPU queue lock. Configuration:
CLIP_PATH / DINO_PATH or source ../evaluation_models/R81_paths.env first.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import reporting as report

HERE = Path(__file__).resolve().parent
REQUIRED = ('clip_target_similarity', 'clip_directional_similarity',
            'clip_image_similarity', 'dino_full_feature_similarity',
            'dino_pattern_crop_feature_similarity')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(temp, path)


def local_model(variable, project, repo):
    configured = os.environ.get(variable)
    if configured:
        return Path(configured).expanduser().resolve()
    # Read existing Hugging Face refs/main only; never choose a snapshot by output.
    bases = [project / 'evaluation_models',
             Path(os.environ.get('HF_HUB_CACHE', Path.home()/'.cache/huggingface/hub')),
             Path(os.environ.get('HF_HOME', Path.home()/'.cache/huggingface')) / 'hub']
    for base in bases:
        cache = base / ('models--'+repo.replace('/', '--'))
        ref = cache / 'refs/main'
        if ref.is_file():
            candidate = cache / 'snapshots' / ref.read_text().strip()
            if candidate.is_dir():
                return candidate.resolve()
    raise RuntimeError(variable+' unset and no existing refs/main snapshot for '+repo)


def validate_model(variable, path):
    if not (path/'config.json').is_file() or not (path/'preprocessor_config.json').is_file():
        raise RuntimeError(variable+' missing config/processor: '+str(path))
    if not list(path.glob('*.safetensors')) + list(path.glob('pytorch_model*.bin')):
        raise RuntimeError(variable+' missing local weights: '+str(path))
    for index in path.glob('*.index.json'):
        for shard in set(report.read(index).get('weight_map', {}).values()):
            if not (path/shard).is_file():
                raise RuntimeError('Missing evaluator shard: '+str(path/shard))
    if variable == 'CLIP_PATH' and not (path/'tokenizer.json').is_file() and not all((path/n).is_file() for n in ('vocab.json','merges.txt')):
        raise RuntimeError('CLIP tokenizer files missing: '+str(path))


def readiness(project):
    versions = {}
    for package in ('torch', 'transformers', 'numpy', 'Pillow'):
        versions[package] = importlib.metadata.version(package)
    try:
        versions['scikit-image'] = importlib.metadata.version('scikit-image')
    except importlib.metadata.PackageNotFoundError:
        versions['scikit-image'] = 'unavailable; evaluator exports pixel_ssim_status'
    paths = dict(CLIP_PATH=local_model('CLIP_PATH', project, 'openai/clip-vit-large-patch14'),
                 DINO_PATH=local_model('DINO_PATH', project, 'facebook/dinov2-base'))
    for variable, path in paths.items():
        validate_model(variable, path)
        os.environ[variable] = str(path)
    return dict(versions=versions, paths={k:str(v) for k,v in paths.items()},
                readiness_scope='installed metadata and local files; model load occurs in evaluator')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=HERE, help='Experiment folder with plan.json and results/')
    args = p.parse_args()
    root = args.root.resolve()
    out = root/'results'
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS=os.environ.get('EVAL_CPU_THREADS','2'),
               MKL_NUM_THREADS=os.environ.get('EVAL_CPU_THREADS','2'))
    # Only serialize this experiment's writes; unrelated GPU work proceeds freely.
    with report.file_lock(root/'.evaluation.lock'), report.file_lock(root/'.results.lock'):
        plan = report.read(root/'plan.json')
        config = dict(status='preparing', device='cpu', executable=sys.executable,
                      required_metrics=list(REQUIRED), lpips_status='not_requested', root=str(root))
        failure = None
        try:
            if root == HERE:
                report.package(root)
            config.update(readiness(root.parent))
            env.update(config['paths'])
            report.save(out/'evaluation_config.json', config)
            command = [sys.executable, '-u', str(HERE/'evaluate.py'), '--from-results', str(out),
                       '--clip-path', config['paths']['CLIP_PATH'], '--dino-path', config['paths']['DINO_PATH'],
                       '--device', 'cpu', '--data-dir', str(root/'data')]
            if os.environ.get('LPIPS_PATH') or os.environ.get('LPIPS_TRUNK_PATH'):
                if not all(os.environ.get(k) for k in ('LPIPS_PATH','LPIPS_TRUNK_PATH')):
                    raise RuntimeError('LPIPS_PATH and LPIPS_TRUNK_PATH must both be set')
                importlib.metadata.version('lpips')
                command += ['--lpips-path', os.environ['LPIPS_PATH'], '--lpips-trunk-path', os.environ['LPIPS_TRUNK_PATH']]
                config['lpips_status'] = 'requested'
            config['command'] = command
            began = time.time_ns()
            config.update(status='running', pid=os.getpid(), started=time.time(), device='cpu')
            report.save(out/'evaluation_config.json', config)
            result = subprocess.run(command, env=env)
            config['evaluator_returncode'] = result.returncode
            csv_path = out/'evaluation.csv'
            if result.returncode and (not csv_path.exists() or csv_path.stat().st_mtime_ns < began):
                raise RuntimeError('Evaluator exited before exporting new rows (code '+str(result.returncode)+')')
        except Exception as exc:
            failure = str(exc)
            traceback.print_exc()
        existing = {}
        if (out/'evaluation.csv').exists():
            with (out/'evaluation.csv').open(encoding='utf-8-sig', newline='') as f:
                existing = {r['id']:r for r in csv.DictReader(f)}
        rows = []
        for job in plan['jobs']:
            row = dict(existing.get(job['id'], {}))
            row.update(id=job['id'], case=job['case'], task=job['task'], seed=job['seed'],
                       arm=job.get('arm', job.get('label')), target_category=job.get('target_category',''))
            for metric in REQUIRED:
                row.setdefault(metric, '')
            if failure:
                row.update(eval_status='pipeline_error', eval_error=failure)
                for metric in REQUIRED:
                    row[metric] = ''
            elif row.get('eval_status') != 'ok' or any(row.get(m) in ('', None) for m in REQUIRED):
                row.update(eval_status=('incomplete_metrics' if row.get('eval_status') == 'ok' else row.get('eval_status', 'missing_evaluation')),
                           eval_error=row.get('eval_error') or 'Required metric missing; inspect per-metric status')
            row['lpips_status'] = row.get('lpips_status', config['lpips_status'])
            rows.append(row)
        write_csv(out/'evaluation.csv', rows)
        incomplete = [r['id'] for r in rows if r.get('eval_status') != 'ok' or any(r.get(m) in ('',None) for m in REQUIRED)]
        config.update(status='failed' if failure else ('incomplete' if incomplete else 'completed'),
                      error=failure, planned=len(rows), incomplete_ids=incomplete)
        report.save(out/'evaluation_config.json', config)
        # Export references separately; their compact JPEGs cannot supply native metrics.
        if root == HERE:
            try:
                report.package(root)
                cmd = [sys.executable, str(HERE/'analyze.py'), '--metrics', str(out/'comparison_metrics.csv'),
                       '--evaluation', str(out/'evaluation.csv'), '--reference', 'AUTO_DETAIL']
                with (out/'statistics.json.tmp').open('w', encoding='utf-8') as f:
                    result = subprocess.run(cmd, env=env, stdout=f)
                if result.returncode:
                    raise RuntimeError('analyze.py failed; see evaluation.log')
                os.replace(out/'statistics.json.tmp', out/'statistics.json')
                report.package(root)
            except Exception as exc:
                config.update(status='failed', export_error=str(exc))
                report.save(out/'evaluation_config.json', config)
                traceback.print_exc()
        # A dependency failure still yields an explicit, portable metric-status export.
        import zipfile
        target = out/'evaluation_export.zip'
        with zipfile.ZipFile(target.with_suffix('.zip.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
            for name in ('evaluation.csv','evaluation_config.json','statistics.json','category_summary.csv'):
                if (out/name).is_file():
                    z.write(out/name, name)
        os.replace(target.with_suffix('.zip.tmp'), target)
        print(json.dumps(config, ensure_ascii=False), flush=True)
        return 0 if config['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
