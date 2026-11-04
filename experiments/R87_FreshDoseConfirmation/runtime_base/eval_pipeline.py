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
        candidate = Path(configured).expanduser().resolve()
        try:
            validate_model(variable, candidate)
            return candidate
        except RuntimeError:
            pass  # A stale paths.env must not hide a valid shared cache.
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


def pipeline_status(root, status, **values):
    data = dict(status=status, updated=time.time(), pid=os.getpid(), **values)
    report.save(root/'pipeline_status.json', data)
    print(json.dumps(data, ensure_ascii=False), flush=True)
    return data


def has_required(row):
    import math
    try:
        if row.get('eval_status') != 'ok':
            return False
        if row.get('lpips_status') != 'ok' or not math.isfinite(float(row.get('lpips_full', ''))):
            return False
        preservation = row.get('lpips_preserved_status')
        if preservation == 'ok_receptive_field_spill':
            if not math.isfinite(float(row.get('lpips_preserved', ''))):
                return False
        elif preservation != 'empty_region':
            return False
        directional = 'clip_directional_similarity'
        if not all(math.isfinite(float(row[m])) for m in REQUIRED if m != directional):
            return False
        # An explicitly undefined zero direction is a completed measurement,
        # not a zero score or a failed evaluator. Keep the score empty.
        if row.get('clip_directional_status') == 'zero_direction' and row.get(directional) in (None, ''):
            return True
        return math.isfinite(float(row[directional]))
    except (ValueError, TypeError, KeyError):
        return False


def publish(root, status, output_name):
    os.environ['OUTPUT_NAME'] = output_name
    report.package(root)
    # Preserve compact media policy; add only small status/log files.
    import zipfile
    archive = report.locations(root)[2]
    with zipfile.ZipFile(archive, 'a', zipfile.ZIP_DEFLATED) as z:
        z.write(root/'pipeline_status.json', 'pipeline_status.json')
        if (root/'evaluation_prepare_status.json').is_file():
            z.write(root/'evaluation_prepare_status.json', 'evaluation_prepare_status.json')
        for name in ('evaluation.log','evaluation_prepare.log'):
            path = root/name
            if path.is_file():
                with path.open(encoding='utf-8', errors='replace') as f:
                    from collections import deque
                    tail = ''.join(deque(f, maxlen=100))[-32000:]
                z.writestr(name+'.tail.txt', tail)
    print('PUBLISHED '+status+' '+str(archive), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=HERE, help='Original experiment directory')
    p.add_argument('--status', help='Write orchestration status only; no model loading or packaging')
    p.add_argument('--skip', action='store_true')
    p.add_argument('--preparation-error-file', type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    if args.status:
        pipeline_status(root, args.status)
        return 0
    out = root/'results'
    plan_path = root/'plan.json'
    if not plan_path.is_file():
        plan_path = out/'plan.json'
    plan = report.read(plan_path)
    out.mkdir(parents=True, exist_ok=True)
    output_name = os.environ.get('OUTPUT_NAME') or root.name
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS=os.environ.get('EVAL_CPU_THREADS','2'),
               MKL_NUM_THREADS=os.environ.get('EVAL_CPU_THREADS','2'))
    with report.file_lock(root/'.evaluation.lock'), report.file_lock(root/'.results.lock'):
        config = dict(status='preparing', device='cpu', executable=sys.executable,
                      required_metrics=list(REQUIRED), root=str(root), lpips_status=os.environ.get('LPIPS_STATUS','not_requested'))
        config['native_png_audit'] = [dict(id=j['id'], raw_exists=(out/j['id']/'raw.png').is_file(),
            record_exists=(out/j['id']/'record.json').is_file(),
            source_exists=(out/'sources'/(j['case']+'_source_eval.png')).is_file()) for j in plan['jobs']]
        config['lpips_policy'] = 'full LPIPS required; preserved LPIPS required except explicit empty region; spatial-map mask weighting allows receptive-field spill'
        config['clip_directional_policy'] = 'finite score or explicitly reported zero_direction with empty score; other four required metrics must be finite'
        report.save(out/'evaluation_config.json', config)
        failure = None
        use_existing = False
        if args.preparation_error_file:
            detail = report.read(args.preparation_error_file) if args.preparation_error_file.is_file() else {}
            failure = 'Model preparation failed: '+str(detail.get('error', args.preparation_error_file))
            config['preparation'] = detail
        elif not args.skip:
            try:
                config.update(readiness(root.parent))
                env.update(config['paths'])
                command = [sys.executable, '-u', str(HERE/'evaluate.py'), '--from-results', str(out),
                           '--clip-path', config['paths']['CLIP_PATH'], '--dino-path', config['paths']['DINO_PATH'],
                           '--device', 'cpu', '--data-dir', str(root/'data'),
                           '--lpips-status', os.environ.get('LPIPS_STATUS','not_requested'),
                           '--lpips-error', os.environ.get('LPIPS_ERROR','')]
                for variable, option in (('LPIPS_PATH','--lpips-path'),('LPIPS_TRUNK_PATH','--lpips-trunk-path')):
                    if os.environ.get(variable):
                        command += [option, os.environ[variable]]
                config.update(command=command, status='running', started=time.time(), pid=os.getpid())
                pipeline_status(root, 'evaluating')
                report.save(out/'evaluation_config.json', config)
                began = time.time_ns()
                result = subprocess.run(command, env=env)
                config['evaluator_returncode'] = result.returncode
                csv_path = out/'evaluation.csv'
                use_existing = csv_path.is_file() and (result.returncode == 0 or csv_path.stat().st_mtime_ns >= began)
                if result.returncode:
                    failure = 'Evaluator exited with code '+str(result.returncode)+'; inspect evaluation.log'
            except Exception as exc:
                failure = str(exc)
                traceback.print_exc()
        existing = {}
        if use_existing:
            with (out/'evaluation.csv').open(encoding='utf-8-sig', newline='') as f:
                existing = {row['id']:row for row in csv.DictReader(f)}
        rows = []
        for job in plan['jobs']:
            row = dict(existing.get(job['id'], {}))
            row.update(id=job['id'], case=job['case'], task=job['task'], seed=job['seed'],
                       arm=job.get('arm', job.get('label')), target_category=job.get('target_category',''))
            for metric in REQUIRED:
                row.setdefault(metric, '')
            if args.skip:
                row.update(eval_status='skipped_by_request', eval_error='SKIP_EVALUATION=1 explicitly requested')
            elif not has_required(row):
                row.update(eval_status=('preparation_failed' if args.preparation_error_file else
                                        'pipeline_error' if not use_existing else
                                        'incomplete_metrics' if row.get('eval_status')=='ok' else row.get('eval_status','missing_evaluation')),
                           eval_error=row.get('eval_error') or failure or 'Required metric missing; inspect metric status')
            row.setdefault('lpips_status', config['lpips_status'])
            row.setdefault('lpips_error', os.environ.get('LPIPS_ERROR',''))
            rows.append(row)
        write_csv(out/'evaluation.csv', rows)
        incomplete = [row['id'] for row in rows if not has_required(row)]
        config.update(status='skipped' if args.skip else 'failed' if failure else 'incomplete' if incomplete else 'completed',
                      error=failure, planned=len(rows), incomplete_ids=incomplete,
                      lpips_status_counts={s:sum(r['lpips_status']==s for r in rows) for s in sorted({r['lpips_status'] for r in rows})},
                      raw_audit={field:sum(str(r.get(field, '')).lower() == 'true' for r in rows) for field in ('eval_raw_exists','eval_source_exists','eval_record_exists')},
                      raw_audit_planned=len(rows))
        report.save(out/'evaluation_config.json', config)
        if root == HERE:
            try:
                # Refresh tables under a pending name, never the final archive name.
                pipeline_status(root, 'evaluation_'+config['status'], error=failure)
                publish(root, config['status'], output_name+'_evaluation_pending' if config['status']=='completed' else output_name+'_evaluation_'+config['status'])
                if config['status'] == 'completed':
                    reference = os.environ.get('EVAL_REFERENCE_ARM') or 'U25_KV'
                    config.update(statistics_status='running', statistics_reference=reference)
                    report.save(out/'evaluation_config.json', config)
                    pipeline_status(root, 'computing_statistics', reference=reference)
                    metrics_path = out/'comparison_metrics.csv'
                    with metrics_path.open(encoding='utf-8-sig', newline='') as f:
                        arms = {r.get('arm') or r.get('label') for r in csv.DictReader(f)}
                    if reference not in arms:
                        raise RuntimeError('Required evaluation reference arm absent: '+reference)
                    cmd = [sys.executable, str(HERE/'analyze.py'), '--metrics', str(metrics_path),
                           '--evaluation', str(out/'evaluation.csv'), '--reference', reference]
                    with (out/'statistics.json.tmp').open('w', encoding='utf-8') as f:
                        result = subprocess.run(cmd, env=env, stdout=f)
                    if result.returncode:
                        raise RuntimeError('analyze.py failed; inspect evaluation.log')
                    os.replace(out/'statistics.json.tmp', out/'statistics.json')
                    overnight = [sys.executable, '-u', str(HERE/'overnight_stats.py'), '--results', str(out)]
                    config['overnight_statistics_command'] = overnight
                    report.save(out/'evaluation_config.json', config)
                    result = subprocess.run(overnight, env=env)
                    config['overnight_statistics_returncode'] = result.returncode
                    if result.returncode:
                        raise RuntimeError('overnight_stats.py failed; inspect evaluation.log')
                    required_statistics = ('statistics.json', 'category_summary.csv',
                                           'evidence/overnight_source_values.csv',
                                           'evidence/overnight_arm_summary.csv',
                                           'evidence/overnight_comparisons.csv',
                                           'evidence/overnight_evidence.json')
                    missing = [name for name in required_statistics if not (out/name).is_file()]
                    if missing:
                        raise RuntimeError('Statistics output missing: '+', '.join(missing))
                    config.update(statistics_status='completed', statistics_files=list(required_statistics))
                    report.save(out/'evaluation_config.json', config)
                pipeline_status(root, config['status'], error=config.get('error'))
                publish(root, config['status'], output_name if config['status']=='completed' else output_name+'_evaluation_'+config['status'])
            except Exception as exc:
                config.update(status='failed', export_error=str(exc))
                if config.get('statistics_status') == 'running':
                    config['statistics_status'] = 'failed'
                report.save(out/'evaluation_config.json', config)
                pipeline_status(root, 'evaluation_failed', error=str(exc))
                traceback.print_exc()
        else:
            pipeline_status(root, config['status'], error=config.get('error'))
        import zipfile
        target = out/'evaluation_export.zip'
        with zipfile.ZipFile(target.with_suffix('.zip.tmp'), 'w', zipfile.ZIP_DEFLATED) as z:
            for name in ('evaluation.csv','evaluation_config.json'):
                if (out/name).is_file():
                    z.write(out/name, name)
            for path in report.statistics_artifacts(out):
                z.write(path, path.relative_to(out).as_posix())
            z.write(root/'pipeline_status.json', 'pipeline_status.json')
            if (root/'evaluation_prepare_status.json').is_file():
                z.write(root/'evaluation_prepare_status.json', 'evaluation_prepare_status.json')
        os.replace(target.with_suffix('.zip.tmp'), target)
        print(json.dumps(config, ensure_ascii=False), flush=True)
        return 0 if config['status'] in ('completed','skipped') else 2


if __name__ == '__main__':
    raise SystemExit(main())
