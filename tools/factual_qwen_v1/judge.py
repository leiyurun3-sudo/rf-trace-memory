"""Offline factual observations + frozen rules; no reuse of old judge labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback
import zipfile

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
HERE = Path(__file__).resolve().parent
VERSION = 'factual-judge-v1'
from calibration_lib import decode_json, prompt as control_prompt, validate as control_validate
from aggregate import summarize


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conjunction(values):
    values = list(values)
    if any(v == 0 for v in values if v is not None):
        return 0
    return 1 if values and all(v == 1 for v in values) else None


def factual_target(facts, target):
    if facts is None:
        return None
    return conjunction(None if facts[k] in (None, 'uncertain') else int(facts[k] == v)
                       for k, v in target.items())


def edit_label(source, candidate, task):
    end = factual_target(candidate, task['target_all'])
    start = factual_target(source, task['target_all'])
    # A candidate failing the target fails. A positive needs a discernible change.
    if end == 0:
        return 0
    return 1 if end == 1 and start == 0 else None


def observation_prompt(task):
    return ('Describe only visible facts in this ONE image. No desired edit is supplied. '
            'Do not assume a transformation or guess hidden details. Answer all fields below '
            'as one flat JSON object, with no Markdown. Boolean fields allow true, false, null; '
            'null means visually undecidable. Categorical fields use one listed value exactly. '
            'Use uncertain for ambiguous appearance; absent for an absent queried object. '
            'No explanation/evidence field is required. Fields: ' + json.dumps(task['fields'], ensure_ascii=False))


def validate_facts(value, task):
    if not isinstance(value, dict):
        raise ValueError('Expected flat JSON object')
    result = {}
    for key, spec in task['fields'].items():
        if key not in value:
            raise ValueError('Missing factual field: ' + key)
        v = value[key]
        if spec.get('type') == 'boolean':
            if v is not None and type(v) is not bool:
                raise ValueError(key + ' requires true, false, or null')
        elif v not in spec['values']:
            raise ValueError(key + ' requires one listed categorical value')
        result[key] = v
    return result


def comparison_prompt(task, reverse=False):
    ref, cand = ('SECOND', 'FIRST') if reverse else ('FIRST', 'SECOND')
    return (f'The {ref} image is the REFERENCE and the {cand} is the CANDIDATE. '
            'Compare only the protected content listed below. Method names and editing scores are hidden. '
            'Allowed changes: ' + task['allowed_changes'] + '\n'
            'For each check return retained, changed, not_visible_in_reference, or uncertain. '
            'Use not_visible_in_reference only if the check cannot be inspected because all its '
            'listed content is absent from the REFERENCE. Content present in the reference but '
            'missing from the candidate is changed, not not_visible_in_reference. '
            'Use uncertain when identity/detail cannot be resolved. Allow the stated changes and '
            'associated occlusion, without excusing unrelated changes. '
            'Also report severe_artifact: true if the candidate has clear unintended duplicated '
            'anatomy/objects, ghosting, broken geometry or severe rendering corruption; false if '
            'visibly absent; null if unclear. Intentional stylization is not itself an artifact. '
            'Return one flat JSON object, no Markdown, keys: '
            + json.dumps({p['id']: p['text'] for p in task['protected_checks']}, ensure_ascii=False)
            + ', plus severe_artifact. No explanation/evidence field is required.')


def validate_comparison(value, task):
    if not isinstance(value, dict):
        raise ValueError('Expected flat comparison JSON')
    result = {}
    for check in task['protected_checks']:
        key = check['id']
        if value.get(key) not in ('retained', 'changed', 'not_visible_in_reference', 'uncertain'):
            raise ValueError(key + ' requires a listed comparison status')
        result[key] = value[key]
    if 'severe_artifact' not in value:
        raise ValueError('Missing severe_artifact')
    v = value['severe_artifact']
    if v is not None and type(v) is not bool:
        raise ValueError('severe_artifact requires true, false, or null')
    result['severe_artifact'] = v
    return result


def preservation(comparison, task):
    if comparison is None:
        return None, None
    states = [comparison[p['id']] for p in task['protected_checks']]
    scores = [None if s == 'uncertain' else int(s == 'retained') for s in states
              if s != 'not_visible_in_reference']
    artifact = comparison['severe_artifact']
    return conjunction(scores), None if artifact is None else int(artifact)


def preflight(root, model, out):
    from PIL import Image
    plan, definitions = read(HERE / 'r85_plan.json'), read(HERE / 'tasks.json')
    tasks = {t['case']: t for t in definitions}
    source_tasks = {t['case']: t for t in plan['tasks']}
    if len(tasks) != 40 or set(tasks) != set(source_tasks) or len(plan['jobs']) != 520:
        raise ValueError('Expected frozen 40-source/520-output plan')
    for task in tasks.values():
        if not task['protected_checks'] or not set(task['target_all']) <= set(task['fields']):
            raise ValueError('Invalid fixed task schema: ' + task['case'])
        validate_facts({k: (None if s.get('type') == 'boolean' else s['values'][0])
                        for k, s in task['fields'].items()}, task)
        for k, value in task['target_all'].items():
            spec = task['fields'][k]
            if spec.get('type') == 'boolean' and type(value) is not bool:
                raise ValueError('Non-boolean target')
            if 'values' in spec and value not in spec['values']:
                raise ValueError('Target outside allowed values')
    inputs, sources, outputs = [], {}, {}
    def inspect(path, expected=None):
        if not path.is_file():
            raise FileNotFoundError('Required native input: ' + str(path))
        with Image.open(path) as im:
            if im.format != 'PNG':
                raise ValueError('Expected native PNG: ' + str(path))
            im.verify()
        digest = sha(path)
        if expected and expected != digest:
            raise ValueError('Source differs from frozen plan: ' + str(path))
        inputs.append({'path': str(path), 'sha256': digest})
    for case, task in source_tasks.items():
        path = root / task['source_image']
        inspect(path, plan.get('input_sha256', {}).get(task['source_image']))
        sources[case] = path
    for job in plan['jobs']:
        options = [root / 'results' / job['id'] / 'raw.png',
                   root / 'results' / 'R85' / job['id'] / 'raw.png', root / job['id'] / 'raw.png']
        path = next((p for p in options if p.is_file()), options[0])
        inspect(path)
        outputs[job['id']] = path
    index = read(model / 'model.safetensors.index.json')
    for shard in set(index['weight_map'].values()):
        p = model / shard
        if not p.is_file() or p.stat().st_size < 1024:
            raise FileNotFoundError('Incomplete local weight: ' + str(p))
    for name in ('config.json', 'preprocessor_config.json', 'tokenizer_config.json'):
        read(model / name)
    source_code = {name: sha(HERE / name) for name in ('judge.py', 'aggregate.py', 'tasks.json', 'r85_plan.json', 'calibration_lib.py')}
    protocol = {'version': VERSION, 'files': source_code, 'inputs': inputs,
                'model': str(model), 'model_config_sha256': sha(model / 'config.json'),
                'model_index_sha256': sha(model / 'model.safetensors.index.json'),
                'decoding': {'do_sample': False, 'max_new_tokens': 384, 'max_pixels': 262144, 'use_fast': True},
                'scope': 'Local VLM estimates, not human ratings. Calibration is task-limited.',
                'rules': 'Fixed source/target-text task definitions; goals hidden during factual observation. '
                         'Target matches need observed source-target change. Pink is not red. '
                         'Unknown stays unknown. Pair comparisons assess allowed-change-adjusted preservation.'}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol['fingerprint'] = fingerprint
    protocol_path = out / 'protocol.json'
    if protocol_path.exists() and read(protocol_path)['fingerprint'] != fingerprint:
        raise ValueError('Existing output belongs to a different protocol/input. Set OUT to a new directory.')
    save(protocol_path, protocol)
    for name in ('tasks.json', 'r85_plan.json', 'judge.py', 'aggregate.py', 'calibration_lib.py', 'run.sh', 'start.sh'):
        if (HERE / name).exists():
            target = out / 'code' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((HERE / name).read_bytes())
    print('PREFLIGHT OK: 40 sources, 520 native outputs; local weights found.', flush=True)
    return plan, tasks, sources, outputs


class Observer:
    def __init__(self, modelpath, out):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; refusing an unintended CPU run')
        self.torch, self.out = torch, out
        self.processor = AutoProcessor.from_pretrained(str(modelpath), local_files_only=True,
                                                       use_fast=True, max_pixels=262144)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(modelpath), local_files_only=True, device_map='auto',
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.model.eval()
        self.model.generation_config.do_sample = False
        for name in ('temperature', 'top_p', 'top_k'):
            setattr(self.model.generation_config, name, None)
        if any(str(d) in ('cpu', 'disk') for d in getattr(self.model, 'hf_device_map', {}).values()):
            raise RuntimeError('Model offloaded to CPU/disk; free GPU memory before retrying')
        self.consecutive_errors = 0
        save(out / 'runtime.json', {'torch': torch.__version__, 'transformers': transformers.__version__,
                                   'device': torch.cuda.get_device_name(), 'start': time.time()})

    def observe(self, key, paths, text, validator):
        from PIL import Image
        path = self.out / 'observations' / (key + '.json')
        if path.exists():
            previous = read(path)
            if previous.get('status') == 'ok':
                self.consecutive_errors = 0
                return previous['parsed']
        else:
            previous = {}
        attempts = list(previous.get('attempts', []))
        images = []
        for p in paths:
            with Image.open(p) as im:
                images.append(im.convert('RGB'))
        base_text, error = text, None
        parsed = None
        for attempt in range(2):
            content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': text}]
            chat = self.processor.apply_chat_template([{'role': 'user', 'content': content}],
                                                      tokenize=False, add_generation_prompt=True)
            started, raw, tensor_audit = time.time(), None, None
            try:
                inputs = self.processor(text=[chat], images=images, padding=True, return_tensors='pt')
                grid = inputs['image_grid_thw']
                expected = int((grid.prod(dim=1) // (int(self.processor.image_processor.merge_size) ** 2)).sum())
                actual = int((inputs['input_ids'] == self.model.config.image_token_id).sum())
                finite = bool(self.torch.isfinite(inputs['pixel_values']).all())
                tensor_audit = {'images': len(images), 'grid': grid.tolist(), 'expected_tokens': expected,
                                'actual_tokens': actual, 'finite_pixels': finite}
                if expected != actual or not finite or len(grid) != len(images):
                    raise RuntimeError('Visual input/token mapping mismatch')
                inputs = inputs.to(self.model.device)
                with self.torch.inference_mode():
                    generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=384)
                raw = self.processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:],
                                                  skip_special_tokens=True)[0]
                parsed = validator(decode_json(raw))
                error = None
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
            attempts.append({'prompt': text, 'raw_response': raw, 'error': error,
                             'tensor_audit': tensor_audit, 'seconds': time.time() - started})
            save(path, {'status': 'ok' if error is None else 'error', 'parsed': parsed,
                        'image_count': len(images), 'attempts': attempts})
            if error is None:
                self.consecutive_errors = 0
                return parsed
            # Retry schema problems only. Never repair facts using expected answers.
            if raw is None:
                break
            text = base_text + '\nYour previous output did not match the requested schema: ' + error + '\nReturn only the requested JSON fields.'
        self.consecutive_errors += 1
        print('OBSERVATION ERROR', key, error, flush=True)
        if self.consecutive_errors >= 3:
            raise RuntimeError('Three consecutive technical observation failures; see observations/*.json')
        return None


def controls(observer, out):
    from PIL import Image, ImageDraw
    specs = [('blue', 'green', False), ('red', 'green', False), ('blue', 'red', False),
             (None, 'green', False), ('blue', None, False), (None, 'green', True)]
    rows, paths = [], []
    colors = {'blue': (20, 40, 220), 'red': (220, 20, 30), 'green': (20, 150, 40)}
    for i, (circle, square, triangle) in enumerate(specs):
        im = Image.new('RGB', (448, 448), 'white')
        d = ImageDraw.Draw(im)
        if circle:
            d.ellipse((35, 130, 185, 280), fill=colors[circle])
        if square:
            d.rectangle((270, 130, 410, 270), fill=colors[square])
        if triangle:
            d.polygon([(110, 120), (30, 280), (190, 280)], fill=colors['blue'])
        path = out / 'controls' / ('shape_%d.png' % i)
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path)
        paths.append(path)
        expected = {'circle_present': circle is not None, 'circle_color': circle or 'absent',
                    'square_present': square is not None, 'square_color': square or 'absent',
                    'triangle_present': triangle}
        actual = observer.observe('control_shape_%d' % i, [path], control_prompt('shapes'),
                                  lambda v: control_validate(v, 'shapes'))
        rows.append({'id': i, 'expected': expected, 'observed': actual, 'correct': actual == expected})
        print('CONTROL', i + 1, '/', len(specs), 'ok' if actual == expected else 'failed', flush=True)
    pair_task = {'protected_checks': [{'id': 'p1', 'text': 'The square keeps its green color, shape and position.'}],
                 'allowed_changes': 'The circle may change color. The square must stay unchanged.'}
    pair_rows = []
    for index, (candidate, expected_status) in enumerate(((0, 'retained'), (1, 'retained'), (2, 'changed'))):
        observed = observer.observe('control_pair_%d' % index, [paths[0], paths[candidate]],
                                    comparison_prompt(pair_task), lambda v: validate_comparison(v, pair_task))
        expected = {'p1': expected_status, 'severe_artifact': False}
        pair_rows.append({'id': index, 'expected': expected, 'observed': observed, 'correct': observed == expected})
        print('PAIR CONTROL', index + 1, '/ 3', 'ok' if observed == expected else 'failed', flush=True)
    audit = {'total': len(rows), 'correct': sum(r['correct'] for r in rows), 'rows': rows,
             'pair_total': len(pair_rows), 'pair_correct': sum(r['correct'] for r in pair_rows), 'pair_rows': pair_rows,
             'scope': 'Simple known visual facts only; these do not establish benchmark-wide accuracy.'}
    save(out / 'control_audit.json', audit)
    if not all(row['correct'] for row in rows + pair_rows):
        raise RuntimeError('Known-image controls failed; full evaluation stopped before spending GPU on 520 outputs')
    return audit


def pack(out):
    target = out.with_name(out.name + '.zip')
    temp = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and not path.name.endswith('.tmp'):
                archive.write(path, out.name + '/' + path.relative_to(out).as_posix())
    temp.replace(target)
    print('RESULT ZIP:', target, flush=True)


def run(args):
    out = args.out.resolve()
    plan, tasks, sources, outputs = preflight(args.root.resolve(), args.model.resolve(), out)
    if args.preflight:
        return
    records = {}
    controls_audit = None
    fatal = None
    done = 0
    started = time.time()
    try:
        observer = Observer(args.model, out)
        controls_audit = controls(observer, out)
        source_facts = {}
        for case, task in tasks.items():
            source_facts[case] = observer.observe('source_' + case, [sources[case]], observation_prompt(task),
                                                   lambda v, t=task: validate_facts(v, t))
            print('SOURCE', len(source_facts), '/ 40', case, flush=True)
        save(out / 'source_audit.json', {case: {'facts': source_facts[case],
             'source_already_matches_target': factual_target(source_facts[case], task['target_all'])}
             for case, task in tasks.items()})
        jobs = list(plan['jobs'])
        random.Random(850711).shuffle(jobs)
        for job in jobs:
            task, case, jid = tasks[job['case']], job['case'], job['id']
            blind_id = hashlib.sha256(jid.encode()).hexdigest()[:24]
            facts = observer.observe('output_' + blind_id, [outputs[jid]], observation_prompt(task),
                                     lambda v, t=task: validate_facts(v, t))
            # Randomize image order while keeping the reference explicit.
            reverse = bool(int(blind_id[0], 16) % 2)
            pair = [outputs[jid], sources[case]] if reverse else [sources[case], outputs[jid]]
            comparison = observer.observe('pair_' + blind_id, pair, comparison_prompt(task, reverse),
                                           lambda v, t=task: validate_comparison(v, t))
            edit = edit_label(source_facts[case], facts, task)
            protected, artifact = preservation(comparison, task)
            joint = conjunction([edit, protected, None if artifact is None else 1 - artifact])
            status = 'error' if facts is None or comparison is None or source_facts[case] is None else (
                     'ok' if all(v is not None for v in (edit, protected, artifact, joint)) else 'unresolved')
            record = {'edit': edit, 'protected': protected, 'artifact': artifact, 'joint': joint,
                      'status': status, 'detail': {'blind_id': blind_id, 'facts': facts,
                      'comparison': comparison, 'reference_second': reverse,
                      'source_target_match': factual_target(source_facts[case], task['target_all'])}}
            records[jid] = record
            save(out / 'records' / (jid + '.json'), record)
            done += 1
            elapsed = time.time() - started
            save(out / 'progress.json', {'stage': 'benchmark', 'completed': done, 'planned': 520,
                                        'seconds': elapsed, 'updated': time.time()})
            print('%d / 520 %s edit=%s preserve=%s artifact=%s joint=%s' % (
                  done, status, edit, protected, artifact, joint), flush=True)
        save(out / 'progress.json', {'stage': 'complete', 'completed': done, 'planned': 520,
                                    'seconds': time.time() - started, 'updated': time.time()})
    except Exception as exc:
        fatal = type(exc).__name__ + ': ' + str(exc)
        save(out / 'failure.json', {'error': fatal, 'traceback': traceback.format_exc(), 'updated': time.time()})
        print('STOPPED:', fatal, flush=True)
    finally:
        # Recover already completed records on a resumed or interrupted execution.
        for job in plan['jobs']:
            path = out / 'records' / (job['id'] + '.json')
            if path.exists() and job['id'] not in records:
                records[job['id']] = read(path)
        result = summarize(plan, plan['jobs'], records, out)
        result.update({'run_error': fatal, 'version': VERSION, 'controls': controls_audit,
                       'measure': 'Local Qwen factual-rule edit + protected-content + severe-artifact evaluation',
                       'old_labels_imported': False, 'human_ratings': False})
        save(out / 'summary.json', result)
        pack(out)
    if fatal:
        raise SystemExit(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    run(parser.parse_args())
