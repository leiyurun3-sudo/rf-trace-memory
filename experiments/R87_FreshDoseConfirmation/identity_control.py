"""Standalone, local-only 40-source identity calibration for frozen Qwen v2 rules.

Default: 40 fresh reference observations + 40 separately keyed fresh candidate
observations of the SAME native source PNGs. Only the factual edit channel runs.
No artifact/preservation calls, benchmark jobs, downloads, or label repairs.

Example (support files may instead be copied into ./identity_support):
 python identity_control.py --judge-dir /path/A_Qwen_FactualJudge_v2 \
   --tasks /path/tasks.json --plan /path/r85_plan.json \
   --root /path/R85_IndependentComparison --model /path/local/Qwen \
   --out /path/R87_identity_results
Optional: --reuse-source-dir /path/previous_frozen_observations
Reuse is fail-closed: incomplete compatibility evidence means fresh reference
calls, with reasons recorded. Ordinary legacy v2 protocol.json omits processor/
weight/runtime evidence and therefore normally falls back to 80 observations.
A prior run made with --strict-model-hash supplies full content evidence.
Default fresh mode records metadata/index inventory, NOT weight-content verification.
--reuse-source-dir automatically enables strict weight hashing; incompatible or
legacy source facts still fall back to fresh reference observations.

Requires unchanged judge.py, aggregate.py, calibration_lib.py as support files,
plus torch/transformers/Pillow and the already-installed local Qwen dependencies.
Resume only in the same --out with identical protocol/input fingerprints. Failed
or interrupted observations are terminal unknowns, not silently retried on resume.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for _key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
    os.environ[_key] = '1'
VERSION = 'r87-source-as-output-edit-only-v2'
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def load_judge(directory):
    directory = directory.resolve()
    for name in ('judge.py', 'aggregate.py', 'calibration_lib.py'):
        if not (directory / name).is_file():
            raise FileNotFoundError('Required unchanged support file: ' + str(directory / name))
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('r87_frozen_qwen_v2', directory / 'judge.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.VERSION != 'factual-judge-v2-separate-channels':
        raise ValueError('Expected the frozen factual-judge-v2-separate-channels implementation')
    return module


def preflight(args, judge):
    from PIL import Image
    definitions, plan = read(args.tasks), read(args.plan)
    tasks = {str(t['case']): t for t in definitions}
    originals = {str(t['case']): t for t in plan['tasks']}
    if len(definitions) != 40 or len(tasks) != 40 or len(originals) != 40 or set(tasks) != set(originals):
        raise ValueError('Require exactly 40 unique matching source cases in tasks and R85 plan')
    sources, image_fingerprints, prompts = {}, {}, {}
    for case in sorted(tasks):
        task = tasks[case]
        if not task.get('target_all') or not set(task['target_all']) <= set(task['fields']):
            raise ValueError('Invalid frozen factual rule for ' + case)
        # Validate target value types/enums without supplying targets to the observer.
        for field, target in task['target_all'].items():
            schema = task['fields'][field]
            if schema.get('type') == 'boolean' and type(target) is not bool:
                raise ValueError('Non-boolean target: ' + case + '/' + field)
            if 'values' in schema and target not in schema['values']:
                raise ValueError('Target outside frozen schema: ' + case + '/' + field)
        relative = originals[case]['source_image']
        path = (args.root / relative).resolve()
        if not path.is_relative_to(args.root.resolve()):
            raise ValueError('Source path escapes --root: ' + relative)
        if not path.is_file():
            raise FileNotFoundError('Native source PNG missing: ' + str(path))
        with Image.open(path) as im:
            if im.format != 'PNG':
                raise ValueError('Native PNG required, not a preview: ' + str(path))
            im.verify()
        value = digest(path)
        expected = plan.get('input_sha256', {}).get(relative)
        if expected is not None and expected != value:
            raise ValueError('Source differs from frozen R85 plan: ' + case)
        sources[case] = path
        image_fingerprints[case] = value
        prompts[case] = judge.observation_prompt(task)
    index = read(args.model / 'model.safetensors.index.json')
    for name in set(index['weight_map'].values()):
        path = (args.model / name).resolve()
        if not path.is_relative_to(args.model.resolve()) or not path.is_file() or path.stat().st_size < 1024:
            raise FileNotFoundError('Missing/invalid local model shard: ' + name)
    for name in ('config.json', 'preprocessor_config.json', 'tokenizer_config.json'):
        read(args.model / name)
    # Fresh mode inventories weights without reading their multi-GB contents.
    # Small processor/tokenizer/config files remain content-bound. Strict mode
    # is required for cross-run frozen-reference reuse, or may be requested to
    # create a future reusable reference archive. Inventory is NOT content proof.
    asset_paths = [p for p in sorted(args.model.rglob('*')) if p.is_file()
                   and p.suffix.lower() in ('.json', '.safetensors', '.txt', '.model', '.tiktoken')]
    strict = bool(args.strict_model_hash or args.reuse_source_dir)
    inventory = {p.relative_to(args.model).as_posix():
                 {'bytes': p.stat().st_size, **({} if strict else {'mtime_ns': p.stat().st_mtime_ns})}
                 for p in asset_paths}
    content_hashes = {p.relative_to(args.model).as_posix(): digest(p)
                      for p in asset_paths if strict or p.suffix.lower() != '.safetensors'}
    model_evidence = {
        'mode': 'strict_content_sha256' if strict else 'fresh_metadata_index_inventory',
        'inventory': inventory, 'content_sha256': content_hashes,
        'weight_contents_verified': strict,
        'scope': ('Full weight and processor/tokenizer/config content hashes.' if strict else
                  'Weight filenames, sizes and modification times plus index/config/processor/tokenizer hashes; '
                  'weight inventory is not weight-content verification. Resume binding has this same limitation.')}
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback or download is attempted')
    versions = {}
    for package in ('torch', 'transformers', 'Pillow', 'tokenizers', 'torchvision', 'accelerate', 'safetensors'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    compatibility = {
        'judge_version': judge.VERSION,
        'support_sha256': {n: digest(args.judge_dir / n) for n in ('judge.py', 'aggregate.py', 'calibration_lib.py')},
        'tasks_sha256': digest(args.tasks), 'source_png_sha256': image_fingerprints,
        'observation_prompts': prompts, 'local_model_evidence': model_evidence,
        'runtime_versions': versions,
        'dtype': 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16',
        'decoding': {'do_sample': False, 'max_new_tokens': 384, 'max_pixels': 262144, 'use_fast': True},
        'schema_retry_policy': 'Original Observer.observe: at most two attempts per new observation; terminal across resumes',
    }
    return tasks, sources, prompts, compatibility


def reusable_references(args, compatibility, judge):
    reused, reasons = {}, []
    if args.reuse_source_dir is None:
        return reused, ['No frozen reference directory requested; default 80 fresh observations.']
    root = args.reuse_source_dir.resolve()
    task_map = {str(t['case']): t for t in read(args.tasks)}
    try:
        if not compatibility['local_model_evidence']['weight_contents_verified']:
            raise ValueError('Strict weight-content evidence required for frozen-reference reuse')
        protocol_path = root / 'identity_control_protocol.json'
        if not protocol_path.is_file():
            raise ValueError('Legacy/missing compatibility manifest: complete model, processor, prompt and runtime compatibility cannot be verified')
        old = read(protocol_path)
        if old.get('identity_compatibility') != compatibility:
            raise ValueError('Frozen reference protocol/model/processor/tasks/input/runtime mismatch')
        audit = read(root / 'source_audit.json')
        for case in sorted(compatibility['source_png_sha256']):
            try:
                entry = audit[case]
                raw = (root / entry['observation_relative_path']).resolve()
                if not raw.is_relative_to(root) or digest(raw) != entry['observation_sha256']:
                    raise ValueError('Reference observation missing or changed')
                observation = read(raw)
                expected = signature({'compatibility': signature(compatibility), 'case': case,
                                      'role': 'reference', 'image_sha256': compatibility['source_png_sha256'][case],
                                      'prompt': compatibility['observation_prompts'][case]})
                binding = read(root / 'cache_bindings' / ('reference_' + case + '.json'))
                if (entry['observation_fingerprint'] != expected or binding.get('fingerprint') != expected
                        or binding.get('status') != 'complete' or binding.get('raw_sha256') != digest(raw)
                        or observation.get('status') != 'ok'):
                    raise ValueError('Reference provenance/cache binding is incompatible')
                facts = judge.validate_facts(observation['parsed'], task_map[case])
                if facts != entry['facts']:
                    raise ValueError('Frozen reference facts differ from raw validated observation')
                reused[case] = {'facts': facts, 'provenance': entry, 'directory': str(root)}
            except (KeyError, ValueError, TypeError, OSError) as exc:
                reasons.append(case + ': fresh reference required: ' + str(exc))
    except (KeyError, ValueError, TypeError, OSError) as exc:
        reasons.append('Frozen reference reuse rejected; fresh references: ' + str(exc))
    return reused, reasons


class IdentityObserver:
    def __init__(self, judge, model, out, compatibility):
        self.judge, self.model, self.out, self.compatibility = judge, model, out, compatibility
        self.engine = None
        self.new_observations = 0

    def observe(self, case, role, image, prompt, task):
        # Distinct, fixed keys; the same image bytes never imply shared facts.
        key = role + '_' + case
        fp = signature({'compatibility': signature(self.compatibility), 'case': case, 'role': role,
                        'image_sha256': self.compatibility['source_png_sha256'][case], 'prompt': prompt})
        binding_path = self.out / 'cache_bindings' / (key + '.json')
        raw = self.out / 'observer' / 'observations' / (key + '.json')
        if binding_path.exists():
            binding = read(binding_path)
            if binding.get('fingerprint') != fp:
                raise RuntimeError('Cache fingerprint mismatch; refusing observation: ' + key)
            if binding.get('status') != 'complete':
                return None, {'role': role, 'status': 'interrupted_unknown', 'fingerprint': fp}
            if not raw.is_file() or digest(raw) != binding.get('raw_sha256'):
                raise RuntimeError('Cached raw observation missing/changed: ' + key)
            record = read(raw)
            facts = self.judge.validate_facts(record['parsed'], task) if record.get('status') == 'ok' else None
            return facts, {'role': role, 'status': 'cached_' + record.get('status', 'error'),
                           'fingerprint': fp, 'raw_path': str(raw)}
        if raw.exists():
            raise RuntimeError('Unbound raw cache; refusing to reuse or overwrite: ' + key)
        if self.engine is None:
            self.engine = self.judge.Observer(self.model, self.out / 'observer')
        save(binding_path, {'fingerprint': fp, 'status': 'started', 'started': time.time(),
                            'role': role, 'case': case, 'maximum_attempts': 2})
        self.new_observations += 1
        # Schema failures are terminal for that source/channel, not a reason to
        # abandon the other 39 controls. The per-observation two-attempt cap stays.
        self.engine.consecutive_errors = 0
        facts = self.engine.observe(key, [image], prompt,
                                    lambda value: self.judge.validate_facts(value, task))
        self.engine.consecutive_errors = 0
        record = read(raw)
        save(binding_path, {'fingerprint': fp, 'status': 'complete', 'raw_sha256': digest(raw),
                            'role': role, 'case': case, 'attempts': len(record.get('attempts', [])),
                            'observation_status': record.get('status'), 'finished': time.time()})
        return facts, {'role': role, 'status': record.get('status', 'error'),
                       'fingerprint': fp, 'raw_path': str(raw)}


def counts(rows):
    n = len(rows)
    fp = sum(r['edit'] == 1 for r in rows)
    zero = sum(r['edit'] == 0 for r in rows)
    unknown = n - fp - zero
    return {'planned_sources': n, 'false_positive': fp, 'edit_zero': zero, 'unknown': unknown,
            'technical_or_missing_sources': sum(r['technical_or_missing'] for r in rows),
            'false_positive_over_planned': fp / n if n else None,
            'false_positive_over_resolved': fp / (fp + zero) if fp + zero else None,
            'resolved_denominator': fp + zero,
            'missing_label_lower': fp / n if n else None,
            'missing_label_upper': (fp + unknown) / n if n else None}


def report(out, rows, observer, error=None):
    ordered = [rows[k] for k in sorted(rows)]
    attempt_rows = []
    for path in sorted((out / 'observer' / 'observations').glob('*.json')):
        attempt_rows.extend(read(path).get('attempts', []))
    cost = {'recorded_attempts_all_invocations': len(attempt_rows),
            'recorded_attempt_wall_seconds_all_invocations': sum(float(a.get('seconds', 0)) for a in attempt_rows),
            'scope': 'GPU model inference has nonzero cost. Attempt wall time includes processing/decoding/validation; it excludes model loading and is not GPU kernel time. Failed attempts need not reach model.generate. No peak-VRAM measurement is available from this driver.'}
    summary = {'version': VERSION, 'gpu_cost': cost, 'channel': 'factual edit only; source-as-output',
               'complete_run': all(r['processed'] for r in ordered), 'run_error': error,
               'overall': counts(ordered),
               'reference_target_strata': {name: counts([r for r in ordered if r['reference_target_stratum'] == name])
                                          for name in ('already_satisfies', 'does_not_satisfy', 'unknown')},
               'new_observations_this_invocation': observer.new_observations,
               'interpretation': 'This diagnoses identity edit false positives/specificity, not complete evaluator validity. Every source-as-output positive is an edit false positive. Missing-label bounds are deterministic, not confidence intervals. Reference strata are model-observed, not human ground truth. Greedy repeated observations are separately executed, not independent random samples.',
               'missing_policy': 'Any absent/technically failed factual observation makes the control edit unknown. This is explicit control-level missingness, rather than counting a candidate known-failure with missing reference as a resolved negative.'}
    save(out / 'identity_control_summary.json', summary)
    fields = ['case', 'processed', 'reference_origin', 'reference_target_match', 'candidate_target_match',
              'reference_target_stratum', 'edit', 'technical_or_missing', 'reference_status', 'candidate_status']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(ordered)
    target = out / 'identity_control_rows.csv'
    temp = target.with_name(target.name + '.tmp')
    temp.write_text(stream.getvalue(), encoding='utf-8')
    temp.replace(target)
    save(out / 'identity_control_records.json', rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge-dir', type=Path, default=HERE / 'identity_support')
    parser.add_argument('--tasks', type=Path, help='Frozen v2 tasks/rules JSON; default --judge-dir/tasks.json')
    parser.add_argument('--plan', type=Path, help='Frozen R85 source plan; default --judge-dir/r85_plan.json')
    parser.add_argument('--root', type=Path, required=True, help='Local root resolving plan source_image paths')
    parser.add_argument('--model', type=Path, required=True, help='Complete existing local Qwen2.5-VL-7B-Instruct directory')
    parser.add_argument('--out', type=Path, required=True, help='NEW dedicated identity namespace; never existing 520-score output')
    parser.add_argument('--reuse-source-dir', type=Path, help='Optional frozen source observation directory; fail-closed compatibility checks')
    parser.add_argument('--strict-model-hash', action='store_true', help='Hash full model weights; automatically enabled by --reuse-source-dir. Default fresh mode uses metadata/index inventory, not weight-content validation.')
    args = parser.parse_args()
    args.judge_dir = args.judge_dir.resolve()
    args.tasks = (args.tasks or args.judge_dir / 'tasks.json').resolve()
    args.plan = (args.plan or args.judge_dir / 'r85_plan.json').resolve()
    args.root, args.model, args.out = args.root.resolve(), args.model.resolve(), args.out.resolve()
    if args.out in (args.root, args.model, args.judge_dir) or (args.reuse_source_dir and args.out == args.reuse_source_dir.resolve()):
        raise ValueError('--out must be a separate identity-only output namespace')
    protocol_path = args.out / 'identity_control_protocol.json'
    if args.out.exists() and any(args.out.iterdir()) and not protocol_path.is_file():
        raise ValueError('Nonempty output is not an identity-control namespace; existing scores will not be touched')
    judge = load_judge(args.judge_dir)
    tasks, sources, prompts, compatibility = preflight(args, judge)
    reused, reasons = reusable_references(args, compatibility, judge)
    protocol = {'version': VERSION, 'driver_sha256': digest(Path(__file__)),
                'identity_compatibility': compatibility, 'plan_sha256': digest(args.plan),
                'reference_reuse_source': str(args.reuse_source_dir.resolve()) if args.reuse_source_dir else None,
                'reference_reuse_cases': sorted(reused), 'reference_reuse_decisions': reasons,
                'scope': '40 source controls; no modifications to original 520 evaluation',
                'nominal_fresh_observations': 80 - len(reused)}
    protocol['fingerprint'] = signature(protocol)
    if protocol_path.exists() and read(protocol_path).get('fingerprint') != protocol['fingerprint']:
        raise ValueError('Resume protocol mismatch; refusing reuse, overwrite, or automatic new-key retry')
    save(protocol_path, protocol)
    observer = IdentityObserver(judge, args.model, args.out, compatibility)
    rows = {case: {'case': case, 'processed': False, 'reference_origin': 'not_processed',
                  'reference_target_match': None, 'candidate_target_match': None,
                  'reference_target_stratum': 'unknown', 'edit': None,
                  'technical_or_missing': True, 'reference_status': 'missing', 'candidate_status': 'missing'}
            for case in tasks}
    source_audit, error = {}, None
    try:
        for case in sorted(tasks):
            task = tasks[case]
            if case in reused:
                reference = reused[case]['facts']
                reference_info = {'status': 'compatible_frozen_reference'}
                origin = 'frozen_reference'
            else:
                reference, reference_info = observer.observe(case, 'reference', sources[case], prompts[case], task)
                origin = 'fresh_reference_namespace'
                if reference is not None:
                    raw = Path(reference_info['raw_path'])
                    source_audit[case] = {'facts': reference,
                        'observation_relative_path': raw.relative_to(args.out).as_posix(),
                        'observation_sha256': digest(raw),
                        'observation_fingerprint': reference_info['fingerprint']}
            # Always a separate candidate call/key, even for identical pixels.
            candidate, candidate_info = observer.observe(case, 'fresh_candidate', sources[case], prompts[case], task)
            start = judge.factual_target(reference, task['target_all'])
            end = judge.factual_target(candidate, task['target_all'])
            missing = reference is None or candidate is None
            edit = None if missing else judge.edit_label(reference, candidate, task)
            rows[case] = {'case': case, 'processed': True, 'reference_origin': origin,
                          'reference_target_match': start, 'candidate_target_match': end,
                          'reference_target_stratum': 'already_satisfies' if start == 1 else ('does_not_satisfy' if start == 0 else 'unknown'),
                          'edit': edit, 'technical_or_missing': missing,
                          'reference_status': reference_info['status'], 'candidate_status': candidate_info['status'],
                          'reference_facts': reference, 'candidate_facts': candidate,
                          'candidate_observation': candidate_info}
            save(args.out / 'source_audit.json', source_audit)
            report(args.out, rows, observer)
            print(f'{sum(r["processed"] for r in rows.values())}/40 {case} source_target={start} edit={edit}', flush=True)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        save(args.out / 'source_audit.json', source_audit)
        summary = report(args.out, rows, observer, error)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
