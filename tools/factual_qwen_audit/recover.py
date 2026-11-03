"""Recover explicit answers from saved Qwen JSON. No models, downloads, or inference.

Only observations rejected by the original validator are canonicalized. Valid
observations remain unchanged. Conflicting attempts become unknown, never a
choice of the more favorable answer. Task target values are NOT given to the
canonicalizer. Unknown pair comparisons are not repaired or imputed.
"""
import argparse
import copy
import csv
import io
import json
from collections import Counter
from pathlib import Path
import zipfile

from aggregate import summarize

VERSION = 'v2-offline-format-recovery-1'


def json_object(raw):
    raw = raw.strip()
    if raw.startswith('```'):
        lines = raw.splitlines()
        if lines[-1].strip() == '```':
            raw = '\n'.join(lines[1:-1])
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError('Not an object')
    return value


def scalar(value, spec):
    if spec.get('type') == 'boolean':
        if value is None or type(value) is bool:
            return value
        if type(value) is str and value in ('true', 'false', 'null'):
            return {'true': True, 'false': False, 'null': None}[value]
    elif type(value) is str and value in spec['values']:
        return value
    raise ValueError('Not an explicit permitted scalar')


def field(value, spec, key):
    if not isinstance(value, dict):
        return scalar(value, spec), 'scalar_or_boolean_string'
    # Explicit answer keys have priority over schema metadata.
    selected = []
    for answer_key in ('value', 'answer'):
        if answer_key in value:
            selected.append((scalar(value[answer_key], spec), answer_key))
    if selected:
        if any(x[0] != selected[0][0] for x in selected):
            raise ValueError('Conflicting explicit answer keys')
        return selected[0]
    if spec.get('type') == 'boolean':
        # Qwen sometimes writes type:"true" instead of a boolean answer.
        if value.get('type') in ('true', 'false', 'null'):
            return scalar(value['type'], spec), 'boolean_answer_in_type'
    else:
        options = value.get('values')
        if isinstance(options, str):
            return scalar(options, spec), 'categorical_answer_in_values'
        if isinstance(options, list) and len(options) == 1:
            return scalar(options[0], spec), 'singleton_selected_value'
        # One observed wrapper in this archive, restricted to this exact key.
        if key == 'background' and set(value) == {'predominant_distant_background'}:
            return scalar(value['predominant_distant_background'], spec), 'named_background_answer'
    raise ValueError('No explicit single answer; schema lists are not answers')


def recover_observation(observation, fields):
    if observation['status'] == 'ok':
        return observation['parsed'], None
    attempts = []
    candidates = []
    for index, attempt in enumerate(observation.get('attempts', []), 1):
        audit = {'attempt': index, 'raw_response': attempt.get('raw_response')}
        try:
            obj = json_object(attempt['raw_response'])
            result, rules = {}, {}
            for key, spec in fields.items():
                if key not in obj:
                    raise ValueError('Missing field: ' + key)
                result[key], rules[key] = field(obj[key], spec, key)
            candidates.append(result)
            audit.update(parsed=result, rules=rules)
        except (KeyError, ValueError, TypeError) as exc:
            audit['unrecovered_reason'] = str(exc)
        attempts.append(audit)
    merged, conflicts = None, []
    if candidates:
        merged = dict(candidates[0])
        for key, spec in fields.items():
            if any(c[key] != merged[key] for c in candidates[1:]):
                merged[key] = None if spec.get('type') == 'boolean' else 'uncertain'
                conflicts.append(key)
    return merged, {'attempts': attempts, 'canonicalized': merged,
                    'conflicting_fields_kept_unknown': conflicts}


def conjunction(values):
    values = list(values)
    if any(v == 0 for v in values if v is not None):
        return 0
    return 1 if values and all(v == 1 for v in values) else None


def target_match(facts, target):
    if facts is None:
        return None
    return conjunction(None if facts[k] in (None, 'uncertain') else int(facts[k] == v)
                       for k, v in target.items())


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run(archive, out):
    if out.exists():
        raise ValueError('Choose a new output directory; original and prior audits stay intact')
    with zipfile.ZipFile(archive) as z:
        prefix = next(n[:-len('summary.json')] for n in z.namelist()
                      if n.endswith('/summary.json') and n.count('/') == 1)
        def read(name):
            return json.loads(z.read(prefix + name))
        plan = read('code/r85_plan.json')
        definitions = read('code/tasks.json')
        tasks = {t['case']: t for t in definitions}
        observations = {Path(n).stem: json.loads(z.read(n)) for n in z.namelist()
                        if n.startswith(prefix + 'observations/') and n.endswith('.json')}
        records = {j['id']: read('records/' + j['id'] + '.json') for j in plan['jobs']}
        original_summary = read('summary.json')
        controls = read('control_audit.json')
        runtime = read('runtime.json')
    assert len(plan['jobs']) == len(records) == 520 and len(tasks) == 40
    recovered, audits = {}, {}
    for case, task in tasks.items():
        key = 'source_' + case
        recovered[key], audit = recover_observation(observations[key], task['fields'])
        if audit is not None:
            audits[key] = audit
    for job in plan['jobs']:
        original = records[job['id']]
        key = 'output_' + original['detail']['blind_id']
        recovered[key], audit = recover_observation(observations[key], tasks[job['case']]['fields'])
        if audit is not None:
            audits[key] = audit
    updated, changed = {}, []
    for job in plan['jobs']:
        jid, task = job['id'], tasks[job['case']]
        original = records[jid]
        record = copy.deepcopy(original)
        details = record['detail']
        source = recovered['source_' + job['case']]
        candidate = recovered['output_' + details['blind_id']]
        match, start = target_match(candidate, task['target_all']), target_match(source, task['target_all'])
        edit = 0 if match == 0 else (1 if match == 1 and start == 0 else None)
        record['edit'] = edit
        # These two channels are copied byte-for-value from the original record.
        assert record['protected'] == original['protected']
        assert record['artifact'] == original['artifact']
        record['joint'] = conjunction([edit, record['protected'],
                                      None if record['artifact'] is None else 1 - record['artifact']])
        record['status'] = 'error' if candidate is None or source is None or details['comparison'] is None or details['artifact_observation'] is None else (
            'ok' if all(record[k] is not None for k in ('edit', 'protected', 'artifact', 'joint')) else 'unresolved')
        details['facts'], details['source_target_match'] = candidate, start
        if record != original:
            changed.append({'id': jid, 'case': job['case'], 'arm': job['arm'],
                            'before': {k: original[k] for k in ('status', 'edit', 'protected', 'artifact', 'joint')},
                            'after': {k: record[k] for k in ('status', 'edit', 'protected', 'artifact', 'joint')}})
        updated[jid] = record
    out.mkdir(parents=True)
    summary = summarize(plan, plan['jobs'], updated, out)
    summary.update(version=VERSION, source_archive=archive.name,
                   inference_calls=0, human_ratings=False,
                   controls=controls, runtime=runtime,
                   observation_errors_before=len(audits),
                   observation_errors_after=sum(recovered[k] is None for k in audits),
                   schema_recovered_observations=sum(recovered[k] is not None for k in audits),
                   conflict_observations=sum(bool(a['conflicting_fields_kept_unknown']) for a in audits.values()),
                   affected_records=len(changed),
                   artifact_sensitivity='Unvalidated: all 520 outputs marked false; controls contain clean artifacts only.',
                   interpretation='Exploratory local-VLM estimates. Original frozen task criteria retained. Unknown comparisons are not imputed.')
    dump(out / 'summary.json', summary)
    dump(out / 'original_summary.json', original_summary)
    dump(out / 'recovery_audit.json', audits)
    dump(out / 'changed_records.json', changed)
    dump(out / 'records.json', updated)
    dump(out / 'source_facts.json', {k: v for k, v in recovered.items() if k.startswith('source_')})
    dump(out / 'tasks.json', definitions)
    print(json.dumps({k: summary[k] for k in ('observation_errors_before', 'observation_errors_after',
          'schema_recovered_observations', 'conflict_observations', 'affected_records')}, ensure_ascii=False))
    for arm, result in summary['arms'].items():
        print(arm, result['record_status_counts'], {k: [s['positive_outputs'], s['negative_outputs'], s['unknown_outputs']]
              for k, s in result['criteria'].items()})
    print(json.dumps(summary['paired_joint'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.archive, args.out)
