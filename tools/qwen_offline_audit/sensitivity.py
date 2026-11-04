"""Compare archived, harmonized pre-recovery, and recovered labels; no inference.

Pre and post use identical frozen task rules and the same three-valued AND.
Only explicitly parsed observations accepted by the original validator enter pre.
The old archived joint labels are reported separately, never used as the baseline.
"""
import copy
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import zipfile

from aggregate import summarize, _bootstrap
from recover import conjunction, target_match

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path('C:/Users/14725/Downloads/A_Qwen_FactualJudge_v2_results.zip')
OUT = ROOT / 'results' / 'sensitivity'
LABELS = ['edit', 'protected', 'artifact', 'joint']


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def counts(records):
    return {k: {str(v): sum(r[k] == v for r in records.values()) for v in (1, 0, None)} for k in LABELS}


def main():
    with zipfile.ZipFile(ARCHIVE) as z:
        prefix = next(n[:-len('summary.json')] for n in z.namelist()
                      if n.endswith('/summary.json') and n.count('/') == 1)
        read = lambda n: json.loads(z.read(prefix + n))
        plan = read('code/r85_plan.json')
        tasks = {t['case']: t for t in read('code/tasks.json')}
        observations = {Path(n).stem: json.loads(z.read(n)) for n in z.namelist()
                        if n.startswith(prefix + 'observations/') and n.endswith('.json')}
        archived = {j['id']: read('records/' + j['id'] + '.json') for j in plan['jobs']}
    recovered = json.loads((ROOT / 'results' / 'records.json').read_text(encoding='utf-8'))
    pre = copy.deepcopy(archived)
    for job in plan['jobs']:
        r = pre[job['id']]
        task = tasks[job['case']]
        source_obs = observations['source_' + job['case']]
        output_obs = observations['output_' + r['detail']['blind_id']]
        source = source_obs['parsed'] if source_obs['status'] == 'ok' else None
        output = output_obs['parsed'] if output_obs['status'] == 'ok' else None
        start, match = target_match(source, task['target_all']), target_match(output, task['target_all'])
        r['edit'] = 0 if match == 0 else (1 if match == 1 and start == 0 else None)
        r['joint'] = conjunction([r['edit'], r['protected'], None if r['artifact'] is None else 1-r['artifact']])
    OUT.mkdir(parents=True, exist_ok=True)
    pre_s = summarize(plan, plan['jobs'], pre, OUT / 'before')
    post_s = summarize(plan, plan['jobs'], recovered, OUT / 'after')
    rows = []
    for arm in pre_s['arms']:
        a, b = pre_s['arms'][arm], post_s['arms'][arm]
        row = {'arm': arm, 'planned': a['planned_outputs']}
        for prefix, stats in [('before', a), ('after', b)]:
            for metric in ('edit', 'joint'):
                s = stats['criteria'][metric]
                for key in ('positive_outputs', 'negative_outputs', 'unknown_outputs', 'output_rate_lower', 'output_rate_upper'):
                    row[f'{prefix}_{metric}_{key}'] = s[key]
        rows.append(row)
    with (OUT / 'recovery_counts.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # Restrict both versions to exactly the same complete-source intersection.
    all_means = {}
    for version, records in [('before', pre), ('after', recovered)]:
        clusters = defaultdict(list)
        for j in plan['jobs']:
            clusters[(j['arm'], j['case'])].append(records[j['id']]['joint'])
        all_means[version] = {key: sum(v)/len(v) for key,v in clusters.items() if all(x is not None for x in v)}
    paired = []
    routing = 'RELEASE_CORR25'
    for arm in pre_s['arms']:
        if arm == routing: continue
        shared = sorted(case for case in tasks if all((a,case) in all_means[v]
                        for v in ('before','after') for a in (routing,arm)))
        row = {'reference': arm, 'fixed_sources': shared, 'n': len(shared)}
        for v in ('before','after'):
            diff = [all_means[v][(routing,c)]-all_means[v][(arm,c)] for c in shared]
            row[v] = {'difference': sum(diff)/len(diff) if diff else None,
                      'ci95': _bootstrap(diff, ['recovery_fixed', routing, arm])}
        paired.append(row)
    changes = {k: sum(pre[j][k] != recovered[j][k] for j in pre) for k in LABELS}
    transitions = Counter((pre[j]['joint'], recovered[j]['joint']) for j in pre)
    result = {'purpose': 'Post-hoc format-recovery sensitivity; same frozen rules and same three-valued logic in both versions.',
              'inference_calls': 0, 'archived_counts': counts(archived),
              'before_harmonized_counts': counts(pre), 'after_counts': counts(recovered),
              'label_changes': changes, 'joint_transitions': [{'before':a,'after':b,'n':n} for (a,b),n in transitions.items()],
              'paired_before': pre_s['paired_joint'], 'paired_after': post_s['paired_joint'],
              'fixed_cohort_paired': paired,
              'notes': ['The pre-recovery labels are recomputed with the same three-valued logic as post-recovery. In this archive they exactly reproduce the stored edit and joint labels.',
                        'Protected and artifact channels are unchanged; no unknown protected label is imputed.',
                        'Bounds are deterministic missing-label bounds, not confidence intervals or full instruction accuracy.',
                        'Paired bootstrap uses 5000 source resamples; pointwise and unadjusted.']}
    write_json(OUT / 'recovery_sensitivity.json', result)
    assert all(pre[j][k] == archived[j][k] for j in pre for k in LABELS)
    short = {'RELEASE_CORR25':'CORR25', 'RELEASE_BOTH25':'BOTH25', 'ENERGY_MATCH25':'ENERGY',
             'U25_KV':r'U25\_KV', 'FLOWEDIT_N26':'FlowEdit', 'ETA03':'ETA03', 'RFEDIT_I08':'RF-Edit'}
    def triple(row, v):
        return '/'.join(str(row[f'{v}_joint_{k}_outputs']) for k in ('positive','negative','unknown'))
    def bounds(row, v):
        return '[%.2f, %.2f]' % (100*row[f'{v}_joint_output_rate_lower'],100*row[f'{v}_joint_output_rate_upper'])
    lines = [r'\begin{table*}[t]', r'\centering\small',
        r'\caption{Offline format-recovery sensitivity with identical frozen rules and three-valued logic. Counts are joint-proxy pass/fail/unknown; intervals are deterministic planned-output percentage bounds, not confidence intervals. No new inference is performed.}',
        r'\label{tab:recovery-counts}', r'\begin{tabular}{lrrrr}',r'\toprule',
        r'Arm & Before P/F/U & After P/F/U & Before bounds (\%) & After bounds (\%)\\',r'\midrule']
    for arm in short:
        row = next(r for r in rows if r['arm']==arm)
        lines.append(f"{short[arm]} & {triple(row,'before')} & {triple(row,'after')} & {bounds(row,'before')} & {bounds(row,'after')}"+r'\\')
    lines += [r'\midrule',r'All outputs & 103/332/85 & 104/358/58 & [19.81, 36.15] & [20.00, 31.15]\\',r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (OUT/'qwen_recovery_counts.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    lines = [r'\begin{table*}[t]',r'\centering\small',
        r'\caption{CORR25 minus each comparator in the joint factual proxy before and after format recovery. Each stage uses its own complete-source intersection. Intervals use 5,000 paired source-bootstrap resamples and are pointwise, unadjusted. On the fixed intersection shared by both stages, every paired difference is unchanged.}',
        r'\label{tab:recovery-pairs}',r'\begin{tabular}{lrrlrrl}',r'\toprule',
        r'Comparator & $n_{\rm pre}$ & $\Delta_{\rm pre}$ & 95\% CI & $n_{\rm post}$ & $\Delta_{\rm post}$ & 95\% CI\\',r'\midrule']
    for before in pre_s['paired_joint']:
        after = next(r for r in post_s['paired_joint'] if r['reference']==before['reference'])
        def cells(d): return f"{d['paired_complete_sources']} & {d['difference']:+.4f} & [{d['ci95_low']:.4f}, {d['ci95_high']:.4f}]"
        lines.append(short[before['reference']]+' & '+cells(before)+' & '+cells(after)+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    (OUT/'qwen_recovery_pairs.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    for row in rows:
        print(row['arm'], [(row[f'{v}_joint_positive_outputs'], row[f'{v}_joint_negative_outputs'], row[f'{v}_joint_unknown_outputs']) for v in ('before','after')])


if __name__ == '__main__':
    main()
