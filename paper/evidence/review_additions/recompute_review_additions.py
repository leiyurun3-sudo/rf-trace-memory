"""Post-hoc dose contrasts from frozen exported measurements. No model calls."""
import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCES = EVIDENCE / 'dose_profiles_20261103/source_arm_values.csv'
PILOT = EVIDENCE / 'r86_actual_dose'

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def number(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None

def paired_summary(values, seed, exact=False):
    ds = list(values.values())
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(ds, k=len(ds))) for _ in range(10000))
    result = dict(n_sources=len(ds), difference=statistics.mean(ds),
                  ci95=[boot[250], boot[9749]], source_differences=values,
                  bootstrap_seed=seed)
    if exact:
        threshold = abs(sum(ds))
        result['p_exact'] = sum(
            abs(sum(v*s for v, s in zip(ds, signs))) >= threshold-1e-12
            for signs in itertools.product((-1, 1), repeat=len(ds))) / 2**len(ds)
    return result

def main():
    data = [r for r in rows(SOURCES) if r['dataset'] == 'R85']
    by = {(r['arm'], r['source']): r for r in data}
    cases = sorted({r['source'] for r in data if r['arm'] == 'RELEASE_CORR25'})
    eligible = [s for s in cases if all(number(by[a,s]['lpips_preserved']) is not None
                for a in ('RELEASE_CORR25', 'U25_KV'))]
    checks = {}
    for name, population in [('all_sources', cases), ('nonempty_preservation', eligible)]:
        vals = {a: {s: number(by[a,s]['I']) for s in population}
                for a in ('RELEASE_CORR25', 'U25_KV')}
        if any(v is None for arm in vals.values() for v in arm.values()):
            raise ValueError('Missing archived impulse')
        means = {a: statistics.mean(v.values()) for a,v in vals.items()}
        diff = {s: vals['U25_KV'][s]-vals['RELEASE_CORR25'][s] for s in population}
        checks[name] = dict(means=means, ratio_of_means=means['U25_KV']/means['RELEASE_CORR25'],
                            u25_greater_sources=sum(x>0 for x in diff.values()),
                            u25_minus_corr=paired_summary(diff, 86101+len(population)))

    plan = json.loads((PILOT/'plan.json').read_text(encoding='utf-8-sig'))
    jobs = {j['id']: j for j in plan['jobs']}
    anchors = {j['id']: j for j in plan['r86_anchor_cells']}
    obs = {}
    for row in rows(PILOT/'evaluation.csv'):
        job = jobs[row['id']]
        obs[job['arm'], job['case']] = number(row['lpips_preserved'])
    for row in rows(PILOT/'anchor_evaluation.csv'):
        obs['CORR_1', anchors[row['id']]['case']] = number(row['lpips_preserved'])
    pilot_sources = sorted({j['case'] for j in jobs.values()})
    dose = []
    paired_rows = []
    for k, direction in enumerate(('CORR', 'GLOBAL')):
        a, b = direction+'_07', direction+'_1'
        complete = [s for s in pilot_sources if obs.get((a,s)) is not None and obs.get((b,s)) is not None]
        values = {s: obs[a,s]-obs[b,s] for s in complete}
        summary = paired_summary(values, 86201+k, exact=True)
        summary.update(comparison=a+' - '+b, metric='lpips_preserved',
                       mean_beta_07=statistics.mean(obs[a,s] for s in complete),
                       mean_beta_1=statistics.mean(obs[b,s] for s in complete),
                       missing_sources=[s for s in pilot_sources if s not in complete])
        dose.append(summary)
        paired_rows.extend(dict(comparison=summary['comparison'], source=s,
                                beta07=obs[a,s], beta1=obs[b,s], difference=v)
                           for s,v in values.items())
    previous = 0
    for index, row in enumerate(sorted(dose, key=lambda r:r['p_exact'])):
        previous = max(previous, min(1, (2-index)*row['p_exact']))
        row['p_holm_two_posthoc_dose_contrasts'] = previous
    result = dict(
        scope='Post-hoc additions; no new model calls or confirmatory tests.',
        inputs={'source_means':str(SOURCES),'pilot':str(PILOT)},
        impulse_definition='I = sum_i h_i RMS(applied correction velocity_i); stored I is already integrated.',
        sampling='Average the two seeds within each R85 source; R86 has one seed per source. Sources are bootstrap units.',
        intervals='10000 source bootstrap draws; percentile order statistics [250,9749]; pointwise 95% intervals.',
        dose_tests='Two-sided exhaustive source sign flips; optional Holm over only these two post-hoc dose contrasts, not the original direction family.',
        r85_impulse=checks, r86_within_direction_dose=dose)
    (OUT/'review_additions.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (OUT/'within_direction_source_pairs.csv').open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(paired_rows[0]));writer.writeheader();writer.writerows(paired_rows)
    print(json.dumps({
        'r85': {k:{'means':v['means'],'ratio':v['ratio_of_means'],'n':v['u25_minus_corr']['n_sources'],
                   'diff':v['u25_minus_corr']['difference'],'ci':v['u25_minus_corr']['ci95'],
                   'positive':v['u25_greater_sources']} for k,v in checks.items()},
        'r86':[{k:v for k,v in r.items() if k!='source_differences'} for r in dose]}, indent=2))

if __name__ == '__main__':
    main()
