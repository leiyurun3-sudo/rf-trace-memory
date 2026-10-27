"""Reproduce source-clustered R73 statistics from frozen records; standard library only."""
from pathlib import Path
from collections import defaultdict
from itertools import product
from statistics import mean
from math import comb
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'tables'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q):
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = ROOT / 'results' / 'R73' / 'metrics.csv'
    comparison_path = ROOT / 'results' / 'R73' / 'comparison_metrics.csv'
    jobs = read(metrics_path)
    audit = []
    for row in jobs:
        valid = (row['status'] == 'completed'
                 and row['correction_budget_status'] == 'matched'
                 and float(row['correction_budget_relative_error']) <= .01)
        audit.append(dict(id=row['id'], generation_status=row['status'],
                          budget_status=row['correction_budget_status'],
                          relative_error=row['correction_budget_relative_error'],
                          protocol_valid=valid))
    write('r73_protocol_audit.csv', audit)
    if not all(row['protocol_valid'] for row in audit):
        raise ValueError('Budget protocol violations recorded in r73_protocol_audit.csv; resolve before pooling.')

    pairs = defaultdict(dict)
    for row in read(comparison_path):
        if row['arm'] in ('C50', 'BUDGET_EARLY', 'BUDGET_LATE'):
            if row['status'] != 'completed':
                raise ValueError('Incomplete comparison: ' + row['id'])
            key = (row['case'], row['task'], row['seed'])
            if row['arm'] in pairs[key]:
                raise ValueError('Duplicate comparison arm: ' + row['id'])
            pairs[key][row['arm']] = float(row['pattern_ncc_shift6'])

    contrasts = [('early_minus_constant', 'BUDGET_EARLY', 'C50'),
                 ('late_minus_constant', 'BUDGET_LATE', 'C50'),
                 ('early_minus_late', 'BUDGET_EARLY', 'BUDGET_LATE')]
    effects, summaries = [], []
    digest = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    for label, first, second in contrasts:
        clusters = defaultdict(list)
        tasks = defaultdict(set)
        for (case, task, seed), arms in sorted(pairs.items()):
            clusters[case].append(arms[first] - arms[second])
            tasks[case].add(task)
        values = [mean(ds) for ds in clusters.values()]
        observed = mean(values)
        n = len(values)
        # Enumerate the complete empirical source-bootstrap distribution (4**4).
        boot = sorted(mean(draw) for draw in product(values, repeat=n))
        flips = [mean(v * sign for v, sign in zip(values, signs))
                 for signs in product((-1, 1), repeat=n)]
        nonzero = [v for v in values if v != 0]
        positive = sum(v > 0 for v in nonzero)
        tail = min(positive, len(nonzero) - positive)
        sign_p = min(1., 2 * sum(comb(len(nonzero), k) for k in range(tail + 1)) / 2**len(nonzero))
        for case, ds in clusters.items():
            effects.append(dict(contrast=label, case=case, tasks=';'.join(sorted(tasks[case])),
                                n_pairs=len(ds), mean_delta=mean(ds),
                                min_pair_delta=min(ds), max_pair_delta=max(ds),
                                positive_pairs=sum(v > 0 for v in ds), negative_pairs=sum(v < 0 for v in ds)))
        summaries.append(dict(contrast=label, metric='pattern_ncc_shift6',
                              n_sources=n, n_pairs=len(pairs), source_mean_delta=observed,
                              bootstrap_percentile_95_low=percentile(boot, .025),
                              bootstrap_percentile_95_high=percentile(boot, .975),
                              bootstrap_ordered_resamples=len(boot),
                              source_sign_test_two_sided_p=sign_p,
                              source_signflip_two_sided_p=sum(abs(v) >= abs(observed) - 1e-12 for v in flips) / len(flips),
                              source_signflip_assignments=len(flips),
                              p_adjustment='none_exploratory',
                              comparison_csv_sha256=digest))
    write('r73_source_effects.csv', effects)
    write('r73_source_statistics.csv', summaries)
    for row in summaries:
        print(row['contrast'], 'delta=', round(row['source_mean_delta'], 7),
              'CI=', (round(row['bootstrap_percentile_95_low'], 7), round(row['bootstrap_percentile_95_high'], 7)),
              'sign_p=', row['source_sign_test_two_sided_p'],
              'signflip_p=', row['source_signflip_two_sided_p'])
    print('Budget-protocol valid:', sum(r['protocol_valid'] for r in audit), '/', len(audit))


if __name__ == '__main__':
    main()
