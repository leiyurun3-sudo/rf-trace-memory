"""Print frozen experiment aggregates; standard-library only, no GPU or writes."""
from pathlib import Path
import csv
from collections import defaultdict
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('mae_vs_source', 'object_mae', 'bg_mae', 'pattern_ncc_fixed', 'pattern_ncc_shift6')

def load(round_name, name='metrics.csv'):
    with (ROOT / 'results' / round_name / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['status'] == 'completed':
            groups[row['task'], row['arm']].append(row)
    return [(task, arm, len(rs), {k: mean(float(r[k]) for r in rs if r.get(k) not in ('', None))
                                for k in METRICS if any(r.get(k) not in ('', None) for r in rs)})
            for (task, arm), rs in groups.items()]

def main():
    total = 0
    for round_name in ('R71', 'R72', 'R73'):
        rows = load(round_name)
        total += len(rows)
        print(round_name, 'new_outputs=', len(rows), 'failed=', sum(r['status'] == 'failed' for r in rows))
        for task, arm, n, values in aggregate(rows):
            print(task, arm, n, ' '.join(f'{k}={v:.6f}' for k, v in values.items()))
    rows = load('R73')
    print('Total unique new jobs across rounds:', total)
    print('R73 maximum relative budget error:', max(float(r['correction_budget_relative_error']) for r in rows))
    print('R73 actual NFE:', sum(int(r['nfe_total']) for r in rows))
    print('R73 standalone NFE:', sum(int(r['nfe_standalone']) for r in rows))
    print('R73 sum of per-job seconds:', sum(float(r['seconds']) for r in rows))

if __name__ == '__main__':
    main()
