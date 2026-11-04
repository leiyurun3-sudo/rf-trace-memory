"""Analyze R80 metrics on CPU. Example:
python analyze.py --metrics D:/results/metrics.csv --reference ONLINE25 --seed 42
Writes JSON to stdout; redirect it to a file if desired. No model inference is run.
"""
import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ('pattern_ncc_fixed', 'pattern_ncc_shift6', 'mae_vs_source',
           'object_mae', 'bg_mae', 'clip_target_similarity',
           'clip_directional_similarity', 'clip_image_similarity',
           'dino_full_feature_similarity', 'dino_pattern_crop_feature_similarity',
           'lpips_full', 'pixel_mse', 'pixel_psnr_db', 'pixel_ssim',
           'correction_impulse', 'nfe_total', 'seconds')
LABELS = ('edit_ok', 'subject_ok', 'pattern_ok', 'artifact')


def num(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def joint(row):
    if row['status'] in ('failed', 'error'):
        return 0
    if row['status'] != 'completed':
        return None
    values = [num(row.get(k)) for k in LABELS]
    if any(v not in (0, 1) for v in values):
        return None
    return int(values == [1, 1, 1, 0])


def value(row, metric):
    if metric == 'joint_success':
        return joint(row)
    if row['status'] != 'completed':
        return None
    # The evaluator may have partially populated values before an error.
    if metric.startswith(('clip_', 'dino_', 'lpips_', 'pixel_')) and row.get('eval_status') not in ('', None, 'ok'):
        return None
    return num(row.get(metric))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', type=Path, nargs='+', required=True, help='One or more compatible metrics.csv files, e.g. R80 and R81')
    p.add_argument('--evaluation', type=Path, help='Optional evaluate.py evaluation.csv, joined by id')
    p.add_argument('--reference', required=True, help='Reference arm; also compares every other pair')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--replicates', type=int, default=10000)
    args = p.parse_args()
    if args.replicates < 1:
        p.error('--replicates must be positive')
    try:
        rows = []
        for metrics_path in args.metrics:
            with metrics_path.open(encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                required = {'id', 'case', 'task', 'seed', 'status'}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError('Required metric columns missing: '+str(metrics_path))
                rows.extend(reader)
        if not rows:
            raise ValueError('metrics.csv has no rows')
        ids = [r['id'] for r in rows]
        if any(not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError('metrics.csv contains blank or duplicate ids')
        if args.evaluation:
            with args.evaluation.open(encoding='utf-8-sig', newline='') as f:
                extra = list(csv.DictReader(f))
            mapping = {}
            for r in extra:
                ident = r.get('id')
                if not ident or ident in mapping:
                    raise ValueError('evaluation.csv contains blank or duplicate ids')
                if ident not in ids:
                    raise ValueError('evaluation.csv id absent from metrics.csv: ' + ident)
                mapping[ident] = r
            for row in rows:
                row.update({k: v for k, v in mapping.get(row['id'], {}).items()
                            if k.startswith(('clip_', 'dino_', 'lpips_', 'pixel_', 'eval_'))})
        groups = defaultdict(dict)
        for row in rows:
            diagnostic = row.get('diagnostic_only', '').strip().lower()
            if diagnostic in ('true', '1') or row.get('cohort') == 'reconstruction':
                continue
            arm = row.get('arm') or row.get('label')
            if not arm or any(not row.get(k) for k in ('case', 'task', 'seed', 'status')):
                raise ValueError('Missing arm, case, task, seed or status in row ' + row['id'])
            cohort = row.get('cohort', '')
            key = (row['case'], row['task'], row['seed'])
            if key in groups[(cohort, arm)]:
                raise ValueError('Duplicate cohort/arm/case/task/seed: ' + str((cohort, arm, key)))
            groups[(cohort, arm)][key] = row
        if not groups or not any(arm == args.reference for _, arm in groups):
            raise ValueError('Reference arm absent from non-diagnostic rows: ' + args.reference)
        rng = np.random.default_rng(args.seed)
        metrics = [m for m in METRICS if any(m in r for r in rows)] + ['joint_success']
        summaries, comparisons = [], []
        for (cohort, arm), keyed in sorted(groups.items()):
            rr = list(keyed.values())
            outcomes = [joint(r) for r in rr]
            decided = sum(x is not None for x in outcomes)
            successes = sum(x == 1 for x in outcomes)
            summary = dict(cohort=cohort, arm=arm, planned=len(rr),
                           completed=sum(r['status'] == 'completed' for r in rr),
                           failed=sum(r['status'] in ('failed', 'error') for r in rr),
                           joint_decided=decided, joint_successes=successes,
                           joint_success_rate=successes / len(rr) if decided == len(rr) else None,
                           joint_lower_bound=successes / len(rr),
                           joint_upper_bound=(successes + len(rr) - decided) / len(rr), metrics={})
            for metric in metrics:
                vals = [v for r in rr if (v := value(r, metric)) is not None]
                # A partial joint mean is not an aggregate success rate.
                summary['metrics'][metric] = dict(available=len(vals), missing=len(rr)-len(vals),
                    mean=(sum(vals)/len(vals) if vals and (metric != 'joint_success' or decided == len(rr)) else None))
            summaries.append(summary)
        cohorts = sorted({c for c, _ in groups})
        for cohort in cohorts:
            arms = sorted(a for c, a in groups if c == cohort)
            arms.sort(key=lambda a: (a != args.reference, a))
            for ref, arm in itertools.combinations(arms, 2):
                left, right = groups[(cohort, ref)], groups[(cohort, arm)]
                common = sorted(left.keys() & right.keys())
                for metric in metrics:
                    by_source = defaultdict(list)
                    for key in common:
                        a, b = value(right[key], metric), value(left[key], metric)
                        if a is not None and b is not None:
                            by_source[key[0]].append(a-b)
                    source_means = [sum(v)/len(v) for _, v in sorted(by_source.items())]
                    n = len(source_means)
                    mean = float(np.mean(source_means)) if n else None
                    ci = [None, None]
                    pvalue = None
                    if n >= 2:
                        samples = rng.integers(0, n, size=(args.replicates, n))
                        ci = [float(x) for x in np.quantile(np.asarray(source_means)[samples].mean(axis=1), [.025, .975])]
                    if n >= 2:
                        data = np.asarray(source_means)
                        if n <= 16:
                            signs = np.asarray(list(itertools.product((-1,1), repeat=n)))
                            pvalue = float(np.mean(np.abs((signs*data).mean(axis=1)) >= abs(mean)-1e-12))
                        else:
                            signs = rng.choice((-1,1), size=(args.replicates,n))
                            pvalue = float((1+np.sum(np.abs((signs*data).mean(axis=1)) >= abs(mean)-1e-12))/(args.replicates+1))
                    if metric == 'joint_success' and (len(common) != len(left) or len(common) != len(right) or sum(map(len, by_source.values())) != len(common)):
                        mean, ci, pvalue = None, [None, None], None
                    comparisons.append(dict(cohort=cohort, arm=arm, reference=ref, metric=metric,
                        planned_reference=len(left), planned_arm=len(right), matched_planned=len(common),
                        unmatched_reference=len(left.keys()-right.keys()), unmatched_arm=len(right.keys()-left.keys()),
                        paired_available=sum(map(len, by_source.values())),
                        matched_missing=len(common)-sum(map(len, by_source.values())), sources=n,
                        source_weighted_difference=mean, ci95_low=ci[0], ci95_high=ci[1], source_signflip_p=pvalue))
        # Holm correction within each cohort and metric, over reported arm pairs.
        families = defaultdict(list)
        for row in comparisons:
            if row['source_signflip_p'] is not None:
                families[(row['cohort'],row['metric'])].append(row)
        for family in families.values():
            ordered = sorted(family,key=lambda r:r['source_signflip_p'])
            previous = 0.
            for i,row in enumerate(ordered):
                previous = max(previous,min(1.,(len(ordered)-i)*row['source_signflip_p']))
                row['holm_p'] = previous
        print(json.dumps(dict(seed=args.seed, bootstrap_replicates=args.replicates,
            estimator='mean of within-source paired differences, then equal-weight mean across sources',
            confidence_interval='percentile source bootstrap; unavailable with fewer than two sources',
            pairing='within cohort, case, task and seed; differences are arm minus reference',
            missing='completed numeric values only; nonfinite values missing; failed joint outcomes are zero; pending or unlabeled outcomes unknown',
            summaries=summaries, comparisons=comparisons), ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, csv.Error) as exc:
        p.exit(2, 'error: ' + str(exc) + '\n')


if __name__ == '__main__':
    main()

