"""Source-clustered evidence from metrics.csv and optional evaluation.csv; CPU only."""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from evidence_stats import inference, number, read_csv, write_csv, joint_value, merge_judgments

METRICS = ('joint_success', 'edit_ok', 'preserved_region_mae', 'object_mae',
           'clip_target_similarity', 'clip_directional_similarity',
           'clip_image_similarity', 'dino_full_feature_similarity', 'lpips_full', 'lpips_preserved',
           'pixel_ssim', 'nfe_total', 'runtime_seconds_synchronized')
PRIMARY = ('U25_KV', 'ENERGY_MATCH25', 'FLOWEDIT_N26', 'RFEDIT_I08', 'ETA03')
BASE = 'RELEASE_CORR25'
HUMAN = {'joint_success', 'edit_ok'}


def metric(row, name):
    status = row.get('status', '')
    if name in HUMAN:
        if status in ('failed', 'error'):
            return 0.0
        if status != 'completed':
            return None
        if name == 'edit_ok':
            v = number(row.get('edit_ok'))
            return v if v in (0, 1) else None
        return joint_value(row)
    if status != 'completed':
        return None
    if name.startswith(('clip_', 'dino_', 'lpips_')) and row.get('eval_status') not in (None, '', 'ok'):
        return None
    if name == 'preserved_region_mae':
        if row.get('bg_mae_status') not in (None, '', 'ok') or row.get('eval_pixel_region') == 'none':
            return None
        v = number(row.get(name))
        return v if v is not None else number(row.get('bg_mae'))
    return number(row.get(name))


def summarize(vals, rng, repeats):
    if not vals:
        return (None, None, None, None)
    return inference(vals, rng, repeats)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, default=Path(__file__).resolve().parent / 'results')
    ap.add_argument('--replicates', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261030)
    ap.add_argument('--judgments', type=Path, help='Optional task-specific judgments; default results/judgments.csv')
    ap.add_argument('--export-judgment-template', type=Path, help='Write CSV template for planned result IDs; refuses overwrite')
    args = ap.parse_args()
    if args.replicates < 1:
        ap.error('--replicates must be positive')
    root = args.results.resolve()
    rows = read_csv(root / 'metrics.csv')
    if not rows:
        raise SystemExit('Missing or empty metrics.csv: ' + str(root))
    ids = [r.get('id') for r in rows]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise SystemExit('metrics.csv has blank or duplicate IDs')
    evaluation = read_csv(root / 'evaluation.csv')
    eval_ids = [r.get('id') for r in evaluation]
    if any(not x for x in eval_ids) or len(eval_ids) != len(set(eval_ids)):
        raise SystemExit('evaluation.csv has blank or duplicate IDs')
    # The saved plan supplies planned denominators even if metrics.csv is partial.
    plan_path = root / 'plan.json'
    if not plan_path.is_file():
        plan_path = Path(__file__).resolve().parent / 'plan.json'
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
        by_id = {r['id']: r for r in rows}
        planned = plan.get('jobs', [])
        if planned:
            plan_ids = [j.get('id') for j in planned]
            if any(not x for x in plan_ids) or len(plan_ids) != len(set(plan_ids)):
                raise SystemExit('Plan has blank or duplicate job IDs')
            if set(by_id) - set(plan_ids):
                raise SystemExit('metrics.csv contains IDs absent from plan')
            rows = [dict(dict(j, status='planned'), **by_id.get(j['id'], {})) for j in planned]
    if args.export_judgment_template:
        if args.export_judgment_template.exists():
            raise SystemExit('Refusing to overwrite judgment template')
        fields = ['id', 'subject_applicable', 'pattern_applicable', 'edit_ok', 'subject_ok', 'pattern_ok', 'artifact']
        with args.export_judgment_template.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dict(id=r['id'], subject_applicable=1, pattern_applicable=1) for r in rows)
    if args.judgments and not args.judgments.is_file():
        raise SystemExit('Judgment CSV not found: '+str(args.judgments))
    merge_judgments(rows, args.judgments or root/'judgments.csv')
    extra = {r['id']: r for r in evaluation}
    for row in rows:
        ev = extra.get(row['id'], {})
        row.update({k: v for k, v in ev.items() if k.startswith(('clip_', 'dino_', 'lpips_', 'pixel_', 'eval_'))})
    rows = [r for r in rows if str(r.get('diagnostic_only', '')).lower() not in ('1', 'true')]
    grouped = defaultdict(list)
    for row in rows:
        arm, case = row.get('arm'), row.get('case')
        if not arm or not case:
            raise SystemExit('Missing arm or case in metrics.csv')
        grouped[(arm, case)].append(row)
    arms = sorted({a for a, _ in grouped})
    if BASE not in arms:
        raise SystemExit('Missing baseline arm: ' + BASE)
    # Fixed R85 primary family; reserve all hypotheses even when a result is pending.
    # For R84 use all planned non-baseline arms, as exported in metrics.csv.
    is_r85 = 'R85' in Path(__file__).resolve().parent.name
    comparators = list(PRIMARY) + ['RELEASE_BOTH25'] if is_r85 else [a for a in arms if a != BASE]
    rng = random.Random(args.seed)
    source_rows, summaries, comparisons = [], [], []
    values = {}
    for (arm, case), rr in sorted(grouped.items()):
        seeds = [str(r.get('seed', '')) for r in rr]
        if '' in seeds or len(seeds) != len(set(seeds)):
            raise SystemExit('Duplicate or missing seed for '+arm+'/'+case)
        for m in METRICS:
            vals = [metric(r, m) for r in rr]
            # Do not silently change a two-seed source into a single-seed source.
            v = statistics.mean(vals) if all(x is not None for x in vals) else None
            values[(arm, case, m)] = v
            source_rows.append(dict(arm=arm, case=case, metric=m, value=v,
                planned_jobs=len(rr), available_jobs=sum(x is not None for x in vals), seeds=','.join(seeds)))
    for arm in arms:
        cases = sorted(c for a,c in grouped if a == arm)
        for m in METRICS:
            vals = [values[(arm,c,m)] for c in cases]
            available = [x for x in vals if x is not None]
            complete = len(available) == len(cases)
            # Human outcome rates remain pending until all planned sources are decided.
            use = available if m not in HUMAN or complete else []
            mean, lo, hi, _ = summarize(use, rng, args.replicates)
            rr = [r for c in cases for r in grouped[(arm,c)]]
            applicability = {}
            for field in ('subject', 'pattern'):
                flags = [number(r.get(field+'_applicable')) if str(r.get(field+'_applicable', '')).strip() else 1 for r in rr]
                applicability[field+'_applicable_jobs'] = sum(x == 1 for x in flags)
                applicability[field+'_na_jobs'] = sum(x == 0 for x in flags)
                applicability[field+'_rated_applicable_jobs'] = sum(flag == 1 and number(r.get(field+'_ok')) in (0, 1) for r, flag in zip(rr, flags))
            summaries.append(dict(arm=arm, metric=m, source_n=len(use), planned_sources=len(cases),
                **applicability,
                complete_sources=len(available), planned_jobs=len(rr),
                failed_jobs=sum(r.get('status') in ('failed','error') for r in rr),
                pending_jobs=sum(metric(r,m) is None for r in rr), mean=mean, ci95_low=lo, ci95_high=hi))
    for other in comparators:
        base_cases = {c for a,c in grouped if a == BASE}
        other_cases = {c for a,c in grouped if a == other}
        shared = sorted(base_cases & other_cases)
        for m in METRICS:
            ds = [values[(BASE,c,m)]-values[(other,c,m)] for c in shared
                  if values[(BASE,c,m)] is not None and values[(other,c,m)] is not None]
            complete = len(ds) == len(base_cases) == len(other_cases) and bool(base_cases)
            use = ds if m not in HUMAN or complete else []
            mean, lo, hi, p = summarize(use, rng, args.replicates)
            comparisons.append(dict(arm=BASE, comparator=other, metric=m,
                comparison_role='primary' if other in PRIMARY else 'secondary',
                source_n=len(use), matched_sources=len(shared), complete_sources=len(ds),
                baseline_planned_sources=len(base_cases), comparator_planned_sources=len(other_cases),
                difference=mean, ci95_low=lo, ci95_high=hi, p_two_sided_signflip=p,
                p_holm=None, holm_family='all planned arm comparisons within metric',
                seed_aggregation='mean within source independently for each arm; RFEDIT single seed'))
    # Holm within each metric over the entire planned comparator family; missing p is
    # conservatively treated as 1 in multiplicity, but its output remains blank.
    for m in METRICS:
        family = [r for r in comparisons if r['metric'] == m]
        ordered = sorted(family, key=lambda r: r['p_two_sided_signflip'] if r['p_two_sided_signflip'] is not None else 1.0)
        previous = 0.0
        for i, r in enumerate(ordered):
            p = r['p_two_sided_signflip']
            previous = max(previous, min(1.0, (len(ordered)-i)*(p if p is not None else 1.0)))
            if p is not None:
                r['p_holm'] = previous
            r['holm_family_size'] = len(ordered)
    out = root / 'evidence'; out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'overnight_source_values.csv', source_rows)
    write_csv(out/'overnight_arm_summary.csv', summaries)
    write_csv(out/'overnight_comparisons.csv', comparisons)
    metadata = dict(baseline=BASE, primary_comparisons=list(PRIMARY), comparator_family=comparators,
        metrics=list(METRICS), ci='source bootstrap percentile 95%',
        signflip='two-sided exact at <=16 paired sources; Monte Carlo otherwise, plus-one correction',
        replicates=args.replicates, rng_seed=args.seed,
        holm='within each metric across all planned comparator arms; missing p reserved as 1',
        seed_policy='average all planned seeds per source and arm; RFEDIT single seed versus other arm two-seed mean',
        human_policy='failed generation zero; pending applicable items unknown; subject/pattern applicability defaults 1; inapplicable requirements omitted from joint, not counted as success; rates withheld until complete',
        numeric_missing_policy='complete within-source seed set required; other missing sources excluded and counts reported',
        evaluation_available=bool(evaluation), arm_summary=summaries, comparisons=comparisons)
    (out/'overnight_evidence.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Saved source-clustered evidence:', out)


if __name__ == '__main__':
    main()
