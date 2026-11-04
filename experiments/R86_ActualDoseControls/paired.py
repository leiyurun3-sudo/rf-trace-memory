import argparse,csv,json,math,random,statistics
from pathlib import Path
from collections import defaultdict

def load_table(path):
    if not path.is_file(): return {}
    with path.open(encoding='utf-8-sig',newline='') as f: return {r['id']:r for r in csv.DictReader(f)}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--r85',required=True,type=Path); parser.add_argument('--runtime',required=True,type=Path); args=parser.parse_args()
    plan=json.loads((args.runtime/'plan.json').read_text(encoding='utf-8')); anchors={x['id']:x for x in plan['r86_anchor_cells']}; expected=defaultdict(set); jobs={x['id']:x for x in plan['jobs']}; by={}
    for job in plan['jobs']: expected[job['case']].add(str(job['seed']))
    for root,original in ((args.r85/'results',True),(args.runtime/'results',False)):
        combined=load_table(root/'comparison_metrics.csv'); evaluated=load_table(args.runtime/'results'/'anchor_evaluation.csv' if original else root/'evaluation.csv')
        for ident,row in combined.items():
            if original and ident not in anchors: continue
            if not original and ident not in jobs: continue
            row=dict(row); row.update({k:v for k,v in evaluated.get(ident,{}).items() if v not in ('',None)})
            arm='CORR_1' if original else jobs[ident]['arm']
            meta=anchors[ident] if original else jobs[ident]
            by[(arm,meta['case'],str(meta['seed']))]=row
    pilot=any(j['arm']=='CORR_07' for j in jobs.values())
    contrasts=[('CORR_1 - GLOBAL_1',{'CORR_1':1,'GLOBAL_1':-1},'primary'),('CORR_1 - SHIFT_1',{'CORR_1':1,'SHIFT_1':-1},'primary')]
    if pilot: contrasts += [('CORR_07 - GLOBAL_07',{'CORR_07':1,'GLOBAL_07':-1},'exploratory'),('(CORR_1 - GLOBAL_1) - (CORR_07 - GLOBAL_07)',{'CORR_1':1,'GLOBAL_1':-1,'CORR_07':-1,'GLOBAL_07':1},'exploratory')]
    metrics=['clip_target_similarity','clip_directional_similarity','dino_full_feature_similarity','lpips_full','lpips_preserved','bg_mae']; out=[]; rng=random.Random(86001)
    for name,weights,role in contrasts:
        for metric in metrics:
            vals=[]; missing=[]
            for case,seeds in sorted(expected.items()):
                per_seed=[]
                for seed in sorted(seeds):
                    try:
                        values={arm:float(by[(arm,case,seed)][metric]) for arm in weights}
                        if not all(math.isfinite(x) for x in values.values()): raise ValueError('nonfinite')
                        per_seed.append(sum(weights[arm]*v for arm,v in values.items()))
                    except (KeyError,TypeError,ValueError): break
                if len(per_seed)==len(seeds): vals.append(statistics.mean(per_seed))
                else: missing.append(case)
            n=len(vals); estimates=sorted(statistics.mean(rng.choices(vals,k=n)) for _ in range(10000)) if n>=2 else []
            out.append(dict(comparison=name,role=role,metric=metric,complete_sources=n,planned_sources=len(expected),difference=statistics.mean(vals) if vals else None,ci95_low=estimates[250] if estimates else None,ci95_high=estimates[9749] if estimates else None,missing_cases=';'.join(missing),interval='pointwise source bootstrap, 10000 resamples, seed 86001; no multiplicity adjustment'))
    with (args.runtime/'results/r86_paired.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=out[0]); w.writeheader(); w.writerows(out)
    print('Wrote primary and exploratory source-paired estimates; missing labels remain missing.')
if __name__=='__main__': main()
