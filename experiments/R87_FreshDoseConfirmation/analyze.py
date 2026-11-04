"""Frozen source-level R87 contrasts and actual per-step RMS audit, CPU only."""
import argparse,csv,json,math
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
ARMS=['REQ','GLB','SFT','PERM']
METRICS=['lpips_preserved','clip_target_similarity','clip_directional_similarity','dino_full_feature_similarity','lpips_full','bg_mae']
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def num(v):
    try:
        x=float(v);return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvout(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def exact_p(ds):
    if not ds:return None
    values=np.array([0.],dtype=np.float64)
    for d in ds:values=np.concatenate([values-d,values+d])
    return float(np.mean(np.abs(values)>=abs(sum(ds))-1e-12))
def holm(rows,key):
    usable=sorted([r for r in rows if r['p_exact'] is not None],key=lambda r:r['p_exact']);prev=0.
    for i,r in enumerate(usable):
        prev=max(prev,min(1.,(len(rows)-i)*r['p_exact']));r[key]=prev
    for r in rows:r.setdefault(key,None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=ROOT/'runtime_base');a=p.parse_args();out=a.runtime/'results';plan=read(a.runtime/'plan.json')
    cases=sorted(t['case'] for t in plan['tasks']);jobs={j['id']:j for j in plan['jobs']}
    with (out/'evaluation.csv').open(encoding='utf-8-sig',newline='') as f:ev=list(csv.DictReader(f))
    with (out/'metrics.csv').open(encoding='utf-8-sig',newline='') as f: measured={r['id']:r for r in csv.DictReader(f)}
    rows={r['id']:{**measured.get(r['id'],{}),**r} for r in ev};index={(j['arm'],j['case'],int(j['seed'])):rows.get(j['id'],{}) for j in jobs.values()}
    complete={};source_rows=[];means=[]
    for arm in ARMS:
        for metric in METRICS:
            vals={}
            for case in cases:
                pair=[num(index.get((arm,case,s),{}).get(metric)) for s in (0,17)]
                value=float(np.mean(pair)) if all(v is not None for v in pair) else None
                vals[case]=value;source_rows.append(dict(arm=arm,case=case,metric=metric,seed0=pair[0],seed17=pair[1],mean=value))
            complete[arm,metric]=vals;usable=[v for v in vals.values() if v is not None]
            means.append(dict(arm=arm,metric=metric,sources=len(usable),planned_sources=20,mean=float(np.mean(usable)) if usable else None))
    paired=[];rng=np.random.default_rng(87004)
    for other in ('GLB','SFT','PERM'):
        for metric in METRICS:
            ds={c:complete['REQ',metric][c]-complete[other,metric][c] for c in cases if complete['REQ',metric][c] is not None and complete[other,metric][c] is not None}
            values=list(ds.values());n=len(values)
            boot=np.mean(rng.choice(values,size=(10000,n),replace=True),axis=1) if n else []
            ci=list(map(float,np.quantile(boot,[.025,.975]))) if n else [None,None]
            role='primary' if other=='GLB' and metric=='lpips_preserved' else 'secondary_position' if metric=='lpips_preserved' else 'exploratory'
            paired.append(dict(comparison='REQ - '+other,metric=metric,role=role,sources=n,planned_sources=20,
                difference=float(np.mean(values)) if n else None,ci95_low=ci[0],ci95_high=ci[1],p_exact=exact_p(values),
                missing_sources=[c for c in cases if c not in ds],source_differences=ds))
    holm([r for r in paired if r['role']=='secondary_position'],'p_holm_secondary2')
    holm([r for r in paired if r['metric']=='lpips_preserved'],'p_holm_preservation3')
    holm(paired,'p_holm_all18')
    for r in paired:
        for k in ('p_holm_secondary2','p_holm_preservation3'):r.setdefault(k,None)
    audit=[];locations=[];step_rows=[]
    for job in jobs.values():
        folder=out/job['id'];f=folder/'steps.json'
        if not f.exists():audit.append(dict(id=job['id'],arm=job['arm'],status='missing_steps'));continue
        steps=read(f).get('edit',{}).get('steps',[])
        anchor=out/(job.get('r86_anchor') or job['id'])/'steps.json'
        anchors=read(anchor).get('edit',{}).get('steps',[]) if anchor.exists() else []
        errors=[];impulse=0.;expected_impulse=0.;invalid=[]
        for i,s in enumerate(steps):
            actual=num(s.get('actual_rms'));target=num(anchors[i].get('actual_rms')) if i<len(anchors) else None;h=num(s.get('h'))
            if actual is None or target is None or h is None or target<=0:invalid.append(i);continue
            error=abs(actual-target);errors.append(error/target);impulse+=abs(h)*actual;expected_impulse+=abs(h)*target
            step_rows.append(dict(id=job['id'],arm=job['arm'],case=job['case'],seed=job['seed'],step=i,actual_rms=actual,target_rms=target,h=h,absolute_error=error,relative_error=error/target))
        audit.append(dict(id=job['id'],arm=job['arm'],status='ok' if len(steps)==28 and not invalid else 'incomplete',
            steps=len(steps),invalid_steps=invalid,max_relative_error=max(errors,default=None),impulse=impulse,anchor_impulse=expected_impulse))
        npz=folder/'request_maps.npz'
        if npz.exists():
            from PIL import Image
            with np.load(npz) as maps:
                mask=np.asarray(Image.open(a.runtime/'data'/job['case']/'outer.png').convert('L').resize((64,64),Image.Resampling.NEAREST))>0
                for name in ('E_original','E_shifted','E_permuted'):
                    if name not in maps:continue
                    e=maps[name].squeeze()
                    if e.shape==(64,64) and mask.any() and (~mask).any():
                        locations.append(dict(id=job['id'],case=job['case'],seed=job['seed'],arm=job['arm'],map=name,L=float(e[mask].mean()-e[~mask].mean())))
    evidence=out/'evidence';evidence.mkdir(exist_ok=True)
    result=dict(protocol=read(ROOT/'protocol.json'),planned_outputs=160,evaluated_rows=len(rows),status_counts=dict(Counter(r.get('eval_status') for r in ev)),
        arm_means=means,paired=paired,dose=audit,missing_job_ids=sorted(set(jobs)-set(rows)),
        inference='Source is the independent unit; first average both planned seeds. Missing paired observations are listed, never replaced by zeros. CIs are pointwise source bootstrap intervals. Primary test and secondary family were frozen before generation; all-18 sensitivity also reported.')
    save(evidence/'r87_statistics.json',result);csvout(evidence/'r87_source_values.csv',source_rows);csvout(evidence/'r87_arm_means.csv',means)
    csvout(evidence/'r87_paired.csv',[{k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in paired])
    csvout(evidence/'r87_actual_profiles.csv',step_rows);csvout(evidence/'r87_localization.csv',locations)
    primary=next(r for r in paired if r['role']=='primary');print(json.dumps(primary,ensure_ascii=False))
    if any(r['status']!='ok' or r.get('max_relative_error',0)>1e-5 for r in audit):raise RuntimeError('RMS profile audit incomplete/above numerical tolerance; results kept for inspection.')
    print('Statistics saved; no automatic method ranking or semantic-success claim.')
if __name__=='__main__':main()
