"""Recompute R86 compact-export statistics; no model inference or external data."""
import csv,itertools,json,math,random,statistics,collections
from pathlib import Path
R=Path(__file__).resolve().parent.parent; OUT=R/'analysis'/'audit.json'
def read_csv(name):
 with (R/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def num(v):
 try:
  x=float(v);return x if math.isfinite(x) else None
 except (TypeError,ValueError):return None
def holm(rows,key='p_exact'):
 ordered=sorted(rows,key=lambda r:r[key]); n=len(rows);prev=0
 for i,r in enumerate(ordered):prev=max(prev,min(1.,(n-i)*r[key]));r['_holm']=prev
 return {id(r):r['_holm'] for r in rows}
def main():
 plan=json.loads((R/'plan.json').read_text(encoding='utf-8-sig'));jobs=plan['jobs'];anchors=plan['r86_anchor_cells'];cases=sorted({j['case'] for j in jobs});assert len(jobs)==40 and len(anchors)==10 and len(cases)==10
 metrics=['clip_target_similarity','clip_directional_similarity','dino_full_feature_similarity','lpips_full','lpips_preserved','bg_mae']
 by={};statuses=collections.Counter()
 ev=read_csv('evaluation.csv');ae=read_csv('anchor_evaluation.csv'); jmap={j['id']:j for j in jobs};amap={a['id']:a for a in anchors}
 for r in ev:
  j=jmap[r['id']];statuses['new_'+r['eval_status']]+=1; by[(j['arm'],j['case'])]=r
 for r in ae:
  a=amap[r['id']];statuses['anchor_'+r['eval_status']]+=1;by[('CORR_1',a['case'])]=r
 arms=['CORR_1','GLOBAL_1','SHIFT_1','CORR_07','GLOBAL_07']; means={};complete={}
 for arm in arms:
  means[arm]={};complete[arm]={}
  for m in metrics:
   available={c:num(by.get((arm,c),{}).get(m)) for c in cases}; vals=[v for v in available.values() if v is not None];means[arm][m]={'n':len(vals),'mean':statistics.mean(vals) if vals else None,'missing_cases':[c for c,v in available.items() if v is None]};complete[arm][m]=len(vals)
 comparisons=[('CORR_1 - GLOBAL_1',{'CORR_1':1,'GLOBAL_1':-1},'primary'),('CORR_1 - SHIFT_1',{'CORR_1':1,'SHIFT_1':-1},'primary'),('CORR_07 - GLOBAL_07',{'CORR_07':1,'GLOBAL_07':-1},'exploratory'),('(CORR_1 - GLOBAL_1) - (CORR_07 - GLOBAL_07)',{'CORR_1':1,'GLOBAL_1':-1,'CORR_07':-1,'GLOBAL_07':1},'exploratory')]
 archived={(r['comparison'],r['metric']):r for r in read_csv('r86_paired.csv')};results=[];rng=random.Random(86001)
 for name,w,role in comparisons:
  for m in metrics:
   differences={}
   for c in cases:
    v={arm:num(by.get((arm,c),{}).get(m)) for arm in w}
    if all(x is not None for x in v.values()):differences[c]=sum(w[a]*x for a,x in v.items())
   ds=list(differences.values());n=len(ds);mean=statistics.mean(ds) if ds else None;boot=sorted(statistics.mean(rng.choices(ds,k=n)) for _ in range(10000)) if n>=2 else []
   ci=[boot[250],boot[9749]] if boot else [None,None]
   if n:
    threshold=abs(sum(ds));p=sum(abs(sum(x*sgn for x,sgn in zip(ds,signs)))>=threshold-1e-12 for signs in itertools.product((-1,1),repeat=n))/(2**n)
   else:p=None
   old=archived[(name,m)];oldci=[num(old['ci95_low']),num(old['ci95_high'])];oldmean=num(old['difference'])
   results.append({'comparison':name,'role':role,'metric':m,'n':n,'difference':mean,'ci95':ci,'source_differences':differences,'missing_cases':[c for c in cases if c not in differences],'p_exact':p,'archive_mean':oldmean,'archive_ci95':oldci,'archive_n':int(old['complete_sources']),'matches_archive':n==int(old['complete_sources']) and all((a==b if a is None or b is None else abs(a-b)<1e-12) for a,b in zip([mean]+ci,[oldmean]+oldci))})
 prim=[r for r in results if r['role']=='primary' and r['p_exact'] is not None]
 global12=holm(prim)
 for r in prim:r['p_holm_12_endpoints']=global12[id(r)]
 for m in metrics:
  group=[r for r in prim if r['metric']==m];corrected=holm(group)
  for r in group:r['p_holm_2_per_endpoint']=corrected[id(r)]
 for r in results:r.pop('_holm',None)
 dose={};errors=[]
 for j in jobs:
  f=R/j['id']/'step_summary.json';x=json.loads(f.read_text(encoding='utf-8-sig'));steps=x['steps'];vals=[]
  for s in steps:
   actual=num(s.get('actual_rms'));target=num(s.get('r86_target_rms'));h=num(s.get('h'))
   if actual is None or target is None or h is None or target<=0:errors.append((j['id'],s.get('step_index')));continue
   vals.append({'step':s.get('step_index'),'absolute':abs(actual-target),'relative':abs(actual-target)/target,'update_absolute':abs(h*actual-h*target),'reported_relative':num(s.get('r86_relative_rms_error'))})
  dose.setdefault(j['arm'],[]).append({'id':j['id'],'steps':len(steps),'valid':len(vals),'max_abs':max((v['absolute'] for v in vals),default=None),'max_rel':max((v['relative'] for v in vals),default=None),'max_update_abs':max((v['update_absolute'] for v in vals),default=None),'reported_relative_max_difference':max((abs(v['relative']-v['reported_relative']) for v in vals if v['reported_relative'] is not None),default=None)})
 dose_summary={arm:{'jobs':len(rows),'steps':sum(r['valid'] for r in rows),'max_abs':max(r['max_abs'] for r in rows if r['max_abs'] is not None),'max_rel':max(r['max_rel'] for r in rows if r['max_rel'] is not None),'max_update_abs':max(r['max_update_abs'] for r in rows if r['max_update_abs'] is not None),'jobs_details':rows} for arm,rows in dose.items()}
 loc=read_csv('localization_L.csv');local={};consistency=[]
 for arm in arms:
  rows=[r for r in loc if r['arm']==arm];values=[num(r['L']) for r in rows if num(r['L']) is not None]
  for r in rows:
   if r['status']=='ok':consistency.append(abs(float(r['L'])-(float(r['inside_mean'])-float(r['outside_mean']))))
  local[arm]={'statuses':dict(collections.Counter(r['status'] for r in rows)),'n':len(values),'mean_L':statistics.mean(values) if values else None,'median_L':statistics.median(values) if values else None,'per_case':{r['case']:num(r['L']) for r in rows}}
 locpair={}
 for left,right in [('CORR_07','SHIFT_1'),('GLOBAL_1','SHIFT_1')]:
  a=local[left]['per_case'];b=local[right]['per_case'];d={c:a[c]-b[c] for c in cases if a.get(c) is not None and b.get(c) is not None};locpair[left+' - '+right]={'n':len(d),'mean_difference':statistics.mean(d.values()) if d else None,'per_case':d}
 config=json.loads((R/'evaluation_config.json').read_text(encoding='utf-8-sig'))
 evaluation_audit={'config_status':config.get('status'),'config_lpips_status':config.get('lpips_status'),'config_lpips_status_counts':config.get('lpips_status_counts'),'config_incomplete_ids':config.get('incomplete_ids'),'new_rows':dict(collections.Counter(r.get('lpips_status') for r in ev)),'anchor_rows':dict(collections.Counter(r.get('lpips_status') for r in ae)),'new_preserved_status':dict(collections.Counter(r.get('lpips_preserved_status') for r in ev)),'anchor_preserved_status':dict(collections.Counter(r.get('lpips_preserved_status') for r in ae)),'weight_hash_pairs':dict(collections.Counter((r.get('lpips_calibration_sha256') or '')+' / '+(r.get('lpips_trunk_sha256') or '') for r in ev+ae)),'interpretation':'config lpips_status is initial configuration value; per-output rows and config lpips_status_counts report actual evaluation outcome. Investigate output rows and error fields rather than initial setting.'}
 audit={'evaluation_audit':evaluation_audit,'scope':'Post-hoc R86 pilot, 10 selected existing R85 sources, one seed, 10 reused anchors plus 40 newly generated. Six endpoints; two primary contrasts. CIs pointwise; Holm families selected post hoc. No inference about independent test/general population.','methods':{'ci':'Reproduction of archive: Python random.Random(86001), 10000 source bootstraps; sequential comparison and endpoint order; sorted order statistics [250] and [9749].','p_exact':'Two-sided exhaustive 2^n source sign flips; |sum| at least observed with absolute tolerance 1e-12; no Monte Carlo.','holm':'Two primary contrasts per endpoint; sensitivity family is 2 contrasts x 6 endpoints = 12 tests across endpoints, all post hoc.','localization':'L means and per-case contrasts recomputed from exported L CSV; masks not in compact export so individual values cannot be independently regenerated here.'},'population':{'sources':cases,'anchors':len(anchors),'new_jobs':len(jobs),'eval_status':dict(statuses)},'metric_completeness':complete,'arm_means':means,'paired':results,'archive_all_matches':all(r['matches_archive'] for r in results),'dose':{'arms':dose_summary,'invalid_steps':errors,'archived_audit':json.loads((R/'dose_audit.json').read_text())},'localization':{'arms':local,'paired':locpair,'max_csv_L_identity_error':max(consistency,default=None),'metadata':json.loads((R/'localization_metadata.json').read_text())}}
 OUT.write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');print('saved',OUT,'archive_all_matches',audit['archive_all_matches']);
 for r in results:
  if r['role']=='primary':print(r['comparison'],r['metric'],'n',r['n'],'diff',round(r['difference'],6),'CI',*[round(v,6) for v in r['ci95']],'p',r['p_exact'],'holm2',r['p_holm_2_per_endpoint'],'holm12',r['p_holm_12_endpoints'])
 print('loc',[(k,v['n'],v['mean_L']) for k,v in local.items()]);print('dose',[(k,v['max_rel']) for k,v in dose_summary.items()])
if __name__=='__main__':main()
