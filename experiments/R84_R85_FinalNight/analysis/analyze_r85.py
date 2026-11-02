import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[3]/'results'/'R85')
parser.add_argument('--out',type=Path)
args=parser.parse_args()
P=args.results; O=args.out or P/'analysis'; O.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(P/'evaluation.csv',dtype={'case':str}); ev=json.loads((P/'evidence/overnight_evidence.json').read_text()); st=json.loads((P/'statistics.json').read_text())
metrics=['clip_target_similarity','clip_directional_similarity','clip_image_similarity','dino_full_feature_similarity','lpips_full','lpips_preserved','pixel_ssim','mae_vs_source','object_mae','bg_mae','nfe_total']
extra=['pattern_ncc_fixed']; allm=metrics+extra
arms=['RELEASE_CORR25','RELEASE_BOTH25','ENERGY_MATCH25','U25_KV','FLOWEDIT_N26','ETA03','RFEDIT_I08']
comps=['U25_KV','ENERGY_MATCH25','FLOWEDIT_N26','RFEDIT_I08','ETA03','RELEASE_BOTH25']; rng=np.random.default_rng(20261102)
sv=d.groupby(['arm','case'])[allm].agg(lambda x: float(x.mean()) if x.notna().all() else np.nan)
sv.to_csv(O/'source_equal_values.csv'); audit=[]
alias=lambda m:'preserved_region_mae' if m=='bg_mae' else m
for x in ev['arm_summary']:
 m='bg_mae' if x['metric']=='preserved_region_mae' else x['metric']
 if m in allm and x['mean'] is not None:
  a=sv.loc[x['arm'],m].dropna(); audit.append(abs(a.mean()-x['mean']))

def holm(ps):
 p=np.asarray(ps,float); ix=np.argsort(p); q=np.empty(len(p)); q[ix]=np.minimum(1,np.maximum.accumulate(p[ix]*(len(p)-np.arange(len(p))))); return q

def infer(a):
 a=np.asarray(a); n=len(a); idx=rng.integers(0,n,(10000,n)); lo,hi=np.quantile(a[idx].mean(1),[.025,.975]); obs=abs(a.mean()); perm=(rng.choice([-1.,1.],(10000,n))*a).mean(1); p=(np.count_nonzero(np.abs(perm)>=obs-1e-15)+1)/10001; return float(lo),float(hi),float(p)
summ=[]
for arm in arms:
 for m in allm:
  a=sv.loc[arm,m].dropna(); e=next((x for x in ev['arm_summary'] if x['arm']==arm and x['metric']==alias(m)),None)
  if e and e['mean'] is not None: lo,hi=e['ci95_low'],e['ci95_high']; origin='overnight source bootstrap'
  else: lo,hi,_=infer(a); origin='recomputed source bootstrap'
  summ.append(dict(arm=arm,metric=m,n_sources=len(a),n_jobs=int(d[d.arm==arm][m].notna().sum()),mean=float(a.mean()),ci95_low=lo,ci95_high=hi,ci_origin=origin))
summary=pd.DataFrame(summ); summary.to_csv(O/'method_means_ci.csv',index=False)
wide=summary.pivot(index='arm',columns='metric',values='mean').reindex(arms); wide.to_csv(O/'publication_means.csv')
rows=[]; delta_errors=[]; existing_holm_errors=[]
for c in comps:
 for m in allm:
  a=sv.loc['RELEASE_CORR25',m]; b=sv.loc[c,m]; pairs=pd.concat([a,b],axis=1).dropna(); z=pairs.iloc[:,0]-pairs.iloc[:,1]
  e=next((x for x in ev['comparisons'] if x['comparator']==c and x['metric']==alias(m)),None)
  if e and e['difference'] is not None:
   delta_errors.append(abs(z.mean()-e['difference'])); lo,hi,p=e['ci95_low'],e['ci95_high'],e['p_two_sided_signflip']; origin='overnight source bootstrap/signflip'
  else: lo,hi,p=infer(z); origin='recomputed source bootstrap/signflip'
  old=next((x for x in st['comparisons'] if x['arm']=='RELEASE_CORR25' and x['reference']==c and x['metric']==m),None)
  if old is None:
   old=next((x for x in st['comparisons'] if x['arm']==c and x['reference']=='RELEASE_CORR25' and x['metric']==m),None)
  rows.append(dict(arm='RELEASE_CORR25',reference=c,metric=m,n_sources=len(z),difference=float(z.mean()),ci95_low=lo,ci95_high=hi,p_two_sided_signflip=p,p_holm_6=None,existing_p_holm_21=old.get('holm_p') if old else None,inference_origin=origin))
r=pd.DataFrame(rows)
for m,g in r.groupby('metric'):
 r.loc[g.index,'p_holm_6']=holm(g.p_two_sided_signflip)
 for ix,x in g.iterrows():
  e=next((v for v in ev['comparisons'] if v['comparator']==x.reference and v['metric']==alias(m)),None)
  if e and e['p_holm'] is not None: existing_holm_errors.append(abs(r.loc[ix,'p_holm_6']-e['p_holm']))
r['p_holm_global_66_numeric']=np.nan
ix=r.index[r.metric.isin(metrics)]; r.loc[ix,'p_holm_global_66_numeric']=holm(r.loc[ix,'p_two_sided_signflip'])
# Verify the original statistics.json Holm families directly, retaining missing hypotheses as p=1.
st_holm=[]
for m in set(x['metric'] for x in st['comparisons']):
 g=[x for x in st['comparisons'] if x['metric']==m and x.get('source_signflip_p') is not None]; pp=[x['source_signflip_p'] for x in g]; q=holm(pp)
 for x,v in zip(g,q):
  if x.get('holm_p') is not None: st_holm.append(abs(x['holm_p']-v))
# Orient BOTH minus CORR, retaining two-sided p values and reversing CI endpoints.
both=r[r.reference=='RELEASE_BOTH25'].copy(); both['arm']='RELEASE_BOTH25'; both['reference']='RELEASE_CORR25'; both['difference']=-both.difference; ll=both.ci95_low.copy(); both['ci95_low']=-both.ci95_high; both['ci95_high']=-ll
pub=pd.concat([r[r.reference!='RELEASE_BOTH25'],both],ignore_index=True); pub.to_csv(O/'paired_comparisons_ci_pHolm.csv',index=False)
# Joint descriptive comparisons use identical source support for semantic and retention coordinates.
sem=['clip_target_similarity','clip_directional_similarity']; ret=['clip_image_similarity','dino_full_feature_similarity','lpips_full','lpips_preserved','pixel_ssim']; direction={m:(-1 if m.startswith('lpips') else 1) for m in sem+ret}
joint=[]
for a,b in [('RELEASE_CORR25',c) for c in comps if c!='RELEASE_BOTH25']+[('RELEASE_BOTH25','RELEASE_CORR25')]:
 for sm in sem:
  for rm in ret:
   x=sv.loc[a,[sm,rm]].join(sv.loc[b,[sm,rm]],lsuffix='_a',rsuffix='_b').dropna(); ds=x[sm+'_a']-x[sm+'_b']; dr=(x[rm+'_a']-x[rm+'_b'])*direction[rm]
   joint.append(dict(arm=a,reference=b,semantic=sm,retention=rm,n_sources=len(x),semantic_difference=ds.mean(),retention_difference_oriented_higher_better=dr.mean(),both_improved_sources=int(((ds>0)&(dr>0)).sum()),semantic_up_retention_down_sources=int(((ds>0)&(dr<0)).sum()),semantic_down_retention_up_sources=int(((ds<0)&(dr>0)).sum()),both_worse_sources=int(((ds<0)&(dr<0)).sum()),ties_sources=int(((ds==0)|(dr==0)).sum()),mean_relation='dominates' if ds.mean()>=0 and dr.mean()>=0 else ('dominated' if ds.mean()<=0 and dr.mean()<=0 else 'tradeoff')))
pd.DataFrame(joint).to_csv(O/'joint_semantic_retention.csv',index=False)
pareto=[]
for label,mm in [('full40',sem+ret[:3]),('preserved_common33',sem+ret)]:
 common=set.intersection(*(set(sv.loc[a,mm].dropna().index) for a in arms)); mat=pd.DataFrame({a:sv.loc[a].loc[sorted(common),mm].mean() for a in arms}).T
 mat.to_csv(O/('joint_means_'+label+'.csv')); oriented=mat*np.array([direction[m] for m in mm])
 for a in arms:
  dom=[b for b in arms if b!=a and (oriented.loc[b]>=oriented.loc[a]).all() and (oriented.loc[b]>oriented.loc[a]).any()]
  pareto.append(dict(support=label,n_sources=len(common),arm=a,dominated_by=dom,pareto_optimal=not dom))
meta=dict(dataset=str(P),jobs=len(d),sources=int(d['case'].nunique()),jobs_by_arm=d.groupby('arm').size().to_dict(),estimator='Equal source mean of per-arm complete seed mean; paired difference of these source means. RFEDIT uses its single deterministic result, never duplicated.',seed_policy='seeds 0,17; RFEDIT seed 0 only',bootstrap='10000 source-resampling percentile 95% intervals, pointwise, not multiplicity-adjusted',p_test='two-sided source sign flip, 10000 Monte Carlo replicates, plus-one correction; assumes source-level sign symmetry under null',multiplicity=dict(p_holm_6='six planned CORR comparisons separately per metric; includes BOTH and RF',existing_p_holm_21='21 all-arm pairwise contrasts separately per metric from statistics.json; separate MC stream',p_holm_global_66_numeric='additional sensitivity: 6 comparisons x 11 requested numeric outcomes; excludes pattern diagnostic and unavailable human outcomes'),audit=dict(max_abs_overnight_mean_error=max(audit),max_abs_overnight_paired_delta_error=max(delta_errors),max_abs_overnight_holm_error=max(existing_holm_errors),max_abs_statistics_holm_available_family_error=max(st_holm),statistics_family_note='21 valid pairs per requested numeric metric; correction_impulse diagnostic uses only 10 valid pairs (missing p omitted)',duplicate_arm_source_seed=int(d.duplicated(['arm','case','seed']).sum()),seed_sets_complete=all(set(g.seed)==({0} if a=='RFEDIT_I08' else {0,17}) for (a,c),g in d.groupby(['arm','case'])),all_eval_ok=bool((d.eval_status=='ok').all()),regions=dict(preserved_sources=33,empty_preserved_sources=7,pixel_scope=d.pixel_scope.unique().tolist()),human_rated=0),definitions=dict(clip='CLIP ViT-L/14',dino='DINOv2-base full-image similarity',lpips='AlexNet LPIPS; preserved map weighted by area-resampled official edit-mask complement; receptive fields cross region boundary',ssim='existing background-region metric; local windows may cross mask boundary',bg_mae='existing bg_mae, alias overnight preserved_region_mae',object_mae='existing object_mae; region identity follows dataset masks',mae_vs_source='full-image source MAE',nfe='nfe_total, complete standalone cost; constants by arm, structural budget differences',joint='descriptive source-level semantic/retention quadrants and mean Pareto; no human joint-success labels available'),interpretation=['CORR versus ENERGY: similar semantic means; full LPIPS improves and survives global 66-test Holm; preserved LPIPS and SSIM improve under 6-comparison Holm but not 21-comparison or global 66-test Holm.', 'U25 descriptively dominates CORR across all five full-image semantic/retention means on 40 sources, at 91 versus 97 NFE. On the common 33 preserved-region sources CORR target CLIP is higher, so strict dominance disappears.', 'FLOW versus CORR has stronger directional editing and lower LPIPS/full and preserved plus higher SSIM, with 52 versus 97 NFE; CORR DINO mean is higher but uncertain.', 'BOTH versus CORR improves target and directional CLIP while worsening LPIPS, DINO and SSIM; semantic and LPIPS changes survive global 66-test Holm, DINO and SSIM do not.', 'RF versus CORR is a strong retention versus semantic tradeoff. No human joint-success measurements are available.', 'Pareto sets are descriptive coordinatewise comparisons, not inferential dominance or semantic equivalence; nonsignificant semantic differences do not establish equivalence.'],pareto=pareto,means=summary.to_dict('records'),comparisons=pub.to_dict('records'))
def clean(x):
 if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,float) and not math.isfinite(x):return None
 return x
(O/'r85_numeric_results.json').write_text(json.dumps(clean(meta),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('MEANS\n',wide.round(6).to_string()); print('AUDIT',meta['audit']); print('PARETO',pareto); print('KEY COMPARISONS\n',pub[pub.metric.isin(sem+['dino_full_feature_similarity','lpips_full','lpips_preserved','pixel_ssim'])][['arm','reference','metric','difference','ci95_low','ci95_high','p_holm_6','existing_p_holm_21','p_holm_global_66_numeric']].round(6).to_string(index=False)); print('OUTPUT',str(O))
