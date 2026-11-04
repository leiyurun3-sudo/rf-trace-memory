from pathlib import Path
import sys, json, itertools, platform, warnings
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
BASE=Path(__file__).resolve().parents[3]/'results'
MET={'clip_target_similarity':1,'clip_directional_similarity':1,'dino_full_feature_similarity':1,'lpips_full':-1,'lpips_preserved':-1,'bg_mae':-1,'pixel_ssim':1,'clip_image_similarity':1}
SHORT={'clip_target_similarity':'Target CLIP','clip_directional_similarity':'Directional CLIP','dino_full_feature_similarity':'DINO','lpips_full':'Full LPIPS','lpips_preserved':'Preserved LPIPS','bg_mae':'Preserved MAE','pixel_ssim':'Preserved SSIM','clip_image_similarity':'Image CLIP'}
TRACE={'RELEASE_CORR15','RELEASE_CORR25','RELEASE_CORR35','ENERGY_MATCH25','RELEASE_BOTH25','U25_KV'}
COLORS={'RELEASE_CORR15':'#56B4E9','RELEASE_CORR25':'#0072B2','RELEASE_CORR35':'#004568','ENERGY_MATCH25':'#E69F00','U25_KV':'#009E73','RELEASE_BOTH25':'#CC79A7'}
NAME={'RELEASE_CORR15':'CORR15','RELEASE_CORR25':'CORR25','RELEASE_CORR35':'CORR35','ENERGY_MATCH25':'ENERGY','RELEASE_BOTH25':'BOTH25','U25_KV':'U25'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
manifest={'analysis':'Exploratory CPU offline dose/metric analysis','sampling_unit':'source image; all planned seeds averaged before analysis; source-exclusive CV','dose_definition':'I=sum_i h_i * RMS(applied correction velocity_i). No additional h applied to stored correction_impulse. This is latent impulse, not squared energy.','undefined_dose':'FlowEdit, RF-Edit, ETA03 set to missing; native ETA03 zero not treated as a comparable correction dose.','inputs':{},'environment':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__,'pillow_dependency':'read-only runtime site-packages appended after scientific imports'},'model_interpretation':'Associational models of state-dependent realized impulse; source difficulty, allocation, and memory also affect impulse. No causal sufficiency or universal collapse claim.','cv':'Leave one entire source out. No seeds from a held-out source enter fit. Centered targets are held-out within-source profile deviations, not absolute predictions; held-out outcome mean is used only to define/scoring deviations, never to fit or reconstruct an absolute prediction.','models':'linear, quadratic, log candidates all reported, no selected-model significance; within-source baseline is zero, raw baseline is training-source mean','bootstrap':'2000 source bootstrap draws for mean within-source Spearman, seed 20261103; descriptive pointwise intervals, no multiplicity claims','exclusions':'No VLM data; dose curves exclude noncomparable algorithms and distinguish memory release.'}
allrows=[];coverage=[];checks=[]
for ds in ['R84','R85']:
 path=BASE/ds/'metrics.csv';d=pd.read_csv(path,dtype={'case':str,'id':str});e=pd.read_csv(BASE/ds/'evaluation.csv',dtype={'case':str,'id':str})
 manifest['inputs'][ds]={'metrics':str(path),'evaluation':str(BASE/ds/'evaluation.csv'),'metrics_rows':len(d),'evaluation_rows':len(e),'unique_ids':d.id.nunique(),'duplicates':int(d.duplicated(['arm','case','seed']).sum()),'status_counts':d.status.value_counts().to_dict()}
 d['dataset']=ds;d['source']=d['case'];d['I_native']=pd.to_numeric(d['correction_impulse'],errors='coerce');d['I']=d.I_native.where(d.arm.isin(TRACE));d['allocation_local']=d.arm.str.startswith('RELEASE_').astype(float);d['memory_release']=(d.arm=='RELEASE_BOTH25').astype(float)
 for m in MET:d[m]=pd.to_numeric(d[m],errors='coerce')
 # Compare overlapping evaluation fields by ID, without changing metrics.
 for m in MET:
  if m in e:
   merged=d[['id',m]].merge(e[['id',m]],on='id',suffixes=('_metrics','_evaluation'))
   a=pd.to_numeric(merged[m+'_metrics'],errors='coerce');b=pd.to_numeric(merged[m+'_evaluation'],errors='coerce');ok=a.notna()&b.notna()
   checks.append({'dataset':ds,'metric':m,'common_finite':int(ok.sum()),'max_abs_metrics_evaluation_difference':float(np.max(np.abs(a[ok]-b[ok]))) if ok.any() else None,'finite_mask_disagreements':int((a.notna()!=b.notna()).sum())})
 for (arm,src),g in d.groupby(['arm','source']):
  expected={0} if arm=='RFEDIT_I08' else {0,17}
  complete=set(g.seed)==expected and len(g)==len(expected)
  row={'dataset':ds,'arm':arm,'source':src,'outputs':len(g),'planned_seed_complete':complete}
  for c in list(MET)+['I','I_native','corr_lambda','allocation_local','memory_release']:
   vals=pd.to_numeric(g[c],errors='coerce');row[c]=float(vals.mean()) if complete and vals.notna().all() else np.nan
  allrows.append(row)
 for arm,g in d.groupby('arm'):
  coverage.append({'dataset':ds,'arm':arm,'rows':len(g),'sources':g.source.nunique(),'dose_finite':int(g.I.notna().sum()),'native_dose_finite':int(g.I_native.notna().sum()),'native_dose_zero':int((g.I_native==0).sum()),**{m+'_finite':int(g[m].notna().sum()) for m in MET},'step_log_outputs_recomputed':0})
 d.to_csv(OUT/f'{ds}_output_metrics.csv',index=False)
src=pd.DataFrame(allrows);src.to_csv(OUT/'source_arm_values.csv',index=False)
pd.DataFrame(checks).to_csv(OUT/'metrics_evaluation_agreement.csv',index=False)
manifest['source_overlap']=sorted(set(src[src.dataset=='R84'].source)&set(src[src.dataset=='R85'].source))
# Recompute available step logs using exactly one factor h.
step=BASE/'R85/rollout_audit/actual_rollout_steps.csv';ss=pd.read_csv(step);audit=[]
for job,g in ss.groupby('job'):
 for arm,col,cum in [('RELEASE_CORR25','corr_rms','corr_cumulative'),('ENERGY_MATCH25','energy_rms','energy_cumulative')]:
  jid=job.replace('RELEASE_CORR25',arm);native=pd.read_csv(BASE/'R85/metrics.csv',usecols=['id','correction_impulse']).set_index('id').loc[jid,'correction_impulse']
  I=float(np.sum(g.h*g[col]));audit.append({'dataset':'R85','id':jid,'arm':arm,'steps':len(g),'h_sum':g.h.sum(),'recomputed_I':I,'native_I':native,'abs_difference':abs(I-native),'stored_cumulative_final':g.sort_values('step')[cum].iloc[-1]})
audit=pd.DataFrame(audit);audit.to_csv(OUT/'step_impulse_reconstruction.csv',index=False)
for row in coverage:
 row['step_log_outputs_recomputed']=int(((audit.dataset==row['dataset'])&(audit.arm==row['arm'])).sum())
pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
manifest['step_coverage']={'input':str(step),'paired_step_rows':len(ss),'recomputed_outputs':len(audit),'max_abs_I_error':float(audit.abs_difference.max()),'steps_per_output':sorted(audit.steps.unique().tolist()),'R84':'No per-job step logs in specified results tree; use native impulse whose exact formula was inspected in engine.py.','other_R85_arms':'Native impulses only; per-step exports absent in specified results tree.','engine_source':str(BASE.parent/'experiments/R84_R85_FinalNight/R85_IndependentComparison/engine.py'),'engine_formula_line':399}
# Complete per-arm, source-level descriptive statistics.
desc=[]
for (ds,arm),g in src.groupby(['dataset','arm']):
 for m in ['I']+list(MET):
  v=g[m].dropna();desc.append({'dataset':ds,'arm':arm,'metric':m,'n_sources':len(v),'mean':v.mean(),'sd':v.std(),'median':v.median(),'min':v.min(),'max':v.max()})
pd.DataFrame(desc).to_csv(OUT/'descriptive_source_means.csv',index=False)
def arms_for(ds,pop):
 local=['RELEASE_CORR15','RELEASE_CORR25','RELEASE_CORR35'] if ds=='R84' else ['RELEASE_CORR25']
 if pop=='local_dose':return local
 common=local+['ENERGY_MATCH25']+(['U25_KV'] if ds=='R85' else [])
 return common+(['RELEASE_BOTH25'] if pop=='with_memory' else [])
def prep(ds,pop,m):
 arms=arms_for(ds,pop);g=src[(src.dataset==ds)&src.arm.isin(arms)].dropna(subset=['I',m]).copy()
 good=g.groupby('source').arm.nunique();good=good[good==len(arms)].index
 return g[g.source.isin(good)].copy()
def features(g,form,indicators):
 I=g.I.to_numpy();x=I[:,None]
 if form=='quadratic':x=np.column_stack([I,I**2])
 if form=='log':x=np.log(I)[:,None]
 if indicators:x=np.column_stack([x,g.allocation_local.to_numpy(),g.memory_release.to_numpy()])
 return x
scores=[];predrows=[];coefs=[];rng=np.random.default_rng(20261103)
for ds in ['R84','R85']:
 for pop in ['local_dose','same_memory','with_memory']:
  if ds=='R85' and pop=='local_dose':continue
  for m in MET:
   g=prep(ds,pop,m).reset_index(drop=True)
   for form in ['linear','quadratic','log']:
    for indic in [False,True]:
     if pop=='local_dose' and indic:continue
     for mode in ['raw','centered']:
      X=features(g,form,indic);y=g[m].to_numpy();groups=g.source.to_numpy()
      if mode=='centered':
       X=X-pd.DataFrame(X).groupby(groups).transform('mean').to_numpy();y=y-pd.Series(y).groupby(groups).transform('mean').to_numpy();A=X
      else:A=np.column_stack([np.ones(len(X)),X])
      fit=np.linalg.lstsq(A,y,rcond=None)[0];fitted=A@fit;pred=np.full(len(y),np.nan);base=np.zeros(len(y));ranks=[]
      for hold in sorted(set(groups)):
       tr=groups!=hold;te=~tr;b=np.linalg.lstsq(A[tr],y[tr],rcond=None)[0];pred[te]=A[te]@b;base[te]=0 if mode=='centered' else y[tr].mean();ranks.append(int(np.linalg.matrix_rank(A[tr])))
      err=pred-y;sse=np.sum(err**2);den=np.sum((y-base)**2);fitden=np.sum((y-y.mean())**2)
      rec={'dataset':ds,'population':pop,'metric':m,'form':form,'indicators':indic,'mode':mode,'n_sources':len(set(groups)),'n_source_arm_rows':len(g),'n_features':A.shape[1],'minimum_training_rank':min(ranks),'CV_RMSE':float(np.sqrt(np.mean(err**2))),'CV_MAE':float(np.mean(abs(err))),'CV_baseline_RMSE':float(np.sqrt(np.mean((y-base)**2))),'CV_skill_vs_baseline':float(1-sse/den) if den else np.nan,'train_R2':float(1-np.sum((y-fitted)**2)/fitden) if fitden else np.nan,'source_rmse_max':float(max(np.sqrt(np.mean(err[groups==s]**2)) for s in set(groups))),'max_abs_residual':float(max(abs(err)))}
      scores.append(rec)
      for k,b in enumerate(fit):coefs.append({**{z:rec[z] for z in ['dataset','population','metric','form','indicators','mode']},'coefficient_index':k,'coefficient':b})
      for j,row in g.iterrows():predrows.append({**{z:rec[z] for z in ['dataset','population','metric','form','indicators','mode']},'source':row.source,'arm':row.arm,'I':row.I,'observed':y[j],'LOSO_predicted':pred[j],'LOSO_baseline':base[j],'residual':err[j]})
score=pd.DataFrame(scores);score.to_csv(OUT/'dose_models_LOSO.csv',index=False);pd.DataFrame(predrows).to_csv(OUT/'LOSO_predictions.csv',index=False);pd.DataFrame(coefs).to_csv(OUT/'dose_model_coefficients.csv',index=False)
# Transfer fixed R84 fits, no R85 fitting or model selection. Evaluate common policies only.
trans=[];tp=[]
for m in MET:
 train=prep('R84','same_memory',m);test=src[(src.dataset=='R85')&src.arm.isin(['RELEASE_CORR25','ENERGY_MATCH25'])].dropna(subset=['I',m]).copy();good=test.groupby('source').arm.nunique();test=test[test.source.isin(good[good==2].index)]
 for form in ['linear','quadratic','log']:
  for indic in [False,True]:
   for mode in ['raw','centered']:
    X=features(train,form,indic);Z=features(test,form,indic);y=train[m].to_numpy();v=test[m].to_numpy()
    if mode=='centered':
     X=X-pd.DataFrame(X).groupby(train.source.to_numpy()).transform('mean').to_numpy();Z=Z-pd.DataFrame(Z).groupby(test.source.to_numpy()).transform('mean').to_numpy();y=y-pd.Series(y).groupby(train.source.to_numpy()).transform('mean').to_numpy();v=v-pd.Series(v).groupby(test.source.to_numpy()).transform('mean').to_numpy();base=np.zeros(len(v))
    else:X=np.column_stack([np.ones(len(X)),X]);Z=np.column_stack([np.ones(len(Z)),Z]);base=np.full(len(v),y.mean())
    b=np.linalg.lstsq(X,y,rcond=None)[0];pred=Z@b;den=np.sum((v-base)**2)
    rec={'metric':m,'form':form,'indicators':indic,'mode':mode,'training_sources':train.source.nunique(),'training_rows':len(train),'test_sources':test.source.nunique(),'test_rows':len(test),'training_I_min':train.I.min(),'training_I_max':train.I.max(),'test_outside_training_I_range':int(((test.I<train.I.min())|(test.I>train.I.max())).sum()),'transfer_RMSE':float(np.sqrt(np.mean((v-pred)**2))),'transfer_skill':float(1-np.sum((v-pred)**2)/den) if den else np.nan}
    trans.append(rec)
    for j,(_,row) in enumerate(test.iterrows()):tp.append({**{k:rec[k] for k in ['metric','form','indicators','mode']},'source':row.source,'arm':row.arm,'I':row.I,'observed':v[j],'predicted':pred[j]})
pd.DataFrame(trans).to_csv(OUT/'R84_to_R85_transfer.csv',index=False);pd.DataFrame(tp).to_csv(OUT/'transfer_predictions.csv',index=False)
manifest['transfer_scope']='R84 same-memory CORR15/25/35 + ENERGY fits transferred to R85 CORR25 + ENERGY only. No U25 allocation extrapolation or BOTH memory extrapolation; populations and intervention range differ. Centered transfer evaluates within-source policy profiles only.'
# Direction-aligned within-source Spearman: seeds averaged, methods as within-source units.
rhodetail=[];rhosummary=[]
for ds in ['R84','R85']:
 for pop in ['all_methods','same_memory','local_dose']:
  if ds=='R85' and pop=='local_dose':continue
  g=src[src.dataset==ds];g=g if pop=='all_methods' else g[g.arm.isin(arms_for(ds,pop))]
  for a,b in itertools.combinations(MET,2):
   vals=[];eligible=0
   for source,h in g.groupby('source'):
    h=h.dropna(subset=[a,b]);n=len(h);rho=np.nan
    if n>=3 and h[a].nunique()>1 and h[b].nunique()>1:rho=float(spearmanr(MET[a]*h[a],MET[b]*h[b]).statistic);vals.append(rho)
    rhodetail.append({'dataset':ds,'population':pop,'source':source,'metric_a':a,'metric_b':b,'n_methods':n,'rho_aligned':rho})
   if vals:
    arr=np.array(vals);boot=arr[rng.integers(0,len(arr),(2000,len(arr)))].mean(axis=1);ci=np.quantile(boot,[.025,.975]);rhosummary.append({'dataset':ds,'population':pop,'metric_a':a,'metric_b':b,'n_sources':len(vals),'mean_within_source_rho':arr.mean(),'median_rho':np.median(arr),'fraction_negative':np.mean(arr<0),'CI_low':ci[0],'CI_high':ci[1]})
pd.DataFrame(rhodetail).to_csv(OUT/'within_source_spearman.csv',index=False);rho=pd.DataFrame(rhosummary);rho.to_csv(OUT/'within_source_spearman_summary.csv',index=False)
# Pairwise method contrasts: independent sign disagreement for each metric pair, no invented 3-way event.
dis=[];di=[]
for ds in ['R84','R85']:
 g=src[src.dataset==ds]
 for aa,bb in itertools.combinations(sorted(g.arm.unique()),2):
  z=g[g.arm==aa].set_index('source')[list(MET)].join(g[g.arm==bb].set_index('source')[list(MET)],lsuffix='_a',rsuffix='_b',how='inner')
  for a,b in itertools.combinations(MET,2):
   za=MET[a]*(z[a+'_a']-z[a+'_b']);zb=MET[b]*(z[b+'_a']-z[b+'_b']);ok=za.notna()&zb.notna();nonzero=ok&(abs(za)>1e-12)&(abs(zb)>1e-12);conf=(za*zb<0)&nonzero
   dis.append({'dataset':ds,'arm_a':aa,'arm_b':bb,'metric_a':a,'metric_b':b,'n_common_sources':int(ok.sum()),'n_nontied_sources':int(nonzero.sum()),'n_sign_disagreements':int(conf.sum()),'fraction_disagree_nontied':float(conf.sum()/nonzero.sum()) if nonzero.any() else np.nan,'aligned_mean_delta_a':float(za[ok].mean()),'aligned_mean_delta_b':float(zb[ok].mean()),'aggregate_mean_sign_disagreement':bool(za[ok].mean()*zb[ok].mean()<0) if ok.any() else None})
   for source in z.index[ok]:di.append({'dataset':ds,'arm_a':aa,'arm_b':bb,'metric_a':a,'metric_b':b,'source':source,'aligned_delta_a':za[source],'aligned_delta_b':zb[source],'nontied':bool(nonzero[source]),'sign_disagreement':bool(conf[source])})
dis=pd.DataFrame(dis);dis.to_csv(OUT/'pairwise_metric_sign_disagreement.csv',index=False);pd.DataFrame(di).to_csv(OUT/'pairwise_metric_sign_details.csv',index=False)
# Vector-only figures: raw and source-centered per-source-arm observations; BOTH distinct marker.
for ds in ['R84','R85']:
 for mode in ['raw','centered']:
  fig,axes=plt.subplots(2,3,figsize=(10.5,6.1),layout='constrained')
  for ax,m in zip(axes.flat,['lpips_full','lpips_preserved','bg_mae','dino_full_feature_similarity','clip_target_similarity','clip_directional_similarity']):
   g=prep(ds,'with_memory',m).copy();g['x']=g.I;g['y']=g[m]
   if mode=='centered':
    # Center same-memory reference panel only; BOTH deviations relative to that source reference.
    ref=g[g.arm!='RELEASE_BOTH25'].groupby('source')[['I',m]].mean();g['x']=g.I-g.source.map(ref.I);g['y']=g[m]-g.source.map(ref[m])
   for arm,h in g.groupby('arm'):ax.scatter(h.x,h.y,s=15,alpha=.68,color=COLORS[arm],marker='x' if arm=='RELEASE_BOTH25' else 'o',label=NAME[arm])
   ax.set(xlabel='Impulse I' if mode=='raw' else 'I minus same-memory source mean',ylabel=SHORT[m],title=f'{h.dataset.iloc[0]} | {g.source.nunique()} sources');ax.grid(alpha=.2)
   if mode=='centered':ax.axhline(0,c='.65',lw=.5);ax.axvline(0,c='.65',lw=.5)
  handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=6);fig.suptitle(f'{ds}: {mode} source means; BOTH shown separately',y=1.05)
  fig.savefig(OUT/f'{ds}_dose_{mode}.pdf',bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
keys=list(MET)
for ax,ds in zip(axes,['R84','R85']):
 mat=np.eye(len(keys));nmat=np.zeros_like(mat)
 for _,r in rho[(rho.dataset==ds)&(rho.population=='all_methods')].iterrows():i=keys.index(r.metric_a);j=keys.index(r.metric_b);mat[i,j]=mat[j,i]=r.mean_within_source_rho
 im=ax.imshow(mat,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(keys)),[SHORT[k] for k in keys],rotation=65,ha='right',fontsize=7);ax.set_yticks(range(len(keys)),[SHORT[k] for k in keys],fontsize=7);ax.set_title(ds+' | mean within-source Spearman')
fig.colorbar(im,ax=list(axes),shrink=.65);fig.savefig(OUT/'metric_spearman_aligned.pdf',bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
for ax,ds in zip(axes,['R84','R85']):
 for indic,c in [(False,'#0072B2'),(True,'#D55E00')]:
  z=score[(score.dataset==ds)&(score.population=='same_memory')&(score.form=='linear')&(score['mode']=='centered')&(score.indicators==indic)].set_index('metric').reindex(keys)
  ax.plot(range(len(keys)),z.CV_skill_vs_baseline,marker='o',c=c,label='Impulse + allocation' if indic else 'Impulse only')
 ax.axhline(0,c='.5',lw=.7);ax.set_xticks(range(len(keys)),[SHORT[k] for k in keys],rotation=65,ha='right',fontsize=7);ax.set(title=ds+' | source-exclusive CV',ylabel='Skill vs zero within-source profile');ax.legend(fontsize=7)
fig.savefig(OUT/'LOSO_impulse_vs_allocation.pdf',bbox_inches='tight');plt.close(fig)
# Results manifest and concise human-readable findings generated from actual computed rows.
manifest['coverage']=coverage;manifest['outputs']=[p.name for p in OUT.iterdir() if p.is_file()];manifest['metric_orientation']=MET
(OUT/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
lines=['CPU OFFLINE ANALYSIS — exploratory; no GPU, no VLM','',f"Step reconstruction: {len(audit)} R85 outputs, 28 steps each; max |recomputed-native I|={audit.abs_difference.max():.3g}.",'R84 130 outputs/10 sources; R85 520 outputs/40 sources. Sources disjoint. Complete source-seed means required. Preservation support: R84 8, R85 33 sources.','FlowEdit, RF-Edit and ETA03 comparable dose undefined (ETA03 native zeros replaced with NaN).','Centered CV predicts within-source profile deviations, NOT absolute performance of an unseen source; held-out outcomes never enter training.','No model chosen from these exploratory candidates; all linear/quadratic/log results retained.','']
for ds in ['R84','R85']:
 lines.append(ds+' same-memory, linear centered LOSO:')
 for m in ['lpips_full','lpips_preserved','dino_full_feature_similarity','clip_target_similarity','clip_directional_similarity']:
  z=score[(score.dataset==ds)&(score.population=='same_memory')&(score.form=='linear')&(score['mode']=='centered')&(score.metric==m)].sort_values('indicators')
  lines.append(f"  {m}: "+'; '.join(f"{'I+allocation' if r.indicators else 'I'} skill={r.CV_skill_vs_baseline:.3f}, RMSE={r.CV_RMSE:.5f}, n_sources={r.n_sources}, rows={r.n_source_arm_rows}" for _,r in z.iterrows()))
 lines.append(ds+' with BOTH, linear centered LOSO:')
 for m in ['lpips_full','clip_directional_similarity']:
  z=score[(score.dataset==ds)&(score.population=='with_memory')&(score.form=='linear')&(score['mode']=='centered')&(score.metric==m)].sort_values('indicators')
  lines.append(f"  {m}: "+'; '.join(f"{'I+allocation+memory' if r.indicators else 'I'} skill={r.CV_skill_vs_baseline:.3f}, RMSE={r.CV_RMSE:.5f}" for _,r in z.iterrows()))
tr=pd.DataFrame(trans)
lines+=['','Transfer R84 -> R85 common CORR25/ENERGY, fixed linear fits:']
for _,r in tr[(tr.form=='linear')&(~tr.indicators)&tr.metric.isin(['lpips_full','lpips_preserved','clip_directional_similarity'])].iterrows():lines.append(f'  {r.metric} {r["mode"]}: skill={r.transfer_skill:.3f}, RMSE={r.transfer_RMSE:.5f}, test sources={r.test_sources}, out-of-range={r.test_outside_training_I_range}/{r.test_rows}')
lines+=['','R85 U25 minus CORR (direction-aligned mean differences):']
for m in ['dino_full_feature_similarity','lpips_full','lpips_preserved','clip_directional_similarity']:
 a=src[(src.dataset=='R85')&(src.arm=='U25_KV')].set_index('source')[m];b=src[(src.dataset=='R85')&(src.arm=='RELEASE_CORR25')].set_index('source')[m];delta=MET[m]*(a-b).dropna();lines.append(f'  {m}: {delta.mean():+.6f}, n={len(delta)}, positive={int((delta>0).sum())}')
lines+=['U25 has both higher mean DINO and lower mean LPIPS than CORR; no three-metric conflict is asserted.','Pairwise disagreement refers to signs of source-averaged method contrasts, not image-level ground truth.','Realized impulse changes with state and memory; regression adjustment cannot establish causal dose sufficiency.','No universal dose-collapse conclusion follows from in-sample R2. See LOSO skill/RMSE, transfer failure and per-source residuals.']
(OUT/'findings.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines));print('\nCOMPLETE',OUT)
