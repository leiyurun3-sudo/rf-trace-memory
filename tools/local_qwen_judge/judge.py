import argparse, csv, hashlib, json, random, statistics, sys
from pathlib import Path

PROMPT = '''You are evaluating an image edit. Image 1 is the source, image 2 is the edited result. Judge only visible content. Source description: {source_prompt}\nRequested target: {target_prompt}\nSpecific edit: {rubric}\nReturn ONLY a JSON object with exactly four keys edit, subject, pattern, artifact. Each value is an object with keys value, applicable, reason. value is integer 0 or 1, or null if uncertain/not applicable; For edit and artifact applicable MUST be true; use value null with a reason when uncertain. For subject and pattern use exactly the frozen subject_applicable and pattern_applicable values. reason is a short concrete visual explanation. edit: 1 if requested change is visibly achieved, 0 if not. subject: 1 if identity/shape and important unrelated content are preserved as appropriate to the request, 0 for substantial unrelated damage; when replacement, removal, posture or shape change is requested, do not penalize that intended change. pattern: assess preservation of meaningful distinctive source patterns/textures outside requested changes; use the frozen source-only rubric applicability exactly; when false or null, value must be null. Do not change applicability based on the edited image. Do not invent patterns. artifact: 1 for severe visual artifacts (broken anatomy, major incoherence, etc.), 0 if absent. If unclear, use null rather than guessing. Global style changes may legitimately alter textures, but preserve relevant scene content. Do not assess by pixel similarity.''' 
VERSION='local-qwen-judge-v4'
RUBRIC_PROMPT='''Inspect ONLY the source image and task descriptions, before seeing any outputs. Source description: {source_prompt}\nRequested target: {target_prompt}\nSpecific edit: {rubric}\nReturn ONLY JSON with exactly: requested_edit (nonempty array of strings), protected_requirements (nonempty array of strings), subject_applicable (true, false, or null if uncertain), subject_reason (string), pattern_applicable (true, false, or null if uncertain), pattern_reason (string). Include task-appropriate preservation requirements; never require preserving a requested identity/shape change. subject_applicable is false if the request replaces, removes, or changes the count of the subject such that preserving the original subject identity/shape is not appropriate. Do not require source subject identity for such tasks; use true if meaningful source identity/shape should be preserved, or null if uncertain. Independently consider protected unrelated content under artifact and edit. pattern_applicable is true only when distinctive source patterns or textures should be preserved outside the requested edit; false when none is relevant or the edit legitimately replaces them. If unclear use null. This rubric is frozen across all outputs.'''

def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
def digest(x): return hashlib.sha256(x).hexdigest()
def filehash(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def safe(root,part):
 p=Path(part)
 if p.is_absolute() or '..' in p.parts or '\\' in str(part) or ':' in str(part): raise ValueError('unsafe relative path')
 q=(root/p).resolve(); q.relative_to(root.resolve()); return q
def dump(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def result_dir(root):
 for p in [root/'results',root/'results'/'R85',root]:
  if (p/'plan.json').is_file(): return p
 raise ValueError('R85 plan.json not found in root/results, root/results/R85 or root')
def prepare(root,out,revision,model_hash):
 root=root.resolve(); plan=load(Path(__file__).with_name('r85_plan.json')); tasks={t['case']:t for t in plan['tasks']}; jobs=[]; missing=[]
 # Raw outputs normally live under results/<job>/raw.png; also accept results/R85/<job>/raw.png.
 choices=[root/'results',root/'results'/'R85',root]
 rd=next((p for p in choices if any((p/j['id']/'raw.png').is_file() for j in plan['jobs'])),None)
 if rd is None: rd=next((p for p in choices if (p/'plan.json').is_file()),root/'results')
 protocol={'version':VERSION,'rubric_prompt':RUBRIC_PROMPT,'prompt':PROMPT,'model':'Qwen/Qwen2.5-VL-7B-Instruct','revision':revision,'model_fingerprint':model_hash,'max_new_tokens':768,'do_sample':False,'max_pixels':512*512,'schema':'four ratings applicability reasons v1'}
 ph=digest(canonical(protocol).encode()); dump(out/'protocol.json',protocol)
 for job in plan['jobs']:
  t=tasks[job['case']]; jid=job['id']
  if not isinstance(jid,str) or not jid or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in jid): raise ValueError('invalid job id')
  raw=rd/jid/'raw.png'; src=safe(root,t['source_image']); paths=[src,raw]
  try:
   for p in paths:
    p.resolve().relative_to(root)
    if not p.is_file() or p.suffix.lower()!='.png': raise FileNotFoundError(p.name)
   info={'source_sha256':filehash(src),'output_sha256':filehash(raw)}
   if info['source_sha256']!=plan['input_sha256'][t['source_image']]: raise ValueError('source image differs from R85 plan: '+t['case'])
  except (ValueError,FileNotFoundError,OSError) as e: missing.append({'id':jid,'error':str(e)}); info=None
  public={'source_prompt':t['source_prompt'],'target_prompt':t['target_prompt'],'rubric':t['edit_rubric']}
  key=digest(canonical({'protocol':ph,'images':info,'task':public}).encode()) if info else None
  jobs.append({'job_id':jid,'blind_id':digest((jid+ph).encode())[:24], 'case':t['case'],'category':str(t['target_category_id']),'arm':job['arm'],'seed':job['seed'],'source':src.relative_to(root).as_posix(),'output':raw.relative_to(root).as_posix(),'content':info,'fingerprint':key,'public':public})
 controls=[]
 for c in sorted(tasks):
  j=next(x for x in jobs if x['case']==c)
  control=dict(j); control.update({'job_id':'control_'+c,'arm':'SOURCE_CONTROL','seed':0,'output':j['source'],'content':({'source_sha256':j['content']['source_sha256'],'output_sha256':j['content']['source_sha256']} if j['content'] else None)})
  control['fingerprint']=digest(canonical({'protocol':ph,'images':control['content'],'task':control['public'],'control':True}).encode()) if control['content'] else None
  control['blind_id']=digest(('control_'+c+ph).encode())[:24];controls.append(control)
 random.Random(85005).shuffle(jobs); dump(out/'controls_private.json',controls); dump(out/'jobs_private.json',jobs); dump(out/'missing_inputs.json',missing)
 print('Prepared',len(jobs),'jobs; missing native inputs:',len(missing)); return jobs,ph

def valid(d):
 if not isinstance(d,dict) or set(d)!={'edit','subject','pattern','artifact'}: raise ValueError('expected exactly four ratings')
 for k,v in d.items():
  if not isinstance(v,dict) or set(v)!={'value','applicable','reason'}: raise ValueError('invalid '+k)
  a=v['applicable']; val=v['value']
  if a is not None and type(a)!=bool: raise ValueError('invalid applicability')
  if val is not None and (type(val)!=int or val not in (0,1)): raise ValueError('invalid value')
  if a is not True and val is not None: raise ValueError('rating with non-applicable/uncertain applicability')
  if k not in ('subject','pattern') and a is not True: raise ValueError('only frozen subject and pattern can be non-applicable; use null value when uncertain')
  if not isinstance(v['reason'],str) or not v['reason'].strip(): raise ValueError('missing reason')
 return d

def preflight(model):
 import torch,transformers
 from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
 from PIL import Image
 import PIL, accelerate
 if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
 if not (model/'config.json').is_file() or not (model/'judge_model_revision.json').is_file(): raise RuntimeError('download model first using download_model.py')
 conf=load(model/'config.json')
 if conf.get('model_type')!='qwen2_5_vl': raise RuntimeError('not Qwen2.5-VL')
 files=list(model.glob('*.safetensors'))
 if not files: raise RuntimeError('missing model weights')
 index=model/'model.safetensors.index.json'
 if index.is_file():
  absent=set(load(index)['weight_map'].values())-{p.name for p in files}
  if absent: raise RuntimeError('missing model shards: '+str(sorted(absent)))
 meta=load(model/'judge_model_revision.json');
 if meta.get('model')!='Qwen/Qwen2.5-VL-7B-Instruct' or len(meta.get('revision',''))!=40: raise RuntimeError('invalid model identity or revision metadata')
 mem=torch.cuda.mem_get_info(); print('GPU:',torch.cuda.get_device_name(0),'free GiB:',round(mem[0]/2**30,2),'torch:',torch.__version__,'transformers:',transformers.__version__,'revision:',meta['revision']); print('Memory is informational; actual model loading determines fit.')
 # Hash all local model files: changed weights/config/tokenizer cannot reuse old decisions.
 inventory=[(p.name,filehash(p)) for p in sorted(model.iterdir()) if p.is_file() and p.name!='judge_model_revision.json']; return meta['revision'],digest(canonical({'files':inventory,'torch':torch.__version__,'transformers':transformers.__version__,'pillow':PIL.__version__,'accelerate':accelerate.__version__}).encode())

def load_model(model):
 import torch
 from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
 print('Loading local model and processor as a GPU smoke check...',flush=True)
 modelobj=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model),torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,device_map='auto',local_files_only=True)
 processor=AutoProcessor.from_pretrained(str(model),local_files_only=True,max_pixels=512*512)
 if not any(p.device.type=='cuda' for p in modelobj.parameters()): raise RuntimeError('model did not load on GPU')
 print('Model load smoke check passed; GPU allocated GiB:',round(torch.cuda.memory_allocated()/2**30,2),flush=True)
 return modelobj,processor

def run(root,out,model,preflight_only,repeat_count=None):
 revision,mh=preflight(model)
 if preflight_only: return
 import torch
 from PIL import Image
 from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
 jobs,ph=prepare(root,out,revision,mh)
 if any(not j['fingerprint'] for j in jobs): raise RuntimeError('Missing native inputs; see missing_inputs.json. No model loaded.')
 modelobj,processor=load_model(model)
 # Stage 1: source-only rubric, frozen for all arms and controls.
 rubrics={}
 for case in sorted({j['case'] for j in jobs}):
  j=next(x for x in jobs if x['case']==case)
  rk=digest(canonical({'protocol':ph,'source':j['content']['source_sha256'],'task':j['public']}).encode())
  rp=out/'rubrics'/f'{rk}.json'; rec=load(rp) if rp.exists() else {}
  if rec.get('status')!='ok':
   rec={'fingerprint':rk,'model_revision':revision,'prompt':RUBRIC_PROMPT.format(**j['public']),'raw_response':None,'status':'failed'}
   try:
    if filehash(safe(root,j['source']))!=j['content']['source_sha256']: raise ValueError('source changed after preparation')
    with Image.open(safe(root,j['source'])) as im: source=im.convert('RGB')
    messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':rec['prompt']}]}]
    template=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    inp=processor(text=[template],images=[source],padding=True,return_tensors='pt').to(modelobj.device)
    with torch.inference_mode(): gen=modelobj.generate(**inp,max_new_tokens=768,do_sample=False)
    rec['raw_response']=processor.batch_decode(gen[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0]
    raw=rec['raw_response'].strip()
    if raw.startswith('```'): raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
    rubric=json.loads(raw)
    if set(rubric)!={'requested_edit','protected_requirements','subject_applicable','subject_reason','pattern_applicable','pattern_reason'}: raise ValueError('rubric keys')
    for k in ('requested_edit','protected_requirements'):
     if not isinstance(rubric[k],list) or not rubric[k] or any(not isinstance(v,str) or not v.strip() for v in rubric[k]): raise ValueError('rubric '+k)
    for k in ('subject','pattern'):
     if rubric[k+'_applicable'] is not None and type(rubric[k+'_applicable']) is not bool: raise ValueError('rubric '+k+' applicability')
     if not isinstance(rubric[k+'_reason'],str) or not rubric[k+'_reason'].strip(): raise ValueError('rubric '+k+' reason')
    rec['rubric']=rubric;rec['status']='ok'
   except Exception as e: rec['error']=str(e);print('RUBRIC FAILED',case,str(e),flush=True)
   dump(rp,rec)
  rubrics[case]=rec
 print('Source-only rubric phase complete; starting output phase.',flush=True)
 if repeat_count is None:
  previous=out/'repeats_private.json'; previous_ids={x['job_id'] for x in load(previous)} if previous.exists() else set()
  selected=[j for j in jobs if j['job_id'] in previous_ids]
 else:
  if repeat_count<0 or repeat_count>len(jobs): raise ValueError('--repeat-count must be between 0 and number of main jobs')
  selected=random.Random(85007).sample(jobs,repeat_count)
 # One extra invocation for each fixed sampled job; main and controls stay separate.
 repeats=[dict(j,audit_repeat=True) for j in selected]
 dump(out/'repeats_private.json',[{'job_id':j['job_id'],'blind_id':j['blind_id']} for j in repeats])
 alljobs=jobs+load(out/'controls_private.json')+repeats;random.Random(85006).shuffle(alljobs)
 for n,j in enumerate(alljobs,1):
  frozen=rubrics[j['case']]
  if frozen.get('status')!='ok': print('UNRESOLVED rubric',j['blind_id'],flush=True);continue
  # Response cache incorporates the exact frozen rubric content.
  j['fingerprint']=digest(canonical({'job':j['fingerprint'],'rubric':frozen['rubric']}).encode())
  dest=out/('repeat_responses' if j.get('audit_repeat') else 'responses')/f"{j['fingerprint']}.json"
  if dest.exists():
   old=load(dest)
   if old.get('status')=='ok' and old.get('fingerprint')==j['fingerprint']: continue
  rec={'blind_id':j['blind_id'],'fingerprint':j['fingerprint'],'protocol_fingerprint':ph,'model_revision':revision,'prompt':PROMPT.format(**j['public'])+'\nFrozen source-only rubric (must follow subject_applicable and pattern_applicable exactly): '+canonical(frozen['rubric']),'raw_response':None,'status':'failed','ratings':None,'audit_repeat':bool(j.get('audit_repeat'))}
  try:
   sp=safe(root,j['source']); op=safe(root,j['output'])
   if filehash(sp)!=j['content']['source_sha256'] or filehash(op)!=j['content']['output_sha256']: raise ValueError('input changed after prepare')
   with Image.open(sp) as im: source=im.convert('RGB')
   with Image.open(op) as im: output=im.convert('RGB')
   messages=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':rec['prompt']}]}]
   template=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
   inputs=processor(text=[template],images=[source,output],padding=True,return_tensors='pt').to(modelobj.device)
   with torch.inference_mode(): generated=modelobj.generate(**inputs,max_new_tokens=768,do_sample=False)
   rec['raw_response']=processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
   raw=rec['raw_response'].strip()
   if raw.startswith('```'):
    raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
   rec['ratings']=valid(json.loads(raw))
   for k in ('subject','pattern'):
    if rec['ratings'][k]['applicable'] is not frozen['rubric'][k+'_applicable']: raise ValueError(k+' applicability contradicts frozen source rubric')
   rec['status']='ok'
  except Exception as e: rec['error']=str(e); print('FAILED',j['blind_id'],str(e),flush=True)
  dump(dest,rec); print(n,'/',len(alljobs),j['blind_id'],rec['status'],flush=True)

def ci(vals,rng,n=2000):
 if len(vals)<2: return None
 vals=sorted(sum(rng.choice(vals) for _ in vals)/len(vals) for _ in range(n)); return [vals[int(.025*(n-1))],vals[int(.975*(n-1))]]
def summarize(out):
 jobs=load(out/'jobs_private.json'); controls=load(out/'controls_private.json'); rows=[]
 for j in jobs+controls:
  protocol=load(out/'protocol.json'); ph=digest(canonical(protocol).encode())
  rk=digest(canonical({'protocol':ph,'source':j['content']['source_sha256'],'task':j['public']}).encode()) if j['content'] else None
  rp=out/'rubrics'/str(rk);rp=rp.with_suffix('.json'); rub=load(rp) if rp.is_file() else {}
  if rub.get('status')=='ok' and j['fingerprint']: j['fingerprint']=digest(canonical({'job':j['fingerprint'],'rubric':rub['rubric']}).encode())
  else: j['fingerprint']=None
  p=out/'responses'/str(j['fingerprint'] or 'missing'); p=p.with_suffix('.json'); r=load(p) if p.is_file() else {}; v=r.get('ratings') if r.get('status')=='ok' and r.get('fingerprint')==j['fingerprint'] else None
  vals={k:(v[k]['value'] if v and v[k]['applicable'] is True else None) for k in ('edit','subject','pattern','artifact')}
  applicability={k:(rub['rubric'][k+'_applicable'] if rub.get('status')=='ok' else None) for k in ('subject','pattern')}
  nas={k:applicability[k] is False for k in ('subject','pattern')}
  joint=None
  if v and all(vals[k] is not None for k in ('edit','artifact')) and all(nas[k] or (applicability[k] is True and vals[k] is not None) for k in ('subject','pattern')):
   joint=int(vals['edit']==1 and vals['artifact']==0 and all(nas[k] or vals[k]==1 for k in ('subject','pattern')))
  details={}
  for k in ('edit','subject','pattern','artifact'):
   details[k+'_reason']=v[k]['reason'] if v else ''
   details[k+'_status']=('not_applicable' if nas.get(k,False) else ('rated' if vals[k] is not None else ('uncertain' if v else 'unresolved')))
  rows.append({'case':j['case'],'category':j['category'],'arm':j['arm'],'seed':j['seed'],'status':r.get('status','missing'),'subject_na':nas['subject'],'pattern_na':nas['pattern'],'joint':joint,**vals,**details})
 with (out/'ratings.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 rng=random.Random(85005); arms=sorted({r['arm'] for r in rows if r['arm']!='SOURCE_CONTROL'}); reports={}
 for arm in arms:
  rr=[r for r in rows if r['arm']==arm]; summary={}
  for k in ('edit','subject','pattern','artifact','joint'):
   per={c:[r[k] for r in rr if r['case']==c and r[k] is not None] for c in sorted({r['case'] for r in rr})}; per={c:statistics.mean(x) for c,x in per.items() if x}; vals=list(per.values())
   eligible=[r for r in rr if k not in ('subject','pattern') or not r[k+'_na']]; rated=sum(r[k] is not None for r in eligible); unresolved=len(eligible)-rated; successes=sum(r[k]==1 for r in eligible)
   summary[k]={'mean_source_seed_average':statistics.mean(vals) if vals else None,'source_count':len(vals),'rated_outputs':rated,'total_outputs':len(rr),'eligible_outputs':len(eligible),'na_outputs':len(rr)-len(eligible),'unresolved_outputs':unresolved,'output_rate_lower':successes/len(eligible) if eligible else None,'output_rate_upper':(successes+unresolved)/len(eligible) if eligible else None,'source_bootstrap_95ci':ci(vals,rng)}
  summary['pattern_na_outputs']=sum(r['pattern_na'] for r in rr);summary['subject_na_outputs']=sum(r['subject_na'] for r in rr); reports[arm]=summary
 base='RELEASE_CORR25'; comparisons={}
 def source_values(arm):
  rr=[r for r in rows if r['arm']==arm and r['joint'] is not None]; cases=sorted({r['case'] for r in rr});return {c:statistics.mean(r['joint'] for r in rr if r['case']==c) for c in cases}
 for arm in arms:
  if arm==base:continue
  a=source_values(base);b=source_values(arm); common=sorted(a.keys()&b.keys()); diff=[a[c]-b[c] for c in common]; comparisons[arm]={'estimand':'CORR25 minus comparator, each available seed averaged within source; complete-case paired sources','paired_sources':len(diff),'mean_difference':statistics.mean(diff) if diff else None,'source_bootstrap_95ci':ci(diff,rng),'note':'RFEDIT seed 0 only when applicable; missing seeds remain excluded, not failures.'}
 audit=[]; repeat_manifest=out/'repeats_private.json'; selected={x['job_id'] for x in load(repeat_manifest)} if repeat_manifest.exists() else set()
 for j in jobs:
  if j['job_id'] not in selected: continue
  # Fingerprints on jobs were resolved during row preparation.
  fp=j['fingerprint']; main=out/'responses'/f'{fp}.json'; rep=out/'repeat_responses'/f'{fp}.json'
  a=load(main) if fp and main.exists() else {};b=load(rep) if fp and rep.exists() else {}
  av=a.get('ratings') if a.get('status')=='ok' else None;bv=b.get('ratings') if b.get('status')=='ok' else None
  for k in ('edit','subject','pattern','artifact'):
   x=av[k]['value'] if av else None;y=bv[k]['value'] if bv else None
   audit.append({'blind_id':j['blind_id'],'criterion':k,'main_value':x,'repeat_value':y,'comparable':x is not None and y is not None,'agree':(x==y) if x is not None and y is not None else None,'main_reason':av[k]['reason'] if av else '', 'repeat_reason':bv[k]['reason'] if bv else ''})
 with (out/'repeat_audit.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['blind_id','criterion','main_value','repeat_value','comparable','agree','main_reason','repeat_reason']);w.writeheader();w.writerows(audit)
 repeat_summary={}
 for k in ('edit','subject','pattern','artifact'):
  rr=[x for x in audit if x['criterion']==k];com=[x for x in rr if x['comparable']]; repeat_summary[k]={'selected':len(rr),'comparable':len(com),'unresolved_or_na':len(rr)-len(com),'agreement':sum(x['agree'] for x in com)/len(com) if com else None}
 with (out/'arm_summary.csv').open('w',encoding='utf-8',newline='') as f:
  fields=['arm','criterion','mean_source_seed_average','source_count','total_outputs','eligible_outputs','na_outputs','rated_outputs','unresolved_outputs','output_rate_lower','output_rate_upper','ci_lower','ci_upper']
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for arm,metrics in reports.items():
   for criterion in ('edit','subject','pattern','artifact','joint'):
    d=dict(metrics[criterion]);bounds=d.pop('source_bootstrap_95ci');w.writerow({'arm':arm,'criterion':criterion,**d,'ci_lower':bounds[0] if bounds else None,'ci_upper':bounds[1] if bounds else None})
 with (out/'paired_summary.csv').open('w',encoding='utf-8',newline='') as f:
  fields=['reference','comparator','paired_sources','mean_difference','ci_lower','ci_upper','estimand','note'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for arm,d in comparisons.items():
   bounds=d['source_bootstrap_95ci'];w.writerow({'reference':base,'comparator':arm,'paired_sources':d['paired_sources'],'mean_difference':d['mean_difference'],'ci_lower':bounds[0] if bounds else None,'ci_upper':bounds[1] if bounds else None,'estimand':d['estimand'],'note':d['note']})
 dump(out/'summary.json',{'repeat_audit':{'selected_jobs':len(selected),'criteria':repeat_summary,'note':'Separate deterministic greedy reruns of fixed selected jobs; reproducibility audit, not an independent rater or guarantee of correctness.'},'label':'VLM-labelled, not human; missing excluded; descriptive source-clustered bootstrap','applicability_denominator':'subject and pattern explicit NA excluded from their denominators; joint ignores explicit NA for both; unknown remains missing; if all NA eligible denominator is zero and rate is null','arms':reports,'paired_joint':comparisons,'source_as_output_calibration':{'total':sum(r['arm']=='SOURCE_CONTROL' for r in rows),'edit_pass':sum(r['arm']=='SOURCE_CONTROL' and r['edit']==1 for r in rows),'edit_fail':sum(r['arm']=='SOURCE_CONTROL' and r['edit']==0 for r in rows),'edit_unresolved':sum(r['arm']=='SOURCE_CONTROL' and r['edit'] is None for r in rows),'interpretation':'Inspect each requested edit; source-as-output is a control, not automatically a failure.'}});print('Wrote ratings.csv and summary.json')

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
 for name in ('preflight','run'):
  q=sub.add_parser(name);q.add_argument('--model',type=Path,required=True);q.add_argument('--root',type=Path);q.add_argument('--out',type=Path);q.add_argument('--repeat-count',type=int,default=None)
 q=sub.add_parser('summarize');q.add_argument('--out',type=Path,required=True)
 a=p.parse_args()
 if a.cmd=='summarize':summarize(a.out)
 elif a.cmd=='preflight':
  preflight(a.model.resolve())
  if a.load_model: load_model(a.model.resolve())
 else:
  if not a.root or not a.out:p.error('run needs --root and --out')
  run(a.root.resolve(),a.out.resolve(),a.model.resolve(),False,a.repeat_count)
if __name__=='__main__':main()
