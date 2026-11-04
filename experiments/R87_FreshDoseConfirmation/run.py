"""R87 fresh-source execution: prepare, generate, evaluate, and compact package."""
import argparse,csv,hashlib,importlib.metadata,importlib.util,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime_base'
DEFAULT_PROJECT=Path('/root/autodl-tmp/MINT_Flow_Jupyter_Ready')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def paths():
 bases=[Path(os.environ.get('R87_SERVER_PROJECT',str(DEFAULT_PROJECT))),DEFAULT_PROJECT]
 for base in bases:
  for n in ('R85_paths.env','paths.env','R84_paths.env'):
   f=base/'evaluation_models'/n
   if not f.is_file():continue
   for line in f.read_text(encoding='utf-8').splitlines():
    tokens=shlex.split(line,comments=True)
    if tokens and tokens[0]=='export':tokens=tokens[1:]
    for token in tokens:
     key,sep,value=token.partition('=')
     if sep and key in ('CLIP_PATH','DINO_PATH','LPIPS_PATH','LPIPS_TRUNK_PATH') and not os.environ.get(key) and Path(value).exists():os.environ[key]=value
 spec=importlib.util.find_spec('lpips');cal=[];trunk=[]
 for base in bases:
  cal.append(base/'evaluation_models/lpips/alex_v0.1.pth');trunk.append(base/'evaluation_models/lpips/alexnet-owt-7be5be79.pth')
 if spec and spec.origin:cal.append(Path(spec.origin).parent/'weights/v0.1/alex.pth')
 trunk.append(Path(os.environ.get('TORCH_HOME',Path.home()/'.cache/torch'))/'hub/checkpoints/alexnet-owt-7be5be79.pth')
 for key,candidates in [('LPIPS_PATH',cal),('LPIPS_TRUNK_PATH',trunk)]:
  if os.environ.get(key) and Path(os.environ[key]).is_file():continue
  found=next((str(p.resolve()) for p in candidates if p.is_file()),None)
  if found:os.environ[key]=found
def preflight_evaluation():
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_DATASETS_OFFLINE='1')
 if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
 import eval_pipeline
 paths();ready=eval_pipeline.readiness(Path(os.environ.get('R87_SERVER_PROJECT',str(DEFAULT_PROJECT))));os.environ.update(ready['paths'])
 for package in ('lpips','torchvision','scikit-image'):importlib.metadata.version(package)
 for name in ('LPIPS_PATH','LPIPS_TRUNK_PATH'):
  if not os.environ.get(name) or not Path(os.environ[name]).is_file():raise RuntimeError('Offline evaluator weight missing: '+name)
 return ready

def preflight_generation(plan):
 ready=preflight_evaluation()
 import model_loader
 model_loader.preflight_runtime(plan)
 # Imports before GPU queue; no model loading or network.
 import torch,numpy,PIL,diffusers,transformers,accelerate,safetensors,lpips,torchvision,skimage
 import engine,controls,rf_kv,attention,mint_backend,spatial_controls
 return ready

def collect_delivery(plan,manifest_path):
 results=RUNTIME/'results'; profiles={}
 for job in plan['jobs']:
  if job['arm']!='REQ':continue
  folder=results/job['id'];record=read(folder/'record.json');stepsfile=folder/'steps.json'
  if record.get('status')!='completed' or sha(stepsfile)!=record.get('artifact_sha256',{}).get('steps.json'):raise RuntimeError('Anchor profile absent or mismatched: '+job['id'])
  steps=read(stepsfile)['edit']['steps']
  if len(steps)!=28:raise RuntimeError('Anchor profile needs 28 steps: '+job['id'])
  profiles[job['id']]={'case':job['case'],'seed':job['seed'],'steps_sha256':sha(stepsfile),'sigma_sha256':record['sigma_sha256'],'nfe_standalone':record['nfe_standalone'],'steps':[{k:s[k] for k in ('step_index','sigma','h','applied_correction_rms')} for s in steps]}
 if len(profiles)!=40:raise RuntimeError('Expected 40 committed REQ profiles')
 write(results/'anchor_profiles.json',profiles)
 shutil.copy2(manifest_path,results/'source_selection_manifest.json')
 for path in ROOT.glob('protocol*'):
  if path.is_file() and path.suffix.lower() in ('.json','.md','.txt','.tex'):
   shutil.copy2(path,results/path.name)
 # Analysis authors can write small outputs under results/evidence or results/analysis.
 for name in ('analysis','evidence'):
  source=ROOT/name
  if source.is_dir():
   for path in source.rglob('*'):
    if path.is_file() and path.suffix.lower() in ('.json','.csv','.txt','.tex'):
     target=results/name/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)

def materialize(manifest_path,data_dir):
 manifest=read(manifest_path);tasks=manifest.get('tasks');template=read(RUNTIME/'r85_template.json')
 if not isinstance(tasks,list) or len(tasks)!=20:raise ValueError('manifest.json must contain exactly 20 tasks')
 seen=set();normalized=[];inputs={}
 for raw in tasks:
  t=dict(raw);case=t.get('case')
  if not isinstance(case,str) or not case or any(c in case for c in '/\\:') or case in seen:raise ValueError('Invalid or duplicate task case: '+str(case))
  seen.add(case);t.setdefault('key','pie_'+case);t.setdefault('task',t['key']);t['seeds']=[0,17]
  for k in ('key','source_prompt','target_prompt','pattern_roi','detail_roi','edit_rubric'):
   if k not in t:raise ValueError(case+' missing task field '+k)
  folder=data_dir/case
  for f in ('source.png','outer.png','full_mask.png','core.png'):
   if not (folder/f).is_file():raise FileNotFoundError(folder/f)
  out=RUNTIME/'data'/case;out.mkdir(parents=True,exist_ok=True)
  for file in folder.iterdir():
   if not file.is_file():continue
   dest=out/file.name
   if dest.exists():
    if sha(dest)!=sha(file):raise RuntimeError('Runtime input differs; use a fresh runtime: '+str(dest))
   else:
    try:os.link(file,dest)
    except OSError:shutil.copy2(file,dest)
   inputs['data/'+case+'/'+file.name]=sha(file)
  normalized.append(t)
 ordered_jobs=[];base=template['anchor_template']
 for t in normalized:
  for seed in (0,17):
   a=dict(base)
   for k in ('case','source_prompt','target_prompt','pattern_roi','detail_roi','edit_rubric'):a[k]=t[k]
   a.update(task=t['key'],seed=seed,id=f"R87_pie_{t['case']}_s{seed}_REQ",arm='REQ',label='REQ',cohort='fresh_confirmation',correction_budget_mode='local_coefficient')
   for k in ('r86_arm','r86_anchor','r86_beta','budget_reference_id','region_gain_cap'):a.pop(k,None)
   for k in ('target_category','target_category_id','eval_pixel_region','roi_scope'):
    if k in t:a[k]=t[k]
   ordered_jobs.append(a)
   for label,kind in [('GLB','GLOBAL'),('SFT','SHIFT'),('PERM','PERM')]:
    j=dict(a);j.update(id=f"R87_pie_{t['case']}_s{seed}_{label}",arm=label,label=label,r86_arm=kind,r86_beta=1.0,r86_anchor=a['id'],correction_budget_mode='r86_fixed_profile');ordered_jobs.append(j)
 plan={k:template[k] for k in ('model_revision','steps','resolution','max_length','numerics')};plan.update(schema_version=1,name='R87_FreshDoseConfirmation',round_id='R87',planned=160,tasks=normalized,jobs=ordered_jobs,input_sha256=inputs,design={'sources':20,'seeds':[0,17],'arms':['REQ','GLB','SFT','PERM'],'outputs':160,'anchor':'REQ; dependent branches replay its per-step realized RMS','sampling':'source/seed pairs fixed in supplied manifest','source_manifest_sha256':sha(manifest_path)})
 target=RUNTIME/'plan.json'
 if (RUNTIME/'results/manifest.json').exists() and target.exists() and read(target)!=plan:raise RuntimeError('Existing runtime plan differs; preserve results and use a separate package directory')
 write(target,plan)
 return plan

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare-only',action='store_true');g.add_argument('--generate',action='store_true');g.add_argument('--evaluate',action='store_true');g.add_argument('--package',action='store_true');p.add_argument('--manifest',type=Path,default=ROOT/'manifest.json');p.add_argument('--data-dir',type=Path,default=ROOT/'data');args=p.parse_args()
 plan=materialize(args.manifest.resolve(),args.data_dir.resolve())
 os.environ.update(OUTPUT_NAME='R87_FreshDoseConfirmation_results',R87_QUEUE_PROJECT=os.environ.get('R87_SERVER_PROJECT',str(DEFAULT_PROJECT)),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
 if args.package:
  collect_delivery(plan,args.manifest.resolve())
  subprocess.run([sys.executable,'-u',str(RUNTIME/'runner.py'),'--pack'],cwd=RUNTIME,check=True);return
 if args.prepare_only or args.generate:preflight_generation(plan)
 else:preflight_evaluation()
 if args.prepare_only:print('Prepared 20 sources / 40 REQ + 40 GLB + 40 SFT + 40 PERM; offline preflight complete.');return
 if args.generate:
  os.environ['R87_PREFLIGHT_PLAN_SHA256']=sha(RUNTIME/'plan.json')
  subprocess.run([sys.executable,'-u',str(RUNTIME/'runner.py')],cwd=RUNTIME,check=True)
  progress=read(RUNTIME/'results/progress.json')
  if progress.get('status')!='completed' or progress.get('completed')!=160:raise RuntimeError('Generation incomplete; inspect progress.json and resume')
  return
 results=RUNTIME/'results'
 if not (results/'progress.json').is_file() or read(results/'progress.json').get('completed')!=160:raise RuntimeError('Evaluation requires 160 completed results')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 command=[sys.executable,'-u',str(RUNTIME/'evaluate.py'),'--from-results',str(results),'--clip-path',os.environ['CLIP_PATH'],'--dino-path',os.environ['DINO_PATH'],'--device','cpu','--data-dir',str(RUNTIME/'data'),'--lpips-path',os.environ['LPIPS_PATH'],'--lpips-trunk-path',os.environ['LPIPS_TRUNK_PATH']]
 subprocess.run(command,cwd=RUNTIME,env=env,check=True)
 sys.path.insert(0,str(RUNTIME));import eval_pipeline
 with (results/'evaluation.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 if len(rows)!=160 or {r['id'] for r in rows}!={j['id'] for j in plan['jobs']} or any(not eval_pipeline.has_required(r) for r in rows):raise RuntimeError('Evaluation incomplete; check evaluation.csv')
 # Materialize source-paired analysis inputs once, without generating a gallery/ZIP.
 import reporting as report
 provenance=report.fingerprint(RUNTIME);report.guard_existing(RUNTIME,provenance)
 rows_by_id={r['id']:r for r in rows};combined=report.rows_for(RUNTIME,plan,provenance['fingerprint'])
 for r in combined:
  extra=rows_by_id[r['id']]
  r.update({k:v for k,v in extra.items() if k.startswith(('eval_','clip_','dino_','lpips_','pixel_'))})
 fields=list(dict.fromkeys(k for r in combined for k in r))
 for name in ('metrics.csv','comparison_metrics.csv'):
  dest=results/name;tmp=dest.with_suffix('.csv.tmp')
  with tmp.open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(combined)
  os.replace(tmp,dest)
 print('Evaluation completed for 160 outputs; run analyze.py then --package for compact ZIP.')
if __name__=='__main__':main()
