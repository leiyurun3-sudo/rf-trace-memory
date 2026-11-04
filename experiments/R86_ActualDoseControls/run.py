import argparse, hashlib, json, math, os, shlex, shutil, subprocess, sys, importlib.util
from pathlib import Path
from r86_compat import MODE, VERSION, fix_engine, fix_runner
HERE=Path(__file__).resolve().parent
read=lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(s,old,new):
    if s.count(old)!=1: raise RuntimeError('Engine/runner version mismatch for patch: '+old[:100])
    return s.replace(old,new)
def cached_evaluator_paths(src):
    bases=[src.parent,src.parent.parent,Path('/root/autodl-tmp/MINT_Flow_Jupyter_Ready')]
    for base in bases:
        for name in ('R85_paths.env','paths.env','R84_paths.env'):
            file=base/'evaluation_models'/name
            if not file.is_file(): continue
            for line in file.read_text(encoding='utf-8').splitlines():
                parts=shlex.split(line,comments=True)
                if parts and parts[0]=='export': parts=parts[1:]
                for token in parts:
                    key,sep,value=token.partition('=')
                    if sep and key in ('CLIP_PATH','DINO_PATH','LPIPS_PATH','LPIPS_TRUNK_PATH') and not os.environ.get(key) and Path(value).exists(): os.environ[key]=value
    spec=importlib.util.find_spec('lpips'); calibration=[]; trunks=[]
    for base in bases:
        calibration.append(base/'evaluation_models/lpips/alex_v0.1.pth')
        trunks.append(base/'evaluation_models/lpips/alexnet-owt-7be5be79.pth')
    if spec and spec.origin: calibration.append(Path(spec.origin).parent/'weights/v0.1/alex.pth')
    trunks.append(Path(os.environ.get('TORCH_HOME',Path.home()/'.cache/torch'))/'hub/checkpoints/alexnet-owt-7be5be79.pth')
    for key,paths in (('LPIPS_PATH',calibration),('LPIPS_TRUNK_PATH',trunks)):
        existing=os.environ.get(key)
        if existing and Path(existing).is_file(): continue
        found=next((x.resolve() for x in paths if x.is_file()),None)
        if found: os.environ[key]=str(found)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--r85',required=True,type=Path); ap.add_argument('--cohort',choices=['pilot','full'],default='pilot'); ap.add_argument('--preflight-only',action='store_true'); args=ap.parse_args()
    src=args.r85.resolve(); base=HERE/('runtime_'+args.cohort+'_v2'); os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    original=read(src/'plan.json'); anchors=read(HERE/'anchor_plan.json'); profiles=read(HERE/'profiles.json'); protocol=read(HERE/'protocol.json'); jobs=anchors['jobs']; all_cases={j['case'] for j in jobs}
    selected=sorted(all_cases,key=lambda c:(hashlib.sha256(('R86-pilot-20261103:'+c).encode('ascii')).hexdigest(),c))[:10]
    if selected!=protocol['case_ids'] or len(all_cases)!=40: raise RuntimeError('Frozen pilot selection mismatch')
    cases=sorted(all_cases) if args.cohort=='full' else selected
    jobs=[j for j in jobs if j['case'] in cases and (args.cohort=='full' or j['seed']==0)]
    if len(jobs)!=(80 if args.cohort=='full' else 10): raise RuntimeError('Unexpected anchor count')
    configs=[('SHIFT_1','SHIFT',1.0),('GLOBAL_1','GLOBAL',1.0)]
    if args.cohort=='pilot': configs += [('CORR_07','CORR',0.7),('GLOBAL_07','GLOBAL',0.7)]
    if original['model_revision']!=anchors['model_revision']: raise RuntimeError('R85 model revision mismatch')
    origin={j['id']:j for j in original['jobs']}; manifest=read(src/'results/manifest.json')
    if manifest.get('plan_sha256')!=sha(src/'plan.json'): raise RuntimeError('R85 manifest plan mismatch')
    for name,digest in manifest.get('code_sha256',{}).items():
        if not (src/name).is_file() or sha(src/name)!=digest: raise RuntimeError('Original R85 code mismatch: '+name)
    for j in jobs:
        required={'arm':'RELEASE_CORR25','corr_lambda':0.25,'correction_shape':'roi_uniform','correction_budget_mode':'local_coefficient','route_mask_source':'prompt_response_reference','route_floor':0.25,'residual_kind':'transport','transport_tau':1.0,'global_update':True,'query_selection':'all_image_queries'}
        if any(j.get(k)!=v for k,v in required.items()) or j.get('memory_response_mode') is not None or j.get('region_gain_cap') is not None: raise RuntimeError('Anchor is not original direct CORR25 configuration: '+j['id'])
        ident=j['id']; folder=src/'results'/ident; record=read(folder/'record.json'); stepsfile=folder/'steps.json'; steps=read(stepsfile)['edit']['steps']; prof=profiles[ident]
        if origin.get(ident)!=j or record.get('planned_job')!=j or record.get('fingerprint')!=manifest.get('fingerprint') or record.get('status')!='completed': raise RuntimeError('R85 anchor configuration/provenance mismatch '+ident)
        for name in ('steps.json','raw.png'):
            if not (folder/name).is_file() or sha(folder/name)!=record.get('artifact_sha256',{}).get(name): raise RuntimeError('R85 anchor artifact absent or mismatch '+str(folder/name))
        if len(steps)!=28 or len(prof)!=28: raise RuntimeError('Profile length mismatch '+ident)
        for i,(s,p) in enumerate(zip(steps,prof)):
            if p['step_index']!=i or any(not math.isfinite(float(p[k])) for k in ('h','sigma','applied_correction_rms')) or p['h']<=0 or p['applied_correction_rms']<=0: raise RuntimeError('Invalid anchor profile '+ident)
            if any(float(s[k])!=float(p[k]) for k in ('sigma','h','applied_correction_rms')): raise RuntimeError('Archived/server profile mismatch '+ident)
        for name in ('source.png','outer.png','full_mask.png'):
            if not (src/'data'/j['case']/name).is_file(): raise RuntimeError('Missing input '+j['case']+'/'+name)
        if sha(src/'data'/j['case']/'source.png')!=record['source_sha256']: raise RuntimeError('Source mismatch '+ident)
    for relative,digest in original.get('input_sha256',{}).items():
        parts=Path(relative).parts
        if len(parts)>=3 and parts[0]=='data' and parts[1] in cases:
            file=(src/relative).resolve()
            if not file.is_relative_to(src) or not file.is_file() or sha(file)!=digest: raise RuntimeError('R85 input missing/changed: '+relative)
    if not (src/'results/comparison_metrics.csv').is_file(): raise RuntimeError('Missing R85 evaluated anchor metrics: results/comparison_metrics.csv')
    for f in ('engine.py','runner.py','controls.py','model_loader.py','eval_pipeline.py','evaluate.py','reporting.py','mint_backend.py','rf_kv.py','attention.py','evaluate_all.sh','analyze.py','overnight_stats.py','evidence_stats.py'):
        if not (src/f).is_file(): raise RuntimeError('Missing R85 code '+f)
    sys.path.insert(0,str(src)); import model_loader, eval_pipeline
    model_loader.preflight_runtime(original); cached_evaluator_paths(src); eval_ready=eval_pipeline.readiness(src.parent)
    os.environ.update(eval_ready['paths'])
    import importlib.metadata
    for pkg in ('lpips','torchvision','scikit-image'): importlib.metadata.version(pkg)
    for key in ('LPIPS_PATH','LPIPS_TRUNK_PATH'):
        if not os.environ.get(key) or not Path(os.environ[key]).is_file(): raise RuntimeError('Set '+key+' to existing offline weight file')
    print('Inputs/cache ready: %d profiles, %d planned new jobs, %d archived anchors; checking patched engine next.'%(len(jobs),len(configs)*len(jobs),len(jobs)),flush=True)
    base.mkdir(exist_ok=True)
    for f in src.glob('*.py'): shutil.copy2(f,base/f.name)
    for f in src.glob('*.sh'): shutil.copy2(f,base/f.name)
    data=base/'data'; data.mkdir(exist_ok=True)
    for case in cases:
        dest=data/case; dest.mkdir(exist_ok=True)
        for source in (src/'data'/case).iterdir():
            if not source.is_file(): continue
            target=dest/source.name
            if target.exists():
                if sha(target)!=sha(source): raise RuntimeError('R86 input copy differs: '+str(target))
            else:
                try: os.link(source,target)
                except OSError: shutil.copy2(source,target)
    new=[]
    for a in jobs:
        for arm,kind,dose in configs:
            j=dict(a); j.update(id=a['id'].replace('R85_','R86_').replace('RELEASE_CORR25',arm),arm=arm,label=arm,r86_arm=kind,r86_beta=dose,r86_anchor=a['id'],correction_budget_mode=MODE); new.append(j)
    plan=dict(original); plan.update(name='R86_ActualDoseControls_'+args.cohort, round_id='R86', tasks=[t for t in original['tasks'] if t['case'] in cases], jobs=new, planned=len(new)); plan['input_sha256']={k:v for k,v in plan.get('input_sha256',{}).items() if k.split('/')[1] in cases}; plan.pop('pre_registered',None); plan['r86_protocol']=protocol; plan['r86_anchor_cells']=[{'id':j['id'],'arm':'CORR_1','case':j['case'],'seed':j['seed'],'reused':True} for j in jobs]; plan['r86_total_cells']=len(jobs)+len(new)
    (base/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    frozen=base/'profiles.json'
    if frozen.exists() and sha(frozen)!=sha(HERE/'profiles.json'): raise RuntimeError('Frozen R86 profile changed; use a fresh package directory')
    if not frozen.exists(): shutil.copy2(HERE/'profiles.json',frozen)
    os.environ['R86_PROFILE_SHA256']=sha(frozen)
    os.environ['R86_SHARED_QUEUE_PROJECT']=str(src.parent.parent)
    engine=(base/'engine.py').read_text(encoding='utf-8')
    engine=replace(engine,"    route_mask = controls.get('route_mask')\n    route_mask64 = controls.get('route_mask64')", """    if job.get('r86_arm'):
        controls['r86_E_original'] = controls['edit_response64'].detach().clone()
        controls['r86_P_original'] = controls['effective_route_mask64'].detach().clone()
    if job.get('r86_arm') == 'SHIFT':
        route64 = controls['route_mask64']
        if route64.shape != (1, 1, 64, 64): raise ValueError('R86 expects 64x64 latent response grid')
        shifted = torch.roll(route64, shifts=(32, 32), dims=(-2, -1))
        if not torch.equal(torch.sort(route64.flatten()).values, torch.sort(shifted.flatten()).values): raise RuntimeError('Shift histogram mismatch')
        controls['route_mask64'].copy_(shifted)
        channels = controls['route_mask'].shape[-1] // 4
        controls['route_mask'].copy_(pack(shifted.expand(-1, channels, -1, -1)))
        controls['effective_route_mask64'] = 0.25 + 0.75*shifted
    route_mask = controls.get('route_mask')
    route_mask64 = controls.get('route_mask64')""")
    engine=replace(engine,"            full_weighted = controls['calibration']*lam*correction\n            actual_rms = rms(mask*weighted)","""            r86_target = None
            if job.get('r86_arm'):
                if budget_profile is None or len(budget_profile) != len(sigmas)-1: raise RuntimeError('Missing fixed R86 profile')
                anchor_step = budget_profile[i]
                if anchor_step['step_index'] != i or abs(float(anchor_step['sigma'])-sigma)>1e-8 or abs(float(anchor_step['h'])-h)>1e-8: raise RuntimeError('R86 schedule mismatch')
                r86_target = float(job['r86_beta']) * float(anchor_step['applied_correction_rms'])
                candidate = controls['calibration'] * (correction if job['r86_arm']=='GLOBAL' else spectral_candidate(correction, job, route_mask=route_mask)[0])
                if not bool(torch.isfinite(candidate).all()): raise FloatingPointError('Nonfinite R86 candidate')
                candidate_rms = rms(mask*candidate)
                if not math.isfinite(r86_target) or r86_target <= 0 or not math.isfinite(candidate_rms) or candidate_rms <= 0: raise FloatingPointError('Invalid or zero R86 target/candidate RMS')
                gain = r86_target / candidate_rms
                if not math.isfinite(gain): raise FloatingPointError('Nonfinite R86 gain')
                weighted = candidate * gain
                if not bool(torch.isfinite(weighted).all()): raise FloatingPointError('Nonfinite weighted correction')
            full_weighted = controls['calibration']*lam*correction
            actual_rms = rms(mask*weighted)
            if r86_target is not None and (not math.isfinite(actual_rms) or abs(actual_rms-r86_target)>max(1e-8,1e-5*r86_target)): raise FloatingPointError('R86 realized RMS mismatch')""")
    engine=replace(engine,"                control_kind=control_kind, candidate_local_rms=candidate_local_rms, actual_rms=actual_rms,","                control_kind=control_kind, candidate_local_rms=candidate_local_rms, actual_rms=actual_rms, r86_target_rms=r86_target, r86_beta=job.get('r86_beta'), r86_applied_update_rms=None if r86_target is None else h*actual_rms, r86_target_update_rms=None if r86_target is None else h*r86_target, r86_abs_rms_error=None if r86_target is None else abs(actual_rms-r86_target), r86_relative_rms_error=None if r86_target is None else abs(actual_rms-r86_target)/r86_target,")
    engine=fix_engine(engine)
    (base/'engine.py').write_text(engine,encoding='utf-8')
    shutil.copy2(HERE/'r86_compat.py',base/'r86_compat.py')
    reporting=(base/'reporting.py').read_text(encoding='utf-8')
    reporting=replace(reporting, "return root / 'results', root.parent.parent / name, root.parent.parent / (name + '.zip')", "return root / 'results', root.parent / name, root.parent / (name + '.zip')")
    reporting=replace(reporting, "'evaluation_config.json', 'statistics.json') if (out / n).exists()]", "'evaluation_config.json', 'statistics.json', 'dose_audit.json', 'r86_paired.csv', 'anchor_evaluation.csv', 'localization_L.csv', 'localization_metadata.json') if (out / n).exists()]")
    reporting=replace(reporting, "'edit_response64.jpg', 'step_summary.json') if (out / job['id'] / n).exists()]", "'edit_response64.jpg', 'step_summary.json', 'request_maps.npz', 'request_maps.json') if (out / job['id'] / n).exists()]")
    reporting=replace(reporting, "    if split_manifest.is_file():\n        paths.append(out / 'split_manifest.json')", "    paths += list((out / 'anchor_previews').glob('*.jpg'))\n    if split_manifest.is_file():\n        paths.append(out / 'split_manifest.json')")
    (base/'reporting.py').write_text(reporting,encoding='utf-8')
    runner=(base/'runner.py').read_text(encoding='utf-8')
    runner=replace(runner, 'with queue_locks(root.parent, root=root):', "with queue_locks(Path(os.environ['R86_SHARED_QUEUE_PROJECT']), root=root):")
    runner=replace(runner, "            # The request probes mutate controls inside execute. Save actual P afterwards.", """            # Exact float maps are diagnostics only; official evaluation masks are never read here.
            if job.get('r86_arm'):
                import numpy as np
                E = controls.get('r86_E_original'); P = controls.get('r86_P_original')
                if E is None or P is None: raise RuntimeError('Missing real request response maps')
                def arr(t):
                    value=t.detach().float().cpu().numpy().squeeze()
                    if value.shape != (64,64) or not np.isfinite(value).all(): raise ValueError('Invalid request map')
                    return value.astype(np.float32,copy=False)
                arrays={'E_original':arr(E),'P_original':arr(P)}
                if job['r86_arm']=='SHIFT':
                    arrays['P_shifted']=arr(controls['effective_route_mask64'])
                    arrays['E_shifted']=np.roll(arrays['E_original'],(32,32),(0,1))
                if job['r86_arm']!='GLOBAL': arrays['P_correction']=arr(controls['effective_route_mask64'])
                tmp=dest/'request_maps.npz.tmp'
                with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
                os.replace(tmp,dest/'request_maps.npz')
                report.save(dest/'request_maps.json',{'schema':'r86-request-maps-v1','case':job['case'],'seed':job['seed'],'arm':job['arm'],'dtype':'float32','shape':[64,64], 'E_original':'mean of three normalized native target/source response magnitudes','P_original':'1 - 0.75 * E_original','route_mask':'engine route_mask64 stores 1-E before floor; packed route mask is same un-floored support','P_shifted':'toroidal roll of P_original by (32,32), only SHIFT','P_correction':'actual correction allocation for CORR/SHIFT; absent for GLOBAL, which uses an unmasked global residual','mask_source':'native prompt-response probes; no official mask','sha256':report.sha(dest/'request_maps.npz')})
            # The request probes mutate controls inside execute. Save actual P afterwards.""")
    runner=replace(runner, "hashes = {name:report.sha(dest / name) for name in ('raw.png', 'preview.jpg', 'pattern.png', 'pattern.jpg', 'steps.json')}", "hashes = {name:report.sha(dest / name) for name in ('raw.png', 'preview.jpg', 'pattern.png', 'pattern.jpg', 'steps.json', 'request_maps.npz', 'request_maps.json')}")
    runner=replace(runner,"            raw_profile = report.exact_raw_profile(root, job, sigma_sha)","""            if job.get('r86_arm'):
                pfile = root / 'profiles.json'
                if report.sha(pfile) != os.environ['R86_PROFILE_SHA256']: raise RuntimeError('R86 frozen profile changed')
                budget_profile = report.read(pfile)[job['r86_anchor']]
                budget_reference_nfe = 97
            raw_profile = None""")
    runner=fix_runner(runner)
    (base/'runner.py').write_text(runner,encoding='utf-8')
    overnight=(base/'overnight_stats.py').read_text(encoding='utf-8')
    overnight=replace(overnight, "BASE = 'RELEASE_CORR25'", "BASE = 'SHIFT_1'")
    (base/'overnight_stats.py').write_text(overnight,encoding='utf-8')
    os.environ['OUTPUT_NAME']='R86_ActualDoseControls_'+args.cohort+'_v2'
    os.environ['EVAL_REFERENCE_ARM']='SHIFT_1'
    for file in base.glob('*.py'):
        compile(file.read_text(encoding='utf-8-sig'),str(file),'exec')
    subprocess.run([sys.executable,str(HERE/'check_engine.py'),'--engine',str(base/'engine.py')],cwd=base,check=True)
    print('R86 fixed-profile engine branch checked on CPU; build='+VERSION+'; runtime='+str(base),flush=True)
    if args.preflight_only:return
    subprocess.run([sys.executable,'-u',str(base/'runner.py')],cwd=base,check=True)
    errors=[]; max_abs=0.; max_rel=0.
    for j in new:
        result=base/'results'/j['id']; rec=result/'record.json'
        if not rec.is_file() or read(rec).get('status')!='completed': errors.append(j['id']); continue
        record=read(rec)
        for name in ('request_maps.npz','request_maps.json'):
            if not (result/name).is_file() or sha(result/name)!=record.get('artifact_sha256',{}).get(name): raise RuntimeError('Missing or changed exact request map: '+str(result/name))
        for step in read(result/'steps.json')['edit']['steps']:
            max_abs=max(max_abs,float(step['r86_abs_rms_error'])); max_rel=max(max_rel,float(step['r86_relative_rms_error']))
    if errors: raise RuntimeError('Incomplete R86 generation; resume before evaluation: '+str(errors))
    (base/'results'/'dose_audit.json').write_text(json.dumps({'profiles':len(jobs),'archived_anchor_nfe_each':97,'anchor_generation_new_nfe':0,'new_outputs':len(new),'new_nfe_each_expected':97,'per_output_with_unshared_anchor_nfe':194,'profile_reuse':True,'total_cells':len(jobs)+len(new),'dose_definition':'target RMS before h; applied state update RMS = h * target','max_absolute_step_rms_error':max_abs,'max_relative_step_rms_error':max_rel},indent=2),encoding='utf-8')
    env=os.environ.copy(); env['PREPARE_EVAL_MODELS']='0'; env['DOWNLOAD_EVAL_MODELS']='0'; env['EVAL_PYTHON']=sys.executable
    subprocess.run(['bash',str(base/'evaluate_all.sh')],cwd=base,env=env,check=True)
    # Re-evaluate the reused anchors with exactly the same offline evaluator and weights.
    aroot=base/'anchor_eval'; aroot.mkdir(exist_ok=True); (aroot/'sources').mkdir(exist_ok=True)
    aplan=dict(plan); aplan.update(jobs=jobs,planned=len(jobs)); (aroot/'plan.json').write_text(json.dumps(aplan,ensure_ascii=False,indent=2),encoding='utf-8')
    def copy_verified(source,destination):
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            if sha(destination)!=sha(source): raise RuntimeError('Anchor evaluation input changed: '+str(destination))
        else:
            try: os.link(source,destination)
            except OSError: shutil.copy2(source,destination)
    for a in jobs:
        for name in ('raw.png','record.json'): copy_verified(src/'results'/a['id']/name,aroot/a['id']/name)
        copy_verified(base/'results'/'sources'/(a['case']+'_source_eval.png'),aroot/'sources'/(a['case']+'_source_eval.png'))
    command=[sys.executable,str(base/'evaluate.py'),'--from-results',str(aroot),'--clip-path',os.environ['CLIP_PATH'],'--dino-path',os.environ['DINO_PATH'],'--device','cpu','--data-dir',str(base/'data'),'--lpips-path',os.environ['LPIPS_PATH'],'--lpips-trunk-path',os.environ['LPIPS_TRUNK_PATH']]
    subprocess.run(command,cwd=base,env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
    import csv
    with (aroot/'evaluation.csv').open(encoding='utf-8-sig',newline='') as f: arows={r['id']:r for r in csv.DictReader(f)}
    if any(a['id'] not in arows or not eval_pipeline.has_required(arows[a['id']]) for a in jobs): raise RuntimeError('Anchor evaluation incomplete; rerun to resume')
    shutil.copy2(aroot/'evaluation.csv',base/'results'/'anchor_evaluation.csv')
    for a in jobs:
        preview=src/'results'/a['id']/'preview.jpg'
        if preview.is_file(): copy_verified(preview,base/'results'/'anchor_previews'/(a['id']+'.jpg'))
    subprocess.run([sys.executable,str(HERE/'paired.py'),'--r85',str(src),'--runtime',str(base)],check=True)
    subprocess.run([sys.executable,str(HERE/'localization.py'),'--runtime',str(base)],check=True)
    subprocess.run([sys.executable,'-c','import reporting; reporting.package()'],cwd=base,check=True)
if __name__=='__main__':main()
