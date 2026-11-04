import argparse,json,sys,types,importlib.util
from pathlib import Path
import torch
torch.set_num_threads(1)
parser=argparse.ArgumentParser();parser.add_argument('--engine',required=True,type=Path);parser.add_argument('--package',type=Path);args=parser.parse_args()
candidates=[args.package, args.engine.resolve().parent.parent, Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent/'R86_ActualDoseControls']
root=next((p for p in candidates if p and (p/'profiles.json').is_file() and (p/'anchor_plan.json').is_file()),None)
if root is None: raise SystemExit('Cannot find profiles.json and anchor_plan.json; use --package')
profiles=json.loads((root/'profiles.json').read_text(encoding='utf-8'));anchor=json.loads((root/'anchor_plan.json').read_text(encoding='utf-8'))['jobs'][0];profile=profiles[anchor['id']]
controls=types.ModuleType('controls');exec('def pack(x):\n b,c,h,w=x.shape\n return x.reshape(b,c,h//2,2,w//2,2).permute(0,2,4,1,3,5).reshape(b,h*w//4,c*4)',controls.__dict__);sys.modules['controls']=controls
class Memory:
 def __init__(self,*a,**k):self.actual=[]
 def off(self):pass
 def capture(self):pass
 def apply(self,*a,**k):pass
 def close(self):pass
attention=types.ModuleType('attention');attention.AttentionMemory=Memory;sys.modules['attention']=attention
sys.path.insert(0,str(args.engine.resolve().parent))
spec=importlib.util.spec_from_file_location('r86_actual_test_engine',args.engine.resolve());engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
class Backend:
 def __init__(self):self.device='cpu';self.nfe=0;self.pipe=types.SimpleNamespace(transformer=None);self.empty_embeddings=0
 def schedule(self,n):return [x['sigma'] for x in profile]+[profile[-1]['sigma']-profile[-1]['h']]
 def field(self,z,sigma,embedding,ids,guidance):
  self.nfe+=1;grid=torch.linspace(0,0.1,z.shape[1])[None,:,None];return .05+.001*z+float(embedding)*grid
shape=(1,1024,64);y=torch.ones(shape);trajectory=[torch.ones(shape)*(1+.001*i) for i in range(29)]
def make_controls():
 z=torch.ones(shape);return {'mask':z.clone(),'allowed':z.clone(),'calibration':z.clone(),'route_mask':z.clone(),'route_mask64':torch.ones(1,1,64,64),'query_mask':torch.ones(1,1024,1)}
def run(job,budget):
 b=Backend();return b,lambda:engine.execute(b,y,None,0.,1.,make_controls(),job,trajectory,lambda *x:None,None,budget)
for arm,kind,beta in [('SHIFT_1','SHIFT',1.),('GLOBAL_1','GLOBAL',1.),('CORR_07','CORR',.7),('GLOBAL_07','GLOBAL',.7)]:
 job=dict(anchor,r86_arm=kind,r86_beta=beta,r86_anchor=anchor['id'],correction_budget_mode='r86_fixed_profile');b,call=run(job,profile);out,stats=call();assert len(stats['steps'])==28
 for i,row in enumerate(stats['steps']):
  target=beta*profile[i]['applied_correction_rms'];assert abs(row['actual_rms']-target)<=max(1e-8,1e-5*target),(arm,i,row['actual_rms'],target)
  assert abs(row['r86_target_rms']-target)<1e-12
 assert stats['correction_budget_status']=='fixed_profile_matched'
 assert stats['external_trajectory_budget_matched'] is True
 expected=sum(beta*r['h']*r['applied_correction_rms'] for r in profile)
 assert abs(stats['budget_target']-expected)<1e-10
 assert stats['correction_budget_relative_error']<1e-5
 assert b.nfe==69 and stats['memory_active_steps']==7
 assert torch.isfinite(out).all();print(arm,'28 steps; max relative error',max(r['r86_relative_rms_error'] for r in stats['steps']),flush=True)
for label,job,budget in [('missing_r86_profile',dict(anchor,r86_arm='CORR',r86_beta=1.,r86_anchor=anchor['id'],correction_budget_mode='r86_fixed_profile'),None),('legacy_local_with_profile',dict(anchor,correction_budget_mode='local_coefficient'),profile)]:
 b,call=run(job,budget)
 try:call()
 except (ValueError,RuntimeError,FloatingPointError) as exc:
  assert b.nfe==0,(label,'field calls before rejection',b.nfe);print(label,'rejected before field:',str(exc),flush=True)
 else:raise AssertionError(label+' incorrectly accepted')
print('PASS: true supplied engine, 4 variants x 28, strict early guards; CPU fake fields only',flush=True)
