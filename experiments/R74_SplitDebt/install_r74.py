# -*- coding: utf-8 -*-
"""R74：残差分解 + 记忆门控。自动探测 R73 代码目录，生成 R74 副本与新计划。"""
import json
import shutil
import sys
from pathlib import Path

TASKS = ['cup_night', 'printed_shirt_studio', 'butterfly_sunset', 'dog_rain', 'dog_snow', 'rug_floor']
ARMS = [
    ('DISC_ONLY', 1.0, 0.0, 'constant', 0.3),
    ('COND_ONLY', 0.0, 1.0, 'constant', 0.3),
    ('GATE_FADE', 1.0, 1.0, 'fade', 0.3),
    ('DISC_FADE', 1.0, 0.0, 'fade', 0.3),
]

ROOT = Path.cwd()


def looks_like_code(d: Path):
    try:
        names = {p.name for p in d.iterdir() if p.is_file()}
    except OSError:
        return False
    return {'engine.py', 'runner.py', 'plan.json'} <= names


cands = []
for d in [ROOT] + [p for p in ROOT.rglob('*') if p.is_dir()]:
    try:
        if len(d.relative_to(ROOT).parts) > 3:
            continue
    except ValueError:
        continue
    if looks_like_code(d):
        cands.append(d)

if not cands:
    print('未找到含 engine.py / runner.py / plan.json 的目录。当前目录内容：')
    for p in sorted(ROOT.iterdir()):
        print('  ', p.name)
    sys.exit(1)


def score(d):
    s = 0.0
    try:
        name = json.loads((d / 'plan.json').read_text(encoding='utf-8-sig')).get('name', '')
    except Exception:
        name = ''
    if 'R73' in name:
        s += 200
    if 'R73' in d.name:
        s += 100
    try:
        s += (d / 'plan.json').stat().st_mtime % 1000 / 1000.0
    except OSError:
        pass
    return s


cands.sort(key=score, reverse=True)
print('候选目录:')
for c in cands[:5]:
    print('  ', c.relative_to(ROOT))
src = cands[0]
plan0 = json.loads((src / 'plan.json').read_text(encoding='utf-8-sig'))
print('选用:', src.relative_to(ROOT), '| plan =', plan0.get('name'), '| jobs =', len(plan0.get('jobs', [])))

if src == ROOT:
    dst = ROOT / 'R74_SplitDebt'
    dst.mkdir(exist_ok=True)
    for p in list(ROOT.glob('*.py')) + list(ROOT.glob('*.sh')):
        shutil.copy2(p, dst / p.name)
else:
    dst = src.parent / 'R74_SplitDebt'
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        'results', 'cache*', '__pycache__', '*.log', 'references', '*.zip'))
(dst / 'plan.json').write_text(json.dumps(plan0, ensure_ascii=False, indent=1), encoding='utf-8')
print('已复制到:', dst.relative_to(ROOT))

# rf_kv.py
rk = dst / 'rf_kv.py'
s = rk.read_text(encoding='utf-8')
if 'fields=[]' in s:
    print('rf_kv.py 已是改好的')
else:
    ok = True
    if 'states=[z.cpu().clone()];logs=[]' in s:
        s = s.replace('states=[z.cpu().clone()];logs=[]', 'states=[z.cpu().clone()];logs=[];fields=[]', 1)
    else:
        ok = False
    anchor = "z=z+h*((1-j['gamma'])*v+j['gamma']*(noise-z)/(1-t))"
    if anchor in s:
        s = s.replace(anchor, anchor + '\n            fields.append(v.detach().float().cpu())', 1)
    else:
        ok = False
    ret = 'return list(reversed(states)),dict('
    if ok and ret in s:
        save = ("import os\n    os.makedirs('cache_r74',exist_ok=True)\n"
                "    torch.save(dict(fields=list(reversed(fields))),os.path.join('cache_r74',"
                "str(j['case'])+'_'+str(seed)+'.pt'))\n    ")
        s = s.replace(ret, save + ret, 1)
        rk.write_text(s, encoding='utf-8')
        print('rf_kv.py 已改')
    else:
        print('rf_kv.py 结构不符，停止（原文件未动）')
        sys.exit(1)

# engine.py
en = dst / 'engine.py'
s = en.read_text(encoding='utf-8')
helper = '''_CACHE = {}


def _split_correction(path, source_field, job, i, b):
    """c_disc = p + f_empty ; c_cond = f_src - f_empty (eta=0)"""
    import os as _os
    import torch as _t
    key = (str(job.get('case')), str(job.get('seed')))
    if key not in _CACHE:
        fp = _os.path.join('cache_r74', key[0] + '_' + key[1] + '.pt')
        _CACHE[key] = _t.load(fp)['fields'] if _os.path.exists(fp) else None
    f = _CACHE[key]
    if f is None:
        raise RuntimeError('R74 missing inversion field cache: ' + str(key))
    vf = f[i].to(b.device).float()
    return (float(job.get('corr_lambda_disc', 1.0)) * (path + vf)
            + float(job.get('corr_lambda_cond', 1.0)) * (source_field - vf))


'''
if 'def _split_correction' in s:
    print('engine.py 已是改好的')
else:
    sig = '@torch.inference_mode()\ndef execute('
    old = '                correction = path - source_velocity'
    if sig not in s or old not in s:
        print('engine.py 找不到插入点，停止。sig:', sig in s, '| correction:', old in s)
        sys.exit(1)
    s = s.replace(sig, helper + sig, 1)
    s = s.replace(old, old + "\n                if job.get('corr_split'):\n"
                             "                    correction = _split_correction(path, source_field, job, i, b)", 1)
    en.write_text(s, encoding='utf-8')
    print('engine.py 已改')

# plan
jobs = plan0['jobs']
by_task = {}
for j in jobs:
    by_task.setdefault(j['task'], []).append(j)

tpl = None
for pref in ('C50', 'BUDGET_EARLY', 'CORR_KV'):
    for j in jobs:
        if j.get('label') == pref and float(j.get('corr_lambda', 0)) > 0 and j.get('mode') != 'off':
            tpl = j
            break
    if tpl:
        break
if tpl is None:
    print('找不到模板 job（C50/BUDGET_EARLY/CORR_KV）。现有 label：',
          sorted({j.get('label') for j in jobs}))
    sys.exit(1)
print('模板:', tpl['id'])

new_jobs = []
for task in TASKS:
    seeds = sorted({j['seed'] for j in by_task.get(task, [])})
    if not seeds:
        print('跳过无此任务:', task)
        continue
    for seed in seeds[:2]:
        base = next((j for j in by_task[task] if j['seed'] == seed and j.get('label') == tpl.get('label')), None)
        if base is None:
            base = next(j for j in by_task[task] if j['seed'] == seed)
        for label, ld, lc, gs, ge in ARMS:
            n = json.loads(json.dumps(base))
            n.update(seed=seed, label=label, id=f"R74_{task}_s{seed}_{label}",
                     corr_lambda=1.0, corr_schedule='constant', corr_policy='constant',
                     corr_split=True, corr_lambda_disc=ld, corr_lambda_cond=lc,
                     gate_schedule=gs, gate_end=ge)
            for k in ('corr_weights', 'budget_h', 'r66_reference', 'reuse', 'reused'):
                n.pop(k, None)
            n.setdefault('correction_budget_target', 1.0)
            new_jobs.append(n)

assert len({j['id'] for j in new_jobs}) == len(new_jobs)
plan = dict(plan0)
plan.update(name='R74_SplitDebt', planned=len(new_jobs), jobs=new_jobs)
plan['tasks'] = [dict(key=t, case=next(j['case'] for j in new_jobs if j['task'] == t), task=t,
                      seeds=sorted({j['seed'] for j in new_jobs if j['task'] == t}),
                      edit_rubric=['requested_change'], preservation_rubric='source markings',
                      review=dict(light='requested', environment='requested', content='markings'))
                 for t in TASKS if any(j['task'] == t for j in new_jobs)]
plan.pop('references', None)
plan.pop('reference_results', None)
(dst / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')
print('plan.json:', len(new_jobs), 'jobs |', len(plan['tasks']), 'tasks')

venv = None
for up in (ROOT, ROOT.parent, ROOT.parent.parent):
    c = up / '.venv' / 'bin' / 'python'
    if c.exists():
        venv = c
        break
if venv is None:
    venv = 'python3'
sh = dst / 'run_all.sh'
sh.write_text('#!/usr/bin/env bash\ncd "$(dirname "$0")" || exit 1\nexec "%s" runner.py\n' % venv,
              encoding='utf-8')
print('python:', venv)
print('\n完成。运行: bash %s/run_all.sh' % dst.relative_to(ROOT))
