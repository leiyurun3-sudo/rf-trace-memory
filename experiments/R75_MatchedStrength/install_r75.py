# -*- coding: utf-8 -*-
"""R75 安装器：匹配强度对比。
新增 C25 / C75（自有方法强度曲线）+ FLOWEDIT 两档更强引导（guidance 5.0 / 6.0）。
只读你的 R74_MethodStudy 目录，复制成 R75_MatchedStrength，不改原目录。
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path.cwd()
OLD = 'R74_MethodStudy'
NEW = 'R75_MatchedStrength'


def arms_of(d: Path):
    try:
        j = json.loads((d / 'plan.json').read_text(encoding='utf-8-sig'))
    except Exception:
        return None, set()
    arms = {str(x.get('arm') or x.get('label')) for x in j.get('jobs', [])}
    return j, arms


cands, info = [], []
for d in [ROOT] + [p for p in ROOT.rglob('*') if p.is_dir()]:
    try:
        if len(d.relative_to(ROOT).parts) > 3:
            continue
    except ValueError:
        continue
    if not (d / 'plan.json').is_file():
        continue
    j, arms = arms_of(d)
    if j is None:
        continue
    score = 0
    if 'FLOWEDIT' in arms:
        score += 1000
    if 'MethodStudy' in str(j.get('name', '')) or 'MethodStudy' in d.name:
        score += 200
    score += min(len(j.get('jobs', [])), 500)
    info.append((score, d, j, arms))

if not info:
    print('[FATAL] 没找到任何含 plan.json 的目录。当前目录：')
    for p in sorted(ROOT.iterdir()):
        print('   ', p.name)
    sys.exit(1)

print('[0] 候选目录:')
for score, d, j, arms in sorted(info, key=lambda x: -x[0]):
    print('    %-40s plan=%-28s jobs=%-4d arms=%s' % (d.relative_to(ROOT), str(j.get('name'))[:28],
                                                       len(j.get('jobs', [])), 'FLOWEDIT' in arms))

info.sort(key=lambda x: -x[0])
_, src, plan0, arm_set = info[0]
if 'FLOWEDIT' not in arm_set:
    print('[FATAL] 选中的目录没有 FLOWEDIT 臂，说明真正的 MethodStudy 结果目录不在当前路径下。')
    print('       请把上面 [0] 的输出发回来。')
    sys.exit(1)
print('[1] 源目录 :', src.relative_to(ROOT))
print('    plan   :', plan0.get('name'), '| jobs', len(plan0.get('jobs', [])))

dst = src.parent / NEW
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
    'results*', 'cache*', '__pycache__', '*.log', '*.zip', 'references*', 'judge*'))
print('[2] 复制到 :', dst.relative_to(ROOT))

renamed = []
for p in list(dst.glob('*.py')) + list(dst.glob('*.sh')):
    s = p.read_text(encoding='utf-8', errors='replace')
    t = s.replace(OLD, NEW)
    if t != s:
        p.write_text(t, encoding='utf-8')
        renamed.append(p.name)
print('[3] 输出路径改名:', renamed if renamed else '无（可能由 plan.name 决定，正常）')

# FLOWEDIT 分支怎么被识别
hits = []
for p in dst.glob('*.py'):
    for i, line in enumerate(p.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
        if 'FLOWEDIT' in line:
            hits.append(f'{p.name}:{i}: {line.strip()[:110]}')
print('[4] FLOWEDIT 路由线索:')
for h in hits[:6]:
    print('    ', h)
if not hits:
    print('     ! 没找到 FLOWEDIT 字样，外部基线可能是单独脚本，本轮不重跑它')

by_arm = {}
for j in plan0['jobs']:
    by_arm.setdefault(j.get('arm') or j.get('label'), j)
for need in ('C50', 'FLOWEDIT'):
    if need not in by_arm:
        print('[FATAL] 缺少模板臂', need, '| 现有:', sorted(by_arm))
        sys.exit(1)

tasks = [t for t in sorted({j['task'] for j in plan0['jobs']}) if not t.endswith('_recon')]
seeds = {}
for j in plan0['jobs']:
    if not j['task'].endswith('_recon'):
        seeds.setdefault(j['task'], set()).add(j['seed'])
seeds = {t: sorted(v)[:2] for t, v in seeds.items()}
print('[5] 任务与种子:', {t: seeds[t] for t in tasks})

# 新臂：FLOW 系列保持 arm='FLOWEDIT' 以命中代码分支，靠 id 区分强度
ARMS = [
    ('C25', 'C50', 'C25', dict(corr_lambda=0.25), 'C25'),
    ('C75', 'C50', 'C75', dict(corr_lambda=0.75), 'C75'),
    ('FLOWEDIT_G50', 'FLOWEDIT', 'FLOWEDIT', dict(guidance=5.0), 'FLOW_G50'),
    ('FLOWEDIT_G60', 'FLOWEDIT', 'FLOWEDIT', dict(guidance=6.0), 'FLOW_G60'),
]

jobs = []
for task in tasks:
    per_arm = {}
    for j in plan0['jobs']:
        if j['task'] == task:
            per_arm.setdefault(j.get('arm') or j.get('label'), j)
    for seed in seeds[task]:
        for jid_suffix, tpl_arm, arm_name, over, _tag in ARMS:
            tpl = per_arm.get(tpl_arm) or by_arm[tpl_arm]
            n = json.loads(json.dumps(tpl))
            n.update(task=task, seed=seed, arm=arm_name, label=arm_name,
                     id=f'R75_{task}_s{seed}_{jid_suffix}', **over)
            for k in ('r66_reference', 'reuse', 'reused', 'corr_weights', 'budget_h'):
                n.pop(k, None)
            jobs.append(n)

ids = [j['id'] for j in jobs]
assert len(ids) == len(set(ids)), 'id 重复'
print('[6] 生成 jobs:', len(jobs), '| 每任务', len(jobs) // max(1, len(tasks)))

old_tasks = {t.get('key') or t.get('task'): t for t in plan0.get('tasks', [])}
new_tasks = []
for t in tasks:
    base = dict(old_tasks.get(t, {'key': t, 'task': t}))
    base['key'] = t
    base['task'] = t
    base['seeds'] = seeds[t]
    base.setdefault('case', next(j['case'] for j in jobs if j['task'] == t))
    new_tasks.append(base)

plan = dict(plan0)
plan.update(name=NEW, planned=len(jobs), jobs=jobs, tasks=new_tasks)
for k in ('references', 'reference_results'):
    plan.pop(k, None)
(dst / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding='utf-8')

REQ_JOB = ('id', 'task', 'case', 'seed', 'arm', 'label', 'source_prompt', 'target_prompt')
bad_jobs = [(j['id'], [k for k in REQ_JOB if k not in j]) for j in jobs]
bad_jobs = [(i, m) for i, m in bad_jobs if m]
print('[7] job 字段检查:', 'OK' if not bad_jobs else f'{len(bad_jobs)} 条缺失 -> {bad_jobs[:3]}')
rmiss = [k for k in ('source_prompt', 'target_prompt') if k not in new_tasks[0]]
print('    任务表字段:', 'OK' if not rmiss else f'缺 {rmiss}')
if bad_jobs or rmiss:
    print('[FATAL] 字段不全，停止（不写 plan）')
    sys.exit(1)

venv = None
for up in (ROOT, ROOT.parent, ROOT.parent.parent):
    c = up / '.venv' / 'bin' / 'python'
    if c.exists():
        venv = c
        break
venv = venv or 'python3'
run_src = src / 'run_all.sh'
if run_src.exists():
    txt = run_src.read_text(encoding='utf-8').replace(OLD, NEW)
    (dst / 'run_all.sh').write_text(txt, encoding='utf-8')
else:
    (dst / 'run_all.sh').write_text(
        '#!/usr/bin/env bash\ncd "$(dirname "$0")" || exit 1\nexec "%s" runner.py\n' % venv,
        encoding='utf-8')
print('[8] python :', venv)
print()
print('===== 完成 =====')
print('运行: bash %s/run_all.sh' % dst.relative_to(ROOT))
print('张数: %d（%d 任务 x %d 臂 x 2 seed）' % (len(jobs), len(tasks), len(jobs) // max(1, len(tasks) * 2)))
