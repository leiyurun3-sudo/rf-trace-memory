# -*- coding: utf-8 -*-
"""人评套件（0 GPU）。扫描结果目录里的 preview.jpg，按任务出 8 格判定图，生成待填 CSV。

用法: python3 judge_sheets.py <结果目录>
输出: <结果目录>/judge/sheet_<task>_<i>.jpg 与 edit_ok_template.csv
"""
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
out = root / 'judge'
out.mkdir(exist_ok=True)

items = []
for p in root.rglob('preview.jpg'):
    name = p.parent.name
    segs = name.split('_')
    idx = next((i for i, s in enumerate(segs) if len(s) > 1 and s[0] == 's' and s[1:].isdigit()), None)
    if idx is None or idx < 1 or idx >= len(segs) - 1:
        continue
    items.append(('_'.join(segs[1:idx]), '_'.join(segs[idx + 1:]), segs[idx], p))

if not items:
    print('没有找到 preview.jpg；请把结果目录作为参数传入')
    sys.exit(1)

print(f'找到 {len(items)} 张，{len({i[0] for i in items})} 个任务')
try:
    F = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
except Exception:
    try:
        F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    except Exception:
        F = ImageFont.load_default()

tpl = []
by_task = {}
for it in items:
    by_task.setdefault(it[0], []).append(it)

for task in sorted(by_task):
    group = sorted(by_task[task], key=lambda x: (x[2], x[1]))
    for i in range(0, len(group), 8):
        chunk = group[i:i + 8]
        cell = 240
        sh = Image.new('RGB', (4 * cell, 2 * (cell + 20)), 'white')
        d = ImageDraw.Draw(sh)
        for k, (t, arm, seed, p) in enumerate(chunk):
            c, r = k % 4, k // 4
            d.text((c * cell + 3, r * (cell + 20) + 2), f'{arm} {seed}', fill='black', font=F)
            try:
                im = Image.open(p).convert('RGB').resize((cell, cell), Image.LANCZOS)
            except Exception as exc:
                print('跳过损坏图片', p, exc)
                continue
            sh.paste(im, (c * cell, r * (cell + 20) + 20))
            tpl.append(dict(job_id=p.parent.name, task=t, arm=arm, seed=seed,
                            edit_ok='', subject_ok='', pattern_ok='', artifact=''))
        sh.save(out / f'sheet_{task}_{i // 8}.jpg', quality=88)

with (out / 'edit_ok_template.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['job_id', 'task', 'arm', 'seed',
                                      'edit_ok', 'subject_ok', 'pattern_ok', 'artifact'])
    w.writeheader()
    w.writerows(tpl)
print(f'写出 {out} | 模板 {len(tpl)} 行 | 判定图 {len(list(out.glob("sheet_*.jpg")))} 张')
