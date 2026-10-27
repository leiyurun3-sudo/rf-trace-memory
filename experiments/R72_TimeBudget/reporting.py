"""CPU-only provenance, preview measurements and incremental delivery."""
import contextlib
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def locations(root=ROOT):
    root = Path(root)
    return root / 'results', root.parent.parent / 'R72_TimeBudget_Images', root.parent.parent / 'R72_TimeBudget_Results.zip'


def safe_name(value):
    if not isinstance(value, str) or not value or value in ('.', '..') or any(c in value for c in '/\\:<>"|?*') or value.endswith((' ', '.')):
        raise ValueError('Unsafe path component: ' + repr(value))
    return value


def valid_roi(roi):
    return (isinstance(roi, (list, tuple)) and len(roi) == 4
            and all(type(v) is int for v in roi)
            and 0 <= roi[0] < roi[2] <= 512 and 0 <= roi[1] < roi[3] <= 512)


def validate_job(job, tasks):
    task = tasks.get(job.get('task'))
    if task is None:
        raise ValueError('Unknown task for ' + str(job.get('id')))
    for field in ('case', 'source_prompt', 'target_prompt', 'pattern_roi', 'detail_roi'):
        if field not in job or job[field] != task[field]:
            raise ValueError(f"{job.get('id')}: task mismatch: {field}")
    if type(job.get('seed')) is not int or job['seed'] not in task['seeds']:
        raise ValueError(str(job.get('id')) + ': seed outside task seeds')
    if 'edit_rubric' in job and job['edit_rubric'] != task['edit_rubric']:
        raise ValueError(str(job.get('id')) + ': edit_rubric mismatch')
    for field in ('pattern_roi', 'detail_roi'):
        if not valid_roi(job[field]):
            raise ValueError(str(job.get('id')) + ': invalid 512-coordinate ' + field)
    if job.get('corr_policy', 'constant') not in ('constant', 'conflict', 'norm_matched'):
        raise ValueError('Unknown corr_policy')
    return task


def load_plan(root=ROOT):
    plan = read(Path(root) / 'plan.json')
    tasks = {}
    for task in plan['tasks']:
        key = task['key']
        if key in tasks:
            raise ValueError('Duplicate task: ' + key)
        safe_name(task['case'])
        for field in ('source_prompt', 'target_prompt'):
            if not isinstance(task[field], str):
                raise ValueError('Task prompt must be a string')
        for field in ('pattern_roi', 'detail_roi', 'seeds', 'edit_rubric'):
            if field not in task:
                raise ValueError('Task missing ' + field)
        tasks[key] = task
    ids = set()
    for job in plan['jobs']:
        safe_name(job['id'])
        if job['id'] in ids:
            raise ValueError('Duplicate job: ' + job['id'])
        ids.add(job['id'])
        validate_job(job, tasks)
    if 'planned' in plan and plan['planned'] != len(plan['jobs']):
        raise ValueError('planned must equal len(jobs)')
    return plan, tasks


def fingerprint(root=ROOT):
    root = Path(root)
    plan, _ = load_plan(root)
    code = {p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.suffix in ('.py', '.sh')}
    inputs = {}
    for case in sorted({j['case'] for j in plan['jobs']}):
        folder = root / 'data' / case
        for required in ('source.png', 'outer.png'):
            if not (folder / required).is_file():
                raise FileNotFoundError(folder / required)
        for p in sorted(folder.iterdir()):
            if p.is_file():
                inputs[p.relative_to(root).as_posix()] = sha(p)
    for relative, expected in plan.get('input_sha256', {}).items():
        p = (root / relative).resolve()
        if not p.is_relative_to(root.resolve()) or sha(p) != expected:
            raise ValueError('Input SHA mismatch: ' + relative)
    value = dict(plan_sha256=sha(root / 'plan.json'), code_sha256=code, input_sha256=inputs)
    value['fingerprint'] = digest(canonical(value))
    return value


def guard_existing(root, provenance):
    out, public, _ = locations(root)
    mark = provenance['fingerprint']
    for base in (out, public):
        if not base.exists():
            continue
        manifest = base / 'manifest.json'
        if manifest.exists() and read(manifest).get('fingerprint') != mark:
            raise RuntimeError(f'Fingerprint differs at {base}; use a new build/output directory before continuing.')
        for p in base.glob('*/record.json'):
            if read(p).get('fingerprint') != mark:
                raise RuntimeError(f'Existing result fingerprint differs: {p}; use a new directory.')
        if not manifest.exists() and any(base.iterdir()):
            raise RuntimeError(f'Unidentified existing output at {base}; use a new directory.')


@contextlib.contextmanager
def file_lock(path):
    """Nonblocking process lock; never truncate a lock another runner owns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open('a+b')
    locked = False
    try:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                f.write(b'0'); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            if os.name == 'nt':
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def source_preview(root, case):
    from PIL import Image
    out, _, _ = locations(root)
    path = out / 'sources' / (safe_name(case) + '.jpg')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with Image.open(Path(root) / 'data' / case / 'source.png') as im:
            im.convert('RGB').resize((384, 384), Image.Resampling.LANCZOS).save(path, quality=90)
    return path


def metrics(preview, source, mask_path, roi):
    """Both RGB inputs are saved 384 JPEGs, measured after resize to 512."""
    import numpy as np
    from PIL import Image
    def pixels(path):
        with Image.open(path) as im:
            if im.size != (384, 384) or im.format != 'JPEG':
                raise ValueError('Metrics require saved 384x384 JPEG inputs')
            return np.asarray(im.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS), dtype=np.float64) / 255
    a, b = pixels(preview), pixels(source)
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert('L').resize((512, 512), Image.Resampling.NEAREST), dtype=float) / 255
    diff = np.abs(a - b).mean(-1)
    result = dict(metrics_from='saved_384_JPEG_resized_512', metrics_resolution=512,
                  metrics_status='ok', mae_vs_source=float(diff.mean()))
    for name, weight in (('object', mask), ('bg', 1 - mask)):
        mass = float(weight.sum())
        result[name + '_mae'] = float((diff * weight).sum() / mass) if mass else None
        result[name + '_mae_status'] = 'ok' if mass else 'empty_mask'
    result.update(pattern_ncc_fixed=None, pattern_ncc_shift6=None, pattern_dx=None, pattern_dy=None)
    if not valid_roi(roi):
        result.update(pattern_status='invalid_roi', pattern_fixed_status='invalid_roi', pattern_search_status='invalid_roi')
        return result
    x0, y0, x1, y1 = roi
    ga, gb = a @ np.array([.2126, .7152, .0722]), b @ np.array([.2126, .7152, .0722])
    reference = gb[y0:y1, x0:x1]
    def ncc(x, y):
        x = x.ravel() - x.mean(); y = y.ravel() - y.mean()
        nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
        return float(np.clip(np.dot(x, y) / (nx * ny), -1, 1)) if nx > 1e-8 and ny > 1e-8 else None
    fixed = ncc(ga[y0:y1, x0:x1], reference)
    result['pattern_ncc_fixed'] = fixed
    result['pattern_fixed_status'] = 'ok' if fixed is not None else 'flat_roi'
    best = None
    skipped = 0
    # Tie break favours the smallest displacement; dx/dy shift the output ROI.
    shifts = sorted(((dx, dy) for dy in range(-6, 7) for dx in range(-6, 7)), key=lambda d: (d[0]**2+d[1]**2, d[1], d[0]))
    for dx, dy in shifts:
        if x0 + dx < 0 or y0 + dy < 0 or x1 + dx > 512 or y1 + dy > 512:
            skipped += 1
            continue
        value = ncc(ga[y0+dy:y1+dy, x0+dx:x1+dx], reference)
        if value is not None and (best is None or value > best[0] + 1e-12):
            best = value, dx, dy
    if best is not None:
        result.update(pattern_ncc_shift6=best[0], pattern_dx=best[1], pattern_dy=best[2])
    result.update(pattern_search_status=('boundary_limited' if skipped else 'ok') if best else 'flat_roi',
                  pattern_status='ok' if fixed is not None and best else 'flat_roi',
                  pattern_search_skipped=skipped, pattern_shift_convention='output ROI = source ROI + (dx,dy)')
    return result


def completed(root, job, mark):
    folder = locations(root)[0] / job['id']
    if not (folder / 'record.json').exists():
        return None
    record = read(folder / 'record.json')
    if record.get('fingerprint') != mark:
        raise RuntimeError('Fingerprint mismatch: ' + job['id'])
    if record.get('status') != 'completed':
        return None
    for name in ('preview.jpg', 'pattern.jpg', 'steps.json'):
        p = folder / name
        if not p.is_file() or sha(p) != record.get('artifact_sha256', {}).get(name):
            raise RuntimeError('Completed artifact missing/changed: ' + str(p))
    source = locations(root)[0] / 'sources' / (job['case'] + '.jpg')
    if not source.is_file() or sha(source) != record.get('source_preview_sha256'):
        raise RuntimeError('Completed source preview missing/changed: ' + str(source))
    return record


def rows_for(root, plan, mark):
    out, _, _ = locations(root)
    rows = []
    for job in plan['jobs']:
        record = completed(root, job, mark)
        row = dict(job, status='planned', edit_ok=None, subject_ok=None, pattern_ok=None, artifact=None, judge_status='pending',
                   diagnostic_only=bool(job.get('diagnostic_only') or job.get('cohort') == 'reconstruction' or job['target_prompt'] == job['source_prompt']))
        if record:
            row.update(record['job'])
            row.update(record.get('metrics', {}))
            row.update({k: record.get(k) for k in ('nfe_inversion', 'nfe_edit', 'nfe_total', 'nfe_standalone', 'correction_budget_h', 'gate_budget_h', 'target_nfe', 'source_nfe', 'inversion_cache_hit', 'inversion_owner_job', 'standalone_inversion_nfe', 'seconds', 'generated_width', 'generated_height')})
            row.update(status='completed', preview_sha256=record['preview_sha256'])
            judge_path = out / job['id'] / 'judge.json'
            if judge_path.exists():
                judge = read(judge_path)
                if judge.get('jobid') != job['id'] or judge.get('preview_sha256') != record['preview_sha256'] or judge.get('fingerprint') != mark:
                    raise RuntimeError('Stale judge: ' + job['id'])
                row.update({k: judge[k] for k in ('edit_ok', 'subject_ok', 'pattern_ok', 'artifact', 'judge_status')})
        elif (out / job['id'] / 'error.json').exists():
            error = read(out / job['id'] / 'error.json')
            row.update(status='failed', error=error.get('message', ''))
        rows.append(row)
    return rows


def outcome_summary(rows):
    eligible = [r for r in rows if not r['diagnostic_only']]
    judged = [r for r in eligible if r['status'] == 'completed' and r['judge_status'] == 'done']
    joint = [r for r in judged if r['edit_ok'] == 1 and r['subject_ok'] == 1 and r['pattern_ok'] == 1 and r['artifact'] == 0]
    denominator = len(judged)
    return dict(edit_planned=len(eligible), edit_judged=denominator,
                edit_successes=sum(r['edit_ok'] == 1 for r in judged),
                edit_success_rate=sum(r['edit_ok'] == 1 for r in judged) / denominator if denominator else None,
                joint_successes=len(joint), joint_success_rate=len(joint) / denominator if denominator else None,
                artifact_count=sum(r['artifact'] == 1 for r in judged),
                artifact_rate=sum(r['artifact'] == 1 for r in judged) / denominator if denominator else None,
                rate_denominator_n=denominator,
                diagnostic_planned=sum(r['diagnostic_only'] for r in rows),
                success_rate_denominator='completed non-diagnostic jobs with all four human ratings'), joint


def arm_summaries(rows):
    """All planned cohort/arm groups; descriptive means, no outcome-based arm selection."""
    groups = {}
    for row in rows:
        key = (row.get('cohort', ''), row.get('arm', row.get('label', '')), bool(row['diagnostic_only']))
        groups.setdefault(key, []).append(row)
    summaries = []
    for (cohort, arm, diagnostic), group in sorted(groups.items()):
        completed_rows = [r for r in group if r['status'] == 'completed']
        outcomes, joint = outcome_summary(group)
        summary = dict(cohort=cohort, arm=arm, diagnostic_only=diagnostic,
                       planned=len(group), completed=len(completed_rows),
                       failed=sum(r['status'] == 'failed' for r in group),
                       pending=sum(r['status'] == 'planned' for r in group),
                       judged=sum(r['judge_status'] == 'done' for r in completed_rows), **outcomes,
                       metrics_from='saved_384_JPEG_resized_512',
                       ncc_denominator='finite measured values in named population; missing/flat ROI excluded, never zero-filled')
        for population, members in (('completed', completed_rows), ('joint_success', joint)):
            summary['ncc_' + population + '_population_n'] = len(members)
            for metric in ('pattern_ncc_fixed', 'pattern_ncc_shift6'):
                values = [r[metric] for r in members if type(r.get(metric)) in (int, float) and math.isfinite(r[metric])]
                prefix = metric + '_' + population
                summary[prefix + '_mean'] = sum(values) / len(values) if values else None
                summary[prefix + '_n'] = len(values)
                summary[prefix + '_missing_n'] = len(members) - len(values)
        summaries.append(summary)
    return summaries


def contact_sheets(out, plan, rows):
    """Source plus completed, saved previews in plan order; no pending image tiles."""
    from PIL import Image, ImageDraw, ImageFont
    out = Path(out)
    directory = out / 'sheets'
    paths = []
    font = ImageFont.load_default()
    ascii_text = lambda value: str(value).encode('ascii', 'replace').decode('ascii')
    for task in plan['tasks']:
        for seed in task['seeds']:
            completed_rows = [r for r in rows if r['task'] == task['key'] and r['seed'] == seed and r['status'] == 'completed']
            if not completed_rows:
                continue
            cells = [('SOURCE', out / 'sources' / (task['case'] + '.jpg'))]
            references = [r for r in plan.get('reference_results', []) if r['task'] == task['key'] and r['seed'] == seed]
            cells += [('R71:'+r['arm'], out / r['id'] / 'preview.jpg') for r in references]
            cells += [(r.get('arm', r.get('label', r['id'])), out / r['id'] / 'preview.jpg') for r in completed_rows]
            columns, cell, label_h, header_h = 4, 256, 22, 26
            height = header_h + ((len(cells) + columns - 1) // columns) * (cell + label_h)
            sheet = Image.new('RGB', (columns * cell, height), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((6, 6), ascii_text(f"{task['key']} | seed {seed}"), font=font, fill='black')
            for i, (label, path) in enumerate(cells):
                x, y = (i % columns) * cell, header_h + (i // columns) * (cell + label_h)
                text = ascii_text(label)
                while text and draw.textbbox((0, 0), text, font=font)[2] > cell - 12:
                    text = text[:-1]
                draw.text((x + 6, y + 4), text, font=font, fill='black')
                with Image.open(path) as image:
                    sheet.paste(image.convert('RGB').resize((cell, cell), Image.Resampling.LANCZOS), (x, y + label_h))
            directory.mkdir(exist_ok=True)
            path = directory / (safe_name(task['key']) + '_s' + str(seed) + '.jpg')
            temp = path.with_suffix('.jpg.tmp')
            sheet.save(temp, format='JPEG', quality=85, optimize=True)
            os.replace(temp, path)
            paths.append(path)
    return paths


def write_review(out, plan, rows, mark):
    tasks = {t['key']: t for t in plan['tasks']}
    items = []
    for row in sorted(rows, key=lambda r: digest(mark + r['id'])):
        if row['status'] != 'completed':
            continue
        record = read(out / row['id'] / 'record.json')
        job = record['job']
        items.append(dict(jobid=job['id'], preview_sha256=row['preview_sha256'], source='../sources/'+job['case']+'.jpg',
                          preview='../'+job['id']+'/preview.jpg', source_prompt=job['source_prompt'], target_prompt=job['target_prompt'],
                          rubric=tasks[job['task']]['edit_rubric'], **{k:row[k] for k in ('edit_ok','subject_ok','pattern_ok','artifact')}))
    payload = json.dumps(items, ensure_ascii=False).replace('<', '\\u003c')
    page = '''<!doctype html><meta charset="utf-8"><title>R72 manual review</title>
<style>body{font:16px sans-serif;margin:24px;background:#f5f5f5}article{background:white;padding:20px;margin:16px 0}img{width:384px;max-width:46vw}select{margin:12px}button{padding:12px;position:sticky;top:5px}pre{white-space:pre-wrap}</style>
<h1>R72 manual review</h1><p>Arm labels are hidden. 0 = no, 1 = yes. Artifact: 1 = visible artifact. Unset ratings remain pending.</p>
<button id="export">Export review JSON</button><main></main><script>
const items=PAYLOAD, fingerprint=MARK;
items.forEach((item,i)=>{const a=document.createElement('article');const h=document.createElement('h2');h.textContent='Sample '+(i+1);a.append(h);
const p=document.createElement('pre');p.textContent='Source: '+item.source_prompt+'\\nRequested target: '+item.target_prompt+'\\nRubric: '+JSON.stringify(item.rubric);a.append(p);
for(const [key,label] of [['source','Source'],['preview','Output']]){const img=document.createElement('img');img.src=item[key];img.alt=label;img.title=label;a.append(img);}
for(const key of ['edit_ok','subject_ok','pattern_ok','artifact']){const l=document.createElement('label');l.textContent=key;const s=document.createElement('select');for(const [v,t] of [['','Pending'],['0','0'],['1','1']]){const o=document.createElement('option');o.value=v;o.textContent=t;s.append(o);}s.value=item[key]===null?'':String(item[key]);s.onchange=()=>item[key]=s.value===''?null:Number(s.value);l.append(s);a.append(l);}document.querySelector('main').append(a);});
document.querySelector('#export').onclick=()=>{const reviews=items.map(({jobid,preview_sha256,edit_ok,subject_ok,pattern_ok,artifact})=>({jobid,preview_sha256,edit_ok,subject_ok,pattern_ok,artifact}));const blob=new Blob([JSON.stringify({schema:'r72-human-review-v1',fingerprint,reviews},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='R72_review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
</script>'''.replace('MARK', json.dumps(mark)).replace('PAYLOAD', payload)
    (out / 'review').mkdir(exist_ok=True)
    (out / 'review' / 'index.html').write_text(page, encoding='utf-8')


def package(root=ROOT, plan=None, provenance=None):
    root = Path(root)
    plan = plan if plan is not None else load_plan(root)[0]
    provenance = provenance if provenance is not None else fingerprint(root)
    guard_existing(root, provenance)
    out, public, archive = locations(root)
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'manifest.json', provenance)
    mark = provenance['fingerprint']
    for case in {j['case'] for j in plan['jobs']}:
        source_preview(root, case)
    references = plan.get('reference_results', [])
    for ref in references:
        for name in ('preview.jpg', 'pattern.jpg', 'record.json'):
            destination = out / ref['id'] / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(root / ref['id'] / name, destination)
    rows = rows_for(root, plan, mark)
    sheets = contact_sheets(out, plan, rows)
    summary = {status: sum(r['status'] == status for r in rows) for status in ('completed', 'failed', 'planned')}
    summary.update(reused_reference_images=len(references), total_planned=len(plan['jobs']), pending=summary['planned'], fingerprint=mark)
    outcomes, _ = outcome_summary(rows)
    summary.update(outcomes)
    save(out / 'summary.json', summary)
    arms = arm_summaries(rows)
    with (out / 'arm_summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(arms[0]) if arms else ['cohort', 'arm', 'planned', 'completed', 'judged'])
        writer.writeheader()
        writer.writerows(arms)
    save(out / 'plan.json', plan)
    columns = ['id', 'task', 'case', 'label', 'seed', 'cohort', 'status', 'edit_ok', 'subject_ok', 'pattern_ok', 'artifact', 'judge_status']
    columns += sorted(set().union(*(r.keys() for r in rows)) - set(columns)) if rows else []
    with (out / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    compared = [dict(r, origin_round='R72') for r in rows] + references
    compared_columns = sorted(set().union(*(r.keys() for r in compared)))
    with (out / 'comparison_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=compared_columns)
        writer.writeheader()
        for row in compared:
            writer.writerow({k: canonical(v) if isinstance(v, (dict, list)) else v for k,v in row.items()})
    e = html.escape
    from urllib.parse import quote
    cards = []
    for row in rows:
        title = f"{row['id']} | {row.get('label', '')} | {row['status']}"
        imgs = f'<img src="sources/{quote(row["case"])}.jpg" alt="source">'
        if row['status'] == 'completed':
            imgs += f'<img src="{quote(row["id"])}/preview.jpg" alt="output"><img src="{quote(row["id"])}/pattern.jpg" alt="native pattern crop">'
        cards.append('<article><h3>'+e(title)+'</h3><p>'+e(row['target_prompt'])+'</p>'+imgs+'</article>')
    sheet_links = ''.join('<li><a href="sheets/'+quote(p.name)+'">'+e(p.stem)+'</a></li>' for p in sheets)
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>R72 results</title><style>body{font:16px sans-serif;margin:24px}img{max-width:384px;max-height:384px}article{border-top:1px solid #aaa;padding:12px}</style><h1>R72 TraceMemory</h1><p>'+e(canonical(summary))+'</p><a href="review/index.html">Manual review (arms hidden)</a><ul>'+sheet_links+'</ul>'+''.join(cards), encoding='utf-8')
    write_review(out, plan, rows, mark)
    public.mkdir(parents=True, exist_ok=True)
    # Allowlist: only small deliverables, never latent/trajectory/full-size images.
    paths = [out / n for n in ('manifest.json', 'plan.json', 'summary.json', 'metrics.csv', 'arm_summary.csv', 'comparison_metrics.csv', 'index.html', 'progress.json') if (out / n).exists()]
    paths += list((out / 'sources').glob('*.jpg')) + [out / 'review' / 'index.html']
    paths += sheets
    for job in plan['jobs']:
        paths += [out / job['id'] / n for n in ('preview.jpg', 'pattern.jpg', 'record.json', 'steps.json', 'error.json', 'judge.json') if (out / job['id'] / n).exists()]
    for ref in references:
        paths += [out / ref['id'] / n for n in ('preview.jpg', 'pattern.jpg', 'record.json')]
    for p in paths:
        target = public / p.relative_to(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    # A recovered job must not retain an old error in the public directory.
    for row in rows:
        if row['status'] == 'completed':
            (public / row['id'] / 'error.json').unlink(missing_ok=True)
    temp = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in paths:
            z.write(p, p.relative_to(out).as_posix())
    os.replace(temp, archive)
    print('PACK '+str(archive)+' '+canonical(summary), flush=True)
    return summary
