"""Build a frozen, local CSV benchmark from this package; no model execution."""
import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
SIZE = (512, 512)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_image(path, mode, resampling):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert(mode).resize(SIZE, resampling)


def pixel_hash(image):
    return hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()


def roi(value):
    if not value:
        return [0, 0, 512, 512]
    result = json.loads(value)
    if not (isinstance(result, list) and len(result) == 4 and
            all(type(x) is int for x in result) and
            0 <= result[0] < result[2] <= 512 and 0 <= result[1] < result[3] <= 512):
        raise ValueError('ROI must be [x0,y0,x1,y1] in 512 coordinates')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--arms', default='C50,SPLIT50,FLOWEDIT')
    parser.add_argument('--seeds', default='0,17')
    args = parser.parse_args()
    manifest = args.manifest.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output == ROOT or output.is_relative_to(ROOT):
        raise ValueError('Output must be a new directory outside this package')
    if not output.parent.is_dir():
        raise ValueError('Output parent must already exist')
    arms = [x.strip() for x in args.arms.split(',') if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]
    if not arms or len(set(arms)) != len(arms) or not seeds or len(set(seeds)) != len(seeds) or any(x < 0 for x in seeds):
        raise ValueError('Specify unique arms and nonnegative unique seeds')
    plan = json.loads((ROOT / 'plan.json').read_text(encoding='utf-8-sig'))
    templates = {}
    for job in plan['jobs']:
        if job.get('arm') in arms and job['arm'] not in templates:
            templates[job['arm']] = job
    missing = set(arms) - templates.keys()
    if missing:
        raise ValueError('Missing arm templates: ' + ', '.join(sorted(missing)))
    source_cases = {str(t['case']) for t in plan['tasks']}
    if len(source_cases) != 6:
        raise ValueError('Source plan must contain its six frozen cases before importing')
    source_pixels = set()
    for case in source_cases:
        source = ROOT / 'data' / case / 'source.png'
        source_pixels.add(pixel_hash(load_image(source, 'RGB', Image.Resampling.LANCZOS)))
    required = {'case', 'image', 'source_prompt', 'target_prompt', 'edit_rubric'}
    rows, seen, test_pixels = [], set(), set()
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Manifest requires columns: ' + ', '.join(sorted(required)))
        for line, row in enumerate(reader, 2):
            row = {k: (v or '').strip() for k, v in row.items() if k is not None}
            if any(not row.get(k) for k in required):
                raise ValueError(f'line {line}: required field is empty')
            case = row['case']
            if not re.fullmatch(r'[0-9]+', case):
                raise ValueError(f'line {line}: case must contain only ASCII digits')
            canonical = str(int(case))
            if canonical in seen or any(canonical == str(int(c)) for c in source_cases):
                raise ValueError(f'line {line}: duplicate or source case ID {case}')
            seen.add(canonical)
            image_path = (manifest.parent / row['image']).resolve(strict=True)
            image = load_image(image_path, 'RGB', Image.Resampling.LANCZOS)
            pixels = pixel_hash(image)
            if pixels in source_pixels:
                raise ValueError(f'line {line}: image overlaps source pixels after resize')
            if pixels in test_pixels:
                raise ValueError(f'line {line}: duplicate source image within manifest')
            test_pixels.add(pixels)
            mask_name = row.get('object_mask', '')
            mask = load_image((manifest.parent / mask_name).resolve(strict=True), 'L', Image.Resampling.NEAREST) if mask_name else Image.new('L', SIZE, 0)
            region = row.get('eval_pixel_region', '') or ('object' if mask_name else 'none')
            if region not in ('object', 'background', 'none') or (not mask_name and region != 'none'):
                raise ValueError(f'line {line}: invalid eval_pixel_region or no object_mask')
            pattern, detail = roi(row.get('pattern_roi', '')), roi(row.get('detail_roi', ''))
            rows.append((row, image, mask, pattern, detail, region))
    if not rows:
        raise ValueError('Manifest contains no cases')
    new_plan = copy.deepcopy(plan)
    new_plan['name'] = 'Public benchmark frozen from ' + plan.get('name', 'R74')
    new_plan['tasks'], new_plan['jobs'], new_plan['input_sha256'] = [], [], {}
    task_template = plan['tasks'][0]
    for row, image, mask, pattern, detail, region in rows:
        case = row['case']
        key = 'case_' + case
        fields = dict(case=case, source_prompt=row['source_prompt'], target_prompt=row['target_prompt'],
                      pattern_roi=pattern, detail_roi=detail, edit_rubric=row['edit_rubric'],
                      roi_scope='whole_image' if not row.get('pattern_roi') else 'specified',
                      eval_pixel_region=region)
        task = copy.deepcopy(task_template)
        task.update(fields, key=key, task=key, seeds=seeds, diagnostic_only=False)
        new_plan['tasks'].append(task)
        for seed in seeds:
            for arm in arms:
                job = copy.deepcopy(templates[arm])
                job.update(fields, task=key, seed=seed, arm=arm, label=arm,
                           id=f'benchmark_{case}_s{seed}_{arm}', cohort='public_benchmark', diagnostic_only=False)
                new_plan['jobs'].append(job)
    new_plan['planned'] = len(new_plan['jobs'])
    new_plan['round_id'] = 'R74_Benchmark'
    new_plan['output_name'] = output.name
    new_plan['design'] = dict(stage='frozen_independent_test', independent_source_images=len(rows),
        edit_tasks=len(rows), arms=arms, seeds=seeds, outputs=len(new_plan['jobs']),
        manifest_sha256=digest(manifest), parent_plan_sha256=digest(ROOT/'plan.json'))
    new_plan['preregistration']['selection'] = 'Configuration frozen from development before generation; no test-set selection.'
    output.mkdir(exist_ok=False)
    for p in ROOT.iterdir():
        if p.is_file() and (p.suffix.lower() in ('.py', '.sh', '.md') or p.name.startswith('requirements') or p.name == 'LICENSE'):
            shutil.copy2(p, output / p.name)
    for row, image, mask, _, _, _ in rows:
        folder = output / 'data' / row['case']
        folder.mkdir(parents=True)
        image.save(folder / 'source.png')
        mask.save(folder / 'outer.png')
        Image.new('L', SIZE, 255).save(folder / 'full_mask.png')
        Image.new('L', SIZE, 255).save(folder / 'core.png')
        for p in folder.iterdir():
            new_plan['input_sha256'][p.relative_to(output).as_posix()] = digest(p)
    (output / 'plan.json').write_text(json.dumps(new_plan, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Created {output}: {len(rows)} cases, {len(new_plan['jobs'])} jobs. Set a unique OUTPUT_NAME before running.")


if __name__ == '__main__':
    main()
