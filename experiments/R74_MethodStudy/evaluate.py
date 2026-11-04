"""Offline evaluation of completed R74 results.

Prepare weights separately on a networked machine, then copy both directories:
  huggingface-cli download openai/clip-vit-base-patch32 --local-dir /path/to/clip
  huggingface-cli download facebook/dinov2-base --local-dir /path/to/dinov2
Run: python evaluate.py --from-results /path/to/results --clip-path /path/to/clip --dino-path /path/to/dinov2 --device cpu
The results folder needs plan.json, job folders, sources/*_source_eval.png and,
for background metrics, either data/{case}/outer.png beside results or --data-dir.
All models load locally. No network access, generation, or NFE measurement.
Pixel and feature similarities are reference-difference diagnostics, not edit success.
Set each job eval_pixel_region explicitly to object, background, or none.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, AutoModel


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def safe(value):
    value = str(value)
    if not value or value in ('.', '..') or '/' in value or '\\' in value:
        raise ValueError('Invalid path component: ' + value)
    return value


def image(path):
    with Image.open(path) as im:
        return im.convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)


def cosine(a, b):
    a, b = a.flatten(), b.flatten()
    denom = a.norm() * b.norm()
    return float((a @ b / denom).item()) if denom.item() > 0 else None


def features(model, processor, images, device):
    inputs = processor(images=images, return_tensors='pt').to(device)
    output = model(**inputs)
    return output.last_hidden_state[:, 0, :].float()


def clip_features(model, processor, imgs, texts, device):
    ip = processor(images=imgs, return_tensors='pt').to(device)
    tp = processor(text=texts, padding=True, truncation=True, return_tensors='pt').to(device)
    return model.get_image_features(**ip).float(), model.get_text_features(**tp).float()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-results', type=Path, required=True)
    p.add_argument('--clip-path', type=Path, required=True)
    p.add_argument('--dino-path', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--output', type=Path, help='New CSV path; default results/evaluation.csv')
    args = p.parse_args()
    root = args.from_results.resolve()
    out = args.output or root / 'evaluation.csv'
    plan = read(root / 'plan.json')
    tasks = {t['key']: t for t in plan['tasks']}
    data = args.data_dir or root.parent / 'data'
    for path in (args.clip_path, args.dino_path):
        if not path.is_dir():
            p.error('Local model directory does not exist: ' + str(path))
    start = time.perf_counter()
    clip = CLIPModel.from_pretrained(args.clip_path, local_files_only=True).to(args.device).eval()
    cp = CLIPProcessor.from_pretrained(args.clip_path, local_files_only=True)
    dino = AutoModel.from_pretrained(args.dino_path, local_files_only=True).to(args.device).eval()
    dp = AutoImageProcessor.from_pretrained(args.dino_path, local_files_only=True)
    load_s = time.perf_counter() - start
    existing = {}
    if (root / 'metrics.csv').exists():
        with (root / 'metrics.csv').open(encoding='utf-8-sig', newline='') as f:
            existing = {r['id']: r for r in csv.DictReader(f) if r.get('id')}
    rows = []
    with torch.inference_mode():
        for job in plan['jobs']:
            ident, case = safe(job['id']), safe(job['case'])
            task = tasks.get(job.get('task'), {})
            row = dict(existing.get(ident, {}))
            row.update(id=ident, case=case, eval_status='pending', eval_error='',
                       eval_clip_model=str(args.clip_path.resolve()), eval_dino_model=str(args.dino_path.resolve()),
                       eval_device=args.device, eval_model_load_seconds=load_s,
                       eval_source_path='', eval_raw_path='', eval_seconds='',
                       clip_target_similarity='', clip_source_similarity='', clip_directional_similarity='',
                       dino_full_feature_similarity='', dino_pattern_crop_feature_similarity='',
                       pattern_crop_status='', pixel_scope='', pixel_mse='', pixel_psnr_db='', pixel_ssim='', pixel_status='')
            folder = root / ident
            record_path, raw = folder / 'record.json', folder / 'raw.png'
            source = root / 'sources' / (case + '_source_eval.png')
            if not record_path.exists() or not raw.exists() or not source.exists():
                row['eval_status'] = 'missing_record_or_image'
                rows.append(row); continue
            tick = time.perf_counter()
            try:
                record = read(record_path)
                if record.get('status') != 'completed':
                    row['eval_status'] = 'generation_' + str(record.get('status', 'unknown'))
                    rows.append(row); continue
                for key, value in record.get('metrics', {}).items():
                    if key not in row and not isinstance(value, (dict, list)):
                        row[key] = value
                src, edited = image(source), image(raw)
                row.update(eval_source_path=str(source), eval_raw_path=str(raw))
                source_prompt = job.get('source_prompt', task.get('source_prompt', ''))
                target_prompt = job.get('target_prompt', task.get('target_prompt', ''))
                if not source_prompt or not target_prompt:
                    raise ValueError('Missing source or target prompt')
                im, tx = clip_features(clip, cp, [src, edited], [source_prompt, target_prompt], args.device)
                row['clip_source_similarity'] = cosine(im[1], tx[0])
                row['clip_target_similarity'] = cosine(im[1], tx[1])
                # Directional CLIP uses normalized image and text embeddings before differencing.
                im = torch.nn.functional.normalize(im, dim=-1)
                tx = torch.nn.functional.normalize(tx, dim=-1)
                direction = cosine(im[1] - im[0], tx[1] - tx[0])
                row['clip_directional_similarity'] = '' if direction is None else direction
                row['clip_directional_status'] = 'ok' if direction is not None else 'zero_direction'
                emb = features(dino, dp, [src, edited], args.device)
                row['dino_full_feature_similarity'] = cosine(emb[0], emb[1])
                roi = job.get('pattern_roi', task.get('pattern_roi'))
                if (isinstance(roi, (list, tuple)) and len(roi) == 4 and all(type(x) is int for x in roi)
                    and 0 <= roi[0] < roi[2] <= 512 and 0 <= roi[1] < roi[3] <= 512):
                    crops = features(dino, dp, [src.crop(roi), edited.crop(roi)], args.device)
                    row['dino_pattern_crop_feature_similarity'] = cosine(crops[0], crops[1])
                    row['pattern_crop_status'] = 'ok'
                else:
                    row['pattern_crop_status'] = 'invalid_roi'
                # Pixel differences measure change from the reference, not edit success.
                region = job.get('eval_pixel_region')
                a, b = np.asarray(src, dtype=np.float64) / 255, np.asarray(edited, dtype=np.float64) / 255
                weights = None
                if region == 'none':
                    row['pixel_scope'] = 'none'
                    row['pixel_status'] = 'not_applicable'
                elif region in ('object', 'background'):
                    row['pixel_scope'] = region
                    mask_path = data / case / 'outer.png'
                    if not mask_path.exists():
                        row['pixel_status'] = 'missing_mask'
                    else:
                        with Image.open(mask_path) as mask:
                            weights = np.asarray(mask.convert('L').resize((512, 512), Image.Resampling.NEAREST), dtype=np.float64) / 255
                        if region == 'background':
                            weights = 1 - weights
                else:
                    row['pixel_status'] = 'invalid_or_missing_eval_pixel_region'
                if weights is not None:
                    mass = weights.sum()
                    if mass <= 0:
                        row['pixel_status'] = 'empty_region'
                    else:
                        mse = float((((a - b) ** 2).mean(axis=2) * weights).sum() / mass)
                        row['pixel_mse'] = mse
                        row['pixel_psnr_db'] = 'inf' if mse == 0 else -10 * math.log10(mse)
                        row['pixel_status'] = 'ok'
                        try:
                            from skimage.metrics import structural_similarity
                            _, ssim_map = structural_similarity(a, b, data_range=1.0, channel_axis=2, full=True)
                            row['pixel_ssim'] = float((ssim_map.mean(axis=2) * weights).sum() / mass)
                            row['pixel_ssim_status'] = 'ok_local_windows_may_cross_mask'
                        except ImportError:
                            row['pixel_ssim_status'] = 'skimage_unavailable'
                row['eval_status'] = 'ok'
            except Exception as exc:
                row['eval_status'] = 'error'; row['eval_error'] = str(exc)
            row['eval_seconds'] = round(time.perf_counter() - tick, 4)
            rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    print('Wrote', out, 'jobs:', len(rows), 'evaluated:', sum(r['eval_status'] == 'ok' for r in rows))


if __name__ == '__main__':
    main()
