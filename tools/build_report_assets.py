"""Build report tables and figures from the frozen R71-R73 records (CPU only)."""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import csv
import json

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'docs' / 'tables'
FIGURES = ROOT / 'docs' / 'figures'
METRICS = ('mae_vs_source', 'object_mae', 'bg_mae',
           'pattern_ncc_fixed', 'pattern_ncc_shift6', 'correction_impulse')


def read_rows(round_id, filename='metrics.csv'):
    with (ROOT / 'results' / round_id / filename).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_rows(name, rows):
    with (TABLES / name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def means(rows):
    groups = defaultdict(list)
    for row in rows:
        if row['status'] == 'completed':
            groups[row['task'], row['arm']].append(row)
    out = []
    for (task, arm), items in groups.items():
        result = dict(task=task, arm=arm, n=len(items))
        for name in METRICS:
            values = [float(r[name]) for r in items if r.get(name) not in ('', None)]
            result[name] = mean(values) if values else ''
        out.append(result)
    return out


def ncc(a, b):
    weights = np.array([.2126, .7152, .0722])
    a, b = np.asarray(a, dtype=np.float64) @ weights, np.asarray(b, dtype=np.float64) @ weights
    a, b = (a - a.mean()).ravel(), (b - b.mean()).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denominator, -1, 1)) if denominator > 1e-8 else None


def native_crops(all_rows):
    """Direct 512-grid JPEG crop vs original PNG source crop; no registration."""
    outputs, sources = [], {}
    experiment = {'R71': 'R71_TraceMemory', 'R72': 'R72_TimeBudget', 'R73': 'R73_ResidualBudget'}
    for round_id, rows in all_rows.items():
        for row in rows:
            folder = ROOT / 'results' / round_id / row['id']
            record = json.loads((folder / 'record.json').read_text(encoding='utf-8'))
            w, h = record['generated_width'], record['generated_height']
            key = (round_id, row['case'], w, h)
            if key not in sources:
                path = ROOT / 'experiments' / experiment[round_id] / 'data' / row['case'] / 'source.png'
                with Image.open(path) as image:
                    sources[key] = image.convert('RGB').resize((w, h), Image.Resampling.LANCZOS)
            box = record['pattern_crop_box_native']
            reference = sources[key].crop(box)
            with Image.open(folder / 'pattern.jpg') as image:
                crop = image.convert('RGB')
                if crop.size != reference.size:
                    raise ValueError(f"Native crop size differs: {row['id']}")
                value = ncc(crop, reference)
            outputs.append(dict(round=round_id, id=row['id'], task=row['task'], arm=row['arm'], seed=row['seed'],
                                native_crop_ncc='' if value is None else value,
                                preview_fixed_ncc=row['pattern_ncc_fixed'],
                                width=crop.width, height=crop.height,
                                crop_encoding='JPEG quality 92', reference_encoding='original PNG',
                                status='flat_roi' if value is None else 'ok'))
    write_rows('native_crop_sensitivity.csv', outputs)
    groups = defaultdict(list)
    for row in outputs:
        groups[row['round'], row['task'], row['arm']].append(row)
    summary = []
    for (round_id, task, arm), rows in groups.items():
        values = [float(r['native_crop_ncc']) for r in rows if r['status'] == 'ok']
        summary.append(dict(round=round_id, task=task, arm=arm, n=len(rows),
                            native_crop_ncc=mean(values) if values else '',
                            preview_fixed_ncc=mean(float(r['preview_fixed_ncc']) for r in rows)))
    write_rows('native_crop_summary.csv', summary)
    return outputs


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    all_rows = {r: read_rows(r) for r in ('R71', 'R72', 'R73')}
    for round_id, rows in all_rows.items():
        write_rows(round_id.lower() + '_task_means.csv', means(rows))
    compared = read_rows('R73', 'comparison_metrics.csv')
    write_rows('r73_comparison_task_means.csv', means(compared))
    costs = [dict(id=r['id'], task=r['task'], arm=r['arm'], seed=r['seed'],
                  editing_nfe=r['nfe_edit'], inversion_nfe_charged=r['nfe_inversion'],
                  inversion_cache_hit=r['inversion_cache_hit'], actual_nfe=r['nfe_total'],
                  standalone_nfe=r['nfe_standalone'], seconds=r['seconds']) for r in all_rows['R73']]
    write_rows('r73_compute.csv', costs)
    lookup = {(r['task'], r['seed'], r['arm']): r for r in compared}
    paired = []
    for row in all_rows['R73']:
        c50 = lookup[row['task'], row['seed'], 'C50']
        paired.append(dict(id=row['id'], task=row['task'], seed=row['seed'], arm=row['arm'],
                           coefficient_budget=row['correction_budget_h'],
                           residual_budget=row['correction_impulse'], target_budget=row['correction_budget_target'],
                           relative_error=row['correction_budget_relative_error'],
                           ncc_shift6=row['pattern_ncc_shift6'],
                           delta_shift6_vs_c50=float(row['pattern_ncc_shift6'])-float(c50['pattern_ncc_shift6'])))
    write_rows('r73_paired_budget.csv', paired)
    arms = ('C_EARLY', 'BUDGET_EARLY', 'C50', 'BUDGET_LATE', 'C_LATE')
    stats = []
    for arm in arms:
        rows = [r for r in compared if r['arm'] == arm]
        stats.append(dict(arm=arm, n=len(rows),
                          coefficient_budget=mean(float(r['correction_budget_h']) for r in rows),
                          residual_budget=mean(float(r['correction_impulse']) for r in rows),
                          paired_budget_ratio=mean(float(r['correction_impulse'])/float(lookup[r['task'],r['seed'],'C50']['correction_impulse']) for r in rows),
                          ncc_shift6=mean(float(r['pattern_ncc_shift6']) for r in rows)))
    write_rows('budget_timing.csv', stats)
    native_crops(all_rows)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
    labels = ['Coefficient\nearly', 'Residual\nearly', 'Constant', 'Residual\nlate', 'Coefficient\nlate']
    colors = ['#899aa9', '#276fbf', '#383f51', '#d17b28', '#a4a6aa']
    for ax, key, title, ylabel in [(axs[0], 'paired_budget_ratio', '(a) Actual residual budget', 'Budget / paired constant budget'), (axs[1], 'ncc_shift6', '(b) Pattern correlation', 'Mean shift-compensated NCC')]:
        values = [s[key] for s in stats]
        ax.bar(range(5), values, color=colors, width=.67)
        ax.set_xticks(range(5), labels)
        ax.set_ylabel(ylabel); ax.set_title(title, loc='left', fontweight='bold')
        ax.set_ylim(0, max(values)*1.19)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
        for i, value in enumerate(values): ax.text(i, value+.015*max(values), f'{value:.3f}', ha='center', fontsize=9)
    axs[0].axhline(1, color='#505050', linestyle='--', linewidth=1)
    fig.savefig(FIGURES/'budget_timing.svg')
    fig.savefig(FIGURES/'budget_timing.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'new_outputs':sum(len(v) for v in all_rows.values()), 'budget_timing':stats, 'native_crops':166}, indent=2))


if __name__ == '__main__':
    main()
