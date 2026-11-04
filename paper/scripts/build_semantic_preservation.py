"""Plot stored marginal means/CIs and archived request maps; never infer joint CIs."""
from pathlib import Path
import csv
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
ARCHIVE = BASE.parent / 'results'
STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
         'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5,
         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .6,
         'savefig.bbox': None, 'figure.facecolor': 'white'}
# Okabe-Ito hues plus redundant marker shapes and direct labels.
CONFIGS = {
    'RELEASE_CORR15': ('λ = .15', '#0072B2', 'o'),
    'RELEASE_CORR25': ('CORR', '#0072B2', 'o'),
    'RELEASE_CORR35': ('λ = .35', '#0072B2', 'o'),
    'RELEASE_BOTH25': ('BOTH', '#56B4E9', 's'),
    'ENERGY_MATCH25': ('Energy', '#555555', 'D'),
    'U25_KV': ('U25', '#009E73', 'v'),
    'FLOWEDIT_N26': ('Flow26', '#D55E00', '^'),
    'FLOWEDIT_N28': ('Flow28', '#D55E00', '^'),
    'ETA03': ('ETA', '#000000', 'P'),
    'RFEDIT_I08': ('RF8', '#CC79A7', 'X'),
}


def summary(root):
    path = root / 'evidence/overnight_arm_summary.csv'
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return path, {(r['arm'], r['metric']): r for r in csv.DictReader(handle)}


def values(rows, arm, metric):
    row = rows[arm, metric]
    return tuple(float(row[k]) for k in ('mean', 'ci95_low', 'ci95_high'))


def semantic_figure(r84, r85):
    sources = []
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6))
    fig.subplots_adjust(left=.085, right=.98, bottom=.29, top=.82, wspace=.34)
    panels = [
        (r84, ['RELEASE_CORR15', 'RELEASE_CORR25', 'RELEASE_CORR35',
               'ENERGY_MATCH25', 'FLOWEDIT_N28'], '(a) Dose study: 10 sources'),
        (r85, ['RELEASE_CORR25', 'RELEASE_BOTH25', 'ENERGY_MATCH25', 'U25_KV',
               'FLOWEDIT_N26', 'ETA03', 'RFEDIT_I08'], '(b) Fixed configurations: 40 sources'),
    ]
    # Label positions are presentation-only; points and marginal CIs remain unchanged.
    positions = [
        {'RELEASE_CORR15': (.012, .666), 'RELEASE_CORR25': (.012, .585),
         'RELEASE_CORR35': (.012, .348), 'ENERGY_MATCH25': (.174, .414),
         'FLOWEDIT_N28': (.205, .654)},
        {'ETA03': (.174, .654), 'RELEASE_BOTH25': (.174, .605),
         'ENERGY_MATCH25': (.174, .556), 'RELEASE_CORR25': (.174, .507),
         'U25_KV': (.174, .458), 'FLOWEDIT_N26': (.174, .387),
         'RFEDIT_I08': (.174, .258)},
    ]
    for index, (ax, (root, arms, title)) in enumerate(zip(axes, panels)):
        path, rows = summary(root)
        sources.append(str(path))
        points = {}
        for arm in arms:
            x, xl, xh = values(rows, arm, 'clip_directional_similarity')
            y, yl, yh = values(rows, arm, 'lpips_preserved')
            label, color, marker = CONFIGS[arm]
            ax.errorbar(x, y, xerr=[[x-xl], [xh-x]], yerr=[[y-yl], [yh-y]],
                        fmt='none', ecolor=color, alpha=.42, elinewidth=.7,
                        capsize=2, capthick=.7, zorder=2)
            ax.scatter([x], [y], s=34, c=color, marker=marker, edgecolors='white',
                       linewidths=.5, zorder=4)
            if index == 0 and arm == 'RELEASE_CORR25':
                label = 'λ = .25'
            ax.annotate(label, xy=(x, y), xytext=positions[index][arm],
                        textcoords='data', fontsize=8.5, color='#20252A', va='center',
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .6},
                        arrowprops={'arrowstyle': '-', 'color': color, 'lw': .65},
                        zorder=6)
            points[arm] = (x, y)
        if index == 0:
            dose = [points[a] for a in ['RELEASE_CORR15', 'RELEASE_CORR25', 'RELEASE_CORR35']]
            ax.plot([q[0] for q in dose], [q[1] for q in dose], color='#0072B2', lw=1.0,
                    zorder=3)
        ax.set_title(title, loc='left', weight='bold', pad=21)
        ax.text(0, 1.04, 'LPIPS: 8 nonempty sources' if index == 0 else 'LPIPS: 33 nonempty sources',
                transform=ax.transAxes, fontsize=8.5, color='#555555')
        ax.set_xlim(0, .265)
        ax.set_ylim(.18, .70)
        ax.set_xticks([0, .05, .10, .15, .20, .25])
        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.set_xlabel('Directional CLIP ↑', labelpad=5)
        ax.set_ylabel('Preservation LPIPS ↓', labelpad=5)
        ax.grid(color='#E5E7E9', lw=.45)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(.085, .145,
             'Points: equal-source means. Crossbars: stored marginal 95% bootstrap CIs on each axis.',
             fontsize=8.5)
    fig.text(.085, .090,
             'Only the three same-population λ settings are connected; fixed configurations are separate points.',
             fontsize=8.5)
    fig.text(.085, .035,
             'The panels use different source populations; crossbars are not joint confidence regions.', fontsize=8.5)
    description = ('Directional CLIP (higher) versus preservation LPIPS (lower). Separate R84 dose '
                   'and R85 independent populations. CLIP n=10/40; LPIPS n=8/33 because empty '
                   'preservation masks are excluded. Original source-equal means and marginal 95% '
                   'bootstrap intervals, no joint interval or frontier estimation. Only R84 CORR '
                   'lambda15/25/35 joined. R85 RFEDIT seed0 only. Input tables: ' + '; '.join(sources))
    save(fig, 'semantic_preservation', description)


def save(fig, name, description):
    out = BASE / 'figures' / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'), metadata={'Title': name, 'Subject': description})
    fig.savefig(out.with_suffix('.png'), dpi=600, metadata={'Description': description})
    fig.savefig(out.with_suffix('.svg'), metadata={'Description': description})
    plt.close(fig)


def request_figure(r85):
    case = '000000000100'
    job_id = f'R85_pie_{case}_s0_RELEASE_CORR25'
    job = r85 / job_id
    paths = [r85 / f'sources/{case}.jpg', job / 'edit_response64.jpg',
             job / 'route_mask.jpg', job / 'preview.jpg']
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        print('Request support missing: ' + '; '.join(missing))
        return
    diagnostics = json.loads((job / 'step_summary.json').read_text(encoding='utf-8'))['diagnostics']
    if diagnostics.get('route_preview_from') != 'post_execute_controls_effective_P':
        raise ValueError('Archive does not identify the executed protection map')
    plan = json.loads((r85 / 'plan.json').read_text(encoding='utf-8'))
    task = next(t for t in plan['tasks'] if t['case'] == case)
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.55))
    fig.subplots_adjust(left=.015, right=.985, top=.87, bottom=.25, wspace=.055)
    labels = ['(a) Source', '(b) Request response E', '(c) Executed protection P', '(d) Edited: CORR25']
    for ax, path, label in zip(axes, paths, labels):
        ax.imshow(Image.open(path), cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.set_title(label, fontsize=8.5, pad=7)
        ax.set_axis_off()
    fig.text(.025, .16, 'Request: field → forest.  Sample 000000000100 / seed 0.  P = 0.25 + 0.75(1 − E).', fontsize=8.5)
    fig.text(.025, .085, 'Archived maps: white = high E / high P; E is 64 × 64, P is a 320 × 320 saved preview.', fontsize=8.5)
    fig.text(.025, .015, 'Source, maps and edited image are archived JPEG previews; maps are not inferred from the output.', fontsize=8.5)
    provenance = {'job_id': job_id, 'archive': r85.name, 'path_base': 'archive root',
                  'panels': dict(zip(['source', 'E', 'P', 'edited'], [p.relative_to(r85).as_posix() for p in paths])),
                  'step_summary': (job / 'step_summary.json').relative_to(r85).as_posix(), 'source_prompt': task['source_prompt'],
                  'target_prompt': task['target_prompt'], 'route_preview_from': diagnostics['route_preview_from'],
                  'route_floor': diagnostics['route_floor'], 'probe_estimator': diagnostics['route_probe_estimator'],
                  'probe_nodes': diagnostics['route_probe_nodes'],
                  'encoding': 'Original archived JPEG previews; E64 grayscale, P64 enlarged to320 nearest by runner; no map reconstruction.',
                  'export_code': {'generation_package': 'R85_IndependentComparison',
                                  'files': ['runner.py', 'engine.py'],
                                  'location': 'Separate generation package; file names relative to its root'},
                  'export_lines': 'runner.py:289-302; engine.py:67-99',
                  'display': 'Original grayscale values, no normalization; nearest-neighbor display, source/output full frame.'}
    save(fig, 'figrequest_support', json.dumps(provenance, ensure_ascii=False))
    (BASE / 'figures/request_support.provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Request maps: archived E/P found; provenance saved.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--r84', type=Path, default=ARCHIVE / 'R84')
    parser.add_argument('--r85', type=Path, default=ARCHIVE / 'R85')
    args = parser.parse_args()
    with plt.rc_context(STYLE):
        semantic_figure(args.r84, args.r85)
        request_figure(args.r85)


if __name__ == '__main__':
    main()
