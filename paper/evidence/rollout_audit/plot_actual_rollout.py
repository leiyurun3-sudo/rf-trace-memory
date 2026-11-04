"""Source-level display of measured correction magnitudes; no inference recomputed."""
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--results-root', type=Path,
                    default=Path(__file__).resolve().parent.parent / 'R85_FinalNight',
                    help='Directory containing the R85 per-job step_summary.json files')
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
ROOT, OUT = args.results_root, args.output
OUT.mkdir(parents=True, exist_ok=True)
groups = defaultdict(list)
rows = []
for p in sorted(ROOT.glob('*_RELEASE_CORR25/step_summary.json')):
    q = p.parent.with_name(p.parent.name.replace('_RELEASE_CORR25', '_ENERGY_MATCH25')) / p.name
    a, b = json.loads(p.read_text()), json.loads(q.read_text())
    aa, bb = a['steps'], b['steps']
    if len(aa) != 28 or len(bb) != 28:
        raise ValueError(f'Incomplete trace: {p}')
    h = np.array([v['h'] for v in aa], dtype=float)
    hb = np.array([v['h'] for v in bb], dtype=float)
    if not np.allclose(h, hb, rtol=0, atol=1e-12):
        raise ValueError(f'Different scheduler: {p}')
    ra = np.array([v['applied_correction_rms'] for v in aa], dtype=float)
    rb = np.array([v['applied_correction_rms'] for v in bb], dtype=float)
    ca, cb = np.cumsum(h * ra), np.cumsum(h * rb)
    source = p.parent.name.split('_s')[0]
    groups[source].append(np.stack([ra, rb, ca, cb]))
    for i in range(28):
        rows.append(dict(job=p.parent.name, source=source, step=i+1, h=h[i],
                         corr_rms=ra[i], energy_rms=rb[i],
                         corr_cumulative=ca[i], energy_cumulative=cb[i]))
if len(groups) != 40 or any(len(v) != 2 for v in groups.values()):
    raise ValueError('Expected 40 sources, two paired seeds each')
with (OUT / 'actual_rollout_steps.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
source_values = np.stack([np.mean(v, axis=0) for v in groups.values()])
means = source_values.mean(axis=0)
rel = 100 * (source_values[:, 2] - source_values[:, 3]) / source_values[:, 3]
step = np.arange(1, 29)
style = {'font.family':'DejaVu Sans', 'font.size':8.5, 'axes.titlesize':9,
         'axes.labelsize':8.5, 'xtick.labelsize':8, 'ytick.labelsize':8,
         'legend.fontsize':8, 'pdf.fonttype':42, 'ps.fonttype':42,
         'axes.spines.top':False, 'axes.spines.right':False,
         'axes.linewidth':.6, 'lines.linewidth':1.5, 'svg.fonttype':'none'}
with plt.rc_context(style):
    fig, ax = plt.subplots(1, 3, figsize=(7.05, 2.52))
    fig.subplots_adjust(left=.077, right=.985, bottom=.235, top=.77, wspace=.54)
    colors = ['#0072B2', '#D55E00']
    labels = ['CORR25', 'ENERGY']
    for j in range(2):
        ax[0].plot(step, means[j], color=colors[j], ls=['-','--'][j], label=labels[j])
        ax[1].plot(step, means[j+2], color=colors[j], ls=['-','--'][j])
    for y in rel:
        ax[2].plot(step, y, color='#B3B3B3', lw=.45, zorder=1)
    ax[2].axhline(0, color='#666666', lw=.6, ls=':')
    ax[2].plot(step, rel.mean(axis=0), color='#202020', lw=1.6, zorder=3)
    titles = ['(a) Applied correction', '(b) Accumulated impulse', '(c) Relative impulse']
    ylabs = ['RMS (latent velocity)', r'$\sum h_i\,\mathrm{RMS}$ (latent units)', 'Relative impulse (%)']
    for a, t, y in zip(ax, titles, ylabs):
        a.set_title(t, loc='left', pad=7)
        a.set_xlabel('Generation step'); a.set_ylabel(y, labelpad=4)
        a.set_xlim(1,28); a.set_xticks([1,7,14,21,28]); a.grid(axis='y', color='#E5E5E5', lw=.5)
        a.tick_params(width=.6, length=3)
    ax[0].set_ylim(bottom=0); ax[1].set_ylim(bottom=0)
    handles = ax[0].get_lines()[:2]
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.07,1),
               ncol=2, frameon=False, handlelength=2.6)
    fig.text(.635,.945,'Gray: 40 sources; black: mean', fontsize=8)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT / f'actual_rollout_impulse.{ext}', dpi=220)
    plt.close(fig)
manifest = dict(sources=40, paired_seeds=2, steps=28,
                aggregation='Average two seeds within each source, equal source weights',
                source_glob='R85_FinalNight/*_RELEASE_CORR25/step_summary.json',
                comparator='ENERGY_MATCH25', normalization='100*(source_mean_I_CORR-source_mean_I_ENERGY)/source_mean_I_ENERGY',
                final_corr=float(means[2,-1]), final_energy=float(means[3,-1]),
                final_source_relative_mean_percent=float(rel[:,-1].mean()),
                units='latent model units; not semantic effect or physical energy',
                matplotlib=matplotlib.__version__, numpy=np.__version__)
(OUT / 'actual_rollout_plot_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
