"""Render stored task-weighted estimates at 3.5 in; no statistical recomputation."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
CONTRASTS = [
    ('R78', 'ROUTE25', 'U25_KV', 'ROI spec. − global'),
    ('R78', 'ROI_UNIFORM25', 'ROUTE25', 'Uniform − spectral'),
    ('R78', 'ROUTE25', 'ROUTE_SHIFT25', 'Aligned − shifted'),
    ('R79', 'ROI_ONLINE25', 'ROI_LOCAL25', 'ONLINE − LOCAL'),
    ('R79', 'ROI_ONLINE25_QROI', 'ROI_ONLINE25', 'ROI − all queries'),
    ('R79', 'ROI_LOCAL25', 'ROI_LOCAL25_OFF', 'Memory on − off'),
]


def main():
    with (P / 'tables/evidence_paired_contrasts.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    with plt.style.context(P / 'scripts/paper.mplstyle'), plt.rc_context({
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.bbox': None,
    }):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5))
        fig.subplots_adjust(left=.335, right=.97, bottom=.25, top=.87, hspace=1.0)
        for panel, (ax, title, color) in enumerate(zip(
                axes, ['(a) Spatial allocation', '(b) Magnitude / coverage'],
                ['#0072B2', '#009E73'])):
            ax.axvline(0, color='#858C92', lw=.8, ls='--')
            subset = CONTRASTS[panel * 3:(panel + 1) * 3]
            for y, (study, a, b, _) in enumerate(subset):
                row = next(r for r in rows if r['study'] == study and r['arm_a'] == a
                           and r['arm_b'] == b and r['metric'] == 'pattern_ncc_shift6'
                           and r['weighting'] == 'task_weighted')
                value, low, high = [float(row[k]) for k in ('estimate', 'ci_low', 'ci_high')]
                ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt='o',
                            ms=4.5, color=color, lw=1.1, capsize=3, zorder=3)
            ax.set_yticks(range(3), [r[3] for r in subset])
            ax.set_ylim(2.55, -.55)
            ax.set_xlim(-.12, .52)
            ax.set_xticks([-.1, 0, .2, .4])
            ax.set_title(title, loc='left', weight='bold', pad=10)
            ax.set_xlabel('Difference in pattern NCC', labelpad=5)
            ax.grid(axis='x', lw=.5, color='#E1E5E8')
            ax.set_axisbelow(True)
            ax.tick_params(axis='y', length=0, pad=5)
            for side in ['top', 'right', 'left']:
                ax.spines[side].set_visible(False)
        fig.text(.055, .98, 'Paired pattern-retention effects', fontsize=9,
                 weight='bold', va='top')
        fig.text(.055, .155,
                 'First configuration minus second.\n'
                 'Task-weighted means; 95% cluster-bootstrap CIs.\n'
                 '6 source images; 14 task–seed pairs / contrast.\n'
                 '100,000 resamples; local NCC with ±6 px search.',
                 fontsize=8, va='top', linespacing=1.45)
        description = ('Task-weighted paired pattern_ncc_shift6 differences; stored estimates '
                       'from tables/evidence_paired_contrasts.csv. Six source clusters and '
                       '14 task-seed pairs per contrast; 100000 percentile source-cluster '
                       'bootstrap resamples. Dog/rain and dog/snow share a resampling cluster. '
                       'Regional means spectral ROUTE25; uniform means ROI_UNIFORM25; '
                       'memory on/off uses LOCAL. Exact signs and intervals unchanged.')
        out = P / 'figures/paired_summary'
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix('.pdf'), metadata={'Title': 'Paired pattern-retention effects', 'Subject': description})
        fig.savefig(out.with_suffix('.png'), dpi=600, metadata={'Description': description})
        fig.savefig(out.with_suffix('.svg'), metadata={'Description': description})
        plt.close(fig)


if __name__ == '__main__':
    main()
