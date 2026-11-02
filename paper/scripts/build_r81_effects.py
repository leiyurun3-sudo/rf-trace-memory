"""Plot stored R81 paired estimates, without recomputing statistics.

Input: tables/r81_paired_contrasts.csv, heldout_pilot cohort only.
Output: figures/r81_paired_effects.pdf (vector) and .png (600 dpi).
Final size: 3.5 x 4.9 inches; numerical effects retain their original sign.
Statistics: equal-source means after matching cohort/source/task/seed and
averaging within source; 20,000 source bootstrap resamples, percentile 95% CI;
numpy.random.default_rng(81030), shared sequential stream in analyze_r81.py.
NCC: 24 sources / 48 paired cases. Preservation MAE: 19 / 38; five sources
have empty preservation masks. These are source-retention measurements.
"""
from pathlib import Path
import csv
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / 'tables' / 'r81_paired_contrasts.csv'
OUTPUT = BASE / 'figures' / 'r81_paired_effects'
ARMS = [('U25_KV', 'U25'), ('ETA03', 'ETA'),
        ('FLOWEDIT_N26', 'FlowEdit'), ('RFEDIT', 'RFEDIT')]
PANELS = [
    dict(metric='pattern_ncc_fixed', title='(a) Whole-image NCC',
         subtitle='24 sources · 48 paired cases', n=24, cases=48,
         limits=(-.52, .045), ticks=[-.5, -.4, -.3, -.2, -.1, 0],
         label='AUTO − comparator: NCC',
         direction='Higher = more source retention'),
    dict(metric='preserved_region_mae', title='(b) Preservation MAE',
         subtitle='19 sources · 38 paired cases', n=19, cases=38,
         limits=(-.012, .185), ticks=[0, .05, .10, .15],
         label='AUTO − comparator: MAE',
         direction='Lower = more source retention'),
]


def main():
    with INPUT.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['cohort'] == 'heldout_pilot']
    lookup = {(r['metric'], r['comparator']): r for r in rows}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    style = {
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8.5,
        'axes.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'savefig.bbox': None,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'text.color': '#20252A', 'axes.labelcolor': '#20252A',
        'xtick.color': '#50575D', 'ytick.color': '#20252A',
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.9))
        fig.subplots_adjust(left=.24, right=.97, bottom=.30, top=.84, hspace=1.65)
        for ax, panel in zip(axes, PANELS):
            ax.set_axisbelow(True)
            ax.grid(axis='x', color='#E6E9EC', linewidth=.55)
            ax.axvline(0, color='#858C92', linewidth=.8, linestyle=(0, (3, 3)))
            for y, (arm, _) in zip([3, 2, 1, 0], ARMS):
                row = lookup[panel['metric'], arm]
                if (int(row['source_n']) != panel['n'] or
                        int(row['finite_pair_n']) != panel['cases'] or
                        int(row['bootstrap_replicates']) != 20000 or
                        row['direction'] != 'AUTO_DETAIL_minus_comparator'):
                    raise ValueError('Stored contrast does not match figure description')
                mean, low, high = [float(row[k]) for k in
                                   ('equal_source_delta', 'ci95_low', 'ci95_high')]
                ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]],
                            fmt='o', markersize=4.5, color='#0072B2',
                            markeredgecolor='white', markeredgewidth=.5,
                            elinewidth=1.25, capsize=3, capthick=1.0,
                            zorder=3)
            ax.set_yticks([3, 2, 1, 0], [label for _, label in ARMS])
            ax.set_ylim(-.55, 3.55)
            ax.set_xlim(panel['limits'])
            ax.set_xticks(panel['ticks'])
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda x, pos: '0' if abs(x) < 1e-10 else f'{x:.2f}'.replace('-', '−')))
            ax.tick_params(axis='y', length=0, pad=7)
            ax.tick_params(axis='x', length=3, pad=5)
            for side in ('top', 'right', 'left'):
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color('#A7ADB2')
            ax.set_title(panel['title'], loc='left', fontweight='bold', pad=28)
            ax.text(0, 1.065, panel['subtitle'], transform=ax.transAxes,
                    fontsize=8, color='#50575D', va='bottom')
            ax.set_xlabel(panel['label'] + '\n' + panel['direction'], labelpad=6,
                          linespacing=1.45, fontsize=8)
        # Archive provenance: R81 heldout_pilot, unchanged paired CSV estimates.
        fig.text(.055, .98, 'Paired source-retention effects',
                 fontsize=9, fontweight='bold', va='top')
        fig.text(.055, .155,
                 'Heldout pilot; 2 seeds per source.\n'
                 'Equal-source paired means and 95% CIs.\n'
                 'Percentile source bootstrap: 20,000 resamples.\n'
                 'MAE excludes 5 empty preservation masks.',
                 fontsize=8, color='#50575D', va='top', linespacing=1.45)
        description = ('R81 AUTO_DETAIL minus U25/ETA/Flow/RFEDIT; source-retention metrics. '
                       'Equal-source means, 20000 percentile source-bootstrap resamples, 95% CI; '
                       'NumPy default_rng seed 81030 (analysis-wide sequential stream). '
                       'NCC n=24 sources/48 cases; preservation MAE n=19/38. '
                       'Source data: tables/r81_paired_contrasts.csv.')
        fig.savefig(OUTPUT.with_suffix('.pdf'), facecolor='white',
                    metadata={'Title': 'Paired source-retention effects',
                              'Subject': description, 'Creator': 'build_r81_effects.py'})
        fig.savefig(OUTPUT.with_suffix('.png'), dpi=600, facecolor='white',
                    metadata={'Description': description})
        fig.savefig(OUTPUT.with_suffix('.svg'), facecolor='white',
                    metadata={'Title': 'Paired source-retention effects', 'Description': description})
        plt.close(fig)
    print(OUTPUT.with_suffix('.pdf'))
    print(OUTPUT.with_suffix('.png'))


if __name__ == '__main__':
    main()
