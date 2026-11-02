"""Plot finalized source-level metrics from the released numeric summary."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = BASE.parents[1] / "实验交付/rf-trace-memory/results/R85/evidence/overnight_arm_summary.csv"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=BASE / "figures/final_metrics")
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8-sig", newline="") as handle:
        values = {(r["arm"], r["metric"]): r for r in csv.DictReader(handle)}
    methods = [
        ("RELEASE_CORR25", "CORR25", "#0072B2"),
        ("RELEASE_BOTH25", "BOTH25", "#56B4E9"),
        ("ENERGY_MATCH25", "ENERGY", "#777777"),
        ("U25_KV", "U25", "#009E73"),
        ("FLOWEDIT_N26", "FlowEdit", "#D55E00"),
        ("ETA03", "ETA03", "#666666"),
        ("RFEDIT_I08", "RF-Edit", "#CC79A7"),
    ]
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial", "DejaVu Sans"],
        "font.size":8, "axes.titlesize":9, "axes.labelsize":8,
        "xtick.labelsize":8, "ytick.labelsize":8, "svg.fonttype":"none", "savefig.bbox":None,
        "axes.linewidth":0.6, "pdf.fonttype":42, "ps.fonttype":42, "savefig.dpi":300})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 6.0))
    settings = [("clip_directional_similarity", "(a) Directional CLIP", "Directional CLIP (higher is better)", (0, 0.17)),
                ("lpips_preserved", "(b) Preservation LPIPS", "Preservation LPIPS (lower is better)", (0.20, 0.61))]
    for ax, (metric, title, label, limits) in zip(axes, settings):
        for i, (arm, name, color) in enumerate(methods):
            row = values[arm, metric]
            mean, low, high = [float(row[k]) for k in ["mean", "ci95_low", "ci95_high"]]
            ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="o", color=color,
                        markersize=4.8, elinewidth=1.0, capsize=2.5, capthick=.7, zorder=3)
        ax.set_title(title, loc="left", pad=27, fontweight="bold")
        ax.set_xlim(*limits)
        ax.set_ylim(6.6, -0.6)
        ax.set_yticks(range(7), [m[1] for m in methods])
        ax.grid(axis="x", color="#e3e3e3", linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(label, fontsize=8, labelpad=6)
        ax.text(0, 1.035, "40 sources" if metric == "clip_directional_similarity" else "33 sources",
                transform=ax.transAxes, fontsize=8, va="bottom", color="#50575D")
    fig.subplots_adjust(left=.25, right=.97, bottom=.19, top=.87, hspace=.85)
    fig.text(.055, .98, "Independent-study metrics", fontsize=9, weight="bold", va="top")
    fig.text(.055, .115,
             "Equal-source means; pointwise 95% bootstrap CIs.\n"
             "40 independent sources; preservation uses 33.\n"
             "ENERGY: same-state energy control.\n"
             "BOTH25: secondary; RF-Edit: deterministic seed 0.",
             fontsize=8, color="#444444", va="top", linespacing=1.45)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png", ".svg"]:
        fig.savefig(args.out.with_suffix(suffix), facecolor="white", dpi=600)
    plt.close(fig)

if __name__ == "__main__":
    main()
