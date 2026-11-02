"""Plot finalized source-level metrics from the released numeric summary."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.summary.open(encoding="utf-8-sig", newline="") as handle:
        values = {(r["arm"], r["metric"]): r for r in csv.DictReader(handle)}
    methods = [
        ("RELEASE_CORR25", "TraceMemory (CORR25)", "#0072B2"),
        ("RELEASE_BOTH25", "BOTH25 (secondary)", "#56B4E9"),
        ("ENERGY_MATCH25", "Same-state energy control", "#777777"),
        ("U25_KV", "Uniform correction (U25)", "#009E73"),
        ("FLOWEDIT_N26", "FlowEdit N26", "#D55E00"),
        ("ETA03", "RF-Inversion ETA03", "#666666"),
        ("RFEDIT_I08", "RF-Edit I08", "#CC79A7"),
    ]
    plt.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial", "DejaVu Sans"],
        "font.size":8.5, "axes.titlesize":9.5, "axes.labelsize":8.5,
        "axes.linewidth":0.6, "pdf.fonttype":42, "ps.fonttype":42, "savefig.dpi":300})
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.3), sharey=True)
    settings = [("clip_directional_similarity", "(a) Directional CLIP", "Higher indicates stronger directional alignment", (0, 0.17)),
                ("lpips_preserved", "(b) Preservation LPIPS", "Lower indicates better regional preservation", (0.20, 0.61))]
    for ax, (metric, title, label, limits) in zip(axes, settings):
        for i, (arm, name, color) in enumerate(methods):
            row = values[arm, metric]
            mean, low, high = [float(row[k]) for k in ["mean", "ci95_low", "ci95_high"]]
            ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt="o", color=color,
                        markersize=4.8, elinewidth=1.0, capsize=2.5, capthick=.7, zorder=3)
        ax.set_title(title, loc="left", pad=9, fontweight="bold")
        ax.set_xlim(*limits)
        ax.set_ylim(6.6, -0.6)
        ax.set_yticks(range(7))
        ax.grid(axis="x", color="#e3e3e3", linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(label, fontsize=7.2, labelpad=8)
    axes[0].set_yticklabels([m[1] for m in methods])
    axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(left=.265, right=.98, bottom=.24, top=.88, wspace=.22)
    fig.text(.5, .065, "Equal-source means and pointwise 95% bootstrap intervals; 40 sources (CLIP), 33 (LPIPS).",
             ha="center", fontsize=7.1, color="#444444")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".pdf", ".png"]:
        fig.savefig(args.out.with_suffix(suffix), facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    main()
