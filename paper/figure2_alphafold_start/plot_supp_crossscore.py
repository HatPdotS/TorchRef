#!/usr/bin/env python
"""Supplementary figure: R-factor scorer consistency (REFMAC / PHENIX / TorchRef).

Every AlphaFold-start final model was independently re-scored against identical
data + free flags by three R-factor calculators, each run in a 0-cycle mode that
fits only scaling + bulk solvent (coordinates/B untouched):

  REFMAC    : refmac5 NCYCLES 0          (runs/<arm>/<code>/validate.log)
  PHENIX    : phenix.model_vs_data        (runs/<arm>/<code>/phenix_validate.log)
  TorchRef  : torchref_score.py (-n 0)    (runs/<arm>/<code>/torchref_validate.json)

Pairwise scatter of every refined model (coloured by which engine produced it),
R-work (top) and R-free (bottom). Tight bands parallel to x = y show the three
scorers rank models consistently. Constant offsets below/above x = y are the
scorers' systematic differences (PHENIX lowest, then REFMAC, then TorchRef — the
most conservative); these cancel within a scorer. PHENIX is the main-figure scorer.

Reads runs/metrics/fig_crossscore.csv (analysis/aggregate_crossscore.py).

Usage
-----
    ./.dev/bin/python paper/figure2_alphafold_start/plot_supp_crossscore.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
METRICS = SCRIPT_DIR / "runs" / "metrics" / "fig_crossscore.csv"
OUTPUT_DIR = SCRIPT_DIR / "figures"

# refined-model engines (exclude the AlphaFold prediction to keep axes focused)
ENGINES = [
    ("refmac", "Refmac", "#762a83"),
    ("phenix", "PHENIX", "#2166ac"),
    ("torchref", "TorchRef", "#b2182b"),
]
COLOR = {e: c for e, _, c in ENGINES}
SCNAME = {"refmac": "REFMAC, NCYCLES 0", "phenix": "PHENIX, model_vs_data",
          "torchref": "TorchRef, -n 0"}
SCSHORT = {"refmac": "REFMAC", "phenix": "PHENIX", "torchref": "TorchRef"}
PAIRS = [("refmac", "phenix"), ("refmac", "torchref"), ("phenix", "torchref")]


def setup_matplotlib():
    plt.rcParams.update({
        "font.size": 13, "font.family": "serif",
        "font.serif": ["STIX", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.titlesize": 15,
        "axes.labelsize": 13, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 12,
    })


def panel(ax, df, metric, sx, sy, engines):
    """Scatter scorer sy (y) vs scorer sx (x) for `metric`, over `engines`."""
    sub = df[df.model_engine.isin(engines)]
    wide = sub.pivot_table(index=["code", "model_engine"], columns="scorer",
                           values=metric)
    if sx not in wide.columns or sy not in wide.columns:
        ax.set_visible(False)
        return
    wide = wide.dropna(subset=[sx, sy])
    x, y = wide[sx].values, wide[sy].values

    lo = min(x.min(), y.min()) - 0.01
    hi = max(x.max(), y.max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "--", color="0.3", lw=1.1, alpha=0.7, zorder=1)

    eng = wide.index.get_level_values("model_engine")
    for e, _, c in ENGINES:
        m = eng == e
        ax.scatter(x[m], y[m], s=8, color=c, alpha=0.33, linewidths=0, zorder=2)

    r = np.corrcoef(x, y)[0, 1]
    off = np.median(y - x)
    txt = (f"r = {r:.3f}\noffset = {off:+.4f}\n({SCSHORT[sy]}$-${SCSHORT[sx]})\n"
           f"n = {len(wide)}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    lbl = metric.replace("r_", "R-")
    ax.set_xlabel(f"{lbl} ({SCNAME[sx]})")
    ax.set_ylabel(f"{lbl} ({SCNAME[sy]})")


def create_figure(outpath: str, dpi: int = 300):
    setup_matplotlib()
    df = pd.read_csv(METRICS)
    refiners = ["refmac", "phenix", "torchref"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for row, metric in enumerate(["r_work", "r_free"]):
        for col, (sx, sy) in enumerate(PAIRS):
            panel(axes[row, col], df, metric, sx, sy, refiners)
            axes[row, col].set_title(
                f"{metric.replace('r_', 'R-')}:  {SCSHORT[sx]} vs {SCSHORT[sy]}")

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                      markersize=10, label=l) for _, l, c in ENGINES]
    handles.append(Line2D([0], [0], color="0.3", ls="--", lw=1.1, alpha=0.7,
                          label="x = y (equivalence)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("R-factor scorer consistency: REFMAC / PHENIX / TorchRef "
                 "(same data, same free set)", y=1.03, fontsize=18)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure to: {outpath}")


def main():
    p = argparse.ArgumentParser(description="Scorer-consistency supplementary figure.")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    outpath = args.output or str(OUTPUT_DIR / "figure_supp_crossscore.png")
    create_figure(outpath, dpi=args.dpi)


if __name__ == "__main__":
    main()
