#!/usr/bin/env python
"""Extended Figure 4: single-core wall-clock runtime (reproduce Fig 2c at 1 CPU core).

Single-panel box plot of per-engine wall-clock distribution (log y, minutes), exactly
like Figure 2c (plot_figure_af.py:150-193, plot_runtime_box) but timed at 1 CPU core over
the full conserved AlphaFold-start set. The main Figure-2 benchmark ran all three programs
at 4 cores; this re-levels them onto a clean per-core single-threaded comparison.

Reads data/exF4_singlecore.csv (from aggregate_singlecore.py).

Usage
-----
    ./.dev/bin/python paper/extended_figures/exF4/plot_singlecore.py
    ./.dev/bin/python paper/extended_figures/exF4/plot_singlecore.py --dpi 600
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CSV = SCRIPT_DIR / "data" / "exF4_singlecore.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Paper colors / labels (match plot_figure_af.py:36-41).
COLOR = {"refmac": "#762a83", "phenix": "#2166ac", "torchref": "#b2182b"}
LABEL = {"refmac": "Refmac", "phenix": "PHENIX", "torchref": "TorchRef"}


def setup_matplotlib():
    plt.rcParams.update({
        "font.size": 15,
        "font.family": "serif",
        "font.serif": ["STIX", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
    })


def plot_runtime_box(ax, runtime_long):
    """Per-engine wall-clock distribution (log y), fastest -> slowest. Mirrors Fig 2c."""
    order = ["refmac", "torchref", "phenix"]   # speed order
    rng = np.random.default_rng(0)             # reproducible jitter

    data = [runtime_long.loc[runtime_long.program == e, "wall_s"].dropna().values / 60.0
            for e in order]

    pos = np.arange(len(order))
    bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                    showfliers=False, whis=(5, 95),
                    medianprops=dict(color="black", lw=1.6),
                    whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
    for patch, e in zip(bp["boxes"], order):
        patch.set_facecolor(COLOR[e])
        patch.set_alpha(0.45)
        patch.set_edgecolor(COLOR[e])
    # light jittered strip behind the boxes
    for i, (v, e) in enumerate(zip(data, order)):
        x = rng.normal(pos[i], 0.05, size=len(v))
        ax.scatter(x, v, s=3, color=COLOR[e], alpha=0.12, linewidths=0, zorder=1)

    ax.set_yscale("log")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(pos)
    ax.set_xticklabels([LABEL[e] for e in order])
    ax.set_ylabel("Wall-clock runtime (min)")
    ax.set_title("Wall-clock runtime (1 CPU core)")

    # The pairwise median-runtime ratios (TorchRef vs Refmac / PHENIX) and n are
    # reported in the figure caption / FIGURE_MEDIANS exF4 rather than annotated
    # on the panel.


def main():
    ap = argparse.ArgumentParser(description="exF4 single-core runtime (Fig 2c at 1 core).")
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--dpi", type=int, default=500)
    args = ap.parse_args()

    setup_matplotlib()
    df = pd.read_csv(DATA_CSV)

    n = df.code.nunique()
    print(f"single-core runtime over {n} conserved structures:")
    for e in ("refmac", "torchref", "phenix"):
        v = df.loc[df.program == e, "wall_s"]
        if len(v):
            print(f"  {LABEL[e]:<10s} median {v.median():.1f}s ({v.median()/60:.2f} min)")

    fig, ax = plt.subplots(figsize=(6.2, 5.5))
    plot_runtime_box(ax, df)
    plt.tight_layout()

    out = args.output or str(OUTPUT_DIR / "exF4_singlecore_runtime.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved figure to: {out}")


if __name__ == "__main__":
    main()
