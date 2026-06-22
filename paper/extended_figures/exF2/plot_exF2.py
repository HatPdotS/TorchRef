#!/usr/bin/env python3
"""Extended Figure 2: R-factor gap vs resolution (AlphaFold-start).

Two-panel figure, every refined structure judged by the same independent scorer
(phenix.model_vs_data). TorchRef (locked default arm, xray 1 / geometry 0.2 /
adp 0.02) is compared against both reference programs:
  (A) ΔR-free (TorchRef − PHENIX) and (TorchRef − REFMAC) vs resolution
  (B) ΔR-work (TorchRef − PHENIX) and (TorchRef − REFMAC) vs resolution

Each comparison is a light scatter + a bold running-median line; values below 0
are where TorchRef beats the reference program.

Reads data/exF2_rfactor_by_resolution.csv (collect_exF2_data.py).

Usage:
    ./.dev/bin/python paper/extended_figures/exF2/plot_exF2.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 16,
    "axes.titlesize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})

# reference program -> (scatter colour, median colour, display name)
REFS = {
    "phenix": ("#2166ac", "#0b3a66", "PHENIX"),
    "refmac": ("#9970ab", "#542788", "REFMAC"),
}
DPI = 500

BASE = Path(__file__).resolve().parent
DATA_CSV = BASE / "data" / "exF2_rfactor_by_resolution.csv"
OUTDIR = BASE / "output"


def running_median(x, y, window=200):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    n = len(xs)
    window = max(5, min(window, n if n % 2 else n - 1))
    half = window // 2
    xo, yo = [], []
    for i in range(half, n - half):
        xo.append(xs[i])
        yo.append(np.median(ys[i - half: i + half + 1]))
    return np.array(xo), np.array(yo)


def plot_panel(ax, df, metric, label_letter, xlim, ylim):
    """metric = 'rfree' or 'rwork'."""
    d_min = df["d_min"].values
    name = "R-free" if metric == "rfree" else "R-work"
    txt = []
    for ref, (sc, mc, disp) in REFS.items():
        dr = df[f"delta_{metric}_{ref}"].values
        ax.scatter(d_min, dr, s=7, alpha=0.22, color=sc, edgecolors="none", zorder=2)
        xm, ym = running_median(d_min, dr)
        ax.plot(xm, ym, color=mc, lw=2.2, zorder=4, label=f"− {disp} (median)")
        txt.append(f"vs {disp}: {np.median(dr):+.2f} pp")

    ax.axhline(y=0, color="black", ls="--", lw=0.8, alpha=0.4, zorder=1)
    ax.text(0.97, 0.95, f"Median Δ{name}\n" + "\n".join(txt),
            transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    ax.set_xlabel("Resolution (Å)")
    ax.set_ylabel(f"Δ{name} (TorchRef − reference, pp)")
    ax.set_xlim(*xlim)            # already high→low (inverted)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="lower left")
    ax.text(-0.08, 1.05, label_letter, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="top")


def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} structures")
    d_min = df["d_min"].values

    xlim = (float(np.nanmax(d_min)) + 0.1, float(np.nanmin(d_min)) - 0.1)  # high→low
    cols = [f"delta_{m}_{r}" for m in ("rfree", "rwork") for r in REFS]
    allv = np.concatenate([df[c].values for c in cols])
    pad = 0.5
    ylim = (float(np.nanpercentile(allv, 1)) - pad,
            float(np.nanpercentile(allv, 99)) + pad)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_panel(ax1, df, "rfree", "A", xlim, ylim)
    plot_panel(ax2, df, "rwork", "B", xlim, ylim)
    fig.suptitle("Extended Figure 2 — R-factor gap vs resolution "
                 "(AF-start, PHENIX-scored)", fontsize=17, y=1.02)
    plt.tight_layout()
    out = OUTDIR / "extended_figure2.png"
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    for metric, letter, fname in [("rfree", "A", "exF2_panel_a.png"),
                                  ("rwork", "B", "exF2_panel_b.png")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_panel(ax, df, metric, letter, xlim, ylim)
        plt.tight_layout()
        p = OUTDIR / fname
        plt.savefig(str(p), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
