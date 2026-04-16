#!/usr/bin/env python3
"""Extended Figure 2: Resolution-binned R-factor comparison.

Two-panel figure:
  (A) ΔR-free (TorchRef − Phenix) vs resolution
  (B) ΔR-work (TorchRef − Phenix) vs resolution

Each panel shows a scatter plot with a running median overlay.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style (matching main paper figures) ──────────────────────────────────────
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

COLOR_TORCHREF = "#b2182b"
COLOR_PHENIX = "#2166ac"
DPI = 500

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA_CSV = BASE / "data" / "exF2_rfactor_by_resolution.csv"
OUTDIR = BASE / "output"


def running_median(x, y, window=50):
    """Compute running median of y sorted by x."""
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    n = len(x_sorted)
    half = window // 2
    x_out, y_out = [], []
    for i in range(half, n - half):
        x_out.append(x_sorted[i])
        y_out.append(np.median(y_sorted[i - half : i + half + 1]))
    return np.array(x_out), np.array(y_out)


def plot_panel(ax, d_min, delta_r, ylabel, label_letter, metric_name):
    """Plot one panel: scatter + running median."""
    ax.scatter(d_min, delta_r, s=8, alpha=0.35, color=COLOR_TORCHREF,
               edgecolors="none", zorder=2)

    # Running median
    xm, ym = running_median(d_min, delta_r, window=200)
    ax.plot(xm, ym, color="#67001f", lw=2, zorder=3, label="Running median")

    # Reference line
    ax.axhline(y=0, color="black", ls="--", lw=0.8, alpha=0.3, zorder=1)

    # Annotation
    med = np.median(delta_r)
    ax.text(
        0.97, 0.95,
        f"Median Δ{metric_name} = {med:+.2f} pp",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    ax.set_xlabel("Resolution (Å)")
    ax.set_ylabel(f"Δ{ylabel} (TorchRef − Phenix, pp)")
    ax.invert_xaxis()
    ax.set_xlim(3.1, 1.0)
    ax.set_ylim(-4, 6)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="lower left")

    # Panel label
    ax.text(
        -0.08, 1.05, label_letter,
        transform=ax.transAxes, fontsize=18, fontweight="bold", va="top",
    )


def main():
    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} structures")

    d_min = df["d_min"].values
    delta_rfree = df["delta_rfree"].values
    delta_rwork = df["delta_rwork"].values

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Combined figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plot_panel(ax1, d_min, delta_rfree, "R-free", "A", "R-free")
    plot_panel(ax2, d_min, delta_rwork, "R-work", "B", "R-work")

    fig.suptitle(
        "Resolution-Binned R-Factor Comparison (REFMAC5 Validation)",
        fontsize=17, y=1.02,
    )
    plt.tight_layout()
    out = OUTDIR / "extended_figure2.png"
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Individual panels ──
    for delta, ylabel, letter, metric, fname in [
        (delta_rfree, "R-free", "A", "R-free", "exF2_panel_a.png"),
        (delta_rwork, "R-work", "B", "R-work", "exF2_panel_b.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_panel(ax, d_min, delta, ylabel, letter, metric)
        plt.tight_layout()
        p = OUTDIR / fname
        plt.savefig(str(p), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
