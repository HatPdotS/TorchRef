#!/usr/bin/env python3
"""Extended Figure 3: GPU memory scaling.

Three-panel figure:
  (A) Peak GPU memory vs number of atoms
  (B) Peak GPU memory vs number of reflections
  (C) Memory timeline during refinement (per-stage peak)

Reads data from:
  data/extfig3_memory.csv          (scatter data)
  data/extfig3_memory_timeline.json (timeline data)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style ────────────────────────────────────────────────────────────────────
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

DPI = 500
GPU_TIERS = {8: "8 GB", 16: "16 GB", 24: "24 GB", 40: "40 GB", 80: "80 GB"}

BASE = Path(__file__).resolve().parent
DATA_CSV = BASE / "data" / "extfig3_memory.csv"
DATA_TIMELINE = BASE / "data" / "extfig3_memory_timeline.json"
OUTDIR = BASE / "output"

# Readable labels for timeline stages
STAGE_LABELS = {
    "start": "Start",
    "after_init": "Init\n(load + restraints)",
    "after_scaling": "Scaling",
    "after_create_loss_state_xyz": "Create\nloss state",
    "after_forward_xyz": "Forward\n(XYZ)",
    "after_backward_xyz": "Backward\n(XYZ)",
    "after_refine_xyz": "Full\nrefine XYZ",
    "after_refine_adp": "Full\nrefine ADP",
    "cycle2_after_scaling": "Cycle 2\nscaling",
    "cycle2_after_xyz": "Cycle 2\nXYZ",
    "cycle2_after_adp": "Cycle 2\nADP",
}


def plot_scatter_panel(ax, x, y, d_min, xlabel, label_letter):
    """Plot one memory scaling panel."""
    sc = ax.scatter(x, y, c=d_min, cmap="plasma_r", s=40, edgecolors="black",
                    linewidths=0.3, zorder=3)

    # Trend line (linear fit)
    coeffs = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color="#555555", lw=1.5,
            alpha=0.6, zorder=2, label="Linear fit")

    # GPU tier lines
    y_max = max(y.max() * 1.3, 10)
    for tier_gb, tier_label in GPU_TIERS.items():
        if tier_gb < y_max:
            ax.axhline(y=tier_gb, color="gray", ls=":", lw=0.8, alpha=0.5, zorder=1)
            ax.text(x.max() * 0.98, tier_gb + 0.3, tier_label,
                    fontsize=9, color="gray", ha="right", va="bottom")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="upper left")

    ax.text(-0.08, 1.05, label_letter, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")
    return sc


def plot_timeline_panel(ax, timelines, label_letter):
    """Bar chart showing peak memory at each refinement stage."""
    codes = list(timelines.keys())
    colors = ["#b2182b", "#2166ac", "#4dac26", "#e67e22", "#7c3aed"]

    bar_width = 0.8 / len(codes)

    for ci, code in enumerate(codes):
        tl = timelines[code]
        trace = tl["trace"]

        labels = [STAGE_LABELS.get(pt["label"], pt["label"]) for pt in trace]
        peaks = [pt["peak_mb"] / 1024 for pt in trace]  # Convert to GB

        x = np.arange(len(labels))
        offset = (ci - (len(codes) - 1) / 2) * bar_width
        ax.bar(x + offset, peaks, bar_width * 0.9, label=f"{code} ({tl['n_atoms']} atoms)",
               color=colors[ci % len(colors)], edgecolor="white", linewidth=0.3,
               alpha=0.85)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=0, ha="center")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_title("Memory During Refinement (peak per stage)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10, loc="upper left")

    ax.text(-0.05, 1.05, label_letter, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    has_scatter = DATA_CSV.exists()
    has_timeline = DATA_TIMELINE.exists()

    if has_scatter:
        df = pd.read_csv(DATA_CSV)
        print(f"Loaded {len(df)} structures (scatter)")
        n_atoms = df["n_atoms"].values.astype(float)
        n_refl = df["n_reflections"].values.astype(float)
        mem = df["peak_memory_gb"].values
        d_min = df["d_min"].values

    if has_timeline:
        with open(DATA_TIMELINE) as f:
            timelines = json.load(f)
        print(f"Loaded {len(timelines)} timelines")

    # ── Combined figure ──
    if has_scatter and has_timeline:
        fig = plt.figure(figsize=(16, 10))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 1, 2)

        sc1 = plot_scatter_panel(ax1, n_atoms, mem, d_min, "Number of Atoms", "A")
        sc2 = plot_scatter_panel(ax2, n_refl, mem, d_min, "Number of Reflections", "B")
        plot_timeline_panel(ax3, timelines, "C")

        cbar = fig.colorbar(sc2, ax=[ax1, ax2], shrink=0.8, pad=0.02)
        cbar.set_label("Resolution (Å)", fontsize=14)

        fig.suptitle("GPU Memory Scaling", fontsize=17, y=1.01)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        out = OUTDIR / "extended_figure3.png"
        plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")
    elif has_timeline:
        fig, ax = plt.subplots(figsize=(14, 5))
        plot_timeline_panel(ax, timelines, "")
        plt.tight_layout()
        out = OUTDIR / "extended_figure3.png"
        plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")

    # ── Individual panels ──
    if has_scatter:
        for x_data, xlabel, letter, fname in [
            (n_atoms, "Number of Atoms", "A", "extfig3_panel_a.png"),
            (n_refl, "Number of Reflections", "B", "extfig3_panel_b.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = plot_scatter_panel(ax, x_data, mem, d_min, xlabel, letter)
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label("Resolution (Å)", fontsize=14)
            plt.tight_layout()
            p = OUTDIR / fname
            plt.savefig(str(p), dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {p}")

    if has_timeline:
        fig, ax = plt.subplots(figsize=(14, 5))
        plot_timeline_panel(ax, timelines, "C")
        plt.tight_layout()
        p = OUTDIR / "extfig3_panel_c.png"
        plt.savefig(str(p), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
