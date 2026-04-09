#!/usr/bin/env python3
"""Extended Figure 2: Refinement convergence traces.

Grid of panels (1 per structure), each showing per-cycle R-work and R-free
for both TorchRef and Phenix starting from the same shaken model.

Reads data from data/extfig2_convergence.json (produced by benchmark_extfig2_convergence.py).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

COLOR_TORCHREF = "#b2182b"
COLOR_PHENIX = "#2166ac"
DPI = 500

BASE = Path(__file__).resolve().parent
DATA_JSON = BASE / "data" / "extfig2_convergence.json"
OUTDIR = BASE / "output"


def plot_convergence_panel(ax, entry, label_letter=None):
    """Plot convergence traces for one structure."""
    code = entry["code"]
    d_min = entry.get("d_min", "?")
    n_atoms = entry.get("n_atoms", "?")

    # TorchRef
    if "torchref" in entry:
        tr = entry["torchref"]
        x_tr = list(range(len(tr["rwork"])))
        ax.plot(x_tr, tr["rwork"], "o-", color=COLOR_TORCHREF, markersize=5,
                lw=1.5, label="TorchRef R-work")
        ax.plot(x_tr, tr["rfree"], "o--", color=COLOR_TORCHREF, markersize=5,
                lw=1.5, label="TorchRef R-free")

    # Phenix
    if "phenix" in entry:
        ph = entry["phenix"]
        x_ph = list(range(len(ph["rwork"])))
        ax.plot(x_ph, ph["rwork"], "s-", color=COLOR_PHENIX, markersize=5,
                lw=1.5, label="Phenix R-work")
        ax.plot(x_ph, ph["rfree"], "s--", color=COLOR_PHENIX, markersize=5,
                lw=1.5, label="Phenix R-free")

    d_str = f"{d_min:.2f}" if isinstance(d_min, (int, float)) else d_min
    ax.set_title(f"{code} ({d_str} Å, {n_atoms} atoms)", fontsize=15)
    ax.set_xlabel("Macro Cycle")
    ax.set_ylabel("R-factor")
    ax.grid(True, alpha=0.3)

    if label_letter:
        ax.text(-0.08, 1.05, label_letter, transform=ax.transAxes,
                fontsize=18, fontweight="bold", va="top")


def main():
    import json
    with open(DATA_JSON) as f:
        data = json.load(f)

    structures = data["structures"]
    n = len(structures)
    print(f"Loaded convergence data for {n} structures")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    codes = list(structures.keys())
    letters = [chr(65 + i) for i in range(n)]  # A, B, C, ...

    # ── Layout ──
    if n <= 3:
        nrows, ncols = 1, n
    elif n <= 6:
        nrows, ncols = 2, (n + 1) // 2
    else:
        nrows, ncols = 3, (n + 2) // 3

    # ── Combined figure ──
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (code, letter) in enumerate(zip(codes, letters)):
        plot_convergence_panel(axes[i], structures[code], letter)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Refinement Convergence Traces", fontsize=17, y=1.06)
    plt.tight_layout()
    out = OUTDIR / "extended_figure2.png"
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Individual panels ──
    for i, (code, letter) in enumerate(zip(codes, letters)):
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_convergence_panel(ax, structures[code], letter)
        ax.legend(fontsize=11, loc="upper right")
        plt.tight_layout()
        p = OUTDIR / f"extfig2_panel_{letter.lower()}.png"
        plt.savefig(str(p), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
