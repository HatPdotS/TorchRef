#!/usr/bin/env python3
"""Extended Figure 4: Structure factor splatting optimization breakdown.

Two-panel figure:
  (A) Grouped bar chart — total Fcalc time per approach (log Y)
  (B) Stacked bar chart — time breakdown into splatting/FFT/extraction

Reads data from data/extfig4_splatting.json (produced by benchmark_extfig4_splatting.py).
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
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
})

COLOR_SPLATTING = "#2563eb"
COLOR_FFT = "#e67e22"
COLOR_EXTRACTION = "#10b981"
DPI = 500

BASE = Path(__file__).resolve().parent
DATA_JSON = BASE / "data" / "extfig4_splatting.json"
OUTDIR = BASE / "output"


def load_data():
    import json
    with open(DATA_JSON) as f:
        return json.load(f)


def plot_panel_a(ax, results):
    """Grouped bar chart: total Fcalc time per approach."""
    labels = [r["label"] for r in results]
    total_ms = [r["total"]["mean_ms"] for r in results]
    total_std = [r["total"]["std_ms"] for r in results]

    x = np.arange(len(labels))
    bars = ax.bar(x, total_ms, 0.6, yerr=total_std, capsize=4,
                  color=["#999999", COLOR_SPLATTING, COLOR_SPLATTING,
                         "#ffb347", "#10b981"][:len(labels)],
                  edgecolor="white", linewidth=0.5)

    # Speedup annotations
    baseline = total_ms[0]
    for i, (bar, ms) in enumerate(zip(bars, total_ms)):
        speedup = baseline / ms
        if i > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                f"{speedup:.1f}×",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

    ax.set_yscale("log")
    ax.set_ylabel("Time per F$_{\\mathrm{calc}}$ (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.grid(True, which="both", alpha=0.3, axis="y")
    ax.set_title("Total F$_{\\mathrm{calc}}$ Time")

    # Panel label
    ax.text(-0.08, 1.05, "A", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")


def plot_panel_b(ax, results):
    """Stacked bar chart: stage breakdown."""
    labels = [r["label"] for r in results]
    splatting = [r["stage_a_splatting"]["mean_ms"] for r in results]
    fft = [r["stage_b_fft"]["mean_ms"] for r in results]
    extraction = [r["stage_c_extraction"]["mean_ms"] for r in results]

    x = np.arange(len(labels))
    w = 0.6

    b1 = ax.bar(x, splatting, w, label="Splatting", color=COLOR_SPLATTING,
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x, fft, w, bottom=splatting, label="FFT", color=COLOR_FFT,
                edgecolor="white", linewidth=0.5)
    bottom2 = [s + f for s, f in zip(splatting, fft)]
    b3 = ax.bar(x, extraction, w, bottom=bottom2, label="Extraction",
                color=COLOR_EXTRACTION, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Time per F$_{\\mathrm{calc}}$ (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title("Stage Breakdown")

    # Panel label
    ax.text(-0.08, 1.05, "B", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")


def main():
    data = load_data()
    results = data["results"]
    n_atoms = results[0]["n_atoms"]
    n_refl = results[0]["n_reflections"]
    d_min = results[0]["d_min"]

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Combined figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"F$_{{\\mathrm{{calc}}}}$ Optimization Breakdown "
        f"(1DAW: {n_atoms} atoms, {n_refl} reflections, {d_min:.2f} Å)",
        fontsize=17, y=1.02,
    )

    plot_panel_a(ax1, results)
    plot_panel_b(ax2, results)

    plt.tight_layout()
    out = OUTDIR / "extended_figure4.png"
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Individual panels ──
    for plot_fn, fname in [
        (plot_panel_a, "extfig4_panel_a.png"),
        (plot_panel_b, "extfig4_panel_b.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_fn(ax, results)
        plt.tight_layout()
        p = OUTDIR / fname
        plt.savefig(str(p), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")

    # Print summary table
    print(f"\n{'Label':<35} {'Total':>10} {'Splat':>10} {'FFT':>10} {'Extract':>10} {'Speedup':>8}")
    print("-" * 93)
    for r in results:
        print(
            f"{r['label']:<35} "
            f"{r['total']['mean_ms']:>8.2f}ms "
            f"{r['stage_a_splatting']['mean_ms']:>8.2f}ms "
            f"{r['stage_b_fft']['mean_ms']:>8.2f}ms "
            f"{r['stage_c_extraction']['mean_ms']:>8.2f}ms "
            f"{r['speedup_vs_baseline']:>7.1f}×"
        )


if __name__ == "__main__":
    main()
