#!/usr/bin/env python3
"""Extended Figure 4: Structure factor sampling optimization breakdown.

Three-panel figure:
  (A) Grouped bar chart — total Fcalc time per approach (log Y)
  (B) Stacked bar chart — CPU stage breakdown
  (C) Stacked bar chart — GPU stage breakdown (own y-axis scale)

Reads data from data/exF4_splatting.json (produced by benchmark_exF4_splatting.py).
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

COLOR_SAMPLING = "#2563eb"
COLOR_FFT = "#e67e22"
COLOR_EXTRACTION = "#10b981"
DPI = 500

BASE = Path(__file__).resolve().parent
DATA_JSON = BASE  / "exF4_splatting.json"
OUTDIR = BASE / "output"


def load_data():
    import json
    with open(DATA_JSON) as f:
        return json.load(f)


def _split_by_device(results):
    """Split results into CPU and GPU lists."""
    cpu = [r for r in results if r["device"] == "cpu"]
    gpu = [r for r in results if r["device"] == "cuda"]
    return cpu, gpu


def plot_panel_a(ax, results):
    """Grouped bar chart: total Fcalc time per approach."""
    labels = [r["label"] for r in results]
    total_ms = [r["total"]["mean_ms"] for r in results]
    total_std = [r["total"]["std_ms"] for r in results]

    x = np.arange(len(labels))
    bars = ax.bar(x, total_ms, 0.6, yerr=total_std, capsize=4,
                  color=["#999999", COLOR_SAMPLING, COLOR_SAMPLING,
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

    ax.text(-0.08, 1.05, "A", transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")


def _plot_stacked_breakdown(ax, subset, title, panel_label):
    """Stacked bar chart for a subset of results."""
    labels = [r["label"] for r in subset]
    sampling = [r["stage_a_splatting"]["mean_ms"] for r in subset]
    fft = [r["stage_b_fft"]["mean_ms"] for r in subset]
    extraction = [r["stage_c_extraction"]["mean_ms"] for r in subset]

    x = np.arange(len(labels))
    w = 0.6

    ax.bar(x, sampling, w, label="Sampling", color=COLOR_SAMPLING,
           edgecolor="white", linewidth=0.5)
    ax.bar(x, fft, w, bottom=sampling, label="FFT", color=COLOR_FFT,
           edgecolor="white", linewidth=0.5)
    bottom2 = [s + f for s, f in zip(sampling, fft)]
    ax.bar(x, extraction, w, bottom=bottom2, label="Extraction",
           color=COLOR_EXTRACTION, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Time per F$_{\\mathrm{calc}}$ (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title(title)

    ax.text(-0.08, 1.05, panel_label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top")


def main():
    data = load_data()
    results = data["results"]
    n_atoms = results[0]["n_atoms"]
    n_refl = results[0]["n_reflections"]
    d_min = results[0]["d_min"]
    cpu_results, gpu_results = _split_by_device(results)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── Combined figure (3 panels) ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Extended Figure 4 — F$_{{\\mathrm{{calc}}}}$ Optimization Breakdown "
        f"(pdb_00001daw: {n_atoms} atoms, {n_refl} reflections, {d_min:.2f} Å)",
        fontsize=17, y=1.02,
    )

    plot_panel_a(ax1, results)
    _plot_stacked_breakdown(ax2, cpu_results, "CPU Stage Breakdown", "B")
    _plot_stacked_breakdown(ax3, gpu_results, "GPU Stage Breakdown", "C")

    plt.tight_layout()
    out = OUTDIR / "extended_figure4.png"
    plt.savefig(str(out), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Individual panels ──
    for plot_fn, plot_args, fname in [
        (plot_panel_a, (results,), "exF4_panel_a.png"),
        (_plot_stacked_breakdown, (cpu_results, "CPU Stage Breakdown", "B"), "exF4_panel_b.png"),
        (_plot_stacked_breakdown, (gpu_results, "GPU Stage Breakdown", "C"), "exF4_panel_c.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_fn(ax, *plot_args)
        plt.tight_layout()
        p = OUTDIR / fname
        plt.savefig(str(p), dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p}")

    # Print summary table
    print(f"\n{'Label':<35} {'Total':>10} {'Sample':>10} {'FFT':>10} {'Extract':>10} {'Speedup':>8}")
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
