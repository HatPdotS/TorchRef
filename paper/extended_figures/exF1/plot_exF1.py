#!/usr/bin/env python
"""
Publication-quality plot for the geometry weight screening experiment.

Shows how refinement quality metrics change as a function of the geometry
weight multiplier (1–10), matching the 6 categories from Figure 2 Panel B.
PHENIX medians from the main benchmark are overlaid as reference lines.

Usage
-----
    python plot_exF1.py
    python plot_exF1.py --csv path/to/refmac_metrics.csv --output fig.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR / "output"
PHENIX_METRICS = PAPER_ROOT / "figure2_validation" / "data" / "refmac_metrics.csv"

# ── Colors ───────────────────────────────────────────────────────────────────
COLOR_BOX = "#b2182b"
COLOR_PHENIX = "#2166ac"
COLOR_MEDIAN_LINE = "#333333"

# ── Style ────────────────────────────────────────────────────────────────────
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


def _load_phenix_medians():
    """Load PHENIX median metrics from the figure2 validation benchmark."""
    if not PHENIX_METRICS.exists():
        return {}
    df = pd.read_csv(PHENIX_METRICS)
    phenix = df[df["variant"] == "phenix"]
    if phenix.empty:
        return {}

    medians = {}
    for col in ["r_work", "r_free"]:
        if col in phenix.columns:
            medians[col] = phenix[col].median()

    # RMSZ = rms / sig (already in the REFMAC log columns)
    rmsz_pairs = {
        "bond_rmsz": ("rmsBOND", "sigBOND"),
        "angle_rmsz": ("rmsANGL", "sigANGL"),
        "chiral_rmsz": ("rmsCHIRAL", "sigCHIRAL"),
    }
    for key, (rms_col, sig_col) in rmsz_pairs.items():
        if rms_col in phenix.columns and sig_col in phenix.columns:
            sub = phenix[[rms_col, sig_col]].dropna()
            sig = sub[sig_col].values
            rmsz = np.where(sig > 0, sub[rms_col].values / sig, np.nan)
            rmsz = rmsz[~np.isnan(rmsz)]
            if len(rmsz) > 0:
                medians[key] = np.median(rmsz)

    return medians


def _compute_rmsz(df, rms_col, sig_col):
    """Compute RMSZ = rms / sigma, dropping NaN/zero-sigma rows."""
    sub = df[[rms_col, sig_col]].dropna()
    sig = sub[sig_col].values
    rmsz = np.where(sig > 0, sub[rms_col].values / sig, np.nan)
    return rmsz[~np.isnan(rmsz)]


def plot_weight_sweep(csv_path, output_path, dpi=300):
    """Generate the 2×3 extended figure from REFMAC metrics CSV."""
    setup_matplotlib()

    df = pd.read_csv(csv_path)
    df_sweep = df[df["weight"] > 0].copy()

    weights = sorted(df_sweep["weight"].unique())
    weight_labels = [str(int(w)) for w in weights]

    phenix_medians = _load_phenix_medians()

    # 5 metrics matching Figure 2 Panel B (geometry sweep, no ADP).
    # For geometry metrics we compute RMSZ = rms / sigma per structure.
    metrics = [
        ("r_work",     "R-work",          "R-work",          "r_work"),
        ("r_free",     "R-free",          "R-free",          "r_free"),
        ("bond_rmsz",  "Bond RMSZ",       "Bond RMSZ",       None),
        ("angle_rmsz", "Angle RMSZ",      "Angle RMSZ",      None),
        ("chiral_rmsz","Chiral RMSZ",     "Chiral RMSZ",     None),
    ]

    rmsz_sources = {
        "bond_rmsz":   ("rmsBOND",      "sigBOND"),
        "angle_rmsz":  ("rmsANGL",      "sigANGL"),
        "chiral_rmsz": ("rmsCHIRAL",    "sigCHIRAL"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    panel_labels = "ABCDE"

    # Hide unused panel(s)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)

    for ax, (col, title, ylabel, phenix_key), label in zip(
        axes.flat, metrics, panel_labels
    ):
        # Build per-weight data arrays
        data = []
        for w in weights:
            wdf = df_sweep[df_sweep["weight"] == w]
            if col in rmsz_sources:
                rms_col, sig_col = rmsz_sources[col]
                vals = _compute_rmsz(wdf, rms_col, sig_col)
            else:
                vals = wdf[col].dropna().values
            data.append(vals)

        medians = [np.median(d) for d in data if len(d) > 0]

        bp = ax.boxplot(
            data,
            labels=weight_labels,
            patch_artist=True,
            widths=0.6,
            showfliers=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLOR_BOX)
            patch.set_alpha(0.5)
        for median_line in bp["medians"]:
            median_line.set_color(COLOR_MEDIAN_LINE)
            median_line.set_linewidth(1.5)

        # Connect medians
        positions = range(1, len(weights) + 1)
        if medians:
            ax.plot(positions, medians, "-o", color=COLOR_MEDIAN_LINE,
                    markersize=4, linewidth=1.2, zorder=5)

        # PHENIX median reference line
        pkey = phenix_key if phenix_key else col
        if pkey in phenix_medians:
            ax.axhline(phenix_medians[pkey], color=COLOR_PHENIX,
                       linestyle="--", linewidth=1.5, alpha=0.8,
                       label="PHENIX median")

        ax.set_xlabel("Geometry weight")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        ax.text(
            -0.08, 1.05, label, transform=ax.transAxes,
            fontsize=20, fontweight="bold", va="bottom", ha="right",
        )

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color=COLOR_MEDIAN_LINE, marker="o", markersize=5,
               linewidth=1.2, label="TorchRef median"),
        Line2D([0], [0], color=COLOR_PHENIX, linestyle="--", linewidth=1.5,
               label="PHENIX median"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=2,
        fontsize=14, frameon=True, bbox_to_anchor=(0.5, 1.01),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.0, h_pad=3.0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot geometry weight screening (Extended Figure 1)")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to refmac_metrics.csv (default: auto-detect from latest experiment)")
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_DIR / "exF1.png"),
        help="Output figure path")
    parser.add_argument(
        "--dpi", type=int, default=300, help="Figure DPI (default: 300)")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        experiments_dir = SCRIPT_DIR / "experiments"
        if experiments_dir.exists():
            for d in sorted(experiments_dir.iterdir(), reverse=True):
                candidate = d / "metrics" / "refmac_metrics.csv"
                if candidate.exists():
                    csv_path = str(candidate)
                    print(f"Using {csv_path}")
                    break
        if csv_path is None:
            print("Error: no refmac_metrics.csv found. "
                  "Provide --csv or run the pipeline first.", file=sys.stderr)
            sys.exit(1)

    plot_weight_sweep(csv_path, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
