#!/usr/bin/env python
"""
Create a publication-quality figure for the validation benchmark (Figure 2).

Usage
-----
    python plot_figure2.py
    python plot_figure2.py --output my_figure.png
    python plot_figure2.py --dpi 600

Layout (2x2):
  A) R-work vs R-free scatter   B) Quality radar (box-whisker on polar axes)
  C) Overall B histogram         D) Geometry scatter (Bond vs Angle RMSD)
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
from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# ── Colors ───────────────────────────────────────────────────────────────────
COLOR_PHENIX = "#2166ac"     # blue
COLOR_TORCHREF = "#b2182b"   # red
COLOR_INITIAL = "#4dac26"    # green


# ── Matplotlib style ─────────────────────────────────────────────────────────
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
        "legend.fontsize": 15,
        "figure.titlesize": 21,
    })


# ── Data loading ─────────────────────────────────────────────────────────────
def load_experiment_data(data_dir: Path):
    """Load all metric CSVs, returning per-variant DataFrames.

    Returns
    -------
    torchref, phenix, initial : pd.DataFrame
        Each indexed by PDB code with columns from refmac_metrics + internal_metrics.
    deviations : pd.DataFrame or None
    codes : list[str]
        Common PDB codes across all three variants.
    """
    if not data_dir.exists():
        sys.exit(f"Data directory not found: {data_dir}")

    variant_label = "default"

    # -- Load available metrics --
    refmac_path = data_dir / "refmac_metrics.csv"
    internal_path = data_dir / "internal_metrics.csv"

    df_refmac = pd.read_csv(refmac_path) if refmac_path.exists() else None
    df_internal = pd.read_csv(internal_path) if internal_path.exists() else None

    # Use refmac if available (has geometry + R-factors); fall back to internal
    if df_refmac is not None and not df_refmac.empty:
        df_primary = df_refmac
    elif df_internal is not None:
        df_primary = df_internal
    else:
        sys.exit(f"No metrics found in {data_dir}")

    # Split by variant
    def _extract(df, variant):
        sub = df[df["variant"] == variant].copy()
        sub = sub.set_index("code")
        return sub

    torchref_data = _extract(df_primary, variant_label)
    phenix_data = _extract(df_primary, "phenix")
    initial_data = _extract(df_primary, "initial")

    # Common codes across all three variants
    codes = sorted(
        set(torchref_data.index)
        & set(phenix_data.index)
        & set(initial_data.index)
    )

    # Merge internal metrics columns into primary frames if they came from refmac
    if df_internal is not None and df_refmac is not None and not df_refmac.empty:
        for variant, frame in [(variant_label, torchref_data),
                                ("phenix", phenix_data),
                                ("initial", initial_data)]:
            sub = df_internal[df_internal["variant"] == variant].set_index("code")
            for col in ["log_b_std", "b_mean", "b_std", "b_min", "b_max"]:
                if col in sub.columns and col not in frame.columns:
                    frame[col] = sub[col]

    # Filter to common codes
    torchref_data = torchref_data.loc[torchref_data.index.isin(codes)]
    phenix_data = phenix_data.loc[phenix_data.index.isin(codes)]
    initial_data = initial_data.loc[initial_data.index.isin(codes)]

    # Deviations (optional)
    dev_path = data_dir / "deviations.csv"
    deviations = pd.read_csv(dev_path).set_index("code") if dev_path.exists() else None

    # Runtimes (optional)
    rt_path = data_dir / "runtimes.csv"
    runtimes = pd.read_csv(rt_path).set_index("code") if rt_path.exists() else None

    return torchref_data, phenix_data, initial_data, deviations, runtimes, codes


# ── Panel A: R-work vs R-free scatter ────────────────────────────────────────
def plot_rfactor_scatter(ax, torchref, phenix, initial):
    ax.scatter(
        initial["r_work"], initial["r_free"],
        color=COLOR_INITIAL, alpha=0.35, s=4, linewidths=0, zorder=2,
    )
    ax.scatter(
        phenix["r_work"], phenix["r_free"],
        color=COLOR_PHENIX, alpha=0.35, s=4, linewidths=0, zorder=3,
    )
    ax.scatter(
        torchref["r_work"], torchref["r_free"],
        color=COLOR_TORCHREF, alpha=0.35, s=4, linewidths=0, zorder=3,
    )
    lo, hi = 0.10, 0.45
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.2, linewidth=0.8, zorder=1)

    # Median indicator lines
    for df, color in [(initial, COLOR_INITIAL), (phenix, COLOR_PHENIX),
                      (torchref, COLOR_TORCHREF)]:
        med_rw = df["r_work"].median()
        med_rf = df["r_free"].median()
        ax.axvline(med_rw, color=color, ls="--", lw=0.8, alpha=0.5, zorder=4)
        ax.axhline(med_rf, color=color, ls="--", lw=0.8, alpha=0.5, zorder=4)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("R-work")
    ax.set_ylabel("R-free")
    ax.set_aspect("equal")


# ── Panel B: Quality strip plot (parallel coordinates, stacked) ──────────────
def plot_quality_strips(ax, torchref, phenix, initial):
    """Parallel-coordinates style plot with one row per metric.

    Each row has its own linear scale in real units. Per group we draw a
    shaded 5-95 percentile band and an IQR band (as horizontal spans per
    row), and connect each group's medians across rows with a line so the
    reader can trace a single group's performance through all metrics.
    """
    all_metrics = ["r_work", "r_free", "rmsBOND", "rmsANGL", "rmsCHIRAL", "rmsB_mc_bond"]
    bounds = {
        "r_work": (0.15, 0.4),
        "r_free": (0.15, 0.4),
        "rmsBOND": (0.1, 20.0),
        "rmsANGL": (0.1, 20.0),
        "rmsCHIRAL": (0.1, 20.0),
        "rmsB_mc_bond": (0.1, 20.0),
    }

    all_labels = [
        "R-work", "R-free", "Bond RMSZ",
        "Angle RMSZ", "Chiral RMSZ", "MC B-factor RMSZ",
    ]
    # For geometry / ADP RMSDs we divide by REFMAC's per-structure Av(Sigma)
    # (stored in sig* columns) to get a true RMSZ. If the sigma column is
    # missing the row is dropped from the plot.
    sigma_col = {
        "rmsBOND": "sigBOND",
        "rmsANGL": "sigANGL",
        "rmsCHIRAL": "sigCHIRAL",
        "rmsB_mc_bond": "sigB_mc_bond",
    }
    fmt_map = {
        "r_work": "{:.3f}",
        "r_free": "{:.3f}",
        "rmsBOND": "{:.2f}",
        "rmsANGL": "{:.2f}",
        "rmsCHIRAL": "{:.2f}",
        "rmsB_mc_bond": "{:.2f}",
    }

    data_groups = [
        ("PHENIX", phenix, COLOR_PHENIX),
        ("TorchRef", torchref, COLOR_TORCHREF),
        ("Initial", initial, COLOR_INITIAL),
    ]

    def metric_available(m):
        if not all(m in df.columns for _, df, _ in data_groups):
            return False
        sig = sigma_col.get(m)
        if sig is None:
            return True
        return all(sig in df.columns for _, df, _ in data_groups)

    metrics = [c for c in all_metrics if metric_available(c)]
    labels = [all_labels[all_metrics.index(m)] for m in metrics]
    if len(metrics) < 2:
        ax.text(0.5, 0.5, "Insufficient metrics",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")
        return

    def get_vals(df, m):
        """Return the metric values, converted to RMSZ when a sigma col exists."""
        sig_name = sigma_col.get(m)
        if sig_name is None:
            return df[m].dropna().values
        sub = df[[m, sig_name]].dropna()
        sig = sub[sig_name].values
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(sig > 0, sub[m].values / sig, np.nan)
        return out[~np.isnan(out)]  
    
    LOG_FLOOR = 1e-3  # for log-scaling metrics with long tails, to avoid outliers dominating the plot

    def to_x(m, v):
        lo, hi = bounds[m]
        lo_l = np.log10(lo)
        hi_l = np.log10(hi)
        v_l = np.log10(np.maximum(v, LOG_FLOOR))
        return np.clip((v_l - lo_l) / (hi_l - lo_l + 1e-12), 0.0, 1.0)

    n = len(metrics)
    # Top row = first metric, so reverse y so it reads top-down.
    y_positions = np.arange(n)[::-1].astype(float)
    row_half = 0.32  # vertical half-height of each row's shaded band

    # Light guide rectangles per row
    for y in y_positions:
        ax.axhspan(y - row_half, y + row_half, color="0.96", zorder=0)

    # Draw each group: 5-95% band (light) + IQR band (darker) per row,
    # median markers, and a line connecting the medians across all rows.
    for gname, df, color in data_groups:
        med_xs = []
        for i, m in enumerate(metrics):
            vals = get_vals(df, m)
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            y = y_positions[i]
            x25, x50, x75 = (to_x(m, v) for v in (q25, q50, q75))

            ax.fill_between([x25, x75], y - row_half, y + row_half,
                            color=color, alpha=0.25, linewidth=0, zorder=3)
            med_xs.append(x50)

        # Connecting line through medians + markers on top
        ax.plot(med_xs, y_positions, color=color, lw=1.8, alpha=0.9, zorder=5)
        ax.plot(med_xs, y_positions, "o", color=color, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)

    # Per-row metric label (left) and real-units scale markers (below row).
    # Ticks are geometrically spaced because the x-axis is log10-normalized.
    for i, (m, label) in enumerate(zip(metrics, labels)):
        y = y_positions[i]
        lo, hi = bounds[m]
        geo_mid = np.sqrt(lo * hi)
        fmt = fmt_map.get(m, "{:.3g}")
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=11)
        tick_y = y - row_half - 0.04
        for x_frac, val in [(0.0, lo), (0.5, geo_mid), (1.0, hi)]:
            ax.plot([x_frac, x_frac], [y - row_half, y - row_half - 0.02],
                    color="0.4", lw=0.6, clip_on=False)
            ax.text(x_frac, tick_y, fmt.format(val),
                    ha="center", va="top", fontsize=8, color="0.35")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.8, n - 1 + 0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── Panel C: RMSD from initial structure ─────────────────────────────────────
def plot_rmsd_from_initial(ax, deviations):
    """Histogram of mean per-atom RMSD from the initial (shaken) structure."""
    bins = np.linspace(0, 1.0, 50)
    ax.hist(
        deviations["rmsd_original_phenix_mean"].dropna(), bins=bins,
        color=COLOR_PHENIX, alpha=0.5, edgecolor="none",
        label="Initial \u2192 PHENIX",
    )
    ax.hist(
        deviations["rmsd_original_torchref_mean"].dropna(), bins=bins,
        color=COLOR_TORCHREF, alpha=0.5, edgecolor="none",
        label="Initial \u2192 TorchRef",
    )
    ax.set_xlabel("Mean RMSD from initial (\u00c5)")
    ax.set_ylabel("Count")


# ── Panel D: Runtime comparison ──────────────────────────────────────────────
def plot_runtime_scatter(ax, runtimes, variant_label="default"):
    """Scatter of TorchRef vs Phenix wall-clock runtime (seconds -> minutes)."""
    col_tr = f"wall_s_{variant_label}"
    col_ph = "wall_s_phenix"
    mask = runtimes[col_tr].notna() & runtimes[col_ph].notna()
    tr = runtimes.loc[mask, col_tr].values / 60.0
    ph = runtimes.loc[mask, col_ph].values / 60.0

    ax.scatter(ph, tr, color="gray", alpha=0.35, s=6, linewidths=0, zorder=2)
    lo = min(ph.min(), tr.min()) * 0.8
    hi = max(ph.max(), tr.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=0.8, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PHENIX runtime (min)")
    ax.set_ylabel("TorchRef runtime (min)")
    ax.set_aspect("equal")

    median_ratio = np.median(tr / ph)
    ax.text(
        0.95, 0.05, f"median ratio = {median_ratio:.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
    )


# ── Composite figure ─────────────────────────────────────────────────────────
def create_publication_figure(data_dir: Path, outpath: str, dpi: int = 300):
    setup_matplotlib()

    torchref, phenix, initial, deviations, runtimes, codes = load_experiment_data(data_dir)
    n = len(codes)
    print(f"Loaded data: {n} structures (common across all variants)")

    for label, df in [("TorchRef", torchref), ("PHENIX", phenix), ("Initial", initial)]:
        rw = df["r_work"].median()
        rf = df["r_free"].median()
        print(f"  {label:10s}  median R-work={rw:.4f}  median R-free={rf:.4f}")

    variant_label = "default"

    fig = plt.figure(figsize=(12, 10))

    # A: R-factors
    ax_a = fig.add_subplot(2, 2, 1)
    plot_rfactor_scatter(ax_a, torchref, phenix, initial)

    # B: Quality strips (parallel-coordinates style)
    ax_b = fig.add_subplot(2, 2, 2)
    plot_quality_strips(ax_b, torchref, phenix, initial)

    # C: RMSD from initial
    ax_c = fig.add_subplot(2, 2, 3)
    if deviations is not None and not deviations.empty:
        plot_rmsd_from_initial(ax_c, deviations)
        ax_c.legend(fontsize=11)
    else:
        ax_c.text(0.5, 0.5, "No deviations data", transform=ax_c.transAxes,
                  ha="center", va="center", fontsize=14, color="gray")

    # D: Runtime comparison
    ax_d = fig.add_subplot(2, 2, 4)
    if runtimes is not None and not runtimes.empty:
        plot_runtime_scatter(ax_d, runtimes.loc[runtimes.index.isin(codes)], variant_label)
    else:
        ax_d.text(0.5, 0.5, "No runtime data", transform=ax_d.transAxes,
                  ha="center", va="center", fontsize=14, color="gray")

    # Panel labels
    for label, x, y in [("A", 0.08, 0.95), ("B", 0.52, 0.92),
                          ("C", 0.08, 0.48), ("D", 0.52, 0.45)]:
        fig.text(x, y, label, fontsize=21, fontweight="bold")

    # Shared legend
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PHENIX,
               markersize=10, label="PHENIX refined"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_TORCHREF,
               markersize=10, label="TorchRef"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_INITIAL,
               markersize=10, label="Initial (shaken)"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", ncol=3,
        fontsize=15, frameon=True, bbox_to_anchor=(0.5, 1.01),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved publication figure to: {outpath}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Create Figure 2: validation benchmark (TorchRef vs PHENIX).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to data directory (default: data/ next to this script)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path (default: output/figure2.png next to this script)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI (default: 300)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    if args.output is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        outpath = str(OUTPUT_DIR / "figure2.png")
    else:
        outpath = args.output

    create_publication_figure(data_dir, outpath, dpi=args.dpi)


if __name__ == "__main__":
    main()
