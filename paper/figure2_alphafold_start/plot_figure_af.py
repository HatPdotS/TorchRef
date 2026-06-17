#!/usr/bin/env python
"""Publication figure for the AlphaFold-start refinement benchmark (2x2).

Layout (mimics paper/figure2_validation/plot_figure2.py):
  A) R-work vs R-free scatter (PHENIX-validated): prediction vs Refmac/Phenix/TorchRef
  B) Geometry quality strip (bond/angle/chiral RMSZ + main-chain B-factor RMSZ)
  C) Runtime comparison (TorchRef vs Phenix wall-clock, log-log)
  D) Per-cycle *program-reported* R-work & R-free convergence

Reads the four CSVs produced by analysis/aggregate_figure_metrics.py
(runs/metrics/fig_*.csv).

Usage
-----
    ./.dev/bin/python paper/figure2_alphafold_start/plot_figure_af.py
    ./.dev/bin/python paper/figure2_alphafold_start/plot_figure_af.py -o fig.png --dpi 600
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
METRICS_DIR = SCRIPT_DIR / "runs" / "metrics"
OUTPUT_DIR = SCRIPT_DIR / "figures"

# ── Engines: label, color, z-order (drawn last = on top) ─────────────────────
ENGINES = [
    ("prediction", "Prediction (AlphaFold)", "#4dac26"),  # green
    ("refmac", "Refmac", "#762a83"),                       # purple
    ("phenix", "PHENIX", "#2166ac"),                       # blue
    ("torchref", "TorchRef", "#b2182b"),                   # red
]
COLOR = {e: c for e, _, c in ENGINES}
LABEL = {e: l for e, l, _ in ENGINES}


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
        "figure.titlesize": 21,
    })


# ── Panel A: R-work vs R-free scatter (validated) ────────────────────────────
def plot_rfactor_scatter(ax, by_engine):
    lo, hi = 0.15, 0.55
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.2, linewidth=0.8, zorder=1)
    for i, (engine, _, color) in enumerate(ENGINES):
        df = by_engine.get(engine)
        if df is None or df.empty:
            continue
        ax.scatter(df["r_work"], df["r_free"], color=color, alpha=0.30,
                   s=5, linewidths=0, zorder=2 + i)
        # light median guides
        ax.axvline(df["r_work"].median(), color=color, ls="--", lw=0.7,
                   alpha=0.45, zorder=6)
        ax.axhline(df["r_free"].median(), color=color, ls="--", lw=0.7,
                   alpha=0.45, zorder=6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("R-work")
    ax.set_ylabel("R-free")
    ax.set_aspect("equal")
    ax.set_title("R-factors (PHENIX-validated)")


# ── Panel B: geometry quality strip (parallel coordinates) ───────────────────
def plot_quality_strips(ax, by_engine):
    metrics = ["rmsBOND", "rmsANGL", "rmsCHIRAL", "rmsB_mc_bond"]
    labels = ["Bond RMSZ", "Angle RMSZ", "Chiral RMSZ", "MC B-factor RMSZ"]
    sigma_col = {"rmsBOND": "sigBOND", "rmsANGL": "sigANGL",
                 "rmsCHIRAL": "sigCHIRAL", "rmsB_mc_bond": "sigB_mc_bond"}
    bounds = {m: (0.1, 10.0) for m in metrics}  # RMSZ, log-normalized

    def rmsz(df, m):
        sub = df[[m, sigma_col[m]]].dropna()
        sig = sub[sigma_col[m]].values
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(sig > 0, sub[m].values / sig, np.nan)
        return out[~np.isnan(out)]

    LOG_FLOOR = 1e-3

    def to_x(m, v):
        lo, hi = bounds[m]
        lo_l, hi_l = np.log10(lo), np.log10(hi)
        v_l = np.log10(np.maximum(v, LOG_FLOOR))
        return np.clip((v_l - lo_l) / (hi_l - lo_l + 1e-12), 0.0, 1.0)

    n = len(metrics)
    y_positions = np.arange(n)[::-1].astype(float)
    row_half = 0.32
    for y in y_positions:
        ax.axhspan(y - row_half, y + row_half, color="0.96", zorder=0)

    for engine, _, color in ENGINES:
        df = by_engine.get(engine)
        if df is None or df.empty:
            continue
        med_xs = []
        for i, m in enumerate(metrics):
            vals = rmsz(df, m)
            if len(vals) == 0:
                med_xs.append(np.nan)
                continue
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            y = y_positions[i]
            x25, x75 = to_x(m, q25), to_x(m, q75)
            ax.fill_between([x25, x75], y - row_half, y + row_half,
                            color=color, alpha=0.22, linewidth=0, zorder=3)
            med_xs.append(to_x(m, q50))
        ax.plot(med_xs, y_positions, color=color, lw=1.8, alpha=0.9, zorder=5)
        ax.plot(med_xs, y_positions, "o", color=color, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)

    for i, (m, label) in enumerate(zip(metrics, labels)):
        y = y_positions[i]
        lo, hi = bounds[m]
        geo_mid = np.sqrt(lo * hi)
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=11)
        tick_y = y - row_half - 0.04
        for x_frac, val in [(0.0, lo), (0.5, geo_mid), (1.0, hi)]:
            ax.plot([x_frac, x_frac], [y - row_half, y - row_half - 0.02],
                    color="0.4", lw=0.6, clip_on=False)
            ax.text(x_frac, tick_y, f"{val:g}", ha="center", va="top",
                    fontsize=8, color="0.35")

    ax.set_xlim(-0.18, 1.05)
    ax.set_ylim(-0.8, n - 1 + 0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Geometry (RMSZ vs REFMAC restraints)")


# ── Panel C: runtime comparison ──────────────────────────────────────────────
def plot_runtime_box(ax, runtime_long):
    """Per-engine wall-clock distribution (log y), fastest → slowest."""
    order = ["refmac", "torchref", "phenix"]  # speed order
    rng = np.random.default_rng(0)             # reproducible jitter

    data, meds = [], {}
    for e in order:
        v = runtime_long.loc[runtime_long.engine == e, "wall_s"].dropna().values / 60.0
        data.append(v)
        meds[e] = np.median(v)

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
    # median labels
    for i, e in enumerate(order):
        ax.annotate(f"{meds[e]:.2f} min", (pos[i], meds[e]),
                    xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10, color="0.15")

    ax.set_yscale("log")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(pos)
    ax.set_xticklabels([LABEL[e] for e in order])
    ax.set_ylabel("Wall-clock runtime (min)")
    ax.set_title("Wall-clock runtime")

    # paired median ratios (TorchRef as reference)
    wide = runtime_long.pivot_table(index="code", columns="engine", values="wall_s")
    def pair_ratio(a, b):
        m = wide[a].notna() & wide[b].notna()
        return float(np.median(wide.loc[m, a] / wide.loc[m, b]))
    r_ref = pair_ratio("torchref", "refmac")
    r_phe = pair_ratio("torchref", "phenix")
    txt = (f"TorchRef vs Refmac: {r_ref:.1f}× slower\n"
           f"TorchRef vs PHENIX: {1/r_phe:.1f}× faster")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.85))


# ── Panel D: normalized convergence speed ────────────────────────────────────
def plot_percycle_normalized(ax, percycle):
    """Fraction of each engine's total R-free improvement vs macrocycle.

    Program-reported R-free is normalized *within* each structure/engine
    (start → 0, final → 1) so the panel compares convergence *speed*, not the
    incomparable absolute program-reported R-factors. Structures whose total
    R-free improvement is < GUARD are dropped so a near-flat trajectory cannot
    blow up the ratio.
    """
    GUARD = 0.02  # min total ΔR-free to be included (Å-free units)

    handles = []
    for engine in ("refmac", "phenix", "torchref"):
        sub = percycle[percycle.engine == engine]
        if sub.empty:
            continue
        color = COLOR[engine]
        first, last = sub.cycle.min(), sub.cycle.max()
        piv = sub.pivot_table(index="code", columns="cycle", values="r_free")
        cyc = np.sort(piv.columns.values)
        piv = piv[cyc]
        r0 = piv[first]
        denom = r0 - piv[last]
        keep = denom > GUARD
        piv, r0, denom = piv[keep], r0[keep], denom[keep]

        frac = (r0.values[:, None] - piv.values) / denom.values[:, None]
        x = cyc - first  # rebase so every engine starts at macrocycle 0
        med = np.nanmedian(frac, axis=0)
        q1 = np.nanpercentile(frac, 25, axis=0)
        q3 = np.nanpercentile(frac, 75, axis=0)

        ax.fill_between(x, q1, q3, color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(x, med, "-", color=color, lw=2.2, zorder=4)
        handles.append(Line2D([0], [0], color=color, lw=2.2,
                              label=f"{LABEL[engine]} (n={int(keep.sum())})"))

    ax.axhline(1.0, color="0.6", ls=":", lw=0.8, zorder=1)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Macrocycle (from start)")
    ax.set_ylabel("Fraction of total R-free improvement")
    ax.set_title("Convergence speed (normalized)")
    ax.legend(handles=handles, loc="lower right", fontsize=11, frameon=True)


# ── Composite ────────────────────────────────────────────────────────────────
def create_figure(outpath: str, dpi: int = 300):
    setup_matplotlib()

    rfac = pd.read_csv(METRICS_DIR / "fig_rfactors.csv")
    geom = pd.read_csv(METRICS_DIR / "fig_geometry.csv")
    runtime = pd.read_csv(METRICS_DIR / "fig_runtime.csv")
    percycle = pd.read_csv(METRICS_DIR / "fig_percycle.csv")

    rf_by = {e: rfac[rfac.engine == e] for e in COLOR}
    geom_by = {e: geom[geom.engine == e] for e in COLOR}

    print("Median R-free (validated):")
    for e, lbl, _ in ENGINES:
        df = rf_by[e]
        print(f"  {lbl:<22} n={len(df):>4}  R-work={df.r_work.median():.4f} "
              f" R-free={df.r_free.median():.4f}")

    fig = plt.figure(figsize=(12, 10))
    ax_a = fig.add_subplot(2, 2, 1)
    ax_b = fig.add_subplot(2, 2, 2)
    ax_c = fig.add_subplot(2, 2, 3)
    ax_d = fig.add_subplot(2, 2, 4)
    plot_rfactor_scatter(ax_a, rf_by)
    plot_quality_strips(ax_b, geom_by)
    plot_runtime_box(ax_c, runtime)
    plot_percycle_normalized(ax_d, percycle)

    for ax, label in [(ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")]:
        ax.text(-0.08, 1.12, label, transform=ax.transAxes, fontsize=21,
                fontweight="bold", va="top", ha="left")

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
               markersize=10, label=l)
        for _, l, c in ENGINES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=14,
               frameon=True, bbox_to_anchor=(0.5, 1.005))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved figure to: {outpath}")


def main():
    p = argparse.ArgumentParser(description="AlphaFold-start benchmark figure.")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()
    outpath = args.output or str(OUTPUT_DIR / "figure_af_benchmark.png")
    create_figure(outpath, dpi=args.dpi)


if __name__ == "__main__":
    main()
