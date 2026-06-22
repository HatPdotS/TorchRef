#!/usr/bin/env python3
"""Extended Figure 1: AF-start loss-weight landscape.

Validation-landscape heatmaps for the 10x10 log weight grid (x-ray weight fixed
at 1; geometry & adp weights swept in [1e-3, 1] log-spaced). Each cell is the
MEDIAN over the structure subset of:
  1. Geometry quality   : mean(bond, angle) RMSZ          (diverging @ 1.0)
  2. Main-chain B dist.  : MC-bond B-value RMSZ            (diverging @ 1.0)
  3. R-free - R-work gap : overfitting                     (lower = better)
  4. R-free                                                 (lower = better)

The locked default cell (geometry 0.2 / adp 0.02) is outlined.

Data: figure2_alphafold_start/runs/metrics/weight_grid.csv
      (produced by analysis/submit_weight_grid.py + aggregate_weight_grid.py).

Usage:
    ./.dev/bin/python paper/extended_figures/exF1/plot_exF1.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

BASE = Path(__file__).resolve().parent
CSV = (BASE.parent.parent / "figure2_alphafold_start" / "runs" / "metrics"
       / "weight_grid.csv")
OUTDIR = BASE / "output"
OUT = OUTDIR / "extended_figure1.png"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    if not CSV.exists():
        sys.exit(f"missing {CSV}; run analysis/aggregate_weight_grid.py first")
    rows = list(csv.DictReader(open(CSV)))

    # grid axis values
    geom_by_gi, adp_by_ai = {}, {}
    cells = defaultdict(list)
    for r in rows:
        gi, ai = int(r["gi"]), int(r["ai"])
        geom_by_gi[gi] = float(r["geometry"])
        adp_by_ai[ai] = float(r["adp"])
        cells[(gi, ai)].append(r)
    gis = sorted(geom_by_gi)
    ais = sorted(adp_by_ai)
    ng, na = len(gis), len(ais)
    geom_vals = [geom_by_gi[g] for g in gis]
    adp_vals = [adp_by_ai[a] for a in ais]

    def cell_median(gi, ai, fn):
        vals = [fn(r) for r in cells.get((gi, ai), [])]
        vals = [v for v in vals if v is not None]
        return median(vals) if vals else np.nan

    def mean_geo(r):
        b, a = fnum(r["bond_rmsz"]), fnum(r["angle_rmsz"])
        return (b + a) / 2 if (b is not None and a is not None) else None

    panels = [
        ("Geometry quality  (mean bond/angle RMSZ)",
         lambda r: mean_geo(r), "rmsz"),
        ("Main-chain B distribution  (MC-bond B RMSZ)",
         lambda r: fnum(r["mc_b_rmsz"]), "rmsz"),
        ("R-free − R-work gap  (overfitting)",
         lambda r: (fnum(r["r_free"]) - fnum(r["r_work"]))
         if (fnum(r["r_free"]) is not None and fnum(r["r_work"]) is not None)
         else None, "lower"),
        ("median R-free",
         lambda r: fnum(r["r_free"]), "lower"),
    ]

    def nearest(vals, target):
        return int(np.argmin(np.abs(np.log(np.array(vals)) - np.log(target))))
    # Mark the locked default cell (geom=0.2, adp=0.02).
    gi_def, ai_def = nearest(geom_vals, 0.2), nearest(adp_vals, 0.02)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11))
    for ax, (title, fn, kind) in zip(axes.flat, panels):
        A = np.full((na, ng), np.nan)
        for ci, gi in enumerate(gis):
            for ri, ai in enumerate(ais):
                A[ri, ci] = cell_median(gi, ai, fn)
        extend = "neither"
        if kind == "rmsz":
            finite = A[np.isfinite(A)]
            # Diverge around the ideal 1.0; clip the (catastrophic) under-restrained
            # tail at 3.0 so the useful ~0.5-2 region stays readable.
            cap = 3.0
            vmin = min(float(finite.min()), 0.9) if finite.size else 0.0
            norm = TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=cap)
            if finite.size and float(finite.max()) > cap:
                extend = "max"
            cmap = plt.get_cmap("coolwarm").copy()
        else:
            norm = None
            cmap = plt.get_cmap("viridis_r").copy()
        cmap.set_bad("0.85")
        im = ax.imshow(np.ma.masked_invalid(A), origin="lower", aspect="auto",
                       cmap=cmap, norm=norm)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend=extend)
        if kind == "rmsz":
            cb.set_label("RMSZ (1.0 = ideal; clipped at 3)")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(ng))
        ax.set_xticklabels([f"{v:.3g}" for v in geom_vals], rotation=45, fontsize=7)
        ax.set_yticks(range(na))
        ax.set_yticklabels([f"{v:.3g}" for v in adp_vals], fontsize=7)
        ax.set_xlabel("geometry weight  (rel. to xray=1)")
        ax.set_ylabel("adp weight  (rel. to xray=1)")
        # outline the locked default cell
        ax.add_patch(Rectangle((gi_def - 0.5, ai_def - 0.5), 1, 1, fill=False,
                               edgecolor="lime", lw=2.2))
        ax.plot([], [], color="lime", lw=2, label="default 0.2/0.02")
    axes.flat[0].legend(loc="upper left", fontsize=7, framealpha=0.85)

    fig.suptitle("Extended Figure 1 — AF-start loss-weight landscape (xray=1 "
                 "fixed; median over subset)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
