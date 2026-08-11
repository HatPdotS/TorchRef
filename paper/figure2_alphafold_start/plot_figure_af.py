#!/usr/bin/env python
"""Publication figure for the AlphaFold-start refinement benchmark (2x2).

Layout:
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
import sys
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

sys.path.insert(0, str(SCRIPT_DIR.parent))
from figure_source_data import dump  # noqa: E402

#: Prefix for this run's source-data CSVs, so Figure 2 ("figure2") and ExtFig 5 ("exF5")
#: write side by side instead of overwriting each other. Rebound by create_figure().
SD = "figure2"

# ── Engines: label, color, z-order (drawn last = on top) ─────────────────────
ENGINES = [
    ("prediction", "Prediction (AlphaFold)", "#4dac26"),  # green
    ("refmac", "Refmac", "#762a83"),                       # purple
    ("phenix", "PHENIX", "#2166ac"),                       # blue
    ("torchref", "TorchRef", "#b2182b"),                   # red
]
COLOR = {e: c for e, _, c in ENGINES}
LABEL = {e: l for e, l, _ in ENGINES}

#: Prefix of the metric CSVs to read (``<PREFIX>_rfactors.csv`` etc.), matching
#: ``aggregate_figure_metrics.py --prefix``. Rebound by main().
PREFIX = "fig"

#: The one engine that is a starting model rather than a refinement, so it has no runtime and
#: no per-cycle trajectory. Named once instead of being spelled out in each panel.
PREDICTION = "prediction"

#: Fallback palette for engines introduced via --engines without an explicit colour.
_PALETTE = ["#b2182b", "#2166ac", "#762a83", "#4dac26", "#e08214", "#01665e", "#8c510a"]


def refined_engines():
    """Engine names that have a runtime and a per-cycle trajectory, in ENGINES order."""
    return [e for e, _, _ in ENGINES if e != PREDICTION]


def set_engines(specs):
    """Rebind ENGINES/COLOR/LABEL from ``NAME=LABEL[:COLOR]`` strings.

    Lets ExtFig 5 (four x-ray target modes) reuse this module's exact 2x2 layout instead of
    carrying a second copy of it. Panels C and D derive their engine list from ENGINES via
    :func:`refined_engines`, so a new engine set reaches every panel -- they used to hold
    hardcoded ``("refmac", "phenix", "torchref")`` literals and would have silently dropped
    any engine not in them, blanking three of ExtFig 5's four cells in half the figure.
    """
    global ENGINES, COLOR, LABEL
    out = []
    for i, spec in enumerate(specs):
        if "=" not in spec:
            raise SystemExit(f"--engines expects NAME=LABEL[:COLOR], got {spec!r}")
        name, rest = spec.split("=", 1)
        label, _, colour = rest.partition(":")
        out.append((name, label, colour or _PALETTE[i % len(_PALETTE)]))
    ENGINES = out
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
    sd_rows = []
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.2, linewidth=0.8, zorder=1)
    for i, (engine, _, color) in enumerate(ENGINES):
        df = by_engine.get(engine)
        if df is None or df.empty:
            continue
        sd_rows.extend({"engine": engine, "code": c, "r_work": w, "r_free": f}
                       for c, w, f in zip(df["code"], df["r_work"], df["r_free"]))
        ax.scatter(df["r_work"], df["r_free"], color=color, alpha=0.30,
                   s=5, linewidths=0, zorder=2 + i)
        # light median guides
        ax.axvline(df["r_work"].median(), color=color, ls="--", lw=0.7,
                   alpha=0.45, zorder=6)
        ax.axhline(df["r_free"].median(), color=color, ls="--", lw=0.7,
                   alpha=0.45, zorder=6)
    dump(f"{SD}_panelA_rfactors", sd_rows)
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
    bounds = {m: (0.0, 3.0) for m in metrics}  # RMSZ, linear 0–3

    def rmsz(df, m):
        sub = df[[m, sigma_col[m]]].dropna()
        sig = sub[sigma_col[m]].values
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(sig > 0, sub[m].values / sig, np.nan)
        return out[~np.isnan(out)]

    def to_x(m, v):
        lo, hi = bounds[m]
        return np.clip((v - lo) / (hi - lo + 1e-12), 0.0, 1.0)

    n = len(metrics)
    y_positions = np.arange(n)[::-1].astype(float)
    row_half = 0.32
    for y in y_positions:
        ax.axhspan(y - row_half, y + row_half, color="0.96", zorder=0)

    sd_rows = []
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
            # The RMSZ quartiles the strip draws, in RMSZ units (not the 0-1 axis fraction).
            sd_rows.append({"engine": engine, "metric": labels[i], "n": int(len(vals)),
                            "q25": q25, "median": q50, "q75": q75})
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
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=11)
        tick_y = y - row_half - 0.04
        for val in (0, 1, 2, 3):
            x_frac = (val - lo) / (hi - lo)
            ax.plot([x_frac, x_frac], [y - row_half, y - row_half - 0.02],
                    color="0.4", lw=0.6, clip_on=False)
            ax.text(x_frac, tick_y, f"{val:g}", ha="center", va="top",
                    fontsize=8, color="0.35")

    dump(f"{SD}_panelB_geometry_rmsz", sd_rows)
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
    rng = np.random.default_rng(0)             # reproducible jitter

    # Speed order is MEASURED, not hardcoded. This used to read
    # ["refmac", "torchref", "phenix"], which both fixed the ordering to one arm set and
    # silently dropped any engine absent from the literal.
    vals = {e: runtime_long.loc[runtime_long.engine == e, "wall_s"].dropna().values / 60.0
            for e in refined_engines()}
    order = sorted((e for e in vals if len(vals[e])), key=lambda e: float(np.median(vals[e])))
    # An engine with no runtime rows would otherwise just be an absent box -- visually
    # indistinguishable from a deliberate omission. Say so.
    empty = [e for e in vals if not len(vals[e])]
    if empty:
        print(f"  WARNING panel C: no runtime data for {empty} — box(es) omitted")
    if not order:
        ax.set_title("Wall-clock runtime (no data)")
        return
    data = [vals[e] for e in order]

    # One row per plotted point, in the panel's units (minutes), plus the box position so
    # the left-to-right speed order is reproducible without re-deriving it.
    sd = runtime_long.dropna(subset=["wall_s"])
    dump(f"{SD}_panelC_runtime",
         [{"engine": e, "box_position": i, "code": r.code,
           "runtime_min": r.wall_s / 60.0}
          for i, e in enumerate(order) for r in sd[sd.engine == e].itertuples()])

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

    ax.set_yscale("log")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_xticks(pos)
    ax.set_xticklabels([LABEL[e] for e in order])
    ax.set_ylabel("Wall-clock runtime (min)")
    ax.set_title("Wall-clock runtime")
    # The per-engine median-runtime ratios are reported in the figure caption /
    # FIGURE_MEDIANS Fig2C rather than annotated on the panel.


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
    sd_rows = []
    # Derived from ENGINES, not a literal, for the same reason as panel C.
    for engine in refined_engines():
        sub = percycle[percycle.engine == engine]
        if sub.empty:
            print(f"  WARNING panel D: no per-cycle data for {engine!r} — series omitted")
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

        sd_rows.extend({"engine": engine, "macrocycle": int(xx),
                        "median_fraction": mm, "q25": a, "q75": b,
                        "n_structures": int(keep.sum()), "guard_min_delta_rfree": GUARD}
                       for xx, mm, a, b in zip(x, med, q1, q3))
        ax.fill_between(x, q1, q3, color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(x, med, "-", color=color, lw=2.2, zorder=4)
        handles.append(Line2D([0], [0], color=color, lw=2.2,
                              label=f"{LABEL[engine]} (n={int(keep.sum())})"))

    dump(f"{SD}_panelD_convergence", sd_rows)
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8, zorder=1)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("Macrocycle (from start)")
    ax.set_ylabel("Fraction of total R-free improvement")
    ax.set_title("Convergence speed (normalized)")
    ax.legend(handles=handles, loc="lower right", fontsize=11, frameon=True)


# ── Composite ────────────────────────────────────────────────────────────────
def create_figure(outpath: str, dpi: int = 300, prefix: str = None):
    global SD
    setup_matplotlib()

    pre = prefix or PREFIX
    # "fig" is Figure 2, any other prefix is a reuse of this layout (ExtFig 5 uses "mode").
    SD = "figure2" if pre == "fig" else f"exF5_{pre}" if pre != "mode" else "exF5"
    rfac = pd.read_csv(METRICS_DIR / f"{pre}_rfactors.csv")
    geom = pd.read_csv(METRICS_DIR / f"{pre}_geometry.csv")
    runtime = pd.read_csv(METRICS_DIR / f"{pre}_runtime.csv")
    percycle = pd.read_csv(METRICS_DIR / f"{pre}_percycle.csv")

    # Fail loudly on an engine the CSVs do not contain. Every panel filters by engine name, so
    # a typo in --engines would otherwise render an empty series and look like "that arm has
    # no data yet" rather than "that name is wrong".
    have = set(rfac.engine.unique())
    unknown = [e for e, _, _ in ENGINES if e not in have]
    if unknown:
        raise SystemExit(f"{pre}_rfactors.csv has no rows for engine(s) {unknown}; "
                         f"it contains {sorted(have)}")

    # Conserved set: the identical structures every engine has an R-factor for
    # (written by aggregate_figure_metrics.py). Restricting every panel to it makes
    # the medians comparable — no engine is scored on a different/easier subset.
    conserved_file = METRICS_DIR / ("conserved_codes.txt" if pre == "fig"
                                    else f"{pre}_conserved_codes.txt")
    conserved = set(conserved_file.read_text().split()) if conserved_file.exists() else None
    if conserved:
        for df in (rfac, geom, runtime, percycle):
            df.drop(df.index[~df.code.isin(conserved)], inplace=True)
        print(f"Conserved set: n={len(conserved)} structures (all engines)")

    rf_by = {e: rfac[rfac.engine == e] for e in COLOR}
    geom_by = {e: geom[geom.engine == e] for e in COLOR}

    print("Median R-free (validated, conserved set):")
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
    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(ENGINES)), fontsize=14,
               frameon=True, bbox_to_anchor=(0.5, 1.005))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved figure to: {outpath}")


def main():
    p = argparse.ArgumentParser(
        description="AlphaFold-start benchmark figure.",
        epilog="ExtFig 5 (x-ray target modes) reuses this exact layout:\n"
               "  plot_figure_af.py --prefix mode -o ../extended_figures/exF5/output/"
               "extended_figure5.png \\\n"
               "      --engines ml=ml:#b2182b ml_noalpha=ml_noalpha:#2166ac \\\n"
               "                ml_full=ml_full:#762a83 nll_beta=nll_beta:#e08214",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--prefix", default=PREFIX,
                   help="Metric-CSV prefix to read (default 'fig'); matches "
                        "aggregate_figure_metrics.py --prefix.")
    p.add_argument("--engines", nargs="+", default=None, metavar="NAME=LABEL[:COLOR]",
                   help="Engine set, in draw order (last on top). Default is the four "
                        "Figure-2 engines. Use this for a figure over a different arm set, "
                        "e.g. the x-ray target modes.")
    args = p.parse_args()
    if args.engines:
        set_engines(args.engines)
    outpath = args.output or str(OUTPUT_DIR / "figure_af_benchmark.png")
    create_figure(outpath, dpi=args.dpi, prefix=args.prefix)


if __name__ == "__main__":
    main()
