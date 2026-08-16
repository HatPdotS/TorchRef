#!/usr/bin/env python3
"""3-way comparison of the exF1 weight grids: baseline vs no-SIMU vs locality-only.

For each grid, per cell take the MEDIAN over the common structure set of:
  r_free, gap (=r_free-r_work), mc_b_rmsz (main-chain bond B RMSZ), mean bond/angle RMSZ.
Report, per grid:
  * the locked default cell (geometry 0.2, adp 0.02), and
  * the best-R_free cell (min median r_free), with its coords.

The key question: does removing SIMU (then also the global/KL restraint) degrade
the achievable R_free / overfitting gap / main-chain B behaviour?
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

BASE = Path(__file__).resolve().parent
METRICS = BASE.parent.parent / "figure2_alphafold_start" / "runs" / "metrics"
ABL = BASE / "ablation_runs" / "metrics"
ARCHIVE = BASE.parent.parent / "archive" / "figures_pre_mlrework" / "metrics"
GRIDS = [
    ("baseline (simu+loc+KL)", METRICS / "weight_grid.csv"),
    ("no-SIMU (loc+KL)", ABL / "weight_grid_nosimu.csv"),
    ("locality-only (no SIMU/KL)", ABL / "weight_grid_nosimu_nokl.csv"),
]

#: Named grid sets, so the same per-cell reduction serves more than the SIMU/KL ablation it
#: was written for. `rework` is the one that matters after a numerics change: current build vs
#: the archived pre-rework grid, which answers "did the weight landscape move".
GRID_SETS = {
    "ablation": GRIDS,
    "rework": [
        ("pre-rework (archive)", ARCHIVE / "weight_grid.csv"),
        ("current build", METRICS / "weight_grid.csv"),
    ],
}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    """-> {(gi,ai): {geometry, adp, codes:{code:{r_free,gap,mcb,geo}}}}"""
    cells = {}
    for r in csv.DictReader(open(path)):
        gi, ai = int(r["gi"]), int(r["ai"])
        rf, rw = fnum(r["r_free"]), fnum(r["r_work"])
        b, a = fnum(r["bond_rmsz"]), fnum(r["angle_rmsz"])
        mcb = fnum(r["mc_b_rmsz"])
        c = cells.setdefault((gi, ai), {"geometry": float(r["geometry"]),
                                        "adp": float(r["adp"]), "codes": {}})
        c["codes"][r["code"]] = {
            "r_free": rf, "gap": (rf - rw) if (rf is not None and rw is not None) else None,
            "mcb": mcb, "geo": ((b + a) / 2) if (b is not None and a is not None) else None,
        }
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="ablation", choices=sorted(GRID_SETS),
                   help="Which grid set to compare (default 'ablation'). 'rework' = the "
                        "archived pre-rework grid vs the current build.")
    ap.add_argument("--grids", nargs="+", default=None, metavar="LABEL=PATH",
                   help="Explicit grid list, overriding --set.")
    args = ap.parse_args()

    if args.grids:
        chosen = []
        for spec in args.grids:
            if "=" not in spec:
                raise SystemExit(f"--grids expects LABEL=PATH, got {spec!r}")
            label, path = spec.split("=", 1)
            chosen.append((label, Path(path)))
    else:
        chosen = GRID_SETS[args.set]

    grids = [(name, load(p)) for name, p in chosen if p.exists()]
    for name, p in chosen:
        if not any(name == n for n, _ in grids):
            # Loud, because a silently absent grid turns a two-way comparison into a one-way
            # table that still prints and still looks like a result.
            print(f"  (MISSING grid: {name} -> {p})")
    if len(grids) < len(chosen):
        print(f"  !! {len(grids)} of {len(chosen)} grids loaded — the comparison is partial")

    def med_over(cell, common, key):
        vals = [cell["codes"][c][key] for c in common
                if c in cell["codes"] and cell["codes"][c][key] is not None]
        return median(vals) if vals else float("nan")

    import math

    def nearest(cells, gt, at):
        return min(cells, key=lambda k: abs(math.log(cells[k]["geometry"]) - math.log(gt))
                   + abs(math.log(cells[k]["adp"]) - math.log(at)))

    print(f"\n{'grid':<30} {'cell(geom,adp)':<18} {'R_free':>7} {'gap':>7} "
          f"{'MC-B RMSZ':>10} {'geo RMSZ':>9}")
    print("-" * 86)
    for name, cells in grids:
        common = set.intersection(*[set(c["codes"]) for c in cells.values()]) \
            if cells else set()
        # default cell
        kd = nearest(cells, 0.2, 0.02)
        cd = cells[kd]
        print(f"{name:<30} default {cd['geometry']:.3g}/{cd['adp']:.3g}".ljust(48)
              + f"{med_over(cd, common, 'r_free'):>7.4f} "
              f"{med_over(cd, common, 'gap'):>7.4f} "
              f"{med_over(cd, common, 'mcb'):>10.3f} "
              f"{med_over(cd, common, 'geo'):>9.3f}")
        # best R_free cell
        kb = min(cells, key=lambda k: med_over(cells[k], common, 'r_free'))
        cb = cells[kb]
        print(f"{'':<30} best  {cb['geometry']:.3g}/{cb['adp']:.3g}".ljust(48)
              + f"{med_over(cb, common, 'r_free'):>7.4f} "
              f"{med_over(cb, common, 'gap'):>7.4f} "
              f"{med_over(cb, common, 'mcb'):>10.3f} "
              f"{med_over(cb, common, 'geo'):>9.3f}   (n={len(common)})")
    print("-" * 86)
    print("MC-B RMSZ ~1.0 = ideal B-restraint behaviour; gap lower = less overfit.")


if __name__ == "__main__":
    main()
