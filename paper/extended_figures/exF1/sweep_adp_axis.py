#!/usr/bin/env python3
"""Does the weight optimum shift when SIMU/KL are removed?

For each grid, walk the adp-weight axis (at the geometry weight nearest 0.2) and
report median R_free / gap / main-chain-B RMSZ per adp weight. Also report, per
grid, the BEST cell anywhere for two objectives:
  * min R_free                         (best data fit)
  * min R_free subject to MC-B RMSZ<=2 (good fit AND well-behaved bonded B)
to see whether locality-only can reach baseline B-geometry at some weight, and
what R_free that costs.
"""
import csv
import math
from pathlib import Path
from statistics import median

BASE = Path(__file__).resolve().parent
METRICS = BASE.parent.parent / "figure2_alphafold_start" / "runs" / "metrics"
ABL = BASE / "ablation_runs" / "metrics"
GRIDS = [
    ("baseline", METRICS / "weight_grid.csv"),
    ("no-SIMU", ABL / "weight_grid_nosimu.csv"),
    ("loc-only", ABL / "weight_grid_nosimu_nokl.csv"),
]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    cells = {}
    for r in csv.DictReader(open(path)):
        gi, ai = int(r["gi"]), int(r["ai"])
        rf, rw = fnum(r["r_free"]), fnum(r["r_work"])
        mcb = fnum(r["mc_b_rmsz"])
        c = cells.setdefault((gi, ai), {"geometry": float(r["geometry"]),
                                        "adp": float(r["adp"]), "rows": []})
        c["rows"].append({"r_free": rf,
                          "gap": (rf - rw) if (rf and rw is not None) else None,
                          "mcb": mcb})
    return cells


def med(cell, key):
    v = [r[key] for r in cell["rows"] if r[key] is not None]
    return median(v) if v else float("nan")


def main():
    for name, path in GRIDS:
        if not path.exists():
            print(f"(missing {name})")
            continue
        cells = load(path)
        gis = sorted({k[0] for k in cells})
        # geometry index nearest 0.2
        gi0 = min(gis, key=lambda gi: abs(math.log(cells[(gi, 0)]["geometry"])
                                          - math.log(0.2)))
        gval = cells[(gi0, 0)]["geometry"]
        print(f"\n=== {name}: adp-axis sweep at geometry={gval:.3g} ===")
        print(f"{'adp_w':>8} {'R_free':>7} {'gap':>7} {'MC-B RMSZ':>10}")
        for ai in sorted({k[1] for k in cells if k[0] == gi0}):
            c = cells[(gi0, ai)]
            print(f"{c['adp']:>8.4g} {med(c,'r_free'):>7.4f} "
                  f"{med(c,'gap'):>7.4f} {med(c,'mcb'):>10.3f}")
        # best cells anywhere
        keys = list(cells)
        best_rf = min(keys, key=lambda k: med(cells[k], 'r_free'))
        cb = cells[best_rf]
        feas = [k for k in keys if med(cells[k], 'mcb') <= 2.0]
        print(f"  best R_free anywhere: {med(cb,'r_free'):.4f} "
              f"@ geom={cb['geometry']:.3g}/adp={cb['adp']:.3g} "
              f"(MC-B {med(cb,'mcb'):.2f}, gap {med(cb,'gap'):.3f})")
        if feas:
            kb = min(feas, key=lambda k: med(cells[k], 'r_free'))
            cf = cells[kb]
            print(f"  best R_free with MC-B<=2: {med(cf,'r_free'):.4f} "
                  f"@ geom={cf['geometry']:.3g}/adp={cf['adp']:.3g} "
                  f"(MC-B {med(cf,'mcb'):.2f}, gap {med(cf,'gap'):.3f})")
        else:
            best_mcb = min(keys, key=lambda k: med(cells[k], 'mcb'))
            cm = cells[best_mcb]
            print(f"  NO cell reaches MC-B<=2; best MC-B={med(cm,'mcb'):.2f} "
                  f"@ geom={cm['geometry']:.3g}/adp={cm['adp']:.3g} "
                  f"(R_free {med(cm,'r_free'):.4f})")


if __name__ == "__main__":
    main()
