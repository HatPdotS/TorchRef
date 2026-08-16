#!/usr/bin/env python3
"""Paired comparison of every fine-grid weight cell against the locked default.

The fine grid brackets the library default (geometry 0.2 / adp 0.02) one coarse step
either side at 1.29x per step, so its centre cell *is* the default and doubles as a
null control.

Each cell is compared to that centre cell by the **median of the per-structure
R-free differences** over the structures both refined, with a bootstrap CI on that
median and a Wilcoxon signed-rank p. Difference-of-medians is reported alongside
only for contrast: it is not the paired statistic and can differ in sign.

Usage
-----
    ./.dev/bin/python paper/extended_figures/exF1/compare_fine_grid.py
    ./.dev/bin/python paper/extended_figures/exF1/compare_fine_grid.py --tag ""   # coarse grid
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
METRICS = HERE.parent.parent / "figure2_alphafold_start" / "runs" / "metrics"

#: The locked default the grid is centred on (torchref/refinement/base_refinement.py).
DEFAULT_GEOM, DEFAULT_ADP = 0.2, 0.02


def load(tag: str):
    suffix = f"_{tag}" if tag else ""
    path = METRICS / f"weight_grid{suffix}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run aggregate_weight_grid.py --tag {tag} first")
    cells = defaultdict(dict)          # (gi, ai) -> {code: r_free}
    coords = {}                        # (gi, ai) -> (geometry, adp)
    for r in csv.DictReader(open(path)):
        key = (int(r["gi"]), int(r["ai"]))
        cells[key][r["code"]] = float(r["r_free"])
        coords[key] = (float(r["geometry"]), float(r["adp"]))
    return cells, coords


def manifest_coords(tag: str):
    """(gi, ai) -> (geometry, adp) for every cell the grid defines, populated or not."""
    suffix = f"_{tag}" if tag else ""
    path = METRICS / f"wgrid{suffix}_manifest.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run submit_weight_grid.py first")
    return {(int(r["gi"]), int(r["ai"])): (float(r["geometry"]), float(r["adp"]))
            for r in csv.DictReader(open(path))}


def boot_median_ci(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(np.asarray(d)[idx], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main(tag):
    cells, coords = load(tag)
    # Pick the baseline from the MANIFEST, not from the cells that happen to have rows:
    # on a partial grid the nearest *populated* cell can silently stand in for the
    # default and every delta would then be measured against the wrong baseline.
    # Which cell is nearest the default is a property of the grid, not of its progress.
    manifest = manifest_coords(tag)
    centre = min(manifest,
                 key=lambda k: (abs(manifest[k][0] - DEFAULT_GEOM)
                                + abs(manifest[k][1] - DEFAULT_ADP)))
    if centre not in cells:
        cg, ca = manifest[centre]
        raise SystemExit(
            f"the grid's default cell {centre} (geometry {cg:.4g} / adp {ca:.4g}) has no "
            f"rows yet — nothing to compare against. Wait for the grid to finish.")
    cg, ca = coords[centre]
    print(f"grid tag={tag or '(coarse)'}   cells={len(cells)}")
    print(f"centre cell {centre} = geometry {cg:.4g} / adp {ca:.4g}"
          f"   (library default {DEFAULT_GEOM} / {DEFAULT_ADP})")
    print(f"centre cell structures: {len(cells[centre])}\n")

    rows = []
    for key, per_code in cells.items():
        if key == centre:
            continue
        common = sorted(set(per_code) & set(cells[centre]))
        if len(common) < 20:
            continue
        d = [per_code[c] - cells[centre][c] for c in common]   # negative = cell better
        lo, hi = boot_median_ci(d)
        p = wilcoxon(d).pvalue if any(x != 0 for x in d) else 1.0
        rows.append({
            "gi": key[0], "ai": key[1],
            "geometry": coords[key][0], "adp": coords[key][1],
            "n": len(common), "median_delta": float(np.median(d)),
            "ci_low": lo, "ci_high": hi, "wilcoxon_p": p,
            "better": sum(1 for x in d if x < 0), "worse": sum(1 for x in d if x > 0),
            "delta_of_medians": float(np.median(list(per_code.values()))
                                      - np.median(list(cells[centre].values()))),
        })

    rows.sort(key=lambda r: r["median_delta"])
    print("cells that BEAT the default (paired median ΔR-free < 0), best first:")
    beat = [r for r in rows if r["median_delta"] < 0]
    if not beat:
        print("  none.")
    hdr = (f"  {'geom':>7} {'adp':>8} {'n':>4} {'medΔ':>9} {'95% CI':>20} "
           f"{'wilcox p':>10} {'bet/wor':>9} {'Δmedians':>9}")
    print(hdr)
    for r in beat:
        print(f"  {r['geometry']:7.4g} {r['adp']:8.4g} {r['n']:4d} "
              f"{r['median_delta']:+9.5f} "
              f"[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}] {r['wilcoxon_p']:10.2e} "
              f"{r['better']:4d}/{r['worse']:<4d} {r['delta_of_medians']:+9.5f}")

    sig = [r for r in beat if r["ci_high"] < 0]
    print(f"\n{len(beat)}/{len(rows)} cells have a negative paired median; "
          f"{len(sig)} have a 95% CI entirely below zero.")
    if not sig:
        print("=> no cell beats the locked default at 95% confidence.")

    out = METRICS / f"weight_grid{'_' + tag if tag else ''}_paired.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} cells)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="fine",
                    help="Grid tag; reads weight_grid_<tag>.csv (default 'fine'). "
                         "Pass an empty string for the baseline coarse grid.")
    main(**vars(ap.parse_args()))
