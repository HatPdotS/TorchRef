#!/usr/bin/env python3
"""Paired per-structure comparison of one metric between two benchmark CSVs.

Every figure in this tree is a median over a structure set, and a median moves for two very
different reasons: the metric really shifted, or the *set* changed (a structure that failed
before now succeeds, or vice versa). Comparing two medians cannot tell those apart. This
pairs on the structure code, so the reported delta is a difference on the same structures and
nothing else, and it says out loud which codes each side is missing.

Used to gate the re-run: refine one cell of the ExtFig 1 weight grid with the current build
and compare it, structure by structure, against the archived pre-rework grid at the same
cell. 50 jobs decide whether the remaining 4950 are worth submitting.

Reports the signed median delta (sign convention: **new minus old**, so negative is better
for an R-factor), win/loss counts, a Wilcoxon signed-rank p-value, and a bootstrap CI on the
median delta. The bootstrap is there because a median's uncertainty is not something to
eyeball -- and because a half-split estimate of it is unreliable at these n.

Usage
-----
    # ExtFig 1 locked cell, current build vs the archive
    ./.dev/bin/python paper/compare_to_archive.py \\
        --old paper/archive/figures_pre_mlrework/metrics/weight_grid.csv \\
        --new paper/figure2_alphafold_start/runs/metrics/weight_grid.csv \\
        --filter gi=7 --filter ai=4 --metric r_work --metric r_free

    # Figure 2 R-factors, one engine
    ./.dev/bin/python paper/compare_to_archive.py \\
        --old paper/archive/figures_pre_mlrework/metrics/fig_rfactors.csv \\
        --new paper/figure2_alphafold_start/runs/metrics/fig_rfactors.csv \\
        --filter engine=torchref --metric r_work --metric r_free
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (v != v) else v          # drop NaN


def load(path, key, metrics, filters):
    """-> {code: {metric: value}}, keeping only rows that match every filter."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"missing {path}")
    out, n_rows, n_kept = {}, 0, 0
    with open(path) as f:
        rdr = csv.DictReader(f)
        missing = [c for c in [key, *metrics, *filters] if c not in (rdr.fieldnames or [])]
        if missing:
            sys.exit(f"{path}: no such column(s) {missing}; has {rdr.fieldnames}")
        for r in rdr:
            n_rows += 1
            if any(str(r[c]) != str(v) for c, v in filters.items()):
                continue
            n_kept += 1
            vals = {m: fnum(r[m]) for m in metrics}
            if all(v is None for v in vals.values()):
                continue                     # a failed run: no metric at all
            if r[key] in out:
                sys.exit(f"{path}: duplicate {key}={r[key]!r} after filtering -- the filter "
                         f"is not selective enough to pair on {key}")
            out[r[key]] = vals
    print(f"  {path.name}: {n_rows} rows, {n_kept} after filter, {len(out)} usable")
    return out


def report(name, old, new, rng):
    pairs = [(c, old[c][name], new[c][name]) for c in sorted(set(old) & set(new))
             if old[c][name] is not None and new[c][name] is not None]
    if not pairs:
        print(f"\n{name}: no paired values")
        return
    o = np.array([p[1] for p in pairs])
    n = np.array([p[2] for p in pairs])
    d = n - o                                # new minus old: negative = new is better

    med_d = float(np.median(d))
    # Percentile bootstrap on the median of the paired differences. Resampling the DIFFERENCES
    # (not each arm separately) is what keeps the pairing intact.
    boot = np.median(rng.choice(d, size=(20000, d.size), replace=True), axis=1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))

    wins = int((d < 0).sum())                # new better
    losses = int((d > 0).sum())
    ties = int((d == 0).sum())
    try:
        p = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    except ValueError:
        p = float("nan")

    verdict = "NEW BETTER" if hi < 0 else "NEW WORSE" if lo > 0 else "no clear difference"
    print(f"\n{name}  (n paired = {len(pairs)})")
    print(f"  median old        {float(np.median(o)):.4f}")
    print(f"  median new        {float(np.median(n)):.4f}")
    print(f"  median delta      {med_d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
    print(f"  new better/worse  {wins}/{losses}  (ties {ties})   Wilcoxon p = {p:.3g}")
    worst = sorted(pairs, key=lambda t: t[2] - t[1])[-5:][::-1]
    print(f"  biggest regressions: "
          + ", ".join(f"{c} {b - a:+.4f}" for c, a, b in worst))
    print(f"  mean |delta|      {float(np.abs(d).mean()):.4f}   "
          f"max |delta| {float(np.abs(d).max()):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="Baseline CSV (e.g. the archive).")
    ap.add_argument("--new", required=True, help="CSV from the current build.")
    ap.add_argument("--key", default="code", help="Column to pair on (default 'code').")
    ap.add_argument("--metric", action="append", required=True,
                    help="Metric column; repeatable.")
    ap.add_argument("--filter", action="append", default=[], metavar="COL=VAL",
                    help="Keep only rows with COL == VAL; repeatable. Applied to BOTH files.")
    ap.add_argument("--seed", type=int, default=0, help="Bootstrap seed (default 0).")
    args = ap.parse_args()

    filters = {}
    for f in args.filter:
        if "=" not in f:
            sys.exit(f"--filter expects COL=VAL, got {f!r}")
        c, v = f.split("=", 1)
        filters[c] = v

    print("Paired comparison — sign convention: NEW minus OLD (negative = new better)")
    if filters:
        print("  filter: " + "  ".join(f"{c}={v}" for c, v in filters.items()))
    old = load(args.old, args.key, args.metric, filters)
    new = load(args.new, args.key, args.metric, filters)

    only_old, only_new = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    print(f"  paired {len(set(old) & set(new))}   only-old {len(only_old)}   "
          f"only-new {len(only_new)}")
    # Named, not just counted: an unpaired structure is the mechanism by which a median moves
    # without any structure changing, so it must never be silent.
    if only_old:
        print(f"    missing from new: {' '.join(only_old)}")
    if only_new:
        print(f"    missing from old: {' '.join(only_new)}")

    rng = np.random.default_rng(args.seed)
    for m in args.metric:
        report(m, old, new, rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
