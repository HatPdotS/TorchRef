#!/usr/bin/env python3
"""Aggregate the dev-build TorchRef arms vs the review-build baseline.

Parses each arm's REFMAC 0-cycle ``validate.log`` (R_work/R_free, the
apples-to-apples metric), and the baseline ``baseline/fig_rfactors.csv``
(review-build torchref / phenix / refmac / prediction). Prints a table with the
overfitting gap (R_free − R_work) and paired per-structure deltas, and writes
the final aggregated table to ``results.csv`` (the one results file kept in git;
everything under runs/ is gitignored scratch).

Usage:  ./.dev/bin/python analysis/compare_devbuild.py
        ./.dev/bin/python analysis/compare_devbuild.py --arms torchref_devbuild ...
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_af_pipeline as P  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGDIR = HERE.parent
BASELINE = FIGDIR / "baseline" / "fig_rfactors.csv"
RESULTS_CSV = FIGDIR / "results.csv"


def _med(xs):
    xs = [x for x in xs if x == x]
    return st.median(xs) if xs else float("nan")


def load_baseline():
    out = {}
    if not BASELINE.exists():
        return out
    for r in csv.DictReader(open(BASELINE)):
        try:
            rw, rf = float(r["r_work"]), float(r["r_free"])
        except (ValueError, KeyError):
            continue
        out.setdefault(r["engine"], {})[r["code"]] = (rw, rf)
    return out


def load_arm(arm, codes):
    """{code: (rwork, rfree)} from an arm's REFMAC validate.logs."""
    d, pending = {}, 0
    arm_dir = P.RUNS / arm
    for code in codes:
        log = arm_dir / code / "validate.log"
        if not (log.exists() and P._refmac_complete(log)):
            pending += 1
            continue
        rw, rf = P._parse_refmac(log)
        if rf == rf:
            d[code] = (rw, rf)
    return d, pending


def summarize(d):
    return (len(d), _med([v[0] for v in d.values()]),
            _med([v[1] for v in d.values()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["torchref_devbuild", "torchref_devbuild_xw5"])
    args = ap.parse_args()

    codes = P.load_solved_codes()
    base = load_baseline()

    rows = []  # (label, build, n, rwork, rfree)
    dev = {}   # arm -> {code: (rw, rf)}
    for arm in args.arms:
        d, pending = load_arm(arm, codes)
        dev[arm] = d
        n, rw, rf = summarize(d)
        rows.append((arm, "dev", n, rw, rf))
        print(f"{arm}: validated {n}/{len(codes)}  (pending/failed {pending})")
    for eng in ("torchref", "phenix", "refmac", "prediction"):
        if eng in base:
            n, rw, rf = summarize(base[eng])
            rows.append((f"{eng}_baseline", "review", n, rw, rf))

    # --- table ---
    print(f"\n{'arm/engine':24s} {'build':7s} {'n':>4s} {'R_work':>8s} "
          f"{'R_free':>8s} {'gap':>8s}")
    print("-" * 64)
    for label, build, n, rw, rf in rows:
        print(f"{label:24s} {build:7s} {n:4d} {rw:8.4f} {rf:8.4f} {rf - rw:8.4f}")

    # --- paired deltas: each dev arm vs review-build torchref and phenix ---
    for ref_eng in ("torchref", "phenix"):
        ref = base.get(ref_eng, {})
        if not ref:
            continue
        print(f"\nPaired vs {ref_eng} (review):")
        for arm in args.arms:
            common = sorted(set(dev[arm]) & set(ref))
            if not common:
                continue
            dgap = _med([dev[arm][c][1] - dev[arm][c][0] for c in common])
            rgap = _med([ref[c][1] - ref[c][0] for c in common])
            better = sum(1 for c in common if dev[arm][c][1] < ref[c][1])
            print(f"  {arm:24s} n={len(common)}  "
                  f"median R_free dev={_med([dev[arm][c][1] for c in common]):.4f} "
                  f"{ref_eng}={_med([ref[c][1] for c in common]):.4f}  "
                  f"gap dev={dgap:.4f}/{ref_eng}={rgap:.4f}  "
                  f"R_free-better {better}/{len(common)} ({100*better/len(common):.0f}%)")

    # --- write the one tracked results table ---
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm_or_engine", "build", "n", "median_rwork",
                    "median_rfree", "overfit_gap"])
        for label, build, n, rw, rf in rows:
            w.writerow([label, build, n, f"{rw:.4f}", f"{rf:.4f}", f"{rf - rw:.4f}"])
    print(f"\nWrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
