#!/usr/bin/env python3
"""Test the documented `scale_target='nll'` collapse hazard on real benchmark structures.

``ScalerBase.refine_lbfgs``'s own docstring flags this: ``'ml_noalpha'`` originally replaced
``'nll'`` because a least-squares scale fit collapses in shells where ``F_obs`` is
noise-dominated and uncorrelated with ``F_calc`` -- the per-bin optimum
``k = sum(F_obs*Fc/sigma^2) / sum(Fc^2/sigma^2)`` tends to 0 there, which blows up R -- and it
instructs the reader to **check the per-bin log_scale spread and the post-scaling R-factors
rather than assuming it behaved**. ``'nll'`` is now the default, so that instruction applies
to the default path.

The failure signature is specific: *some* bins diving toward k -> 0 while the middle bins look
fine. A uniform offset between the two targets is not the hazard -- the two objectives simply
have different optima. So the statistic is ``min(k)/median(k)``, which is independent of how
the binner orders bins; an "outer N bins" slice silently assumes an ordering, and getting that
wrong turns a real collapse into a reassuring number.

Runs on the AlphaFold-start placed models (the benchmark's own inputs), not on a separate
data set, so a positive result would be about the structures actually being refined.

Usage
-----
    ./.dev/bin/python paper/probe_scale_collapse.py --codes 1ETE 1FSI 1DAW
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

PLACED = HERE / "figure2_alphafold_start" / "placed"
DATA = HERE / "data"

# Imported rather than hardcoded: the arms are whatever the library currently offers, so a
# renamed or added objective cannot leave this probe quietly testing one target twice.
from torchref.scaling.scaler_base import (  # noqa: E402 - needs the sys.path insert above
    DEFAULT_SCALE_TARGET,
    SCALE_TARGETS,
)


def one(code, target):
    from torchref.refinement.lbfgs_refinement import LBFGSRefinement

    pdb = PLACED / f"{code}_af.pdb"
    mtz = DATA / code / f"{code}.mtz"
    if not pdb.exists() or not mtz.exists():
        return None
    # Scale via the constructor, the way the CLI does. Do NOT call scaler.refine_lbfgs()
    # again afterwards: the refinement has already scaled during __init__, and re-fitting from
    # an already-scaled state drives every scaler parameter to NaN.
    r = LBFGSRefinement(data_file=str(mtz), pdb=str(pdb), verbose=0, scale_target=target)
    with torch.no_grad():
        ls = r.scaler.log_scale.detach().reshape(-1).cpu()
        rw, rf = r.xray_target_work.get_rfactor()
    del r
    return ls, float(rw), float(rf)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", nargs="+", required=True)
    ap.add_argument("--collapse-ratio", type=float, default=0.2,
                    help="Flag min(k)/median(k) below this (default 0.2).")
    args = ap.parse_args()

    verdicts = []
    for code in args.codes:
        print(f"\n=== {code} ===")
        got = {}
        for tgt in SCALE_TARGETS:
            try:
                out = one(code, tgt)
            except Exception as exc:                       # noqa: BLE001 - report, continue
                print(f"  {tgt:7s} FAILED {type(exc).__name__}: {exc}")
                continue
            if out is None:
                print(f"  {code}: inputs not found, skipping")
                break
            got[tgt] = out
        if len(got) < len(SCALE_TARGETS):
            continue
        for tgt, (ls, rw, rf) in got.items():
            k = ls.exp()
            ratio = float(k.min() / k.median())
            flag = "   <-- COLLAPSING" if ratio < args.collapse_ratio else ""
            print(f"  {tgt:7s} R_work={rw:.4f} R_free={rf:.4f}  "
                  f"log_scale {float(ls.min()):+.3f}..{float(ls.max()):+.3f} "
                  f"sd={float(ls.std()):.3f}  min/median k={ratio:.3f}{flag}")
            verdicts.append((code, tgt, ratio))
        rw_n, rf_n = got[DEFAULT_SCALE_TARGET][1], got[DEFAULT_SCALE_TARGET][2]
        for tgt, (_, rw_s, rf_s) in got.items():
            if tgt == DEFAULT_SCALE_TARGET:
                continue
            print(f"  {DEFAULT_SCALE_TARGET} - {tgt}:  "
                  f"R_work {rw_n - rw_s:+.4f}   R_free {rf_n - rf_s:+.4f}")

    bad = [(c, t, r) for c, t, r in verdicts
           if t == DEFAULT_SCALE_TARGET and r < args.collapse_ratio]
    print("\n" + "=" * 70)
    if bad:
        print("HAZARD PRESENT on the default path:")
        for c, _, r in bad:
            print(f"  {c}: min/median k = {r:.3f}")
        alt = [t for t in SCALE_TARGETS if t != DEFAULT_SCALE_TARGET]
        print(f"Consider --scale-target {alt[0]} for the campaign.")
        return 1
    print(f"No per-bin collapse on scale_target={DEFAULT_SCALE_TARGET!r} for "
          f"{len({c for c, _, _ in verdicts})} structure(s) "
          f"(all min/median k >= {args.collapse_ratio}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
