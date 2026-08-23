"""Dump a ranked FRF peak-list fingerprint for cross-tree comparison.

Run with PYTHONPATH pointed at whichever worktree should provide `torchref` and
`alignment_lab`; the file itself is tree-agnostic. Scores are printed at 9
significant figures rather than raw float64 -- the engine's own run-to-run
spread is ~5e-8 relative, so comparing full precision would report noise as a
difference (see `frf_peaklist_reproducibility`).
"""
import argparse
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--n-peaks", type=int, default=500)
    args = ap.parse_args()

    from alignment_lab.lab.benchmark import load_case
    from alignment_lab.lab.frf import FRFConfig, run_frf

    model, data = load_case(args.pdb)[:2]
    res = run_frf(model, data, FRFConfig(n_peaks=args.n_peaks,
                                        lmax_cap=args.lmax_cap))
    peaks = res.peaks if hasattr(res, "peaks") else res[0]
    # Every fingerprint line is prefixed. Loading a structure writes progress to
    # stdout, so a comparison that filters on anything looser (blank lines, a
    # leading '#') silently ingests that chatter as data.
    print(f"#FP pdb={args.pdb} lmax_cap={args.lmax_cap} n_peaks={len(peaks)}")
    for i, p in enumerate(peaks):
        print(f"FP {i:4d} {p.alpha:.9g} {p.beta:.9g} {p.gamma:.9g} "
              f"{p.score:.9g} {p.sigma:.9g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
