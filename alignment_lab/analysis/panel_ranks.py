"""Truth rank over one benchmark structure at seeded orientations.

The acceptance criterion for a change to the rotation function is not the median
rank -- it is whether truth lands inside the candidate window the placement
search carries forward, on nearly every trial, for *every* structure. Seed-to-
seed spread at ``lmax_cap = 64`` is +-4 to 6 ranks, so a bare median hides the
cases that decide it.

Emits one ``ROW`` line per trial so the caller can pair the same seed across two
worktrees. Deliberately reuses the lab's ``seed_for`` / ``rotated_case`` /
``orbit_rank``, which carry the seed contract and the orbit conventions the
earlier sweeps were measured with -- a private reimplementation of any of those
would make the comparison meaningless.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, orbit_rank, rotated_case,  # noqa: E402
                 run_frf, seed_for)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, choices=list(BENCH_PDBS))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--thr-deg", type=float, default=5.0)
    ap.add_argument("--tag", default="?", help="which tree this run came from")
    ap.add_argument("--exclude-h", action="store_true",
                    help="drop hydrogens from F_calc. dev now keeps them by "
                         "default and generates any a file lacks, which is ~47% "
                         "of the atom count and moves |F_calc| by ~7% at the "
                         "median. Whether GENERATED hydrogens belong in a "
                         "molecular-replacement search model is a separate "
                         "question from whether they belong in refinement.")
    args = ap.parse_args()

    cfg = FRFConfig(n_peaks=args.n_peaks, lmax_cap=args.lmax_cap)
    for trial in range(args.trials):
        seed = seed_for(args.pdb, trial)
        model, data, R_true = rotated_case(args.pdb, seed)
        if args.exclude_h:
            model.exclude_H_from_sf = True
        t0 = time.time()
        res = run_frf(model, data, cfg, capture_arf=False, verbose=0)
        seconds = time.time() - t0
        rank, ang = orbit_rank(
            res.peaks, R_true,
            data.spacegroup.matrices.to(torch.float64).cpu(),
            reciprocal_basis=data.cell.reciprocal_basis_matrix.to(
                torch.float64).cpu(),
            side="left", frame="cart", thr_deg=args.thr_deg,
        )
        # orbit_rank returns -1 for "no peak within thr_deg". A miss must not
        # sort as a good rank, so for comparison it counts as worse than the
        # worst hit -- the peak-list length.
        rank_cmp = rank if rank >= 0 else args.n_peaks
        n_h = int((model.pdb["element"].str.strip() == "H").sum())
        print(f"ROW {args.tag} {args.pdb} trial={trial} seed={seed} "
              f"nH={n_h} exclH={int(args.exclude_h)} "
              f"rank={rank} rank_cmp={rank_cmp} found={int(rank >= 0)} "
              f"top20={int(0 <= rank < 20)} "
              f"angle={'' if ang is None else round(float(ang), 3)} "
              f"seconds={seconds:.2f} sg={data.spacegroup.hm}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
