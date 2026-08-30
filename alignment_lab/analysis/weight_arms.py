"""Does the measured model-error curve replace the assumed one?

The two-part system -- one scaler, one weight -- was meant to subsume seven
separate knobs. Four went with the scaler and the weight. These arms test the
last of them: `sigma_A` itself, which is currently a *prior* (a Luzzati falloff
from a coordinate error guessed off the residue count) patched at low resolution
by Babinet's two universal constants.

It does not have to be assumed. Total scattering per shell is rotation-
invariant, so `Sigma_obs(s)/Sigma_calc(s)` measures the model's resolution-
dependent deficiency before the molecule is placed -- and it is safe to take
from the data being scored, because a quantity identical for every candidate
orientation cannot bias the ranking between them.

Ranking is expected to be flat: it has been flat across every configuration of
scaling and weighting tried so far. That is not the question. The question is
whether the measured curve can stand in for the assumed one, so that two
declared objects replace seven knobs rather than four of them.
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

#: Each arm is one deviation from `control`, which is what ships today.
ARMS = {
    # What the engine did before this line of work.
    "luzzati_babinet": {"sigma_a_source": "luzzati", "apply_bulk_solvent": True},
    # Luzzati without the two universal Babinet constants: how much of the
    # low-resolution correction was the solvent term doing?
    "luzzati_only":    {"sigma_a_source": "luzzati", "apply_bulk_solvent": False},
    # sigma_A measured from Sigma_obs/Sigma_calc instead of assumed from an
    # estimated coordinate error. Subsumes Babinet -- the solvent deficit is
    # what the ratio measures -- so the solvent flag is irrelevant here.
    "empirical":       {"sigma_a_source": "empirical"},
    # And the same with no observed-side weight, to check the two halves of the
    # system are still independent of each other.
    "empirical_now":   {"sigma_a_source": "empirical", "obs_weight": "none"},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, choices=list(BENCH_PDBS))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--thr-deg", type=float, default=5.0)
    ap.add_argument("--arms", default="")
    args = ap.parse_args()

    names = [a for a in args.arms.split(",") if a] or list(ARMS)
    for trial in range(args.trials):
        seed = seed_for(args.pdb, trial)
        model, data, R_true = rotated_case(args.pdb, seed)
        sym = data.spacegroup.matrices.to(torch.float64).cpu()
        rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
        okw = dict(side="left", frame="cart", reciprocal_basis=rec,
                   thr_deg=args.thr_deg)
        for name in names:
            cfg = FRFConfig(n_peaks=args.n_peaks, lmax_cap=args.lmax_cap,
                            **ARMS[name])
            t0 = time.time()
            try:
                res = run_frf(model, data, cfg, capture_arf=False, verbose=0)
            except Exception as exc:
                print(f"ROW {name} {args.pdb} trial={trial} seed={seed} rank=-1 "
                      f"rank_cmp={args.n_peaks} found=0 seconds=0.00 "
                      f"error={type(exc).__name__}", flush=True)
                continue
            seconds = time.time() - t0
            rank, ang = orbit_rank(res.peaks, R_true, sym, **okw)
            print(f"ROW {name} {args.pdb} trial={trial} seed={seed} "
                  f"rank={rank} rank_cmp={rank if rank >= 0 else args.n_peaks} "
                  f"found={int(rank >= 0)} top20={int(0 <= rank < 20)} "
                  f"angle={'' if ang is None else round(float(ang), 3)} "
                  f"seconds={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
