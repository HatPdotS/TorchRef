"""What is the observed-side weight worth, and which part of it?

The scaler split scaling from weighting and showed the scaling half is gauge for
a correlation. This is the half that is not.

Arms deviate one thing at a time from `none_shellvar`, which is what the engine
did before any of this. Moving several at once and reading one number is what
made the E-convention panel uninterpretable.

Three things are being asked:

* is a measurement-error weight worth anything at all (`none` vs `information`);
* is folding model error into the same denominator worth more than the
  measurement term alone (`information` vs `inverse_var`) -- they do not
  factorise, so this is the only way to separate them;
* is `apply_shell_variance_weights` doing anything, given it is a per-shell
  weight and per-shell weights are absorbed (`*_shellvar` pairs).

The caps are screened rather than assumed. `trust_cap` is supposed to be a
backstop that never binds, so if it moves the result, sigma_A is wrong.
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
    # What shipped before any of this: unit observed weight (DFAC left with the
    # French-Wilson posterior when the scaler landed) and the per-shell reweight
    # on. The load-bearing control -- without it, "the weight helps" cannot be
    # told apart from "any weight helps".
    "none_shellvar":  {"obs_weight": "none", "shell_variance_weights": True},
    "none":           {"obs_weight": "none"},
    "information":    {"obs_weight": "information"},
    "inverse_var":    {"obs_weight": "inverse_variance"},
    "invvar_shellvar": {"obs_weight": "inverse_variance",
                        "shell_variance_weights": True},
    # Cap screens. The SNR cap sets where measurement error stops limiting; the
    # trust cap is meant to be a backstop, so if these move the result it is
    # binding and sigma_A is wrong.
    "snr_cap_2":      {"obs_weight": "information", "snr_cap": 2.0},
    "snr_cap_10":     {"obs_weight": "information", "snr_cap": 10.0},
    "trust_cap_10":   {"obs_weight": "inverse_variance", "trust_cap": 10.0},
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
