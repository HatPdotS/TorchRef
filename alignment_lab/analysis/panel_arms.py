"""Truth rank per seeded trial under one or more knocked-out engine stages.

Two stages of the rotation function are suspected of earning nothing, each for a
different reason, and both are cheaper to decide by knocking them out than by
reasoning about them:

``no_brel``
    ``fit_relative_wilson_b`` scales ``F_calc`` by ``exp(-B_rel s^2/4)`` and the
    very next step, ``wilson_normalise``, divides each equal-count shell by its
    own ``sqrt(<F_calc^2>)`` -- which removes the shell-mean radial profile,
    B_rel's included. Only the within-shell residual of a smooth exponential can
    survive. If that is below the engine's own spread, the fit is a sort, a
    binning and a regression for nothing.

``no_friedel``
    ``enforce_friedel`` concatenates ``-s`` onto both reflection sets. Only even
    ``l`` are ever computed and ``Y_lm(-s_hat) = Y_lm(s_hat)`` for even ``l``,
    with the intensity duplicated verbatim, so ``c_nlm`` doubles *exactly*. Both
    sides doubled means xi scales by 4, and the mean and standard deviation of
    the rotation function scale with it, so z-scores and the ranking are
    invariant. The prediction is therefore sharp: identical ranks, and the raw
    score up by exactly 4. Reported so it can be checked rather than assumed.

Emits one ``ROW`` line per (arm, trial), carrying the top score so the scaling
prediction is falsifiable from the output.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, orbit_rank, rotated_case,  # noqa: E402
                 run_frf, seed_for)

ARMS = ("production", "no_brel", "no_friedel")


@contextlib.contextmanager
def knocked_out(arm: str):
    """Disable one stage for the duration of a call, then restore it."""
    if arm == "production":
        yield
        return
    if arm == "no_brel":
        import torchref.experimental.alignment.frf.preprocessing as pp
        original = pp.fit_relative_wilson_b
        pp.fit_relative_wilson_b = lambda *a, **k: 0.0
        try:
            yield
        finally:
            pp.fit_relative_wilson_b = original
        return
    if arm == "no_friedel":
        import torchref.experimental.alignment.frf.api as api
        original = api.bessel_sh_expand

        @functools.wraps(original)
        def no_mate(*a, **k):
            k["enforce_friedel"] = False
            return original(*a, **k)

        api.bessel_sh_expand = no_mate
        try:
            yield
        finally:
            api.bessel_sh_expand = original
        return
    raise ValueError(f"unknown arm {arm!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, choices=list(BENCH_PDBS))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--thr-deg", type=float, default=5.0)
    args = ap.parse_args()

    cfg = FRFConfig(n_peaks=args.n_peaks, lmax_cap=args.lmax_cap)
    arms = [a for a in args.arms.split(",") if a]
    # Trial-major so the same seed's arms run back to back: any drift in machine
    # state affects the arms together rather than one of them.
    for trial in range(args.trials):
        seed = seed_for(args.pdb, trial)
        for arm in arms:
            model, data, R_true = rotated_case(args.pdb, seed)
            t0 = time.time()
            with knocked_out(arm):
                res = run_frf(model, data, cfg, capture_arf=False, verbose=0)
            seconds = time.time() - t0
            rank, ang = orbit_rank(
                res.peaks, R_true,
                data.spacegroup.matrices.to(torch.float64).cpu(),
                reciprocal_basis=data.cell.reciprocal_basis_matrix.to(
                    torch.float64).cpu(),
                side="left", frame="cart", thr_deg=args.thr_deg,
            )
            top = res.peaks[0] if res.peaks else None
            print(f"ROW {arm} {args.pdb} trial={trial} seed={seed} "
                  f"rank={rank} rank_cmp={rank if rank >= 0 else args.n_peaks} "
                  f"top20={int(0 <= rank < 20)} "
                  f"top_score={'' if top is None else f'{top.score:.10g}'} "
                  f"top_sigma={'' if top is None else f'{top.sigma:.6g}'} "
                  f"seconds={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
