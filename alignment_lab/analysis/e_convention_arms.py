"""Which E convention ranks the true orientation best?

Layer B of the E-value work: the conformance table says whether a convention
does what E is *supposed* to do; this says whether it makes the rotation
function *work*. When the two disagree, this one decides and the table
diagnoses -- a convention with clean Wilson statistics that ranks truth worse is
not the one to ship, and the table then names the property the winner trades
away.

Headline metric is the fraction of cells at **rank 0**, not "inside the top 20".
The stated target is that truth comes first; the post-merge baseline is 20 of
100, so the bar is a long way up and a metric that saturates hides the climb.

Paired by seed: every convention sees the same rotated case from the same
``seed_for``, so arms are compared cell by cell rather than as two distributions.
The ``default`` arm passes no convention at all, which makes it a control on the
*seam* as well as on the conventions -- if it ever diverges from the production
number, the plumbing changed something rather than the convention did.

The FRF cannot be run once and shared here, unlike the rescore arms: the
convention is what builds the obs expansion, so each arm is a full run. That is
the cost of the question.
"""

from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, e_convention_name,  # noqa: E402
                 orbit_rank, rotated_case, run_frf, seed_for)


def build_arms():
    """Name -> convention class. Built lazily so ``--help`` needs no torchref."""
    from torchref.experimental.alignment.e_values import (
        CalcGlobalE, CalcShellE, FrenchWilsonE, SmoothSigmaE, WilsonShellE,
        WilsonShellEpsE,
    )
    # Mixed arms. The panel's first round put `calc_global` -- a single global
    # RMS on BOTH sides -- ahead of every per-shell convention, which if real
    # says the per-shell flattening is discarding inter-shell amplitude shape
    # the correlation was using. One class sets both sides, so isolating which
    # side carries that needs conventions that differ across the seam. Defined
    # here rather than shipped: they exist to answer one question.
    class GlobalObsShellCalc(CalcGlobalE):
        """Global RMS on obs, per-shell Wilson on calc."""
        calc_companion = WilsonShellE

    class ShellObsGlobalCalc(WilsonShellE):
        """Per-shell Wilson on obs, global RMS on calc."""
        calc_companion = CalcGlobalE

    class FrenchWilsonGlobalCalc(FrenchWilsonE):
        """Production obs side, global RMS on calc."""
        calc_companion = CalcGlobalE

    return {
        # Control: no convention passed, so the production default applies.
        "default":        None,
        # The same thing named explicitly. Must match `default` exactly; if it
        # does not, the seam is not inert and nothing below means anything.
        "french_wilson":  FrenchWilsonE,
        # Drops the measurement-error model entirely -- the size of the gap to
        # `french_wilson` is what sigma_F is worth to the rotation function.
        "wilson":         WilsonShellE,
        # What the rescore uses on its observed side. Running it here asks
        # whether the FRF/rescore disagreement is costing the FRF anything.
        "wilson_eps":     WilsonShellEpsE,
        "calc_shell":     CalcShellE,
        "calc_global":    CalcGlobalE,
        # The divergence candidate: a smooth Chebyshev Sigma(s) instead of
        # per-shell means. Two orders, because the whole question is whether a
        # low-order curve beats 20 independent bins.
        "smooth4":        functools.partial(SmoothSigmaE, n_coeff=4),
        "smooth6":        functools.partial(SmoothSigmaE, n_coeff=6),
        "smooth10":       functools.partial(SmoothSigmaE, n_coeff=10),
        "global_x_shell": GlobalObsShellCalc,
        "shell_x_global": ShellObsGlobalCalc,
        "fw_x_global":    FrenchWilsonGlobalCalc,
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

    arms = build_arms()
    names = [a for a in args.arms.split(",") if a] or list(arms)
    unknown = [a for a in names if a not in arms]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; have {sorted(arms)}")

    for trial in range(args.trials):
        seed = seed_for(args.pdb, trial)
        model, data, R_true = rotated_case(args.pdb, seed)
        sym = data.spacegroup.matrices.to(torch.float64).cpu()
        rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
        okw = dict(side="left", frame="cart", reciprocal_basis=rec,
                   thr_deg=args.thr_deg)

        for name in names:
            cfg = FRFConfig(n_peaks=args.n_peaks, lmax_cap=args.lmax_cap,
                            e_convention=arms[name])
            t0 = time.time()
            try:
                res = run_frf(model, data, cfg, capture_arf=False, verbose=0)
            except Exception as exc:                 # a convention may refuse
                print(f"ROW {name} {args.pdb} trial={trial} seed={seed} "
                      f"rank=-1 rank_cmp={args.n_peaks} found=0 top20=0 "
                      f"seconds=0.00 error={type(exc).__name__}", flush=True)
                continue
            seconds = time.time() - t0
            rank, ang = orbit_rank(res.peaks, R_true, sym, **okw)
            rank_cmp = rank if rank >= 0 else args.n_peaks
            print(f"ROW {name} {args.pdb} trial={trial} seed={seed} "
                  f"rank={rank} rank_cmp={rank_cmp} found={int(rank >= 0)} "
                  f"top20={int(0 <= rank < 20)} "
                  f"angle={'' if ang is None else round(float(ang), 3)} "
                  f"seconds={seconds:.2f} "
                  f"conv={e_convention_name(arms[name])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
