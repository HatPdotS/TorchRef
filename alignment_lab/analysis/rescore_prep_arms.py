"""Does the ML rescore fail because of its MODEL PREPARATION?

The rescore is measurably destructive end-to-end -- dropping it takes pose
recovery from 18/30 to 24/30 -- and the damage is concentrated in the two large
benchmark entries, 4BX9 and 6G9X, which it fails at 52-91 degrees and which the
raw FRF order solves at 3-4 degrees. This harness tests one explanation.

**The FRF and the rescore disagree about the model, inside a single run.** The
pipeline estimates ``model_error_A`` from the model's length, hands it to the
FRF, and applies Phaser's Babinet bulk-solvent term there -- then calls
``m_letf1_rescore`` without passing either, so the rescore falls back to a
hardcoded ``delta_vrms = 0.5`` and no solvent. Every Phaser model-prep knob on
that function defaults OFF and the pipeline overrides none of them.

That predicts the observed failure pattern rather than merely being consistent
with it. Babinet's ``1 - 0.95 exp(-300 s^2/4)`` tends to 0.05 as ``s -> 0``, so
omitting it over-weights the lowest-resolution reflections by up to ~20x, and
low-resolution terms dominate for large molecules. 0.5 A is also furthest from
the truth for a large model, where ``oeffner_vrms`` gives ~0.67 A.

Paired by construction: the FRF runs ONCE per (structure, trial) and every arm
rescores the *same* peak list. The ``none`` arm keeps the FRF order and is the
load-bearing control -- without it, an engine that merely preserves a good input
ranking looks like one that improves it.
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
                 run_frf, run_rescore, seed_for)

#: Each arm is a set of overrides on top of `m_letf1`'s defaults. They are
#: cumulative on purpose: if the whole Phaser prep helps, the interesting
#: question is which piece carries it.
#: `eps_friedel` reproduces the epsilon the rescore used before it was routed
#: through `SpaceGroup.epsilon(friedel=False)`. Passing an explicit `eps_factor`
#: is how the old convention is reproduced without a second worktree, so the
#: comparison stays paired on one FRF peak list.
#: The E-convention arms are the reason this harness is being re-run. The
#: rotation function turned out to be INSENSITIVE to the convention -- 12 of
#: them, 100 paired cells, median rank 2.0 for every one -- which is what a
#: correlation should do, since a global scale cancels out of it. The LLG is a
#: likelihood and has no free scale to cancel, so if the convention matters
#: anywhere it matters here. `no_sigmas` is the control for that: it withholds
#: the sigmas the rescore has only just started receiving.
def _arms():
    from torchref.experimental.alignment.e_values import (
        CalcGlobalE, CalcShellE, FrenchWilsonE, SmoothSigmaE, WilsonShellE,
        WilsonShellEpsE,
    )
    import functools
    return {
        "fw_sigmas":     {"e_convention": FrenchWilsonE},
        "no_sigmas":     {"sig_F_obs": None},
        "wilson":        {"e_convention": WilsonShellE},
        "calc_shell":    {"e_convention": CalcShellE},
        "calc_global":   {"e_convention": CalcGlobalE},
        "smooth6":       {"e_convention": functools.partial(SmoothSigmaE,
                                                            n_coeff=6)},
        "eps_wilson":    {"e_convention": WilsonShellEpsE},
    }


ARMS = {
    "none":            None,                      # control: FRF order
    "default":         {},                        # what ships today
    "eps_friedel":     {"__eps_friedel": True},   # the pre-migration convention
    "vrms":            {"vrms_strategy": "oeffner"},
    "solvent":         {"apply_bulk_solvent": True},
    "vrms_solvent":    {"vrms_strategy": "oeffner", "apply_bulk_solvent": True},
    "full_prep":       {"vrms_strategy": "oeffner", "apply_bulk_solvent": True,
                        "apply_wilson_b": True},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, choices=list(BENCH_PDBS))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--n-refine", type=int, default=20,
                    help="rescore window: the top-N FRF peaks handed to the engine")
    ap.add_argument("--thr-deg", type=float, default=5.0)
    ap.add_argument("--arms", default="")
    args = ap.parse_args()

    ARMS.update(_arms())

    cfg = FRFConfig(n_peaks=args.n_peaks, lmax_cap=args.lmax_cap)
    arms = [a for a in args.arms.split(",") if a] or list(ARMS)

    for trial in range(args.trials):
        seed = seed_for(args.pdb, trial)
        model, data, R_true = rotated_case(args.pdb, seed)
        sym = data.spacegroup.matrices.to(torch.float64).cpu()
        rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
        orbit_kw = dict(side="left", frame="cart", reciprocal_basis=rec,
                        thr_deg=args.thr_deg)

        # The FRF runs once; every arm sees the identical peak list.
        res = run_frf(model, data, cfg, capture_arf=False, verbose=0)
        frf_rank, _ = orbit_rank(res.peaks[: args.n_refine], R_true, sym, **orbit_kw)

        # Same residue estimate the pipeline uses for the FRF, so the `vrms` arm
        # is genuinely "what the FRF was told" and not a second guess.
        n_residues = max(1, int(model.xyz().shape[0] / 8))

        for arm in arms:
            overrides = ARMS[arm]
            if overrides is None:
                rank, seconds = frf_rank, 0.0
            else:
                kw = dict(overrides)
                if kw.pop("__eps_friedel", False):
                    # Friedel-folded epsilon: doubles it on every centric
                    # reflection, which is what the rescore used to get.
                    kw["eps_factor"] = data.spacegroup.epsilon(
                        res.inputs.hkl.to(torch.long), friedel=True,
                    ).to(res.inputs.F_obs.dtype)
                if kw.get("vrms_strategy") == "oeffner":
                    kw["vrms_n_residues"] = n_residues
                t0 = time.time()
                rr = run_rescore(res.peaks, data, res.inputs, engine="m_letf1",
                                 n_refine=args.n_refine, verbose=0, **kw)
                seconds = time.time() - t0
                rank, _ = orbit_rank(rr.peaks, R_true, sym, **orbit_kw)
            # A miss must not sort as a good rank.
            rank_cmp = rank if rank >= 0 else args.n_refine
            print(f"ROW {arm} {args.pdb} trial={trial} seed={seed} "
                  f"frf_rank={frf_rank} rank={rank} rank_cmp={rank_cmp} "
                  f"n_res={n_residues} seconds={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
