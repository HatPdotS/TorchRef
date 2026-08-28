"""Why does the true orientation lose the LLG contest?

The rescore's job is to take a top-20 that already contains truth and put truth
first. On 6G9X it reproducibly does the opposite -- truth goes from FRF rank 1 to
rank 12-17 -- and that survives the full Phaser model preparation, so it is not
explained by sigma_A or the solvent term.

This takes one case with known ground truth and asks where the LLG difference
between truth and the candidate that beats it actually accumulates: per
resolution shell, and split by centric/acentric. A likelihood that prefers the
wrong orientation is either being fed the wrong expected intensity or is summing
a term whose sign is wrong somewhere, and both of those localise.

Per-reflection LL is recomputed here rather than taken from
``_llg_for_orientations``, which sums before returning -- same context, same
``phaser_log_rel_*`` calls, just not reduced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, orbit_rank, rotated_case,  # noqa: E402
                 run_frf, seed_for)


def per_reflection_ll(ctx, alpha, beta, gamma):
    """``(n_orient, N)`` per-reflection log-likelihood -- the unsummed LLG."""
    from torchref.experimental.alignment.distributions import (
        phaser_log_rel_rice, phaser_log_rel_woolfson,
    )
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_matrix_from_edmonds_euler_batch,
    )
    R = rotation_matrix_from_edmonds_euler_batch(
        alpha.to(torch.float64), beta.to(torch.float64), gamma.to(torch.float64),
    ).transpose(-1, -2).to(torch.float32)
    F_calc_m = ctx.interpolator.evaluate(
        R, ctx.unrolled_hkl, ctx.real_cell, return_amplitude=True,
    ).to(ctx.dtype)
    if ctx.dw_per_m is not None:
        F_calc_m = F_calc_m * ctx.dw_per_m.unsqueeze(0)
    E_calc_m = F_calc_m / ctx.sqrt_mean_per_m.unsqueeze(0)
    Esq_m = E_calc_m * E_calc_m
    B = Esq_m.shape[0]
    sum_per_h = torch.zeros(B, ctx.N, dtype=Esq_m.dtype, device=Esq_m.device)
    sum_per_h.scatter_add_(1, ctx.asu_idx.unsqueeze(0).expand(B, -1), Esq_m)
    eImove = ctx.eImove_prefac * sum_per_h
    sqrt_eImove = eImove.clamp(min=1e-30).sqrt()
    ll = torch.where(
        ctx.centric_b,
        phaser_log_rel_woolfson(ctx.E_obs_b, sqrt_eImove, ctx.V_b),
        phaser_log_rel_rice(ctx.E_obs_b, sqrt_eImove, ctx.V_b),
    )
    return ll, eImove


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="6G9X", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--n-refine", type=int, default=20)
    ap.add_argument("--n-shells", type=int, default=10)
    ap.add_argument("--full-prep", action="store_true",
                    help="turn on the Phaser model prep the pipeline omits")
    args = ap.parse_args()

    from torchref.experimental.alignment.ml_rotation import _build_llg_context

    seed = seed_for(args.pdb, args.trial)
    model, data, R_true = rotated_case(args.pdb, seed)
    sym = data.spacegroup.matrices.to(torch.float64).cpu()
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    okw = dict(side="left", frame="cart", reciprocal_basis=rec, thr_deg=5.0)

    res = run_frf(model, data, FRFConfig(n_peaks=500, lmax_cap=args.lmax_cap),
                  capture_arf=False, verbose=0)
    head = res.peaks[: args.n_refine]
    truth_rank, truth_ang = orbit_rank(head, R_true, sym, **okw)
    if truth_rank < 0:
        print(f"truth not in the top {args.n_refine}; nothing for the rescore "
              f"to find here")
        return 0

    inp = res.inputs
    prep = {}
    if args.full_prep:
        prep = dict(vrms_strategy="oeffner",
                    vrms_n_residues=max(1, int(model.xyz().shape[0] / 8)),
                    apply_bulk_solvent=True, apply_wilson_b=True)
    ctx = _build_llg_context(
        inp.F_obs, inp.hkl, inp.s_mag, inp.centric, inp.ll, data.cell,
        data.spacegroup.matrices.to(torch.float64).to(inp.device),
        n_shells=max(20 // 2, 8), batch_size=50, **prep,
    )

    a = torch.tensor([p.alpha for p in head], dtype=torch.float64)
    b = torch.tensor([p.beta for p in head], dtype=torch.float64)
    g = torch.tensor([p.gamma for p in head], dtype=torch.float64)
    ll, eImove = per_reflection_ll(ctx, a, b, g)
    totals = ll.sum(dim=-1)
    order = torch.argsort(totals, descending=True)
    new_rank = int((order == truth_rank).nonzero()[0, 0])
    winner = int(order[0])

    print(f"=== {args.pdb} trial {args.trial} "
          f"({'full prep' if args.full_prep else 'shipped defaults'}) ===")
    print(f"  truth is FRF rank {truth_rank} ({truth_ang:.2f} deg), "
          f"LLG rank {new_rank}; winner is FRF rank {winner}")
    print(f"  LLG(truth)  = {totals[truth_rank]:.4f}")
    print(f"  LLG(winner) = {totals[winner]:.4f}   "
          f"gap = {totals[winner] - totals[truth_rank]:+.4f}")
    if winner == truth_rank:
        print("  truth already wins here")
        return 0

    # Where does the gap accumulate? Equal-count shells in |s|.
    s = inp.s_mag.to(torch.float64).cpu()
    n_sh = args.n_shells
    edge_idx = torch.linspace(0, s.numel() - 1, n_sh + 1).round().long()
    edges = s.sort().values[edge_idx]
    shell = torch.bucketize(s, edges[1:-1])
    d_ll = (ll[winner] - ll[truth_rank]).to(torch.float64).cpu()
    cen = ctx.centric_b[0].cpu()

    print(f"\n  gap by resolution shell (winner - truth; positive = truth loses)")
    print(f"  {'shell':>5s} {'d range (A)':>16s} {'n':>7s} {'d LL':>10s} "
          f"{'cum %':>7s} {'acen':>9s} {'cen':>9s}")
    total_gap = float(d_ll.sum())
    cum = 0.0
    for k in range(n_sh):
        m = shell == k
        if not bool(m.any()):
            continue
        v = float(d_ll[m].sum())
        cum += v
        lo, hi = float(1.0 / edges[k + 1]), float(1.0 / edges[k])
        print(f"  {k:>5d} {f'{hi:6.1f}-{lo:5.2f}':>16s} {int(m.sum()):>7d} "
              f"{v:>+10.3f} {100 * cum / total_gap if total_gap else 0:>6.1f}% "
              f"{float(d_ll[m & ~cen].sum()):>+9.3f} "
              f"{float(d_ll[m & cen].sum()):>+9.3f}")
    print(f"  {'TOTAL':>5s} {'':>16s} {int(s.numel()):>7d} {total_gap:>+10.3f}")

    print(f"\n  expected moving intensity eImove, truth vs winner:")
    for name, idx in (("truth", truth_rank), ("winner", winner)):
        e = eImove[idx].to(torch.float64).cpu()
        print(f"    {name:7s} mean {e.mean():.4e}  median {e.median():.4e}  "
              f"max {e.max():.4e}  frac>E_obs^2 "
              f"{float((e > (ctx.E_obs_b[0].cpu() ** 2)).float().mean()):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
