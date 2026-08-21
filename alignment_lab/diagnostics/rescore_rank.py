"""Does the ML rescore improve the FRF ranking, or damage it?

The FRF reliably puts the true orientation inside the top 20 on this benchmark,
yet end-to-end pose recovery succeeds about half the time. That points at the
rescore, so this measures it directly and in isolation.

For each structure the FRF is run **once**, then every rescore arm is applied to
the *same* peak list -- including a ``none`` control that leaves the FRF order
untouched. The reported quantity is the paired change in the rank of truth
within the rescore window, so an arm that merely preserves a good input ranking
cannot be mistaken for one that improves it.

Rows where truth was never in the window are recorded with ``delta`` empty: the
rescore had nothing to find, and scoring it there would measure the FRF.

Usage::

    python alignment_lab/diagnostics/rescore_rank.py --pdb 1AK5 --trial 0 \
        --engines none,m_letf1,sim --out-csv alignment_lab/runs/rescore.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, ResultWriter, orbit_rank,  # noqa: E402
                 paired_ranks, rotated_case, run_frf, run_rescore, seed_for)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1AK5", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--n-refine", type=int, default=20,
                    help="rescore window: the top-N FRF peaks handed to the engine")
    ap.add_argument("--engines", default="none,m_letf1,sim")
    ap.add_argument("--subpeak-refine", action="store_true")
    ap.add_argument("--orbit-side", default="left", choices=["left", "right"])
    ap.add_argument("--orbit-frame", default="cart", choices=["cart", "frac"])
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    seed = seed_for(args.pdb, args.trial)
    rotated, data, R_true = rotated_case(args.pdb, seed)
    sym = data.spacegroup.matrices.to(torch.float64).cpu()
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    orbit_kw = dict(side=args.orbit_side, frame=args.orbit_frame,
                    reciprocal_basis=rec)

    cfg = FRFConfig(d_min=args.d_min, d_max=args.d_max,
                    n_peaks=args.n_peaks, lmax_cap=args.lmax_cap)
    frf = run_frf(rotated, data, cfg, verbose=args.verbose)
    rank_full, ang_full = orbit_rank(frf.peaks, R_true, sym, **orbit_kw)

    print(f"=== {args.pdb} t{args.trial} seed={seed} {data.spacegroup} "
          f"n_ops={sym.shape[0]} | FRF rank={rank_full} ang={ang_full:.2f} "
          f"({frf.seconds:.1f}s) | window={args.n_refine} ===")
    if rank_full < 0 or rank_full >= args.n_refine:
        print(f"  NOTE truth is outside the rescore window "
              f"(FRF rank {rank_full}); the rescore cannot recover it, so the "
              f"deltas below measure nothing about the rescore.")
    print(f"  {'engine':10s} {'rank_in':>8s} {'rank_out':>9s} {'delta':>6s} "
          f"{'ang_out':>8s} {'secs':>7s}")

    writer = None
    if args.out_csv:
        writer = ResultWriter(args.out_csv, "rescore_rank",
                              extra_fields=("engine", "n_refine", "rank_frf_full",
                                            "rank_frf_window", "rank_rescored",
                                            "delta", "truth_in_window",
                                            "angle_rescored", "rescore_seconds",
                                            "subpeak_refine"))
    for engine in [e.strip() for e in args.engines.split(",") if e.strip()]:
        res = run_rescore(frf.peaks, data, frf.inputs, engine=engine,
                          n_refine=args.n_refine,
                          subpeak_refine=args.subpeak_refine,
                          verbose=args.verbose)
        pr = paired_ranks(frf.peaks, res.peaks, R_true, sym,
                          n_refine=args.n_refine, **orbit_kw)
        delta = pr["delta"]
        print(f"  {engine:10s} {pr['rank_frf']:8d} {pr['rank_rescored']:9d} "
              f"{('' if delta is None else f'{delta:+d}'):>6s} "
              f"{pr['angle_rescored']:8.2f} {res.seconds:7.1f}")
        if writer:
            writer.write(pdb=args.pdb, seed=seed, trial=args.trial,
                         spacegroup=str(data.spacegroup), n_ops=int(sym.shape[0]),
                         truth_rank=pr["rank_rescored"],
                         truth_angle_deg=round(pr["angle_rescored"], 4),
                         orbit_side=args.orbit_side, orbit_frame=args.orbit_frame,
                         lmax_cap=args.lmax_cap, d_min=args.d_min, d_max=args.d_max,
                         device="cpu", engine=engine, n_refine=args.n_refine,
                         rank_frf_full=pr["rank_frf_full"],
                         rank_frf_window=pr["rank_frf"],
                         rank_rescored=pr["rank_rescored"],
                         delta=("" if delta is None else delta),
                         truth_in_window=int(pr["truth_in_window"]),
                         angle_rescored=round(pr["angle_rescored"], 4),
                         rescore_seconds=round(res.seconds, 2),
                         subpeak_refine=int(args.subpeak_refine))
    if args.out_csv:
        print(f"  wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
