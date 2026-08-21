"""Rank of the true orientation in the FRF peak list, per structure.

The basic health check: rotate a deposited model by a seeded random rotation,
run the rotation function against that structure's own measured amplitudes, and
ask where the true orientation lands. Rank 0 means the top peak is correct.

Peaks that outrank truth are the "ghosts" -- genuine correlations between the
model's self-Patterson and the crystal's intermolecular vectors, not noise.

Usage::

    python alignment_lab/diagnostics/frf_rank.py --pdb 1AK5 --trial 0 \
        --lmax-cap 64 --out-csv alignment_lab/runs/rank.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, ResultWriter, orbit_rank,  # noqa: E402
                 rotated_case, run_frf, seed_for)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--orbit-side", default="left", choices=["left", "right"])
    ap.add_argument("--orbit-frame", default="cart", choices=["cart", "frac"])
    ap.add_argument("--thr-deg", type=float, default=5.0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    seed = seed_for(args.pdb, args.trial)
    t0 = time.time()
    rotated, data, R_true = rotated_case(args.pdb, seed)
    load_s = time.time() - t0

    sym = data.spacegroup.matrices.to(torch.float64).cpu()
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    cfg = FRFConfig(d_min=args.d_min, d_max=args.d_max,
                    n_peaks=args.n_peaks, lmax_cap=args.lmax_cap)
    res = run_frf(rotated, data, cfg)
    rank, ang = orbit_rank(res.peaks, R_true, sym, side=args.orbit_side,
                           frame=args.orbit_frame, reciprocal_basis=rec,
                           thr_deg=args.thr_deg)
    truth_sigma = float(res.peaks[rank].sigma) if rank >= 0 else float("nan")

    print(f"{args.pdb:6s} t{args.trial} seed={seed:<6d} {str(data.spacegroup):28s} "
          f"n_ops={sym.shape[0]:2d} n_refl={data.hkl.shape[0]:7d} | "
          f"rank={rank:4d} ang={ang:6.2f} truth_sig={truth_sigma:7.3f} "
          f"map_max={res.map_max_sigma:7.3f} | frf={res.seconds:6.1f}s load={load_s:5.1f}s")

    if args.out_csv:
        w = ResultWriter(args.out_csv, "frf_rank",
                         extra_fields=("truth_sigma", "map_max_sigma", "n_peaks",
                                       "n_ghosts_above", "n_refl",
                                       "frf_seconds", "load_seconds"))
        w.write(pdb=args.pdb, seed=seed, trial=args.trial,
                spacegroup=str(data.spacegroup), n_ops=int(sym.shape[0]),
                truth_rank=rank, truth_angle_deg=round(ang, 4),
                orbit_side=args.orbit_side, orbit_frame=args.orbit_frame,
                lmax_cap=args.lmax_cap, d_min=args.d_min, d_max=args.d_max,
                device="cpu",
                truth_sigma=round(truth_sigma, 4),
                map_max_sigma=round(res.map_max_sigma, 4),
                n_peaks=len(res.peaks),
                n_ghosts_above=(rank if rank >= 0 else len(res.peaks)),
                n_refl=int(data.hkl.shape[0]),
                frf_seconds=round(res.seconds, 2),
                load_seconds=round(load_s, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
