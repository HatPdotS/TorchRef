"""Where do the truth-beating peaks come from? Vary only the observations.

Runs the identical engine on the identical rotated search model, changing
nothing but the observed amplitudes at the same Miller indices:

``real``
    the deposited measurements.
``crystal``
    ``|F_calc|`` of the deposited model in its real space group -- noiseless,
    solvent-free, complete. Ghosts surviving here are not noise, solvent,
    measurement error or missing data.
``molecule``
    ``|F_calc|`` of the same model in **P1** -- the self-Patterson only, with
    the symmetry mates removed. Ghosts vanishing here are intermolecular.

Substituting observations is safe because the FRF reads only ``F``, ``F_sigma``,
``hkl``, ``centric``, ``cell`` and ``spacegroup`` from the dataset.

Usage::

    python alignment_lab/diagnostics/ghost_origin.py --pdb 3K7M --trial 0 \
        --out-csv alignment_lab/runs/ghosts.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, ResultWriter, load_case,  # noqa: E402
                 orbit_rank, random_rotation, run_frf, seed_for)
from lab.truth import angle_to_orbit, symmetry_orbit  # noqa: E402


def substituted_data(data, model, mode: str):
    """Return a dataset whose ``F`` is replaced according to ``mode``."""
    if mode == "real":
        return data
    m = model.copy()
    if mode == "molecule":
        # NOTE: assign the space-group NAME. SpaceGroup is an nn.Module, so
        # assigning the object is intercepted by nn.Module.__setattr__ and the
        # property setter never runs -- a silent no-op that would leave the
        # crystal symmetry in place and quietly invalidate this whole arm.
        m.spacegroup = "P 1"
    else:
        m.spacegroup = data.spacegroup.hm
    m.reset_cache()
    out = data.copy() if hasattr(data, "copy") else data
    F = m(out.hkl).abs().detach().to(out.F.dtype)
    out.F = F
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="3K7M", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--orbit-side", default="left", choices=["left", "right"])
    ap.add_argument("--orbit-frame", default="cart", choices=["cart", "frac"])
    ap.add_argument("--modes", default="real,crystal,molecule")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    seed = seed_for(args.pdb, args.trial)
    model, data = load_case(args.pdb)
    R_true = random_rotation(seed)
    rotated = model.copy().rotate(R_true.to(model.dtype_float),
                                  center=model.xyz().mean(0))
    sym = data.spacegroup.matrices.to(torch.float64).cpu()
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    orbit = symmetry_orbit(R_true, sym, side=args.orbit_side,
                           frame=args.orbit_frame, reciprocal_basis=rec)
    cfg = FRFConfig(d_min=args.d_min, d_max=args.d_max,
                    n_peaks=args.n_peaks, lmax_cap=args.lmax_cap)

    print(f"=== {args.pdb} trial {args.trial} seed {seed} | {data.spacegroup} "
          f"n_ops={sym.shape[0]} | lmax_cap={args.lmax_cap} ===")
    print(f"  {'obs':10s} {'rank':>6s} {'ghosts':>7s} {'truth_sig':>10s} {'map_max':>8s}")

    writer = None
    if args.out_csv:
        writer = ResultWriter(args.out_csv, "ghost_origin",
                              extra_fields=("obs_mode", "n_ghosts_above",
                                            "truth_sigma", "map_max_sigma",
                                            "n_peaks"))
    for mode in args.modes.split(","):
        mode = mode.strip()
        sub = substituted_data(data, model, mode)
        res = run_frf(rotated, sub, cfg)
        rank, ang = orbit_rank(res.peaks, R_true, sym, side=args.orbit_side,
                               frame=args.orbit_frame, reciprocal_basis=rec)
        # Peaks outranking truth, i.e. the ghosts this arm produces.
        n_ghosts = rank if rank >= 0 else len(res.peaks)
        truth_sigma = float(res.peaks[rank].sigma) if rank >= 0 else float("nan")
        print(f"  {mode:10s} {rank:6d} {n_ghosts:7d} {truth_sigma:10.3f} "
              f"{res.map_max_sigma:8.3f}")
        if writer:
            writer.write(pdb=args.pdb, seed=seed, trial=args.trial,
                         spacegroup=str(data.spacegroup), n_ops=int(sym.shape[0]),
                         truth_rank=rank, truth_angle_deg=round(ang, 4),
                         orbit_side=args.orbit_side, orbit_frame=args.orbit_frame,
                         lmax_cap=args.lmax_cap, d_min=args.d_min, d_max=args.d_max,
                         device="cpu", obs_mode=mode, n_ghosts_above=n_ghosts,
                         truth_sigma=round(truth_sigma, 4),
                         map_max_sigma=round(res.map_max_sigma, 4),
                         n_peaks=len(res.peaks))
    if args.out_csv:
        print(f"  wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
