"""Head-to-head: our FRF vs Phaser on identical input.

Both engines get the same rotated search model and the same reflections, and
both truth ranks are computed with the same orbit machinery, so the comparison
isolates the algorithms rather than the data handling.

Read the caveats in :mod:`lab.phaser` before interpreting a result: Phaser
returns ~80-92k densely spaced samples, so a small "closest sample" angle is
expected regardless of whether it ranked truth well.

Usage::

    python alignment_lab/diagnostics/phaser_headtohead.py --pdb 1AK5 --trial 0 \
        --out-csv alignment_lab/runs/h2h.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, FRFConfig, ResultWriter, orbit_rank,  # noqa: E402
                 rotated_case, run_frf, seed_for)
from lab import phaser as ph  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1AK5", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--orbit-side", default="left", choices=["left", "right"])
    ap.add_argument("--orbit-frame", default="cart", choices=["cart", "frac"])
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--timeout-s", type=int, default=5400)
    ap.add_argument("--skip-phaser", action="store_true",
                    help="run only our engine (no phenix on this host)")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    seed = seed_for(args.pdb, args.trial)
    work = Path(args.workdir or (Path(__file__).resolve().parents[1] /
                                 "runs" / f"h2h_{args.pdb}_t{args.trial}")).resolve()
    work.mkdir(parents=True, exist_ok=True)

    rotated, data, R_true = rotated_case(args.pdb, seed)
    sym = data.spacegroup.matrices.to(torch.float64).cpu()
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    orbit_kw = dict(side=args.orbit_side, frame=args.orbit_frame,
                    reciprocal_basis=rec)

    print(f"=== {args.pdb} trial {args.trial} seed {seed} | {data.spacegroup} "
          f"n_ops={sym.shape[0]} ===")

    cfg = FRFConfig(d_min=args.d_min, d_max=args.d_max,
                    n_peaks=args.n_peaks, lmax_cap=args.lmax_cap)
    res = run_frf(rotated, data, cfg)
    our_rank, our_ang = orbit_rank(res.peaks, R_true, sym, **orbit_kw)
    print(f"  OURS   rank={our_rank:5d}  closest={our_ang:6.2f} deg  "
          f"({len(res.peaks)} peaks, {res.seconds:.1f}s, map max {res.map_max_sigma:.2f} sigma)")

    ph_rank, ph_ang, ph_n, ph_secs, ph_rc = -1, float("inf"), 0, 0.0, None
    if not args.skip_phaser:
        model_pdb = work / "rotated.pdb"
        rotated.write_pdb(str(model_pdb))
        _, mtz_path = __import__("lab").case_paths(args.pdb)
        kw = ph.write_frf_keywords(work, mtz_path=mtz_path, model_pdb=model_pdb)
        ph_rc, ph_secs = ph.run_phaser(work, kw, timeout_s=args.timeout_s)
        peaks = ph.parse_rlist(work / "phaser_frf.rlist")
        ph_n = len(peaks)
        if not peaks:
            # rc==0 is not a success test; an empty list is the real signal.
            print(f"  PHASER produced no peaks (rc={ph_rc}); see {work}/phaser.stdout")
        else:
            ph_rank, ph_ang = ph.phaser_truth_rank(peaks, R_true, sym, **orbit_kw)
            print(f"  PHASER rank={ph_rank:5d}  closest={ph_ang:6.2f} deg  "
                  f"({ph_n} samples, {ph_secs:.0f}s)")

    if args.out_csv:
        w = ResultWriter(args.out_csv, "phaser_headtohead",
                         extra_fields=("our_rank", "our_angle_deg", "our_seconds",
                                       "our_n_peaks", "map_max_sigma",
                                       "phaser_rank", "phaser_angle_deg",
                                       "phaser_samples", "phaser_seconds", "phaser_rc"))
        w.write(pdb=args.pdb, seed=seed, trial=args.trial,
                spacegroup=str(data.spacegroup), n_ops=int(sym.shape[0]),
                truth_rank=our_rank, truth_angle_deg=round(our_ang, 4),
                orbit_side=args.orbit_side, orbit_frame=args.orbit_frame,
                lmax_cap=args.lmax_cap, d_min=args.d_min, d_max=args.d_max,
                device="cpu",
                our_rank=our_rank, our_angle_deg=round(our_ang, 4),
                our_seconds=round(res.seconds, 2), our_n_peaks=len(res.peaks),
                map_max_sigma=round(res.map_max_sigma, 4),
                phaser_rank=ph_rank,
                phaser_angle_deg=(round(ph_ang, 4) if ph_n else ""),
                phaser_samples=ph_n, phaser_seconds=round(ph_secs, 1),
                phaser_rc=ph_rc if ph_rc is not None else "")
        print(f"  wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
