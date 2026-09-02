"""Does the anisotropy fix hold on the task the pipeline actually runs?

Every number behind the anisotropy diagnosis was measured with the model in its
DEPOSITED orientation and with lmax / sampling / resolution pinned to Phaser's
own logged values -- one evaluation per structure, truth at the identity. That
was the right setup for a bisection against Phaser, and it is the wrong setup
for deciding a default:

* the pipeline searches a RANDOMLY ROTATED model, not the identity, and the
  ghosts are pose-dependent;
* it runs at the production configuration, not Phaser's pinned one;
* seed-to-seed truth-rank spread at ``lmax_cap = 64`` is +-4 to 6 ranks
  (1AK5 [9, 11, 17], 3K7M [7, 8, 20]), and three earlier findings in this
  investigation looked strong at n <= 7 and vanished at full n.

So this re-measures the arms over seeded random rotations at the production
config, reporting **per-trial paired differences against the production arm**
rather than a bare median -- the same discipline the rest of the lab uses.

Arms are :data:`lab.aniso.ARMS`: ``production``, ``no_aniso``, ``iso_only``,
``fixed_fit``.

Usage
-----
    python -m diagnostics.frf_aniso_rank_sweep --pdb 3GR5 --trials 10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (ANISO_ARMS, BENCH_PDBS, FRFConfig, aniso_arm,  # noqa: E402
                 orbit_rank, rotated_case, run_frf, seed_for, tensor_report)
from lab.results import append_row, provenance  # noqa: E402

EXPERIMENT = "frf_aniso_rank_sweep"


def run_one(pdb: str, trial: int, arm: str, cfg: FRFConfig,
            *, thr_deg: float) -> dict:
    seed = seed_for(pdb, trial)
    model, data, R_true = rotated_case(pdb, seed)
    captured: dict = {}
    t0 = time.time()
    with aniso_arm(arm, data, d_min=cfg.d_min, d_max=cfg.d_max,
                   captured=captured):
        res = run_frf(model, data, cfg, capture_arf=False, verbose=0)
    seconds = time.time() - t0

    rank, ang = orbit_rank(
        res.peaks, R_true, data.spacegroup.matrices.to(torch.float64).cpu(),
        reciprocal_basis=data.cell.reciprocal_basis_matrix.to(torch.float64).cpu(),
        side="right", frame="cart", thr_deg=thr_deg,
    )
    row = {"experiment": EXPERIMENT, "pdb": pdb, "trial": trial, "arm": arm,
           "seed": seed}
    row.update(provenance())
    row.update(cfg.as_row())
    row.update({
        "spacegroup": str(data.spacegroup.hm),
        "truth_rank": rank,
        # orbit_rank returns -1 for "no peak within thr_deg". That must NOT be
        # ordered as a good rank: for paired comparison a miss counts as worse
        # than the worst hit, i.e. the peak-list length.
        "rank_for_compare": rank if rank >= 0 else cfg.n_peaks,
        "found": int(rank >= 0),
        "truth_angle_deg": None if ang is None else round(float(ang), 3),
        "n_peaks_found": len(res.peaks),
        "orbit_side": "left", "orbit_frame": "cart", "thr_deg": thr_deg,
        "seconds": round(seconds, 1),
    })
    for tag in ("raw", "fixed"):
        if tag in captured:
            row.update(tensor_report(captured[tag], tag))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, choices=list(BENCH_PDBS))
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--arms", default=",".join(ANISO_ARMS))
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--d-min", type=float, default=4.0)
    ap.add_argument("--d-max", type=float, default=15.0)
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--thr-deg", type=float, default=5.0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    cfg = FRFConfig(d_min=args.d_min, d_max=args.d_max,
                    n_peaks=args.n_peaks, lmax_cap=args.lmax_cap)
    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parents[1] / "runs" / EXPERIMENT)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{EXPERIMENT}_{args.pdb}.csv"

    print(f"{args.pdb}: truth rank per trial, lmax_cap={args.lmax_cap}",
          flush=True)
    print(f"{'trial':>6}" + "".join(f"{a:>15}" for a in arms), flush=True)
    ranks = {a: [] for a in arms}
    n_fail = 0
    for trial in range(args.trials):
        cells = []
        for arm in arms:
            try:
                row = run_one(args.pdb, trial, arm, cfg, thr_deg=args.thr_deg)
            except Exception as exc:
                n_fail += 1
                ranks[arm].append(None)
                cells.append(f"{type(exc).__name__}")
                print(f"  trial {trial} arm {arm} FAILED: {exc}", flush=True)
                continue
            append_row(csv_path, row)
            ranks[arm].append(row["rank_for_compare"])
            cells.append(str(row["truth_rank"]) if row["found"]
                         else f"miss({row['truth_angle_deg']:.0f}d)")
        print(f"{trial:>6}" + "".join(f"{c:>15}" for c in cells), flush=True)

    # Paired differences against production; per-trial signs, never a bare median.
    base = ranks.get("production")
    if base:
        print("\npaired vs production (negative = better rank):", flush=True)
        for arm in arms:
            if arm == "production":
                continue
            d = [(a - b) for a, b in zip(ranks[arm], base)
                 if a is not None and b is not None]
            if not d:
                print(f"  {arm:<14} no paired trials", flush=True)
                continue
            sd = sorted(d)
            med = (sd[len(sd) // 2] if len(sd) % 2
                   else 0.5 * (sd[len(sd) // 2 - 1] + sd[len(sd) // 2]))
            print(f"  {arm:<14} n={len(d):<3} better={sum(x < 0 for x in d)} "
                  f"same={sum(x == 0 for x in d)} worse={sum(x > 0 for x in d)} "
                  f"median_delta={med:+.1f}  per-trial={d}", flush=True)
    print(f"\nwrote {csv_path}  ({n_fail} failures)", flush=True)
    return 1 if n_fail == len(arms) * args.trials else 0


if __name__ == "__main__":
    raise SystemExit(main())
