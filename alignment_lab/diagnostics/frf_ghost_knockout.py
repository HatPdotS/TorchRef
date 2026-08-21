"""Find which obs-side term creates the trigonal/hexagonal ghosts.

Context. With Phaser's bandwidth, resolution and SO(3) sampling pinned, our FRF
puts truth at rank 0 on five of seven benchmark structures and beats Phaser on
6G9X -- but collapses on 2DQ6 (P 3_1 2 1, truth rank 76799) and 3GR5
(P 6_5 2 2, rank 16855), where Phaser gets rank 0 with a healthy margin. Those
are the only two cases with a 120 degree cell and a 3-fold-containing axis; the
working set is monoclinic / orthorhombic / tetragonal / cubic.

Because lmax and sampling were already pinned to Phaser's own values when the
collapse was measured, bandwidth is eliminated. What remains is the observation-
and calc-side preprocessing chain, where our engine differs from
``DataMR::getELMNxR2`` in ways that are individually documented but never
isolated on these two space groups:

* ``use_epsilon=False`` -- Phaser always normalises by ``sqrt(eps_n * Sigma_N)``
  (DataMR.cc:930). Without the epsilon divisor the axial/zonal reflections are
  over-weighted, and for a 6-fold axis those carry eps up to 6-12.
* ``_orbit_unroll=False`` -- Phaser expands over symmetry but skips duplicate
  P1 indices (``!duplicate(isym,rhkl)``, DataMR.cc:954). We replicate all n_ops
  unconditionally, so reflections on special positions are counted several times.
* the m-symmetry filter, French-Wilson, shell-variance weights, Wilson-B match,
  Oeffner vrms and the Babinet bulk-solvent term.

Each arm flips exactly one of these against a common baseline and reports where
truth lands. The discriminating statistic is ``margin`` -- truth's sigma minus
the strongest non-truth peak's. Negative means the rotation function prefers a
ghost, which is the failure we are chasing; rank alone hides how close the call
was.

Usage
-----
    python -m diagnostics.frf_ghost_knockout --pdb 2DQ6
    python -m diagnostics.frf_ghost_knockout --pdb 3GR5 --arm epsilon
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import FRFConfig, load_case, patched, run_frf  # noqa: E402
from lab.results import append_row, provenance  # noqa: E402

EXPERIMENT = "frf_ghost_knockout"

#: Phaser's own bandwidth / resolution / sampling per case, read from the
#: instrumented run (job 487737). Pinned so every arm differs only in the term
#: under test -- and so the result is comparable with the map-comparison run.
PHASER_PINNED = {
    "1DAW": dict(lmax=58, sampling_deg=6.233148, d_min_eff=5.70),
    "1AK5": dict(lmax=70, sampling_deg=5.155428, d_min_eff=4.84),
    "3K7M": dict(lmax=66, sampling_deg=5.464844, d_min_eff=5.39),
    "4BX9": dict(lmax=96, sampling_deg=3.774036, d_min_eff=6.73),
    "6G9X": dict(lmax=84, sampling_deg=4.314963, d_min_eff=5.92),
    "2DQ6": dict(lmax=76, sampling_deg=4.772056, d_min_eff=6.36),
    "3GR5": dict(lmax=66, sampling_deg=5.483967, d_min_eff=4.10),
}

#: One flipped term per arm. ``baseline`` is our production configuration.
ARMS = {
    "baseline":       {},
    "epsilon":        dict(use_epsilon=True),
    "orbit_unroll":   dict(_orbit_unroll=True),
    "eps+unroll":     dict(use_epsilon=True, _orbit_unroll=True),
    "no_m_filter":    dict(frf_use_m_filter=False),
    "no_french":      dict(frf_use_french_wilson=False),
    "no_shellvar":    dict(frf_use_shell_variance=False),
    "no_bulk_solv":   dict(apply_bulk_solvent=False),
    "no_wilson_b":    dict(apply_wilson_b=False),
    "vrms_fixed":     dict(vrms_strategy="fixed"),
    "acentric_only":  dict(frf_acentric_only=True),
}


def _orbit_of_identity(data):
    """Truth is the identity, up to the point group, in the Cartesian frame."""
    from lab.truth import symmetry_orbit

    symops = data.spacegroup.matrices.to(torch.float64).cpu()
    recip = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    return symmetry_orbit(
        torch.eye(3, dtype=torch.float64), symops,
        side="left", frame="cart", reciprocal_basis=recip,
    )


def _truth_and_margin(arf, orbit):
    """``(rank, sigma, angle, best_ghost_sigma, margin)`` for one map."""
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_matrix_from_edmonds_euler_batch,
    )

    v = arf.values.to(torch.float64).cpu()
    R = rotation_matrix_from_edmonds_euler_batch(
        arf.alphas.to(torch.float64).cpu(),
        arf.betas.to(torch.float64).cpu(),
        arf.gammas.to(torch.float64).cpu(),
    )
    sig = (v - v.mean()) / v.std().clamp(min=1e-30)

    best = None
    tol = 1.5 * float(arf.grid_sampling_deg)
    truth_mask = torch.zeros(v.numel(), dtype=torch.bool)
    for k in range(orbit.shape[0]):
        tr = torch.einsum("nij,ij->n", R, orbit[k])
        ang = torch.rad2deg(torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)))
        truth_mask |= ang <= tol
        j = int(torch.argmin(ang))
        if best is None or v[j] > v[best[0]]:
            best = (j, float(ang[j]))
    j, ang_j = best
    rank = int((v > v[j]).sum())
    ghost_sig = sig.masked_fill(truth_mask, float("-inf"))
    gs = float(ghost_sig.max())
    return rank, float(sig[j]), ang_j, gs, float(sig[j]) - gs


def run_arm(pdb: str, arm: str, extra: dict, *, n_peaks: int = 500) -> dict:
    """One knockout arm on one structure."""
    from torchref.experimental.alignment.frf import api as _api

    pin = PHASER_PINNED[pdb]
    model, data = load_case(pdb)
    orbit = _orbit_of_identity(data)

    def _pinned(model_radius_A, d_min_data, lmax_cap=48):
        return int(pin["lmax"]) + 1, float(pin["d_min_eff"])

    cfg = FRFConfig(
        n_peaks=n_peaks, lmax_cap=int(pin["lmax"]),
        extra={"grid_sampling_deg": float(pin["sampling_deg"]), **extra},
    )
    t0 = time.time()
    with patched(_api, "phaser_lmax_resolution", _pinned):
        res = run_frf(model, data, cfg, capture_arf=True, verbose=0)
    rank, sig, ang, ghost, margin = _truth_and_margin(res.arf, orbit)

    row = {"experiment": EXPERIMENT, "pdb": pdb, "arm": arm}
    row.update(provenance())
    row.update({
        "spacegroup": str(data.spacegroup.hm),
        "n_orbit": int(orbit.shape[0]),
        "lmax": pin["lmax"],
        "sampling_deg": pin["sampling_deg"],
        "d_min_eff": pin["d_min_eff"],
        "n_samples": int(res.arf.values.numel()),
        "truth_rank": rank,
        "truth_sigma": round(sig, 4),
        "truth_angle_deg": round(ang, 3),
        "best_ghost_sigma": round(ghost, 4),
        "margin": round(margin, 4),
        "map_max_sigma": round(float(res.map_max_sigma), 4),
        "seconds": round(time.time() - t0, 1),
    })
    row.update({f"flag_{k}": v for k, v in extra.items()})
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, choices=sorted(PHASER_PINNED))
    ap.add_argument("--arm", help="single arm (default: all)")
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    arms = {args.arm: ARMS[args.arm]} if args.arm else ARMS
    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parents[1] / "runs" / EXPERIMENT
    )
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{EXPERIMENT}_{args.pdb}.csv"

    print(f"{args.pdb}: truth rank / margin per knockout arm", flush=True)
    print(f"{'arm':<15}{'rank':>9}{'truth_sig':>11}{'ghost_sig':>11}{'margin':>9}",
          flush=True)
    failures = 0
    for arm, extra in arms.items():
        try:
            row = run_arm(args.pdb, arm, extra, n_peaks=args.n_peaks)
        except Exception as exc:
            failures += 1
            print(f"{arm:<15}  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        append_row(csv_path, row)
        print(f"{arm:<15}{row['truth_rank']:>9}{row['truth_sigma']:>11.2f}"
              f"{row['best_ghost_sigma']:>11.2f}{row['margin']:>+9.2f}", flush=True)
    print(f"\nwrote {csv_path}", flush=True)
    return 1 if failures == len(arms) else 0


if __name__ == "__main__":
    raise SystemExit(main())
