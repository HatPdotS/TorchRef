"""Is our own anisotropy correction what destroys the hexagonal cases?

The normaliser anatomy (job 489537) decomposed ``log(Esqr_phaser / Esqr_ours)``
and found the disagreement is overwhelmingly **angular**, and only on the two
failing structures:

| pdb  | rms   | eps_n  | iso(|s|) | anisotropy | equivalent B spread |
|------|-------|--------|----------|------------|---------------------|
| 1AK5 |  14%  | 61.5%  | 23.1%    | **0.01%**  | 0.36 A^2            |
| 2DQ6 |  44%  |  1.6%  | 16.4%    | **78.1%**  | 158 A^2             |
| 3GR5 | 116%  |  0.9%  |  6.0%    | **92.2%**  | 189 A^2             |

A radial mis-scaling is nearly harmless to a rotation function; an angular one is
exactly what it measures. So the suspect is our own overall-anisotropy
correction, ``fit_overall_anisotropy`` (``sh.py:445``), which regresses
``ln|F|^2 - ln<|F|^2>_shell`` on ``-2 pi^2 s.U.s`` by unweighted least squares
**with no constant term**. Single-reflection ``ln|F|^2`` is a badly behaved
regressand: its expectation is offset by ``-gamma`` for acentrics and
``-gamma - ln 2`` for centrics, the ``clamp(min=1e-30)`` turns a vanishing
amplitude into ``y ~ -69``, and with no intercept every one of those offsets is
absorbed into the quadratic form.

``symmetrize_anisotropy`` then projects the result onto the point-group-invariant
subspace, and the code comment records that the raw fit gives eigenvalues
``(0.8, 17, 70) A^2`` on a *cubic* dataset where symmetry forces them equal. That
projection is why the damage is invisible on the working structures and not on
these two:

* cubic -> 1 DOF (lambda I): the garbage is annihilated;
* trigonal/hexagonal -> 2 DOF (diag(lambda, lambda, mu)): a **uniaxial tensor
  along c is symmetry-allowed**, so the garbage survives as exactly the fake
  anisotropy the anatomy measures.

Three arms settle it, and the null arm is the one that matters -- if switching the
correction off recovers truth, our correction is not merely imperfect, it is
actively destructive:

* ``production``  -- fitted, symmetrised U;
* ``no_aniso``    -- U = 0, no correction at all;
* ``iso_only``    -- U = (trace/3) I, keeping the radial part and dropping every
  angular component, which the shell means then absorb.

Also reported per structure: the eigenvalues of the fitted tensor before and
after symmetrisation, as B = 8 pi^2 U, so the size of the artefact is visible
next to the rank it costs.

Usage
-----
    python -m diagnostics.frf_aniso_knockout --pdb 3GR5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import (ANISO_ARMS, FRFConfig, aniso_arm, load_case,  # noqa: E402
                 patched, run_frf, tensor_report)
from lab.results import append_row, provenance  # noqa: E402
from diagnostics.frf_ghost_knockout import (  # noqa: E402
    PHASER_PINNED, _orbit_of_identity, _truth_and_margin,
)

EXPERIMENT = "frf_aniso_knockout"

ARMS = ANISO_ARMS


def run_arm(pdb: str, arm: str, *, n_peaks: int = 500) -> dict:
    from torchref.experimental.alignment.frf import api as _api

    pin = PHASER_PINNED[pdb]
    model, data = load_case(pdb)
    orbit = _orbit_of_identity(data)

    seen: dict = {}
    cfg_probe = FRFConfig()

    def _pinned(model_radius_A, d_min_data, lmax_cap=48):
        return int(pin["lmax"]) + 1, float(pin["d_min_eff"])

    cfg = FRFConfig(n_peaks=n_peaks, lmax_cap=int(pin["lmax"]),
                    extra={"grid_sampling_deg": float(pin["sampling_deg"])})
    t0 = time.time()
    with patched(_api, "phaser_lmax_resolution", _pinned), \
         aniso_arm(arm, data, d_min=cfg_probe.d_min, d_max=cfg_probe.d_max,
                   captured=seen):
        res = run_frf(model, data, cfg, capture_arf=True, verbose=0)
    rank, sig, ang, ghost, margin = _truth_and_margin(res.arf, orbit)

    # The tensor actually applied, recomputed the same way align.py does it.
    from torchref.experimental.alignment.sh import (
        hkl_symops_to_cartesian, symmetrize_anisotropy,
    )
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64)
    cart = hkl_symops_to_cartesian(
        data.spacegroup.matrices.to(torch.float64), rec)
    raw = seen.get("raw")

    row = {"experiment": EXPERIMENT, "pdb": pdb, "arm": arm}
    row.update(provenance())
    row.update({
        "spacegroup": str(data.spacegroup.hm),
        "n_ops": int(data.spacegroup.matrices.shape[0]),
        "lmax": pin["lmax"], "sampling_deg": pin["sampling_deg"],
        "d_min_eff": pin["d_min_eff"],
        "n_samples": int(res.arf.values.numel()),
        "truth_rank": rank, "truth_sigma": round(sig, 4),
        "truth_angle_deg": round(ang, 3),
        "best_ghost_sigma": round(ghost, 4), "margin": round(margin, 4),
        "seconds": round(time.time() - t0, 1),
    })
    if raw is not None:
        row.update(tensor_report(raw.cpu(), "raw"))
        row.update(tensor_report(
            symmetrize_anisotropy(raw.to(torch.float64).cpu(), cart.cpu()),
            "sym"))
    if "fixed" in seen:
        row.update(tensor_report(seen["fixed"].cpu(), "fix_raw"))
        row.update(tensor_report(
            symmetrize_anisotropy(seen["fixed"].to(torch.float64).cpu(),
                                  cart.cpu()), "fix_sym"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, choices=sorted(PHASER_PINNED))
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parents[1] / "runs" / EXPERIMENT)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{EXPERIMENT}_{args.pdb}.csv"

    rows = [run_arm(args.pdb, a, n_peaks=args.n_peaks) for a in ARMS]
    r0 = rows[0]
    print(f"\n{args.pdb} ({r0['spacegroup']}): fitted anisotropy as B (A^2) -- "
          f"raw {r0.get('raw_B_min')}..{r0.get('raw_B_max')} "
          f"(spread {r0.get('raw_B_spread')}), after symmetrisation "
          f"{r0.get('sym_B_min')}..{r0.get('sym_B_max')} "
          f"(spread {r0.get('sym_B_spread')})", flush=True)
    print(f"{'arm':<14}{'rank':>8}{'truth_sig':>11}{'ghost_sig':>11}{'margin':>9}",
          flush=True)
    cols = {}
    for r in rows:
        cols.update({k: "" for k in r})
    for r in rows:
        append_row(csv_path, {**cols, **r})
        print(f"{r['arm']:<14}{r['truth_rank']:>8}{r['truth_sigma']:>11.2f}"
              f"{r['best_ghost_sigma']:>11.2f}{r['margin']:>+9.2f}", flush=True)
    print(f"\nwrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
