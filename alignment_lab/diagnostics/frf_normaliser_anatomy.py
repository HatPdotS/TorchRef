"""Decompose the observation-normaliser gap that costs 3GR5 its rank.

Injecting Phaser's per-reflection intensities into our otherwise-unchanged FRF
takes 3GR5 from rank 1995 / margin -1.06 to **rank 0 / margin +4.21** (job
489527), with every other stage already verified exact against Phaser. So the
whole remaining deficit is the normalised observation ``E^2``, and the question
is which part of it.

Phaser's normaliser is (``DataB.cc:1106-1113``)

    sqrt_epsnSN[r] = sqrt( eps_n(h) * binAnisoFactor(bin, ANISO, SOLK, SOLB, K) )

a BEST-curve per-bin Sigma_N clamped to [0.5, 2] of BEST and corrected by a
fitted Wilson K/B, times an anisotropic tensor, plus a bulk-solvent term, times
the reflection multiplicity -- all refined. Ours is equal-count shell means of
``F^2`` (``preprocessing.py:30``) applied to amplitudes that have already had a
separately fitted overall anisotropy divided out (``align.py``:
``apply_overall_anisotropy``), then French-Wilson.

So anisotropy is *not* simply missing on our side; it is removed upstream instead
of being folded into Sigma_N, which is equivalent only if the fitted tensor is
right. This splits ``log(E_phaser / E_ours)`` into pieces that can be fixed
independently:

1. ``eps_n`` -- the multiplicity factor we omit entirely (its docstring in
   ``build_lerf1_intensity`` claims it is "implicit in the symmetry reduction",
   which is not the same thing as dividing by it).
2. the best possible **isotropic** model, a fine step function of ``|s|``.
   Fitting a smooth Sigma_N curve cannot beat this, so it bounds what the curve
   is worth.
3. a general quadratic form in Cartesian ``s`` -- residual **anisotropy** left
   over after our own correction, reported as the eigenvalue spread of the
   equivalent B tensor. This is the piece that matters most for a rotation
   function: an angular error in the observed Patterson is exactly what a
   rotation search is sensitive to, whereas a radial mis-scaling is not.

Whatever variance survives all three is what only Phaser's refined per-bin
treatment could account for.

Our side is *captured from the production path*, not reimplemented: the obs
``s`` and ``eEobs`` are spied out of the engine, so the anisotropy correction,
resolution window, shell edges and French-Wilson posterior are exactly the ones
production uses.

Usage
-----
    python -m diagnostics.frf_normaliser_anatomy --pdb 3GR5 \
        --dumps ../runs/encode_compare_489514/phaser/3GR5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import FRFConfig, load_case, patched  # noqa: E402
from lab.results import append_row, provenance  # noqa: E402
from diagnostics.frf_ghost_knockout import PHASER_PINNED  # noqa: E402
from diagnostics.frf_inject_phaser_obs import _pack  # noqa: E402

EXPERIMENT = "frf_normaliser_anatomy"


def capture_ours(pdb: str):
    """The obs ``s`` and ``eEobs`` the production engine actually expands.

    ``build_lerf1_intensity`` receives ``eEobs`` in the same order as the ``s``
    that reaches ``bessel_sh_expand`` -- both come from step 1's masked arrays
    and nothing between them reorders or filters -- so spying on the two calls
    gives an aligned pair.
    """
    from torchref.experimental.alignment import align as _align
    from torchref.experimental.alignment.frf import api as _api

    pin = PHASER_PINNED[pdb]
    cap: dict = {}
    orig_bessel = _api.bessel_sh_expand
    orig_lerf1 = _api.build_lerf1_intensity

    def spy_bessel(s, vals, **kw):
        if kw.get("zsymm", 1) > 1 and "s" not in cap:
            cap["s"] = s.detach().cpu().to(torch.float64)
        return orig_bessel(s, vals, **kw)

    def spy_lerf1(eEobs, centric, dfac=None, **kw):
        if "eEobs" not in cap:
            cap["eEobs"] = eEobs.detach().cpu().to(torch.float64)
            cap["centric"] = centric.detach().cpu().clone()
            cap["dfac"] = (torch.ones_like(cap["eEobs"]) if dfac is None
                           else dfac.detach().cpu().to(torch.float64))
        return orig_lerf1(eEobs, centric, dfac, **kw)

    model, data = load_case(pdb)

    def _pinned(model_radius_A, d_min_data, lmax_cap=48):
        return int(pin["lmax"]) + 1, float(pin["d_min_eff"])

    cfg = FRFConfig(n_peaks=5, lmax_cap=int(pin["lmax"]))
    with patched(_api, "phaser_lmax_resolution", _pinned), \
         patched(_api, "bessel_sh_expand", spy_bessel), \
         patched(_api, "build_lerf1_intensity", spy_lerf1):
        frf_in = _align._prepare_frf_inputs(
            model, data, d_min=cfg.d_min, d_max=cfg.d_max,
            n_shells=cfg.n_shells, verbose=0,
        )
        _align._run_frf_separate_rotation(
            model, data, frf_in, n_peaks=5, verbose=0,
            lmax_cap=int(pin["lmax"]),
            grid_sampling_deg=float(pin["sampling_deg"]),
        )
    for k in ("s", "eEobs"):
        if k not in cap:
            raise RuntimeError(f"failed to capture {k} from the engine")
    if cap["s"].shape[0] != cap["eEobs"].shape[0]:
        raise RuntimeError(
            f"capture misaligned: s has {cap['s'].shape[0]} rows, eEobs "
            f"{cap['eEobs'].shape[0]} -- something between step 1 and the "
            f"expansion filters the obs set")
    return cap, data


def phaser_esqr_lut(terms_csv: Path, sym_mats: torch.Tensor):
    """Phaser's ``Esqr``, keyed by every P1 index the reflection feeds."""
    d = np.loadtxt(terms_csv, delimiter=",", skiprows=1)
    if d.ndim == 1:
        d = d[None, :]
    hkl = torch.from_numpy(d[:, 0:3]).to(torch.float64)
    esqr = torch.from_numpy(d[:, 8]).to(torch.float64)
    orb = torch.einsum("kji,nj->nki", sym_mats.to(torch.float64), hkl)
    orb = orb.round().to(torch.long).reshape(-1, 3)
    vals = esqr.unsqueeze(1).expand(-1, sym_mats.shape[0]).reshape(-1)
    keys = torch.cat([_pack(orb), _pack(-orb)])
    vals = torch.cat([vals, vals])
    uniq, inv = torch.unique(keys, return_inverse=True)
    lut = torch.zeros(uniq.numel(), dtype=torch.float64)
    lut[inv] = vals
    return uniq, lut


def _quadratic_design(s: torch.Tensor) -> torch.Tensor:
    """``[1, sx^2, sy^2, sz^2, 2 sx sy, 2 sx sz, 2 sy sz]``: constant + 6 aniso."""
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    return torch.stack([torch.ones_like(x), x * x, y * y, z * z,
                        2 * x * y, 2 * x * z, 2 * y * z], dim=1)


def _fit(A: torch.Tensor, b: torch.Tensor):
    sol = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
    return sol, float((b - A @ sol).var(unbiased=False))


def run(pdb: str, dumps: Path) -> dict:
    from torchref.experimental.alignment.frf.preprocessing import compute_epsilon

    cap, data = capture_ours(pdb)
    sg = data.spacegroup.matrices.to(torch.float64).cpu()
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    rec_inv = torch.linalg.inv(rec)

    s = cap["s"]
    hkl = (s @ rec_inv).round().to(torch.long)
    keys, lut = phaser_esqr_lut(dumps / "terms.csv", sg)
    k = _pack(hkl)
    pos = torch.searchsorted(keys, k).clamp(max=keys.numel() - 1)
    found = keys[pos] == k

    esqr_p = lut[pos][found]
    esqr_o = (cap["eEobs"] ** 2)[found]
    s = s[found]
    hkl = hkl[found]
    good = (esqr_p > 1e-12) & (esqr_o > 1e-12)
    esqr_p, esqr_o, s, hkl = esqr_p[good], esqr_o[good], s[good], hkl[good]

    y = torch.log(esqr_p / esqr_o)          # log ratio of NORMALISED intensities
    n = int(y.numel())
    var_tot = float(y.var(unbiased=False))

    eps = compute_epsilon(hkl, sg)
    # Phaser divides intensity by eps_n and we do not, so its Esqr should be
    # SMALLER by that factor: log ratio carries -log(eps).
    y_eps = y + torch.log(eps)
    var_eps = float(y_eps.var(unbiased=False))

    smag = s.norm(dim=-1)
    n_fine = 40
    q = torch.linspace(0, 1, n_fine + 1, dtype=torch.float64)[1:-1]
    fbin = torch.bucketize(smag, torch.quantile(smag, q))
    y_iso = y_eps.clone()
    for b in range(n_fine):
        m = fbin == b
        if int(m.sum()) > 1:
            y_iso[m] = y_eps[m] - y_eps[m].mean()
    var_iso = float(y_iso.var(unbiased=False))

    A = _quadratic_design(s)
    _, var_aniso = _fit(A, y_iso)
    coef, _ = _fit(A, y_eps)
    C = torch.tensor([[coef[1], coef[4], coef[5]],
                      [coef[4], coef[2], coef[6]],
                      [coef[5], coef[6], coef[3]]], dtype=torch.float64)
    # log(E^2) = ... + s^T C s; on intensities a B-factor is exp(-B s^2 / 2),
    # so the equivalent B is -2 C.
    ev = torch.linalg.eigvalsh(-2.0 * C)

    row = {"experiment": EXPERIMENT, "pdb": pdb}
    row.update(provenance())
    row.update({
        "spacegroup": str(data.spacegroup.hm),
        "n_obs_ours": int(cap["s"].shape[0]),
        "n_matched": n,
        "matched_frac": round(float(found.to(torch.float64).mean()), 4),
        "rms_log_ratio_pct": round(100.0 * float(y.std(unbiased=False)), 2),
        "mean_log_ratio": round(float(y.mean()), 4),
        "var_total": var_tot,
        "frac_eps": round(1.0 - var_eps / max(var_tot, 1e-300), 4),
        "frac_iso": round((var_eps - var_iso) / max(var_tot, 1e-300), 4),
        "frac_aniso": round((var_iso - var_aniso) / max(var_tot, 1e-300), 4),
        "frac_unexplained": round(var_aniso / max(var_tot, 1e-300), 4),
        "aniso_B_min": round(float(ev[0]), 2),
        "aniso_B_max": round(float(ev[2]), 2),
        "aniso_B_spread": round(float(ev[2] - ev[0]), 2),
        "n_eps_gt1": int((eps > 1.0001).sum()),
        "eps_max": round(float(eps.max()), 1),
    })
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, choices=sorted(PHASER_PINNED))
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parents[1] / "runs" / EXPERIMENT)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{EXPERIMENT}_{args.pdb}.csv"

    row = run(args.pdb, Path(args.dumps))
    append_row(csv_path, row)
    print(f"\n{args.pdb} ({row['spacegroup']}): variance of "
          f"log(Esqr_phaser / Esqr_ours), n={row['n_matched']} "
          f"({row['matched_frac']:.4f} of our obs matched)", flush=True)
    print(f"  rms disagreement        {row['rms_log_ratio_pct']:6.2f}%   "
          f"mean log ratio {row['mean_log_ratio']:+.4f}", flush=True)
    print(f"  explained by eps_n      {row['frac_eps']*100:6.2f}%   "
          f"({row['n_eps_gt1']} refl with eps>1, max {row['eps_max']})", flush=True)
    print(f"  explained by iso(|s|)   {row['frac_iso']*100:6.2f}%", flush=True)
    print(f"  explained by anisotropy {row['frac_aniso']*100:6.2f}%   "
          f"(equivalent B {row['aniso_B_min']} .. {row['aniso_B_max']} A^2, "
          f"spread {row['aniso_B_spread']})", flush=True)
    print(f"  unexplained             {row['frac_unexplained']*100:6.2f}%", flush=True)
    print(f"\nwrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
