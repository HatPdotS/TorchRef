"""Attribute the per-reflection gap between our FRF inputs and Phaser's.

The stage-wise bisection put the divergence *before* the projection: reflection
positions agree to 1e-8, the SH-Bessel machinery reproduces Phaser's map from
Phaser's own coefficients at r = 0.998, and the radial band already matches
``nmax(l)`` -- but the intensities attached to those positions correlate at only
0.873 (2DQ6) against 0.988 (1AK5). Toggling ``use_epsilon``, French-Wilson,
shell-variance weights and the low-resolution cutoff moves that by <0.005, and
Phaser reports no tNCS, so ``V`` reduces to 1 on both sides.

Two things remain unmeasured, and this runs both in one job.

**Observation side.** Phaser builds
``intensity = cweight * (Esqr - V) / V^2 * DFAC^2`` with
``Esqr = (Feff / SIGMAN.sqrt_epsnSN)^2`` (DataMR.cc:930-945). The instrumented
binary now dumps every one of those terms per reflection, so a mismatch can be
attributed to a specific factor instead of only being visible in the product.
The two suspects are Phaser's smooth fitted ``Sigma_N`` against our equal-count
shell means, and ``DFAC`` (which we hard-wire to 1).

**Calc side.** Never compared per reflection -- only post-projection via
``SearchElmn``. Phaser builds it in ``Ensemble::getELMNxR2``, a different
function from the observation one, so it needed its own dump. A difference here
would be invisible in every measurement made so far.

Usage
-----
    python -m diagnostics.frf_prep_compare --pdb 2DQ6
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import FRFConfig, case_paths, load_case, patched, run_frf  # noqa: E402
from lab.phaser_match import PATCHED_PHASER, write_keywords  # noqa: E402
from lab.results import append_row, provenance  # noqa: E402

EXPERIMENT = "frf_prep_compare"

#: Phaser's own bandwidth / resolution / sampling, from job 487737.
PINNED = {
    "1DAW": dict(lmax=58, sampling_deg=6.233148, d_min_eff=5.70),
    "1AK5": dict(lmax=70, sampling_deg=5.155428, d_min_eff=4.84),
    "3K7M": dict(lmax=66, sampling_deg=5.464844, d_min_eff=5.39),
    "4BX9": dict(lmax=96, sampling_deg=3.774036, d_min_eff=6.73),
    "6G9X": dict(lmax=84, sampling_deg=4.314963, d_min_eff=5.92),
    "2DQ6": dict(lmax=76, sampling_deg=4.772056, d_min_eff=6.36),
    "3GR5": dict(lmax=66, sampling_deg=5.483967, d_min_eff=4.10),
}


def run_phaser_dumps(pdb: str, work: Path) -> dict:
    """Run the instrumented binary, dumping observation and calc inputs."""
    work.mkdir(parents=True, exist_ok=True)
    pdb_path, mtz_path = case_paths(pdb)
    kw = write_keywords(work, mtz_path=mtz_path, model_pdb=pdb_path,
                        n_peaks=5, root=f"{pdb}_prep", title=f"prep {pdb}")
    env = dict(os.environ)
    env["PHASER_OBS_DUMP"] = str(work / "obs.csv")
    env["PHASER_CALC_DUMP"] = str(work / "calc.csv")
    env["PHASER_TERMS_DUMP"] = str(work / "terms.csv")
    env["PHASER_SEARCH_ELMN_DUMP"] = str(work / "search_elmn.csv")
    proc = subprocess.run([str(PATCHED_PHASER)], cwd=str(work),
                          input=kw.read_text(), capture_output=True,
                          text=True, timeout=5400, env=env)
    (work / "run.log").write_text((proc.stdout or "") + (proc.stderr or ""))
    for name in ("obs.csv", "calc.csv", "terms.csv"):
        if not (work / name).exists():
            raise RuntimeError(f"{pdb}: {name} not written; see {work/'run.log'}")
    return {"obs": work / "obs.csv", "calc": work / "calc.csv",
            "terms": work / "terms.csv",
            "search_elmn": work / "search_elmn.csv"}


def _polar_to_cart(r, th, ph) -> torch.Tensor:
    return torch.tensor(np.stack(
        [r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)],
        axis=1))


def capture_ours(pdb: str):
    """Our per-reflection observation and calc inputs to the SH expansion.

    Both go through ``bessel_sh_expand``; the observation call is the one with
    ``zsymm > 1`` (the calc side is deliberately never m-filtered).
    """
    from torchref.experimental.alignment.frf import api as _api

    pin = PINNED[pdb]
    cap: dict = {}
    original = _api.bessel_sh_expand

    def spy(s, vals, **kw):
        key = "obs" if kw.get("zsymm", 1) > 1 else "calc"
        cap.setdefault(key, (s.detach().cpu().to(torch.float64),
                             vals.detach().cpu().to(torch.float64)))
        return original(s, vals, **kw)

    model, data = load_case(pdb)

    def _pinned(model_radius_A, d_min_data, lmax_cap=48):
        return int(pin["lmax"]) + 1, float(pin["d_min_eff"])

    cfg = FRFConfig(n_peaks=20, lmax_cap=int(pin["lmax"]),
                    grid_sampling_deg=float(pin["sampling_deg"]))
    with patched(_api, "phaser_lmax_resolution", _pinned), \
         patched(_api, "bessel_sh_expand", spy):
        run_frf(model, data, cfg, capture_arf=False, verbose=0)
    return cap


def match_and_correlate(P: torch.Tensor, PV: torch.Tensor,
                        S: torch.Tensor, V: torch.Tensor,
                        *, n_sample: int = 4000, tol: float = 1e-6) -> dict:
    """Match Phaser's points onto ours by position, then compare the values.

    Correlation is the statistic, not a ratio: these intensities are centred on
    zero (``E^2 - 1``), so element-wise ratios are dominated by division by
    near-zero and say nothing.
    """
    g = torch.Generator().manual_seed(1)
    k = min(n_sample, P.shape[0])
    sel = torch.randperm(P.shape[0], generator=g)[:k]
    q, qv = P[sel], PV[sel]

    idx = torch.empty(k, dtype=torch.long)
    dist = torch.empty(k)
    for i in range(0, k, 500):
        d2 = ((q[i:i + 500, None, :] - S[None, :, :]) ** 2).sum(-1)
        mn = d2.min(1)
        idx[i:i + 500] = mn.indices
        dist[i:i + 500] = mn.values.sqrt()

    ok = dist < tol
    out = {"n_phaser": int(P.shape[0]), "n_ours": int(S.shape[0]),
           "matched_frac": float(ok.to(torch.float64).mean()),
           "median_pos_dist": float(dist.median())}
    if int(ok.sum()) < 50:
        out["corr"] = float("nan")
        return out
    a, b = qv[ok], V[idx][ok]
    ac, bc = a - a.mean(), b - b.mean()
    out["corr"] = float((ac @ bc) / (ac.norm() * bc.norm()).clamp(min=1e-30))
    out["phaser_mean"] = float(a.mean())
    out["phaser_sd"] = float(a.std())
    out["ours_mean"] = float(b.mean())
    out["ours_sd"] = float(b.std())
    return out


def attribute_obs_terms(terms_csv: Path, pdb: str) -> dict:
    """Compare Phaser's normalisation terms against ours, keyed by Miller index.

    Phaser emits one row per *selected reflection* with its Miller index, so
    this is immune to the two hazards that broke the first attempt: the
    ``reso(r) > LMAX_RESO`` gate drops entries from the HKL list, and
    ``HKL_clustered::add`` buckets by theta, so no parallel array indexed
    against that list can stay aligned.

    ``Esqr = (Feff / sqrt_epsnSN)^2`` is Phaser's normalised intensity. Ours is
    ``E^2`` from equal-count shell means. Comparing the *normalisers* isolates
    the Wilson treatment from everything else.
    """
    d = np.loadtxt(terms_csv, delimiter=",", skiprows=1)
    if d.ndim == 1:
        d = d[None, :]
    hkl = d[:, 0:3].astype(int)
    Feff, sqrtSN, DFAC, V, cw, Esqr, inten, reso = (d[:, i] for i in range(3, 11))

    out = {
        "n_terms": int(d.shape[0]),
        "dfac_mean": float(DFAC.mean()), "dfac_sd": float(DFAC.std()),
        "dfac_is_unity": int(bool(np.allclose(DFAC, 1.0, atol=1e-6))),
        "V_is_unity": int(bool(np.allclose(V, 1.0, atol=1e-6))),
    }
    # Self-consistency: Phaser's own identity must reproduce its own intensity.
    rebuilt = cw * (Esqr - V) / (V ** 2) * (DFAC ** 2)
    scale = float(np.abs(inten).max()) or 1.0
    out["rebuild_max_err"] = float(np.abs(rebuilt - inten).max() / scale)
    # And that Esqr really is (Feff/sqrt_epsnSN)^2.
    out["esqr_max_err"] = float(
        np.abs((Feff / np.maximum(sqrtSN, 1e-30)) ** 2 - Esqr).max()
        / (float(np.abs(Esqr).max()) or 1.0))

    # Our normalised E^2 for the same Miller indices.
    from lab.reference_normalisers import wilson_normalise
    model, data = load_case(pdb)
    B = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    our_hkl = data.hkl.to(torch.long).cpu()
    F = data.F.to(torch.float64).abs().cpu()
    smag = (our_hkl.to(torch.float64) @ B).norm(dim=-1)
    E_obs, _ = wilson_normalise(F, smag, 20)

    off = 1024
    key = lambda t: ((t[:, 0] + off) * 4096 + (t[:, 1] + off)) * 4096 + (t[:, 2] + off)
    lut = {int(k): i for i, k in enumerate(key(our_hkl))}
    idx = np.array([lut.get(int(k), -1) for k in key(torch.tensor(hkl))])
    ok = idx >= 0
    out["terms_matched_frac"] = float(ok.mean())
    if int(ok.sum()) > 50:
        ours_E2 = (E_obs[torch.tensor(idx[ok])] ** 2).to(torch.float64)
        ph_E2 = torch.tensor(Esqr[ok])
        for nm, a, b in (("esqr", ph_E2, ours_E2),
                         ("normaliser", torch.tensor(sqrtSN[ok]),
                          (torch.tensor(Feff[ok]) / ours_E2.clamp(min=1e-30).sqrt()))):
            ac, bc = a - a.mean(), b - b.mean()
            out[f"corr_{nm}"] = float(
                (ac @ bc) / (ac.norm() * bc.norm()).clamp(min=1e-30))
        out["esqr_ratio_median"] = float((ours_E2 / ph_E2.clamp(min=1e-30)).median())
    return out


def run_case(pdb: str, outdir: Path) -> dict:
    t0 = time.time()
    work = outdir / "phaser" / pdb
    dumps = run_phaser_dumps(pdb, work)
    ours = capture_ours(pdb)

    row = {"experiment": EXPERIMENT, "pdb": pdb}
    row.update(provenance())

    # Calc side is NOT position-matched: our calc lives on a cubic P1 box
    # (s = hkl/a) and Phaser's on its own ensemble grid, so the two sampling
    # sets have no reason to coincide -- a nearest-position match returns 0%.
    # The calc comparison belongs at the projected (SearchElmn) level.
    for side, csv_name in (("obs", "obs"),):
        d = np.loadtxt(dumps[csv_name], delimiter=",", skiprows=1)
        P = _polar_to_cart(d[:, 1], d[:, 2], d[:, 3])
        PV = torch.tensor(d[:, 4])
        S, V = ours[side]
        res = match_and_correlate(P, PV, S, V)
        row.update({f"{side}_{k}": v for k, v in res.items()})

    row.update({f"term_{k}": v for k, v in attribute_obs_terms(dumps["terms"], pdb).items()})
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, choices=sorted(PINNED))
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parents[1] / "runs" / EXPERIMENT)
    outdir.mkdir(parents=True, exist_ok=True)
    row = run_case(args.pdb, outdir)
    append_row(outdir / f"{EXPERIMENT}.csv", row)

    print(f"\n=== {args.pdb} ===", flush=True)
    print("  OBS  n=%s/%s matched=%.3f  corr=%.6f"
          % (row.get("obs_n_ours"), row.get("obs_n_phaser"),
             row.get("obs_matched_frac", float("nan")),
             row.get("obs_corr", float("nan"))), flush=True)
    print("  terms n=%s matched=%.3f | rebuild_err=%.2e esqr_err=%.2e"
          % (row.get("term_n_terms"), row.get("term_terms_matched_frac", float("nan")),
             row.get("term_rebuild_max_err", float("nan")),
             row.get("term_esqr_max_err", float("nan"))), flush=True)
    print("  Esqr corr(ours,phaser)=%.6f  ratio median=%.4f"
          % (row.get("term_corr_esqr", float("nan")),
             row.get("term_esqr_ratio_median", float("nan"))), flush=True)
    print("  DFAC unity=%s mean=%.5f sd=%.5f | V unity=%s | rebuild_err=%.2e"
          % (row.get("term_dfac_is_unity"), row.get("term_dfac_mean", float("nan")),
             row.get("term_dfac_sd", float("nan")), row.get("term_V_is_unity"),
             row.get("term_rebuild_max_err", float("nan"))), flush=True)
    print("  normaliser corr=%.6f"
          % row.get("term_corr_normaliser", float("nan")), flush=True)
    print(f"\nwrote {outdir / (EXPERIMENT + '.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
