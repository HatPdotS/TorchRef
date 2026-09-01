"""Run our FRF on Phaser's observation intensities and see where truth lands.

This is the closing experiment of the bisection. Everything downstream of the
per-reflection intensity is now verified exact against Phaser:

* the SH-Bessel projection -- feeding Phaser's own prepared observations and its
  own molecular-transform samples through ``bessel_sh_expand`` reproduces its
  ``DataElmn`` and ``SearchElmn`` at correlation 1.0000, scale 1.0000 at 0
  degrees, residual 0.0000, once Phaser's 1e-3 cos-theta bucketing is replayed
  (job 489517);
* the Wigner contraction, per-beta FFT, interpolation and adaptive sample list
  -- Phaser's ``clmn`` through our evaluator gives r = 0.998 with an identical
  argmax;
* the reciprocal frame (positions agree to 1e-8) and the unroll (our
  the orbit dedup reproduces Phaser's point count exactly).

What is NOT verified is the intensity attached to each position: ours correlates
with Phaser's at 0.988 (1AK5), 0.877 (2DQ6) and 0.711 (3GR5) -- and that ordering
is the performance ordering. So substituting Phaser's intensities into our
otherwise-unchanged pipeline is a decisive test rather than another correlation:
if truth reaches rank 0 on 3GR5, the remaining deficit is entirely in the
intensity computation and nothing else is left to look for. If it does not, there
is a defect outside everything measured so far.

Alignment is by Miller index, not by position. Phaser dumps one row per selected
ASU reflection (``PHASER_TERMS_DUMP``); expanding each over the orbit ``h.W``
gives the P1 index of every point that reflection contributes, and the Friedel
mate carries the same intensity. Reflections our engine keeps but Phaser did not
select are reported rather than silently dropped.

Usage
-----
    python -m diagnostics.frf_inject_phaser_obs --pdb 3GR5 \
        --dumps ../runs/encode_compare_489514/phaser/3GR5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import FRFConfig, load_case, patched, run_frf  # noqa: E402
from lab.results import append_row, provenance  # noqa: E402
from diagnostics.frf_ghost_knockout import (  # noqa: E402
    PHASER_PINNED, _orbit_of_identity, _truth_and_margin,
)

EXPERIMENT = "frf_inject_phaser_obs"

#: Packing base for (h,k,l) -> one int64 key. Miller indices here stay well
#: inside +-1000 at these resolutions, and the base is checked at build time.
_BASE = 2048


def _pack(hkl: torch.Tensor) -> torch.Tensor:
    if int(hkl.abs().max()) >= _BASE // 2:
        raise ValueError(f"Miller index {int(hkl.abs().max())} too large for base {_BASE}")
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    return ((h + _BASE // 2) * _BASE + (k + _BASE // 2)) * _BASE + (l + _BASE // 2)


def build_lut(terms_csv: Path, sym_mats: torch.Tensor):
    """Phaser's per-reflection intensity, keyed by every P1 index it feeds.

    Each ASU row is expanded over the orbit ``h.W`` (Phaser's ``rotMiller`` is
    ``rotsym[isym] * h`` with ``rotsym = W^T``, i.e. the row-vector convention)
    and over the Friedel mate, which carries the same intensity because
    ``|F(-h)| = |F(h)|`` and even-l-only projection is blind to the sign.
    """
    d = np.loadtxt(terms_csv, delimiter=",", skiprows=1)
    if d.ndim == 1:
        d = d[None, :]
    hkl = torch.from_numpy(d[:, 0:3]).to(torch.float64)
    inten = torch.from_numpy(d[:, 9]).to(torch.float64)

    orbits = torch.einsum("kji,nj->nki", sym_mats.to(torch.float64), hkl)
    orbits = orbits.round().to(torch.long).reshape(-1, 3)
    vals = inten.unsqueeze(1).expand(-1, sym_mats.shape[0]).reshape(-1)
    keys = torch.cat([_pack(orbits), _pack(-orbits)])
    vals = torch.cat([vals, vals])

    uniq, inverse = torch.unique(keys, return_inverse=True)
    lut = torch.zeros(uniq.numel(), dtype=torch.float64)
    lut[inverse] = vals
    # Two different ASU reflections mapping onto one P1 index would mean the
    # orbit is still wrong -- the signature of the convention bug that was fixed.
    hi = torch.full((uniq.numel(),), -1e300, dtype=torch.float64)
    lo = torch.full((uniq.numel(),), 1e300, dtype=torch.float64)
    hi.scatter_reduce_(0, inverse, vals, reduce="amax")
    lo.scatter_reduce_(0, inverse, vals, reduce="amin")
    n_conflict = int(((hi - lo).abs() > 1e-12 * hi.abs().clamp(min=1e-30)).sum())
    stats = {"n_asu_terms": int(hkl.shape[0]),
             "n_lut_keys": int(uniq.numel()),
             "n_lut_conflicts": n_conflict}
    return uniq, lut, stats


def run_arm(pdb: str, dumps: Path, arm: str, *, n_peaks: int = 500) -> dict:
    """One arm: ``baseline`` (our intensities) or ``phaser_intensity``."""
    from torchref.experimental.alignment.frf import api as _api

    pin = PHASER_PINNED[pdb]
    model, data = load_case(pdb)
    orbit = _orbit_of_identity(data)
    sg = data.spacegroup.matrices.to(torch.float64).cpu()
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    rec_inv = torch.linalg.inv(rec)

    report = {"n_intercepted": 0}
    original = _api.bessel_sh_expand

    if arm == "phaser_intensity":
        keys, lut, lut_stats = build_lut(dumps / "terms.csv", sg)

        def injected(s, intensity, **kw):
            if kw.get("zsymm", 1) <= 1:          # calc side: untouched
                return original(s, intensity, **kw)
            report["n_intercepted"] += 1
            hkl = (s.to(torch.float64).cpu() @ rec_inv).round().to(torch.long)
            k = _pack(hkl)
            pos = torch.searchsorted(keys, k)
            pos_c = pos.clamp(max=keys.numel() - 1)
            found = keys[pos_c] == k
            report["n_obs"] = int(s.shape[0])
            report["found_frac"] = float(found.to(torch.float64).mean())
            new = lut[pos_c].to(s.dtype).to(s.device)
            return original(s[found], new[found], **kw)

        ctx_name, ctx_val = "bessel_sh_expand", injected
    else:
        lut_stats = {}
        ctx_name, ctx_val = "bessel_sh_expand", original

    def _pinned(model_radius_A, d_min_data, lmax_cap=48):
        return int(pin["lmax"]) + 1, float(pin["d_min_eff"])

    cfg = FRFConfig(n_peaks=n_peaks, lmax_cap=int(pin["lmax"]),
                    extra={"grid_sampling_deg": float(pin["sampling_deg"])})
    t0 = time.time()
    with patched(_api, "phaser_lmax_resolution", _pinned), \
         patched(_api, ctx_name, ctx_val):
        res = run_frf(model, data, cfg, capture_arf=True, verbose=0)
    rank, sig, ang, ghost, margin = _truth_and_margin(res.arf, orbit)

    row = {"experiment": EXPERIMENT, "pdb": pdb, "arm": arm}
    row.update(provenance())
    row.update({
        "spacegroup": str(data.spacegroup.hm),
        "lmax": pin["lmax"], "sampling_deg": pin["sampling_deg"],
        "d_min_eff": pin["d_min_eff"],
        "n_samples": int(res.arf.values.numel()),
        "truth_rank": rank, "truth_sigma": round(sig, 4),
        "truth_angle_deg": round(ang, 3),
        "best_ghost_sigma": round(ghost, 4), "margin": round(margin, 4),
        "seconds": round(time.time() - t0, 1),
    })
    row.update(lut_stats)
    row.update(report)
    if arm == "phaser_intensity" and report["n_intercepted"] != 1:
        raise RuntimeError(
            f"{pdb}: intercepted {report['n_intercepted']} obs expansions, "
            f"expected exactly 1 -- the injection hook is on the wrong call")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True, choices=sorted(PHASER_PINNED))
    ap.add_argument("--dumps", required=True,
                    help="directory holding Phaser's terms.csv for this pdb")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--n-peaks", type=int, default=500)
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parents[1] / "runs" / EXPERIMENT)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{EXPERIMENT}_{args.pdb}.csv"

    print(f"{args.pdb}: truth rank with our vs Phaser's obs intensities",
          flush=True)
    print(f"{'arm':<18}{'rank':>8}{'truth_sig':>11}{'ghost_sig':>11}"
          f"{'margin':>9}{'found':>8}", flush=True)
    rows = []
    for arm in ("baseline", "phaser_intensity"):
        rows.append(run_arm(args.pdb, Path(args.dumps), arm,
                            n_peaks=args.n_peaks))
    cols = {}
    for r in rows:
        cols.update({k: "" for k in r})
    for r in rows:
        full = {**cols, **r}
        append_row(csv_path, full)
        ff = full.get("found_frac", "")
        ff = f"{float(ff):>8.4f}" if ff != "" else f"{'-':>8}"
        print(f"{r['arm']:<18}{r['truth_rank']:>8}{r['truth_sigma']:>11.2f}"
              f"{r['best_ghost_sigma']:>11.2f}{r['margin']:>+9.2f}{ff}",
              flush=True)
    print(f"\nwrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
