"""Feed Phaser's own prepared data into our SH-Bessel encoder.

Every earlier comparison changed two things at once: the *inputs* to the
expansion (normalisation, symmetry unroll, F_calc) and the *expansion itself*.
The per-reflection attribution (job 489442) pinned the input side -- Phaser's
own identities reproduce bit-exactly, DFAC and V are unity, and the residual
disagreement is a roughly uniform ~7-11% in the Wilson normaliser across all
structures, so it does not single out the trigonal/hexagonal failures. That
leaves the encoder untested on its own.

This runs the encoder with Phaser's inputs, so a mismatch can only come from our
projection:

* ``obs_phaser_pts`` -- Phaser's prepared observations (``PHASER_OBS_DUMP``:
  post-normalisation, post-LERF1, post-unroll, post-axis-permutation, in polar
  coordinates, i.e. exactly what ``DataMR::getELMNxR2`` consumes) through
  ``bessel_sh_expand``, against Phaser's ``DataElmn``.
* ``calc_phaser_pts`` -- Phaser's molecular-transform samples
  (``PHASER_CALC_DUMP``, from ``Ensemble::getELMNxR2`` -- a *different* function
  with its own radial scale and its own l != 0 doubling) against Phaser's
  ``SearchElmn``.
* ``obs_phaser_clustered`` / ``calc_phaser_clustered`` -- the same two, but
  replaying Phaser's own angular approximation. Phaser buckets reflections by
  ``|cos(theta) - cos(theta_rep)| < 1e-3`` and evaluates the Legendre functions
  once per bucket from the first member's theta (sphericalY.h:43,
  DataMR.cc:1096); we cluster only on values equal to ~1e-7. So a high-l
  disagreement in the arms above is *expected*, and is Phaser being
  approximate rather than us being wrong. These arms separate the two, and
  answer a question that has never been asked: whether that 1e-3 polar
  smoothing is part of why Phaser is immune to the symmetry-axis ghosts.
* ``obs_ours_unroll`` / ``obs_dedup_unroll`` -- Phaser's ASU-level intensities
  (``PHASER_TERMS_DUMP``, keyed by Miller index) put through *our* two symmetry
  unrolls: the production one, which emits all ``n_ops`` orbit positions, and
  ``SpaceGroup.expand_hkl(include_friedel=False)``, which emits only the
  distinct ones as Phaser does (``!duplicate(isym,rhkl)``, DataMR.cc:954).
  Same intensities, same encoder,
  same target -- so the difference between these two arms is the multiplicity
  handling and nothing else.

Two scalars the expansion needs are not recoverable from the dumped rows -- the
observation-side ``HIRES`` is the *minimum* reso over selected reflections,
one step below the smallest that survives the ``reso(r) > HIRES`` gate -- so the
instrumented binary now writes them to ``<dump>.meta`` and they are read, not
inferred.

Expected relation, if our encoder is right. Phaser projects with ``Y_lm``
(``e^{+im phi}``, Condon-Shortley sign folded into its ``Pmm`` recurrence) while
we project with ``conj(C(m,phi))``; ``bar_P`` carries no CS phase and our
``sign_m`` restores it. Both are real-weighted sums, so

    ours[n, l, m] = k * conj(phaser[l, m, n+1]),   k = 1 (was 2 while the
expansion concatenated the antipodal copy, which Phaser does via cctbx's
conjugate_flag and we no longer do -- see bessel_sh_expand)

with the factor 2 because appending ``-s`` doubles every even-l coefficient
exactly (``Y_lm(-s) = (-1)^l Y_lm(s)``). ``k`` is therefore a prediction, not a
fitted nuisance: a modulus away from 1 or a phase away from 0 is a finding.

Usage
-----
    python -m diagnostics.frf_encode_compare --pdb 2DQ6
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import case_paths, load_case  # noqa: E402
from lab.phaser_match import PATCHED_PHASER, write_keywords  # noqa: E402
from lab.results import append_row, provenance  # noqa: E402

EXPERIMENT = "frf_encode_compare"


# ---------------------------------------------------------------------------
# Phaser side
# ---------------------------------------------------------------------------

def run_phaser_dumps(pdb: str, work: Path) -> dict:
    """One instrumented run producing every stage this comparison needs."""
    work.mkdir(parents=True, exist_ok=True)
    pdb_path, mtz_path = case_paths(pdb)
    kw = write_keywords(work, mtz_path=mtz_path, model_pdb=pdb_path,
                        n_peaks=5, root=f"{pdb}_enc", title=f"encode {pdb}")
    paths = {
        "PHASER_OBS_DUMP": work / "obs.csv",
        "PHASER_CALC_DUMP": work / "calc.csv",
        "PHASER_TERMS_DUMP": work / "terms.csv",
        "PHASER_DATA_ELMN_DUMP": work / "data_elmn.csv",
        "PHASER_SEARCH_ELMN_DUMP": work / "search_elmn.csv",
    }
    env = dict(os.environ)
    for k, v in paths.items():
        env[k] = str(v)
    proc = subprocess.run([str(PATCHED_PHASER)], cwd=str(work),
                          input=kw.read_text(), capture_output=True,
                          text=True, timeout=5400, env=env)
    (work / "run.log").write_text((proc.stdout or "") + (proc.stderr or ""))
    missing = [v.name for v in paths.values() if not v.exists()]
    if missing:
        raise RuntimeError(f"{pdb}: missing dumps {missing}; see {work/'run.log'}")
    # "PHASER_OBS_DUMP" -> "obs": strip both the prefix and the _DUMP suffix.
    return {k[len("PHASER_"):-len("_DUMP")].lower(): v for k, v in paths.items()}


def read_meta(path: Path) -> dict:
    """``<dump>.meta`` -- the scalars the point list cannot carry."""
    meta = Path(str(path) + ".meta")
    if not meta.exists():
        raise RuntimeError(
            f"{meta} missing: rebuild the instrumented binary "
            f"(phaser_src/build/rebuild.sh) -- the Bessel scale would otherwise "
            f"have to be guessed."
        )
    out = {}
    for line in meta.read_text().splitlines()[1:]:
        k, v = line.split(",")
        out[k] = float(v)
    return out


def _cart(r, th, ph) -> torch.Tensor:
    return torch.from_numpy(np.stack(
        [r * np.sin(th) * np.cos(ph),
         r * np.sin(th) * np.sin(ph),
         r * np.cos(th)], axis=1)).to(torch.float64)


def load_points(path: Path):
    """``cluster,r,theta,phi,intensity`` -> our encoder's inputs.

    Returns ``(s_exact, intensity, cos_theta, s_clustered, cluster_stats)``.

    ``s_clustered`` replays Phaser's OWN angular approximation.
    ``HKL_clustered::add`` (sphericalY.h:43) buckets reflections greedily by
    ``|cos(theta) - cos(theta_rep)| < 1e-3`` against the FIRST member of each
    bucket, and the projection then evaluates the associated Legendre functions
    once per bucket from that first member's theta (DataMR.cc:1096) -- while the
    radial Bessel term stays per-reflection. So Phaser's Y_lm carries up to
    1e-3 of cos-theta error, which at l ~ 70 is a percent-level per-coefficient
    error, largest near the poles where sin(theta) is small.

    Our encoder clusters only on values that are equal to ~1e-7, so it is the
    more accurate of the two. That means a high-l disagreement is expected and
    is Phaser's approximation, not our defect -- and it has to be separated
    from a real difference before any residual can be read. Substituting each
    point's bucket-representative ``cos(theta)`` while keeping its own ``r`` and
    ``phi`` reproduces Phaser's evaluation exactly, because ``r`` and ``phi``
    are the only per-reflection quantities Phaser keeps.

    The dump is written before ``HKL_list.shuffle()``, so row order within a
    cluster is insertion order and row 0 of each cluster is the representative.
    """
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    if d.ndim == 1:
        d = d[None, :]
    cid = d[:, 0].astype(np.int64)
    r, th, ph, val = d[:, 1], d[:, 2], d[:, 3], d[:, 4]

    first = np.zeros(cid.max() + 1, dtype=np.int64)
    seen = np.zeros(cid.max() + 1, dtype=bool)
    for i, c in enumerate(cid):
        if not seen[c]:
            seen[c], first[c] = True, i
    th_rep = th[first[cid]]
    stats = {
        "phaser_n_clusters": int(seen.sum()),
        "phaser_cos_spread_max": float(np.abs(np.cos(th) - np.cos(th_rep)).max()),
        "phaser_cluster_size_max": int(np.bincount(cid).max()),
    }
    return (_cart(r, th, ph),
            torch.from_numpy(val).to(torch.float64),
            torch.from_numpy(np.cos(th)).to(torch.float64),
            _cart(r, th_rep, ph),
            stats)


def load_elmn(path: Path, L: int) -> torch.Tensor:
    """Phaser's ``l,m,n`` dump into our ``(N_radial, L, 2L-1)`` layout.

    Phaser's ``n`` is 1-based against our 0-based, and both index the same
    ``u = l + 2n - 1`` radial order, so ``n0 = n - 1``.
    """
    lmax = L - 1
    lmax_even = lmax if lmax % 2 == 0 else lmax - 1
    n_radial = (lmax_even - 2) // 2 + 1
    out = torch.zeros((n_radial, L, 2 * L - 1), dtype=torch.complex128)
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    if d.ndim == 1:
        d = d[None, :]
    l = d[:, 0].astype(int)
    m = d[:, 1].astype(int)
    n0 = d[:, 2].astype(int) - 1
    keep = (l <= lmax_even) & (n0 >= 0) & (n0 < n_radial) & (np.abs(m) <= lmax_even)
    if not keep.all():
        raise RuntimeError(f"{path}: {int((~keep).sum())} rows outside the L={L} band")
    out[n0, l, m + (L - 1)] = torch.from_numpy(d[:, 3] + 1j * d[:, 4])
    return out


def band_mask(L: int, device="cpu") -> torch.Tensor:
    """The (n, l, m) entries Phaser allocates: l even, |m| <= l, n < nmax(l)."""
    lmax = L - 1
    lmax_even = lmax if lmax % 2 == 0 else lmax - 1
    n_radial = (lmax_even - 2) // 2 + 1
    mask = torch.zeros((n_radial, L, 2 * L - 1), dtype=torch.bool, device=device)
    for l in range(2, lmax_even + 1, 2):
        n_l = (lmax_even - l) // 2 + 1
        m_lo, m_hi = (L - 1) - l, (L - 1) + l
        mask[:n_l, l, m_lo:m_hi + 1] = True
    return mask


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def compare_coeffs(ours: torch.Tensor, phaser: torch.Tensor, L: int,
                   *, k_expected: float) -> dict:
    """Our coefficients against ``conj(phaser)`` over Phaser's allocated band.

    Reported quantities:
      ``corr``        modulus of the complex correlation -- shape agreement.
      ``k_mod``/``k_arg_deg``  the fitted complex scale; the prediction is
                      ``k_expected`` at 0 degrees, so a phase here means a
                      convention mismatch, not a scale.
      ``rel_resid``   ``||a - k b|| / ||a||`` after the fitted scale, i.e. what
                      the correlation hides.
      ``pow_offband`` fraction of OUR power sitting where Phaser has exactly
                      zero -- the m-filter / forbidden-m channel.
      ``worst_l``     the even l with the lowest per-l correlation.
    """
    mask = band_mask(L, device=ours.device)
    b = torch.conj(phaser.to(ours.device))
    a = ours
    av, bv = a[mask], b[mask]

    num = torch.vdot(bv, av)                       # sum conj(b)*a
    denom_b = (bv.abs() ** 2).sum()
    out = {
        "n_band": int(mask.sum()),
        "n_phaser_nonzero": int((bv.abs() > 0).sum()),
        "n_ours_nonzero": int((av.abs() > 0).sum()),
    }
    if float(denom_b) == 0.0:
        out.update(corr=float("nan"), k_mod=float("nan"),
                   k_arg_deg=float("nan"), rel_resid=float("nan"))
        return out
    corr = float(num.abs() / (av.norm() * bv.norm()).clamp(min=1e-300))
    k = num / denom_b
    resid = (av - k * bv).norm() / av.norm().clamp(min=1e-300)
    out.update({
        "corr": corr,
        "k_mod": float(k.abs()),
        "k_arg_deg": float(torch.rad2deg(torch.angle(k))),
        "k_expected": float(k_expected),
        "rel_resid": float(resid),
    })
    # Power we place where Phaser has none (inside its own band).
    zero_b = mask & (b.abs() == 0)
    out["pow_offband"] = float(
        (a[zero_b].abs() ** 2).sum() / (a[mask].abs() ** 2).sum().clamp(min=1e-300)
    )
    # Per-l correlation, to see whether a mismatch is radial (high l) or global.
    lmax_even = (L - 1) if (L - 1) % 2 == 0 else (L - 2)
    worst_l, worst_c = -1, 2.0
    per_l = []
    for l in range(2, lmax_even + 1, 2):
        ml = mask[:, l, :]
        al, bl = a[:, l, :][ml], b[:, l, :][ml]
        if float(bl.abs().max()) == 0.0:
            continue
        cl = float(torch.vdot(bl, al).abs()
                   / (al.norm() * bl.norm()).clamp(min=1e-300))
        per_l.append((l, cl))
        if cl < worst_c:
            worst_c, worst_l = cl, l
    out["worst_l"] = worst_l
    out["worst_l_corr"] = worst_c
    out["corr_l2"] = per_l[0][1] if per_l else float("nan")
    out["corr_lmax"] = per_l[-1][1] if per_l else float("nan")
    return out


# ---------------------------------------------------------------------------
# our side
# ---------------------------------------------------------------------------

def encode(s: torch.Tensor, intensity: torch.Tensor, *, L: int,
           h_scale: float, zsymm: int) -> torch.Tensor:
    from torchref.experimental.alignment.frf.data_mr import bessel_sh_expand
    return bessel_sh_expand(
        s, intensity, L=L, bessel_h_scale=h_scale, zsymm=zsymm,
    ).coeffs


def unroll_arms(pdb: str, terms_csv: Path):
    """Phaser's ASU intensities through both of our symmetry unrolls.

    Returns ``(arms, stats)`` where ``arms`` maps name -> ``(s, intensity)``.
    The counts are exact integers, so the multiplicity question is answered by
    arithmetic before any encoding happens.
    """
    d = np.loadtxt(terms_csv, delimiter=",", skiprows=1)
    if d.ndim == 1:
        d = d[None, :]
    hkl = torch.from_numpy(d[:, 0:3]).to(torch.float64)
    inten = torch.from_numpy(d[:, 9]).to(torch.float64)

    _, data = load_case(pdb)
    sg = data.spacegroup.matrices.to(torch.float64).cpu()
    rec = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    n_ops = int(sg.shape[0])

    # Production: every orbit position, duplicates included (align.py:487).
    hkl_all = torch.einsum("kji,nj->kni", sg, hkl).reshape(-1, 3)
    s_all = hkl_all @ rec
    i_all = inten.unsqueeze(0).expand(n_ops, -1).reshape(-1).contiguous()

    # Phaser-faithful: distinct orbit positions only (DataMR.cc:954).
    hkl_ded, asu_idx, _ = data.spacegroup.expand_hkl(
        hkl.to(torch.long), include_friedel=False)
    s_ded = hkl_ded.to(torch.float64) @ rec
    i_ded = inten[asu_idx]

    stats = {
        "n_asu_terms": int(hkl.shape[0]),
        "n_ops": n_ops,
        "n_unroll_all": int(s_all.shape[0]),
        "n_unroll_dedup": int(s_ded.shape[0]),
        "dup_frac": float(1.0 - s_ded.shape[0] / max(1, s_all.shape[0])),
    }
    return {"obs_ours_unroll": (s_all, i_all),
            "obs_dedup_unroll": (s_ded, i_ded)}, stats


def frame_check(s_ours: torch.Tensor, s_phaser: torch.Tensor,
                *, n_sample: int = 2000) -> dict:
    """Do the two Cartesian reciprocal frames coincide?

    ``|s|`` is frame-independent but theta and phi are not, so an orthogonalisation
    convention difference would rotate every coefficient (mixing m) and make the
    coefficient comparison meaningless while leaving the radial part intact.
    Nearest-neighbour distance in Cartesian space tests position, not just radius.
    """
    g = torch.Generator().manual_seed(1)
    k = min(n_sample, int(s_phaser.shape[0]))
    q = s_phaser[torch.randperm(s_phaser.shape[0], generator=g)[:k]]
    dist = torch.empty(k, dtype=torch.float64)
    step = 100  # the (step, N, 3) broadcast is the memory bound, not the (step, N) d2
    for i in range(0, k, step):
        d2 = ((q[i:i + step, None, :] - s_ours[None, :, :]) ** 2).sum(-1)
        dist[i:i + step] = d2.min(1).values.clamp(min=0).sqrt()
    return {"frame_median_dist": float(dist.median()),
            "frame_max_dist": float(dist.max()),
            "frame_matched_frac": float((dist < 1e-9).to(torch.float64).mean())}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(pdb: str, outdir: Path, *, reuse: Path | None = None) -> list:
    work = reuse if reuse is not None else (outdir / "phaser" / pdb)
    if reuse is not None:
        dumps = {n: work / f"{n}.csv" for n in
                 ("obs", "calc", "terms", "data_elmn", "search_elmn")}
        missing = [str(p) for p in dumps.values() if not p.exists()]
        if missing:
            raise RuntimeError(f"--reuse given but missing: {missing}")
    else:
        t0 = time.time()
        dumps = run_phaser_dumps(pdb, work)
        print(f"  phaser dumps in {time.time()-t0:.0f}s", flush=True)

    obs_meta = read_meta(dumps["obs"])
    calc_meta = read_meta(dumps["calc"])
    L = int(obs_meta["lmax"]) + 1
    zsymm = int(obs_meta["zsymm"])
    h_obs = obs_meta["lmax"] * obs_meta["hires"]
    h_calc = calc_meta["lmax"] * calc_meta["max_resolution"]
    print(f"  L={L} zsymm={zsymm} axis={int(obs_meta['axis'])} "
          f"hires={obs_meta['hires']:.6f} h_obs={h_obs:.4f} "
          f"h_calc={h_calc:.4f} (calc max_reso={calc_meta['max_resolution']:.6f})",
          flush=True)
    if int(obs_meta["axis"]) != 3:
        print("  NOTE: high-order axis is not c -- Phaser permutes the frame "
              "(DataMR.cc:984) and we do not; the obs arm is confounded.",
              flush=True)

    data_elmn = load_elmn(dumps["data_elmn"], L)
    search_elmn = load_elmn(dumps["search_elmn"], L)
    s_obs, i_obs, _, s_obs_clu, obs_clu = load_points(dumps["obs"])
    s_calc, i_calc, cos_calc, s_calc_clu, calc_clu = load_points(dumps["calc"])
    calc_clu = {f"calc_{k}": v for k, v in calc_clu.items()}
    print(f"  phaser obs clusters={obs_clu['phaser_n_clusters']} "
          f"(max cos-theta spread {obs_clu['phaser_cos_spread_max']:.2e}, "
          f"largest bucket {obs_clu['phaser_cluster_size_max']}); calc clusters="
          f"{calc_clu['calc_phaser_n_clusters']} "
          f"(spread {calc_clu['calc_phaser_cos_spread_max']:.2e})", flush=True)

    base = {"experiment": EXPERIMENT, "pdb": pdb}
    base.update(provenance())
    base.update({"L": L, "zsymm": zsymm, "axis": int(obs_meta["axis"]),
                 "hires": obs_meta["hires"], "h_obs": h_obs, "h_calc": h_calc})

    rows = []

    def emit(arm, coeffs, target, *, n_points, seconds, extra=None):
        r = dict(base, arm=arm, n_points=int(n_points),
                 seconds=round(seconds, 1))
        r.update(compare_coeffs(coeffs, target, L, k_expected=1.0))
        if extra:
            r.update(extra)
        rows.append(r)

    # --- arm 1: Phaser's own observations through our encoder ---------------
    t0 = time.time()
    c = encode(s_obs, i_obs, L=L, h_scale=h_obs, zsymm=zsymm)
    emit("obs_phaser_pts", c, data_elmn,
         n_points=s_obs.shape[0], seconds=time.time() - t0, extra=obs_clu)

    # --- arm 2: Phaser's observations WITH Phaser's own theta approximation --
    t0 = time.time()
    c = encode(s_obs_clu, i_obs, L=L, h_scale=h_obs, zsymm=zsymm)
    emit("obs_phaser_clustered", c, data_elmn,
         n_points=s_obs_clu.shape[0], seconds=time.time() - t0, extra=obs_clu)

    # --- arm 3: Phaser's calc samples through our encoder -------------------
    # Phaser doubles every l != 0 grid point (Ensemble.cc: `flipped`), because
    # its molecular-transform grid stores only the l >= 0 hemisphere. l == 0 is
    # exactly the s_z == 0 plane for these settings (c* along z), so the flag is
    # recoverable from the dumped theta.
    t0 = time.time()
    flip = (cos_calc.abs() > 1e-12).to(torch.float64) + 1.0
    c = encode(s_calc, i_calc * flip, L=L, h_scale=h_calc, zsymm=1)
    emit("calc_phaser_pts", c, search_elmn, n_points=s_calc.shape[0],
         seconds=time.time() - t0,
         extra=dict(calc_clu, n_l0_plane=int((cos_calc.abs() <= 1e-12).sum())))

    # --- arm 4: the same, with Phaser's theta approximation -----------------
    t0 = time.time()
    c = encode(s_calc_clu, i_calc * flip, L=L, h_scale=h_calc, zsymm=1)
    emit("calc_phaser_clustered", c, search_elmn, n_points=s_calc_clu.shape[0],
         seconds=time.time() - t0, extra=calc_clu)

    # --- arms 5/6: Phaser's ASU intensities through OUR unrolls -------------
    arms, ustats = unroll_arms(pdb, dumps["terms"])
    print(f"  unroll: asu={ustats['n_asu_terms']} x n_ops={ustats['n_ops']} "
          f"= {ustats['n_unroll_all']} all / {ustats['n_unroll_dedup']} dedup "
          f"(phaser obs rows {int(s_obs.shape[0])}; "
          f"dup_frac {ustats['dup_frac']:.4f})", flush=True)
    for arm, (s, val) in arms.items():
        t0 = time.time()
        fc = frame_check(s, s_obs)
        c = encode(s, val, L=L, h_scale=h_obs, zsymm=zsymm)
        emit(arm, c, data_elmn, n_points=s.shape[0], seconds=time.time() - t0,
             extra=dict(ustats, **fc))

    # One schema for every row -- csv.DictWriter fixes fieldnames from the
    # first row it sees, so a later row carrying extra keys would raise.
    cols = {}
    for r in rows:
        cols.update({k: "" for k in r})
    return [{**cols, **r} for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--reuse", default=None,
                    help="directory of existing dumps (skips the Phaser run)")
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parents[1] / "runs" / EXPERIMENT)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{EXPERIMENT}_{args.pdb}.csv"

    rows = run(args.pdb, outdir,
               reuse=Path(args.reuse) if args.reuse else None)
    hdr = (f"{'arm':<20}{'n_pts':>9}{'corr':>9}{'k_mod':>9}{'k_arg':>9}"
           f"{'resid':>9}{'offband':>9}{'worst_l':>9}{'wl_corr':>9}")
    print(f"\n{args.pdb}: our encoder against Phaser's coefficients", flush=True)
    print(hdr, flush=True)
    for r in rows:
        append_row(csv_path, r)
        print(f"{r['arm']:<20}{r['n_points']:>9}{r['corr']:>9.4f}"
              f"{r['k_mod']:>9.4f}{r['k_arg_deg']:>9.2f}{r['rel_resid']:>9.4f}"
              f"{r['pow_offband']:>9.4f}{r['worst_l']:>9}"
              f"{r['worst_l_corr']:>9.4f}", flush=True)
    print(f"\nwrote {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
