"""Compare our FRF array against Phaser's, element-wise, on one shared grid.

Both engines evaluate a rotation function on Phaser's adaptive SO(3) sample
list, so with the sampling matched the two arrays are directly subtractable --
index for index, no interpolation. That makes this a much sharper instrument
than comparing peak lists: a peak-list comparison only sees where the maxima
landed, whereas this sees the whole surface, including how much of the
disagreement is a smooth scale/offset (harmless -- peak *order* is invariant to
an affine map) versus genuine reshaping (not harmless).

What is pinned, and what is not
-------------------------------
Three quantities are read from Phaser's own VERBOSE log and forced onto our
engine: the bandwidth ``lmax``, the resolution the expansion runs at, and the
SO(3) sampling step. They are pinned from the log rather than re-derived,
because they all descend from ``mean_radius()`` and our reimplementation of that
is ~4% off (see :func:`lab.phaser_match.phaser_mean_radius`).

Everything on the observation and calc side keeps our production defaults --
anisotropy correction, symmetry unroll, French-Wilson, shell-variance weights,
Wilson-B match, Oeffner vrms, bulk solvent, dense P1-box calc. That is
deliberate: with the coupled trio pinned, whatever disagreement remains is
attributable to that preprocessing stack, which is what we want localised.

Usage
-----
    python -m diagnostics.frf_map_compare --pdb 1DAW
    python -m diagnostics.frf_map_compare --worklist-index 3
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import BENCH_PDBS, FRFConfig, case_paths, load_case, patched, run_frf  # noqa: E402
from lab.phaser_match import (  # noqa: E402
    PATCHED_PHASER,
    phaser_mean_radius_from_sampling,
    phaser_sampling_from_dump,
    load_phaser_frame,
    load_phaser_map,
    parse_phaser_log,
    phaser_frf_params,
    phaser_mean_radius,
    run_patched_phaser,
    write_keywords,
)
from lab.results import append_row, provenance  # noqa: E402

EXPERIMENT = "frf_map_compare"


def run_phaser_side(pdb: str, work: Path, *, n_peaks: int = 20) -> dict:
    """Run the patched binary and return its parameters plus its map.

    Returns a dict with ``angles`` (N,3 degrees, our sign convention),
    ``values`` (N,), and the parsed log fields.

    Raises
    ------
    RuntimeError
        If the rotation search was short-circuited by the R-factor check, or the
        dump is missing -- both of which Phaser reports as ``EXIT STATUS:
        SUCCESS``, so they must be checked explicitly.
    """
    pdb_path, mtz_path = case_paths(pdb)
    dump = work / f"{pdb}_phaser_map.csv"
    kw = write_keywords(
        work, mtz_path=mtz_path, model_pdb=pdb_path, n_peaks=n_peaks,
        root=f"{pdb}_frf", title=f"FRF map dump {pdb}",
    )
    rc, seconds, log_path = run_patched_phaser(work, kw, dump_path=dump)
    info = parse_phaser_log(log_path)

    if info.get("rotation_search_skipped"):
        raise RuntimeError(
            f"{pdb}: Phaser skipped the rotation search (R-factor short-circuit) "
            f"-- see {log_path}"
        )
    if not dump.exists():
        raise RuntimeError(
            f"{pdb}: no FRF dump written (rc={rc}); is {PATCHED_PHASER} the "
            f"instrumented binary? see {log_path}"
        )
    for key in ("lmax", "sampling_deg"):
        if key not in info:
            raise RuntimeError(f"{pdb}: could not parse {key} from {log_path}")

    angles, values = load_phaser_map(dump)
    # The logged sampling is rounded to 2 decimals and cannot rebuild the grid;
    # the dumped beta step is exact.
    info["sampling_deg_logged"] = info["sampling_deg"]
    info["sampling_deg"] = phaser_sampling_from_dump(angles)
    info.update(angles=angles, values=values, seconds=seconds, log=log_path)
    return info


def run_our_side(pdb: str, info: dict, *, n_peaks: int = 500):
    """Run our FRF with Phaser's bandwidth, resolution and sampling pinned.

    ``phaser_lmax_resolution`` is the single choke point through which the
    bandwidth and resolution reach both the SH expansion and the dense calc
    grid, so overriding it pins both consistently.
    """
    from torchref.experimental.alignment.frf import api as _api

    model, data = load_case(pdb)

    lmax = int(info["lmax"])
    # Resolution the expansion runs at: LMAX_RESO when Phaser's cap bound,
    # otherwise the selected high-resolution limit.
    if info.get("lmax_reso_A") is not None and not info.get("all_data_to_limit", False):
        d_min_eff = float(info["lmax_reso_A"])
    else:
        d_min_eff = float(info.get("selected_d_min", 0.0)) or None
    if d_min_eff is None:
        raise RuntimeError(f"{pdb}: cannot determine Phaser's expansion resolution")

    def _pinned(model_radius_A, d_min_data, lmax_cap=48):
        # Our bandwidth convention is L = lmax + 1.
        return lmax + 1, d_min_eff

    cfg = FRFConfig(
        n_peaks=n_peaks,
        lmax_cap=lmax,
        extra={"grid_sampling_deg": float(info["sampling_deg"])},
    )
    with patched(_api, "phaser_lmax_resolution", _pinned):
        result = run_frf(model, data, cfg, capture_arf=True, verbose=0)
    return model, data, result, d_min_eff


def _orbit_of_identity(data):
    """Rotations equivalent to the deposited orientation under the point group.

    The search model is used unrotated, so "truth" is the identity -- but only
    up to the crystal point group, and the operators must be taken to the
    Cartesian frame the rotation function works in (mixing Cartesian rotations
    with fractional operators is a metric error that inflates ghost counts).
    """
    from lab.truth import symmetry_orbit

    symops = data.spacegroup.matrices.to(torch.float64).cpu()
    recip = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
    I = torch.eye(3, dtype=torch.float64)
    return symmetry_orbit(I, symops, side="right", frame="cart",
                          reciprocal_basis=recip)


def _nearest(R_all: torch.Tensor, R_target: torch.Tensor):
    """Index of the sample rotation closest to ``R_target``, and the angle."""
    tr = torch.einsum("nij,ij->n", R_all, R_target)
    ang = torch.rad2deg(torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)))
    j = int(torch.argmin(ang))
    return j, float(ang[j])


def _truth_rank(values: torch.Tensor, R_all: torch.Tensor, orbit: torch.Tensor):
    """Rank of the best sample lying on the truth orbit.

    Returns ``(rank, sigma, angle_deg)``. The rank is the number of samples
    scoring strictly higher, i.e. 0 means the rotation function put truth first.
    """
    best = None
    for k in range(orbit.shape[0]):
        j, ang = _nearest(R_all, orbit[k])
        if best is None or values[j] > values[best[0]]:
            best = (j, ang)
    j, ang = best
    rank = int((values > values[j]).sum())
    sig = float((values[j] - values.mean()) / values.std().clamp(min=1e-30))
    return rank, sig, ang


def compare(ours, phaser_angles, phaser_values, frame, data, *, topn: int = 20) -> dict:
    """Compare two rotation functions in a common (PDB) frame.

    Element-wise comparison is impossible here and it is worth being explicit
    about why: Phaser samples SO(3) on a grid laid out in the search model's
    **principal frame**, we sample the identically-shaped grid in the PDB frame,
    and the two are related by a rotation (``PR``, and ``axisrot``). The grids
    therefore have the same pitch and the same point count but cover *different*
    rotations -- nearest-neighbour offsets run about half a grid step. On peaks
    only ~6-10 deg wide that annihilates any sample-wise correlation while
    leaving the peak structure intact, so a whole-map Pearson r measures nothing
    but the frame offset. Everything below is therefore computed on rotations,
    via nearest-neighbour lookup, not on indices.

    The headline numbers are ``truth_rank_ours`` and ``truth_rank_phaser``: if
    ours is much worse, the ghost problem is ours; if they agree, the ghosts are
    inherent to the target function and no reimplementation will remove them.
    """
    from torchref.base.alignment.rotation import rotation_matrix_euler_zyz

    a = ours.arf
    ov = a.values.to(torch.float64).cpu()
    R_ours = rotation_matrix_euler_zyz(torch.stack([
        a.alphas.to(torch.float64).cpu(),
        a.betas.to(torch.float64).cpu(),
        a.gammas.to(torch.float64).cpu(),
    ], dim=-1))

    pv = phaser_values.to(torch.float64).cpu()
    ang = phaser_angles.to(torch.float64).cpu()
    R_grid = rotation_matrix_euler_zyz(torch.deg2rad(ang[:, :3]))
    PR, AX = frame["PR"], frame["axisrot"]
    # principal frame -> PDB frame (runMR_FRF.cc:542)
    R_ph = torch.einsum("ij,njk,kl->nil", AX, R_grid, PR)

    osig = (ov - ov.mean()) / ov.std().clamp(min=1e-30)
    psig = (pv - pv.mean()) / pv.std().clamp(min=1e-30)
    out = {
        "n_ours": int(ov.numel()),
        "n_phaser": int(pv.numel()),
        "grid_same_size": int(ov.numel() == pv.numel()),
        "ours_max_sigma": float(osig.max()),
        "phaser_max_sigma": float(psig.max()),
    }
    # Phaser's own statistics, as a check that our sigma means what theirs does.
    if "stats" in frame:
        st = frame["stats"]
        out["phaser_max_sigma_logged"] = (
            (st["max"] - st["mean"]) / st["sigma"] if st["sigma"] else float("nan")
        )

    orbit = _orbit_of_identity(data)
    out["n_orbit"] = int(orbit.shape[0])
    r_o, s_o, a_o = _truth_rank(ov, R_ours, orbit)
    r_p, s_p, a_p = _truth_rank(pv, R_ph, orbit)
    out.update({
        "truth_rank_ours": r_o, "truth_sigma_ours": s_o, "truth_angle_ours": a_o,
        "truth_rank_phaser": r_p, "truth_sigma_phaser": s_p, "truth_angle_phaser": a_p,
        "truth_rank_delta": r_o - r_p,
    })

    # --- Ghost anatomy -----------------------------------------------------
    # A ghost is a peak that outranks truth. The question is not "how many" but
    # "what does the other engine see at exactly that rotation". Three outcomes,
    # each implying a different fix:
    #   * Phaser has a peak there too, but weaker  -> same physics, different
    #     weighting; find the term that suppresses it.
    #   * Phaser has nothing there                 -> we are manufacturing
    #     structure Phaser does not have.
    #   * Phaser has it just as strongly           -> Phaser has the ghost too
    #     and wins somewhere downstream, not in the rotation function.
    # `margin` is the discriminating power that matters: truth's sigma minus the
    # strongest ghost's. Negative means the rotation function prefers a ghost.
    tol = 1.5 * float(a.grid_sampling_deg)

    def _truth_mask(R_all):
        keep = torch.zeros(R_all.shape[0], dtype=torch.bool)
        for k in range(orbit.shape[0]):
            tr = torch.einsum("nij,ij->n", R_all, orbit[k])
            ang = torch.rad2deg(torch.arccos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0)))
            keep |= ang <= tol
        return keep

    m_o, m_p = _truth_mask(R_ours), _truth_mask(R_ph)
    out["n_truthlike_ours"] = int(m_o.sum())
    out["n_truthlike_phaser"] = int(m_p.sum())

    for tag, vals, sig, mask, Rs, other_v, other_R in (
        ("ours", ov, osig, m_o, R_ours, pv, R_ph),
        ("phaser", pv, psig, m_p, R_ph, ov, R_ours),
    ):
        ghost_sig = sig.masked_fill(mask, float("-inf"))
        gi = int(torch.argmax(ghost_sig))
        out[f"best_ghost_sigma_{tag}"] = float(sig[gi])
        out[f"margin_{tag}"] = float(
            out[f"truth_sigma_{tag}"] - float(sig[gi])
        )
        # the same rotation, looked up in the other engine's map
        j, dd = _nearest(other_R, Rs[gi])
        om, osd = other_v.mean(), other_v.std().clamp(min=1e-30)
        out[f"best_ghost_{tag}_seen_by_other_sigma"] = float((other_v[j] - om) / osd)
        out[f"best_ghost_{tag}_seen_by_other_rank"] = int((other_v > other_v[j]).sum())
        out[f"best_ghost_{tag}_lookup_angle"] = dd

    # Cross peak agreement: where do each engine's strongest peaks land in the
    # other's map?
    for label, (vs, Rs, vo, Ro) in {
        "ph_in_ours": (pv, R_ph, ov, R_ours),
        "ours_in_ph": (ov, R_ours, pv, R_grid if False else R_ph),
    }.items():
        if label == "ours_in_ph":
            # our rotations back into the principal frame for lookup
            Rs_use = torch.einsum("ij,njk,kl->nil", AX.T, R_ours, PR.T)
            vs_use, vo_use, Ro_use = ov, pv, R_grid
        else:
            Rs_use, vs_use, vo_use, Ro_use = R_ph, pv, ov, R_ours
        top = torch.topk(vs_use, min(topn, vs_use.numel())).indices
        ranks, sigs, angs = [], [], []
        vo_mean, vo_std = vo_use.mean(), vo_use.std().clamp(min=1e-30)
        for i in top.tolist():
            j, dd = _nearest(Ro_use, Rs_use[i])
            ranks.append(int((vo_use > vo_use[j]).sum()))
            sigs.append(float((vo_use[j] - vo_mean) / vo_std))
            angs.append(dd)
        ranks_t = torch.tensor(ranks, dtype=torch.float64)
        out[f"{label}_median_rank"] = float(ranks_t.median())
        out[f"{label}_median_sigma"] = float(torch.tensor(sigs).median())
        out[f"{label}_median_angle"] = float(torch.tensor(angs).median())
        out[f"{label}_frac_in_top{topn}"] = float((ranks_t < topn).to(torch.float64).mean())
        out[f"{label}_frac_above_5sig"] = float(
            (torch.tensor(sigs) > 5.0).to(torch.float64).mean()
        )
    return out


def run_case(pdb: str, outdir: Path, *, n_peaks: int = 500) -> dict:
    """One structure: Phaser map, our map, comparison row."""
    work = outdir / "phaser" / pdb
    t0 = time.time()
    info = run_phaser_side(pdb, work)
    frame = load_phaser_frame(work / f"{pdb}_phaser_map.csv")
    model, data, ours, d_min_eff = run_our_side(pdb, info, n_peaks=n_peaks)

    row = {"experiment": EXPERIMENT, "pdb": pdb}
    row.update(provenance())
    row.update({
        "phaser_lmax": info["lmax"],
        "phaser_sampling_deg": info["sampling_deg"],
        "phaser_sampling_deg_logged": info.get("sampling_deg_logged"),
        "phaser_lmax_reso_A": info.get("lmax_reso_A"),
        "phaser_all_data_to_limit": int(bool(info.get("all_data_to_limit"))),
        "phaser_selected_d_min": info.get("selected_d_min"),
        "phaser_selected_d_max": info.get("selected_d_max"),
        "phaser_selected_n_refl": info.get("selected_n_refl"),
        "phaser_n_samples_logged": info.get("n_samples"),
        "phaser_seconds": round(info["seconds"], 1),
        "pinned_d_min_eff_A": d_min_eff,
        "ours_seconds": round(ours.seconds, 1),
    })
    # Our radius formula vs Phaser's family, for the record.
    row["our_mean_radius_A"] = float(
        (model.xyz().to(torch.float64) - model.xyz().to(torch.float64).mean(0))
        .norm(dim=-1).mean().item()
    )
    row["phaser_style_mean_radius_A"] = phaser_mean_radius(model)
    pred = phaser_frf_params(
        row["phaser_style_mean_radius_A"],
        float(info.get("selected_d_min") or d_min_eff),
    )
    row["predicted_lmax"] = pred.lmax
    row["predicted_sampling_deg"] = pred.sampling_deg
    # Phaser's own radius, inverted from its exact sampling step.
    row["phaser_true_mean_radius_A"] = phaser_mean_radius_from_sampling(
        info["sampling_deg"], d_min_eff,
    )

    row["high_order_axis"] = frame.get("high_order_axis")
    row.update(compare(ours, info["angles"], info["values"], frame, data))
    row["total_seconds"] = round(time.time() - t0, 1)

    # Keep both arrays so a divergence-vs-beta plot needs no re-run.
    npz = outdir / f"{pdb}_maps.pt"
    torch.save(
        {
            "ours_values": ours.arf.values.cpu(),
            "ours_alphas": ours.arf.alphas.cpu(),
            "ours_betas": ours.arf.betas.cpu(),
            "ours_gammas": ours.arf.gammas.cpu(),
            "phaser_values": info["values"],
            "phaser_angles": info["angles"],
        },
        npz,
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", help="single structure")
    ap.add_argument("--worklist-index", type=int, help="index into BENCH_PDBS")
    ap.add_argument("--all", action="store_true", help="every benchmark structure")
    ap.add_argument("--n-peaks", type=int, default=500)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if args.pdb:
        todo = [args.pdb]
    elif args.worklist_index is not None:
        todo = [BENCH_PDBS[args.worklist_index]]
    elif args.all:
        todo = list(BENCH_PDBS)
    else:
        ap.error("give --pdb, --worklist-index or --all")

    outdir = Path(args.outdir) if args.outdir else (
        Path(__file__).resolve().parents[1] / "runs" / EXPERIMENT
    )
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"{EXPERIMENT}.csv"

    failures = 0
    for pdb in todo:
        try:
            row = run_case(pdb, outdir, n_peaks=args.n_peaks)
        except Exception as exc:  # keep the sweep going, but loudly
            failures += 1
            print(f"[{pdb}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        append_row(csv_path, row)
        print(
            f"[{pdb}] lmax={row['phaser_lmax']} samp={row['phaser_sampling_deg']:.4f}deg "
            f"n={row['n_phaser']} | angle_dev={row['angle_max_dev_deg']:.2e} "
            f"r={row.get('pearson_r', float('nan')):.4f} "
            f"rho={row.get('spearman_r', float('nan')):.4f} "
            f"resid={row.get('resid_rms_frac', float('nan')):.3f} "
            f"top100={row.get('top100_overlap', float('nan')):.2f}",
            flush=True,
        )
    if failures:
        print(f"\n{failures}/{len(todo)} cases FAILED", flush=True)
    print(f"\nwrote {csv_path}", flush=True)
    return 1 if failures == len(todo) else 0


if __name__ == "__main__":
    raise SystemExit(main())
