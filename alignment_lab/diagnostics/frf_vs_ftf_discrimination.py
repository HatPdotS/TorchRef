"""Does the translation function rank FRF peaks better than the FRF's own score?

The FRF is a 3-D check: it correlates Pattersons, so a wrong orientation that
happens to reproduce the intramolecular vector set scores well. The translation
function is a 6-D check -- it has to place the molecule against the *crystal*,
intermolecular contacts included -- so it should separate truth from a ghost by
much more. That is the standard argument for carrying many orientations into the
TF, and it is worth measuring before spending anything on making the TF fast:
if the TF ranks no better than the FRF, carrying 100 orientations buys nothing.

Takes the top ``--n-cand`` raw FRF peaks (no ML rescore -- the rescore is a
separate, and separately measured, reordering), places each one, and reports
where truth lands under four orderings:

``frf``      the FRF's own score, i.e. the baseline
``tf_corr``  top translation peak of the Crowther-Blow amplitude correlation
``tf_llg``   the same peaks re-ranked by the shared-sigma_A Rice/Woolfson LLG
``r``        analytic-scale R at the locally refined t -- what the pipeline
             actually ranks by today

Rank alone understates the question, so each ordering also gets a separation
``z = (score_truth - mean_others) / std_others``: rank 0 by a hair and rank 0 by
five sigma are different claims about discrimination, and only the second one
justifies widening the funnel.

The placement path is the production one -- ``_make_rotated`` then the same
``precompute_G`` / ``amplitude_translation_search`` / ``local_translation_refine``
calls ``_placement_for_candidate`` makes -- with ``do_joint_refine`` off, since
the rigid-body polish is a later stage and would confound the TF's own ranking.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)

from lab import (BENCH_PDBS, ResultWriter, rotated_case, seed_for,  # noqa: E402
                 symmetry_orbit)
from lab.truth import angle_to_orbit  # noqa: E402


def _rank_of_truth(scores, truth_mask, higher_is_better=True):
    """Rank of the best-placed *correct* candidate under this ordering.

    Correct means within the angular threshold, and several candidates can be:
    the peak list carries near-duplicates and symmetry mates. Any of them is a
    solution, so the rank that matters is the first one to appear -- the same
    definition :func:`orbit_rank` uses, and the one the pipeline behaves by.
    """
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s if higher_is_better else s, kind="stable")
    return int(next(i for i, j in enumerate(order) if truth_mask[j]))


def _separation(scores, truth_mask, higher_is_better=True):
    """Best correct candidate's score in sigmas above the *wrong* ones.

    Negated for lower-is-better scores so a larger number always means better
    discrimination, whichever direction the score runs. Every within-threshold
    candidate is held out of the reference pool: leaving a second copy of the
    answer in it would inflate the pool's mean and understate the separation.
    """
    s = np.asarray(scores, dtype=float)
    m = np.asarray(truth_mask, dtype=bool)
    best = float(s[m].max() if higher_is_better else s[m].min())
    others = s[~m]
    if others.size < 2:
        return float("nan")
    sd = float(others.std(ddof=1))
    if sd < 1e-30:
        return float("nan")
    z = (best - float(others.mean())) / sd
    return z if higher_is_better else -z


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--n-cand", type=int, default=25,
                    help="FRF peaks placed. Truth's worst FRF rank over the "
                         "100-cell panel was 21, so 25 covers every case that "
                         "the rotation function gets right at all.")
    ap.add_argument("--n-rotation-peaks", type=int, default=200)
    ap.add_argument("--thr-deg", type=float, default=8.0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    from torchref.experimental.alignment.align import (
        _DirectModelEvaluator, _prepare_frf_inputs,
    )
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_matrix_from_edmonds_euler,
    )
    from torchref.experimental.alignment.pipeline import (
        MolecularReplacementPipeline,
    )
    from torchref.experimental.alignment.translation import (
        amplitude_translation_search, local_translation_refine,
        precompute_G_for_rotation,
    )

    writer = None
    if args.out_csv:
        writer = ResultWriter(
            args.out_csv, "frf_vs_ftf",
            extra_fields=("n_cand", "truth_found", "rank_frf", "rank_tf_corr",
                          "rank_tf_llg", "rank_r", "z_frf", "z_tf_corr",
                          "z_tf_llg", "z_r", "seconds"),
        )

    for trial in range(args.trials):
        seed = seed_for(args.pdb, trial)
        model, data, R_true = rotated_case(args.pdb, seed)
        t0 = time.time()

        pipe = MolecularReplacementPipeline(
            data, model, verbose=0,
            rescore_engine="none", subpeak_refine=False,
            n_rotation_peaks=args.n_rotation_peaks,
            n_rotation_candidates=args.n_cand,
            do_joint_refine=False, dense_rotation_refine=False,
            use_llg_tf=False,
        )
        frf = _prepare_frf_inputs(
            model, data, d_min=pipe.d_min, d_max=pipe.d_max,
            n_shells=pipe.n_shells, verbose=0,
        )
        pipe._frf = frf
        peaks = pipe._rotation_candidates(frf)[: args.n_cand]
        pipe._prepare_translation_arrays()

        orbit = symmetry_orbit(
            R_true, data.spacegroup.matrices.to(torch.float64).cpu(),
            side="left", frame="cart",
            reciprocal_basis=data.cell.reciprocal_basis_matrix.to(
                torch.float64).cpu(),
        )
        eye3 = pipe._eye3
        rows = []
        for k, p in enumerate(peaks):
            ang = angle_to_orbit(
                rotation_matrix_from_edmonds_euler(p.alpha, p.beta, p.gamma),
                orbit,
            )
            rot = pipe._make_rotated(p)[0]
            rot.spacegroup = data.spacegroup.hm
            p1 = rot.copy()
            p1.spacegroup = "P 1"
            ev = _DirectModelEvaluator(p1)
            G, h_R = precompute_G_for_rotation(
                ev, eye3, pipe._hkl_keep, data.spacegroup, data.cell)
            _, _, tp = amplitude_translation_search(
                F_obs=pipe._F_obs_amp, interpolator=ev, R_rotation=eye3,
                hkl=pipe._hkl_keep, spacegroup=data.spacegroup,
                real_cell=data.cell, grid_steps=pipe.translation_grid_steps,
                n_peaks=pipe.n_translation_peaks, cluster_radius=0.05,
                precomputed_G=G, precomputed_h_R=h_R)
            tf_corr = float(tp[0].score)
            # Phaser's TFZ: the top translation peak measured against the
            # spread of *this orientation's own* translation map. Unlike the
            # cross-candidate z reported below it needs no other candidate, so
            # it is the only score here that can stop a sequential search.
            tfz = float(tp[0].sigma)
            llg_peaks = pipe._llg_tf_rescore(tp, G, h_R)
            tf_llg = float(llg_peaks[0].score)
            lv = np.array([q.score for q in llg_peaks], dtype=float)
            llgz = (float((lv[0] - lv[1:].mean()) / lv[1:].std(ddof=1))
                    if lv.size > 2 and lv[1:].std(ddof=1) > 1e-30
                    else float("nan"))
            r_best = float("inf")
            for cand in llg_peaks[: pipe.n_translation_candidates]:
                _, r_a = local_translation_refine(
                    F_obs=pipe._F_obs_amp, interpolator=ev, R_rotation=eye3,
                    hkl=pipe._hkl_keep, spacegroup=data.spacegroup,
                    real_cell=data.cell,
                    t_init=torch.as_tensor(cand.translation,
                                           dtype=torch.float64),
                    radius=0.06, grid_steps=13, n_refinement_passes=1,
                    precomputed_G=G, precomputed_h_R=h_R)
                r_best = min(r_best, r_a)
            rows.append(dict(k=k, ang=ang, frf=float(p.score),
                             tf_corr=tf_corr, tf_llg=tf_llg, r=r_best,
                             tfz=tfz, llgz=llgz))
            print(f"CAND pdb={args.pdb} trial={trial} k={k} ang={ang:.3f} "
                  f"is_truth={int(ang <= args.thr_deg)} frf={p.score:.4f} "
                  f"tf_corr={tf_corr:.5f} tfz={tfz:.3f} tf_llg={tf_llg:.2f} "
                  f"llgz={llgz:.3f} r={r_best:.5f}", flush=True)

        secs = time.time() - t0
        truth = [r for r in rows if r["ang"] <= args.thr_deg]
        if not truth:
            print(f"ROW pdb={args.pdb} trial={trial} truth_found=0 "
                  f"best_ang={min(r['ang'] for r in rows):.2f} "
                  f"seconds={secs:.1f}", flush=True)
            continue
        tmask = [r["ang"] <= args.thr_deg for r in rows]
        ti = rows.index(min(truth, key=lambda r: r["ang"]))
        cols = {"frf": True, "tf_corr": True, "tf_llg": True, "r": False}
        ranks = {c: _rank_of_truth([r[c] for r in rows], tmask, hi)
                 for c, hi in cols.items()}
        zs = {c: _separation([r[c] for r in rows], tmask, hi)
              for c, hi in cols.items()}
        print(f"ROW pdb={args.pdb} trial={trial} seed={seed} truth_found=1 "
              f"n_cand={len(rows)} n_truth={sum(tmask)} "
              + " ".join(f"rank_{c}={ranks[c]}" for c in cols)
              + " " + " ".join(f"z_{c}={zs[c]:.2f}" for c in cols)
              + f" seconds={secs:.1f}", flush=True)
        if writer:
            writer.write(
                pdb=args.pdb, seed=seed, trial=trial,
                spacegroup=str(data.spacegroup),
                n_ops=int(data.spacegroup.matrices.shape[0]),
                truth_rank=ranks["frf"], truth_angle_deg=round(rows[ti]["ang"], 3),
                orbit_side="left", orbit_frame="cart", lmax_cap="",
                d_min=pipe.d_min, d_max=pipe.d_max, device="cpu",
                n_cand=len(rows), truth_found=1,
                **{f"rank_{c}": ranks[c] for c in cols},
                **{f"z_{c}": round(zs[c], 4) for c in cols},
                seconds=round(secs, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
