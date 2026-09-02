"""Is a returned placement the deposited pose, and if not, does the deposited pose score better?

Runs the pipeline on a seeded reorientation, then compares the winner with the
deposited model under the pipeline's own three selection scores, evaluated
through the same ``TranslationObs`` and ``precompute_G`` path. Reports the
rotation and translation error of the winner against the closest symmetry
image of the deposited model, and the raw fractional centroid offset to every
image, so a pseudo-translation shows up as a specific vector.

If the deposited pose scores clearly better than the winner, the translation
search missed it. If they score the same, the data cannot tell them apart.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab import (BENCH_PDBS, allowed_origin_shifts, load_case, pose_error,  # noqa: E402
                 random_rotation, seed_for)


def scores_at(pipe, model_placed):
    """(tf score, R, llg) of an already-placed model through the pipeline's path."""
    from torchref.experimental.alignment.translation import (
        DirectModelEvaluator, analytic_r_at, llg_at_translations,
        prepare_candidate, translation_score_at)
    data, obs = pipe.data, pipe._obs
    m = model_placed.copy()
    if pipe.tf_d_min > 0.0:
        m.max_res = pipe.tf_d_min / 1.5
    m.spacegroup = "P 1"
    cand = prepare_candidate(DirectModelEvaluator(m), obs, data.spacegroup, data.cell)
    t0 = torch.zeros(3, dtype=torch.float64)
    tf = translation_score_at(obs, cand, t0)
    r = analytic_r_at(obs, cand, t0)
    llg = float(llg_at_translations(obs, cand, t0.view(1, 3))[0])
    return tf, r, llg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="2DQ6", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=3)
    ap.add_argument("--rank-by", default="llg")
    ap.add_argument("--tf-d-min", type=float, default=None)
    ap.add_argument("--tf-d-max", type=float, default=None)
    args = ap.parse_args()

    from torchref.experimental.alignment import MolecularReplacementPipeline

    model, data = load_case(args.pdb)
    canonical = model.xyz().clone()
    seed = seed_for(args.pdb, args.trial)
    R_true = random_rotation(seed)
    shifts, polar = allowed_origin_shifts(data.spacegroup)
    print(f"=== {args.pdb} t{args.trial} {data.spacegroup} tf_window=({args.tf_d_max},{args.tf_d_min}) allowed shifts "
          f"{[tuple(round(float(x), 3) for x in u) for u in shifts]} polar dims {polar.shape[1]}")

    search = model.copy()
    search.spacegroup = "P 1"
    search = search.copy().rotate(R_true.to(model.dtype_float), center=canonical.mean(0))
    pipe = MolecularReplacementPipeline(
        data, search, d_min=4.0, d_max=15.0, n_shells=20,
        n_rotation_peaks=200, n_rotation_candidates=25, rank_by=args.rank_by,
        tf_d_min=args.tf_d_min, tf_d_max=args.tf_d_max,
    )
    sols = pipe.run(do_translation=True)
    win = sols[0]

    # Sanity: the deposited model against itself must be (0, 0).
    print("SANITY deposited-vs-deposited rot/trans:",
          pose_error(canonical, canonical, data.cell, data.spacegroup))
    rot, trans = pose_error(win.model.xyz(), canonical, data.cell, data.spacegroup)
    print(f"WINNER rot_deg={rot:.3f} trans_A={trans:.2f} tf={win.translation_score:.5f} "
          f"R={win.r_factor:.5f} llg={win.llg_score:.1f}")

    # Raw fractional centroid offset to every symmetry image of canonical.
    B = data.cell.fractional_matrix.detach().cpu().to(torch.float64)
    Binv = torch.linalg.inv(B)
    S = data.spacegroup.matrices.detach().cpu().to(torch.float64)
    T = data.spacegroup.translations.detach().cpu().to(torch.float64)
    ca = Binv @ win.model.xyz().detach().cpu().to(torch.float64).mean(0)
    cc = Binv @ canonical.detach().cpu().to(torch.float64).mean(0)
    for k in range(S.shape[0]):
        d = ca - (S[k] @ cc + T[k])
        d = d - d.round()
        print(f"  image {k}: centroid offset frac=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) "
              f"|.|={float((B @ d).norm()):.1f} A")

    c_dep, r_dep, llg_dep = scores_at(pipe, model)
    print(f"DEPOSITED tf={c_dep:.5f} R={r_dep:.5f} llg={llg_dep:.1f}")
    c_w, r_w, llg_w = scores_at(pipe, win.model)
    print(f"WINNER(re-scored) tf={c_w:.5f} R={r_w:.5f} llg={llg_w:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
