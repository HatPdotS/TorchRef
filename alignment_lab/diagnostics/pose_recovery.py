"""End-to-end pose recovery for the FRF -> FTF pipeline.

Rank is not the deliverable, pose is. This places a randomly reoriented copy of
the deposited model and asks whether the pipeline gets it back, which is the
only measurement that settles a change to either stage.

The reference number to beat is **24/30** (10 structures x 3 trials): what the
pipeline scored once the ML rescore was taken out of the middle, against 18/30
with it. 2DQ6 and 3GR5 fail in every arm ever measured and cap recovery there.

Arms (``--arms``) sweep how translation candidates are ranked:

``analytic_r``
    the default -- rank each rotation candidate by the analytical-scale R at its
    best translation.
``llg_tf``
    re-rank the translation peaks by the Rice/Woolfson LLG first. At rank level
    the LLG puts truth at rank 0 in 27/30 against analytic R's 22/30; this is
    the arm that says whether that carries through to pose.

Success mirrors the integration test: final coordinates within ``--success-deg``
of canonical, modulo the crystal symmetry.

Usage::

    python alignment_lab/diagnostics/pose_recovery.py --pdb 1DAW --trial 0 \
        --arms analytic_r,llg_tf --out-csv alignment_lab/runs/pose.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# NOTE: no global torch.set_grad_enabled(False) here, unlike the FRF-only
# diagnostics. This runs the full pipeline, whose joint refine and
# rigid-body polish are LBFGS -- they need autograd, and disabling it
# raises "element 0 of tensors does not require grad".

from lab import (BENCH_PDBS, ResultWriter, load_case, random_rotation,  # noqa: E402
                 seed_for)

ARMS = {
    "analytic_r": dict(use_llg_tf=False),
    "llg_tf": dict(use_llg_tf=True),
}


def residual_rotation_deg(aligned_xyz, canonical_xyz, symops) -> float:
    """Smallest angle between the aligned-to-canonical rotation and any symop.

    Kabsch superposition, then compared against every symmetry operator --
    a solution differing from canonical by a crystal symmetry is correct.
    """
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_angular_distance_deg,
    )

    P = canonical_xyz.to(torch.float64)
    Q = aligned_xyz.to(torch.float64)
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = torch.linalg.svd(Qc.T @ Pc)
    d = torch.sign(torch.det(U @ Vt))
    R = U @ torch.diag(torch.tensor([1.0, 1.0, d], dtype=torch.float64)) @ Vt
    return min(float(rotation_angular_distance_deg(R, symops[k]))
               for k in range(symops.shape[0]))


#: The rotation search's own bandwidth constant. Recorded in every row because
#: `align_model_to_data` has no bandwidth argument, so a `--lmax-cap` flag here
#: would name a value the engine never saw.
import importlib as _importlib  # noqa: E402

_LMAX_CAP = _importlib.import_module(
    "torchref.experimental.alignment.rotation_search").LMAX_CAP



def _report_candidates(solutions, R_true, symops, success_deg) -> None:
    """Annotate the pipeline's own candidates with how far each is from truth.

    The pipeline reports every candidate's scores at ``verbose >= 2`` but cannot
    say which was right -- it has no ground truth, and a version of it that did
    would be measuring itself. This joins the two: the ranked solutions it
    returned, against the orientation the benchmark rotated the model by.

    That join is the whole point of driving the pipeline directly rather than
    rebuilding its placement loop in the harness. A reimplementation drifts, and
    then the two disagree about which candidate the pipeline picked -- which is
    exactly what happened here: a harness reported truth top-ranked by analytic
    R in 0 of 10 seeds while the pipeline solved 6 of them, because it fed the
    R-factor a different set of translation peaks.

    ``SOLN`` lines are ordered as the pipeline ranked them, so line 0 is what it
    returned. ``dtruth`` is the angle from that candidate's orientation to the
    true one modulo crystal symmetry; ``pick`` marks the winner and ``true``
    marks every candidate that was in fact correct.
    """
    from torchref.experimental.alignment.frf.rotation_utils import (
        rotation_angular_distance_deg,
    )

    R_t = R_true.to(torch.float64).cpu()
    print("  SOLN rank  rot_score   tf_R    dtruth  flags")
    for i, sol in enumerate(solutions):
        R = torch.as_tensor(sol.rotation, dtype=torch.float64)
        # `rotation` maps the search-model frame onto the crystal frame; the
        # benchmark's R_true is the rotation applied to the coordinates, so the
        # recovered orientation is compared as its transpose.
        d = min(float(rotation_angular_distance_deg(R.T @ R_t, symops[k]))
                for k in range(symops.shape[0]))
        flags = ("pick " if i == 0 else "     ") + ("true" if d <= success_deg else "")
        print(f"  SOLN {i:4d} {sol.rotation_score:10.3f} "
              f"{sol.translation_score:7.4f} {d:8.2f}  {flags}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--arms", default="analytic_r,llg_tf")
    ap.add_argument("--n-rotation-candidates", type=int, default=25)
    ap.add_argument("--n-rotation-peaks", type=int, default=200)
    ap.add_argument("--success-deg", type=float, default=8.0)
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    from torchref.experimental.alignment import MolecularReplacementPipeline

    seed = seed_for(args.pdb, args.trial)
    model, data = load_case(args.pdb)
    canonical_xyz = model.xyz().clone()
    symops = data.spacegroup.matrices.to(torch.float64).cpu()
    R_true = random_rotation(seed)

    print(f"=== {args.pdb} t{args.trial} seed={seed} {data.spacegroup} "
          f"n_ops={symops.shape[0]} | success gate {args.success_deg} deg ===")
    print(f"  {'arm':16s} {'resid_deg':>10s} {'ok':>4s} {'seconds':>9s}")

    writer = None
    if args.out_csv:
        writer = ResultWriter(args.out_csv, "pose_recovery",
                              extra_fields=("arm", "use_llg_tf",
                                            "residual_deg", "success",
                                            "n_rotation_candidates",
                                            "pipeline_seconds"))
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {sorted(ARMS)}")
        flags = ARMS[arm]
        # Fresh copy per arm: rotate/translate mutate in place, and the arms
        # must start from identical coordinates to be comparable.
        search = model.copy()
        search.spacegroup = "P 1"
        search = search.copy().rotate(R_true.to(model.dtype_float),
                                      center=canonical_xyz.mean(0))
        t0 = time.time()
        try:
            # The pipeline rather than `align_model_to_data`, which returns only
            # the winner. Every candidate's score is the diagnosis when a
            # placement goes wrong, and the pipeline already computed them.
            pipe = MolecularReplacementPipeline(
                data, search, d_min=4.0, d_max=15.0, n_shells=20,
                n_rotation_peaks=args.n_rotation_peaks,
                n_rotation_candidates=args.n_rotation_candidates,
                verbose=args.verbose, **flags,
            )
            solutions = pipe.run(do_translation=True)
            aligned = solutions[0].model
            resid = residual_rotation_deg(aligned.xyz(), canonical_xyz, symops)
            err = ""
            if args.verbose >= 2:
                _report_candidates(solutions, R_true, symops, args.success_deg)
        except Exception as exc:  # a crashed arm must not read as a success
            resid, err = float("nan"), f"{type(exc).__name__}: {exc}"
        secs = time.time() - t0
        ok = (resid == resid) and resid <= args.success_deg
        print(f"ROW {arm} {args.pdb} trial={args.trial} "
              f"n_cand={args.n_rotation_candidates} "
              f"resid={resid:.3f} ok={int(bool(ok))} seconds={secs:.1f}",
              flush=True)
        print(f"  {arm:16s} {resid:10.2f} {('yes' if ok else 'NO'):>4s} {secs:9.1f}"
              + (f"   {err}" if err else ""))
        if writer:
            writer.write(pdb=args.pdb, seed=seed, trial=args.trial,
                         spacegroup=str(data.spacegroup), n_ops=int(symops.shape[0]),
                         truth_rank="", truth_angle_deg=(round(resid, 4)
                                                         if resid == resid else ""),
                         orbit_side="kabsch", orbit_frame="cart",
                         lmax_cap=_LMAX_CAP, d_min=4.0, d_max=15.0,
                         device="cpu", arm=arm,
                         use_llg_tf=int(flags["use_llg_tf"]),
                         residual_deg=(round(resid, 4) if resid == resid else ""),
                         success=int(bool(ok)),
                         n_rotation_candidates=args.n_rotation_candidates,
                         pipeline_seconds=round(secs, 1))
    if args.out_csv:
        print(f"  wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
