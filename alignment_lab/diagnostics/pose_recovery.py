"""End-to-end pose recovery: does dropping the ML rescore cost anything?

The rank-level evidence says the rescore only reorders (it leaves every peak's
orientation untouched), that final solutions are selected by R-factor rather
than by the rescore's own score, and that its reordering does not improve
whether truth reaches the translation stage. If all that holds, removing it
should be free -- but rank is not the deliverable, pose is, so this measures the
full pipeline.

Arms (``--arms``):

``m_letf1``
    current default.
``none``
    skip the rescore; raw FRF order straight into the translation search.
``none+subpeak``
    the same, plus quadratic sub-peak refinement -- sharpen each orientation
    in place without reordering.

Success mirrors the integration test: final coordinates within ``--success-deg``
of canonical, modulo the crystal symmetry.

Usage::

    python alignment_lab/diagnostics/pose_recovery.py --pdb 1DAW --trial 0 \
        --arms m_letf1,none,none+subpeak --out-csv alignment_lab/runs/pose.csv
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
    "m_letf1": dict(rescore_engine="m_letf1", subpeak_refine=False),
    "sim": dict(rescore_engine="sim", subpeak_refine=False),
    "none": dict(rescore_engine="none", subpeak_refine=False),
    "none+subpeak": dict(rescore_engine="none", subpeak_refine=True),
    "m_letf1+subpeak": dict(rescore_engine="m_letf1", subpeak_refine=True),
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default="1DAW", choices=list(BENCH_PDBS))
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--arms", default="m_letf1,none,none+subpeak")
    ap.add_argument("--n-rotation-candidates", type=int, default=15)
    ap.add_argument("--n-rotation-peaks", type=int, default=200)
    ap.add_argument("--success-deg", type=float, default=8.0)
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    from torchref.experimental.alignment.align import align_model_to_data

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
                              extra_fields=("arm", "rescore_engine", "subpeak_refine",
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
            aligned = align_model_to_data(
                search, data, d_min=4.0, d_max=15.0, n_shells=20,
                n_rotation_peaks=args.n_rotation_peaks, n_ml_refine=200,
                do_translation=True, do_joint_refine=True,
                n_rotation_candidates=args.n_rotation_candidates,
                verbose=args.verbose, **flags,
            )
            resid = residual_rotation_deg(aligned.xyz(), canonical_xyz, symops)
            err = ""
        except Exception as exc:  # a crashed arm must not read as a success
            resid, err = float("nan"), f"{type(exc).__name__}: {exc}"
        secs = time.time() - t0
        ok = (resid == resid) and resid <= args.success_deg
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
                         rescore_engine=flags["rescore_engine"],
                         subpeak_refine=int(flags["subpeak_refine"]),
                         residual_deg=(round(resid, 4) if resid == resid else ""),
                         success=int(bool(ok)),
                         n_rotation_candidates=args.n_rotation_candidates,
                         pipeline_seconds=round(secs, 1))
    if args.out_csv:
        print(f"  wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
