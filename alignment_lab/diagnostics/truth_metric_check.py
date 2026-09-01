"""Do the two truth metrics in this lab agree about which candidate is correct?

Two answers to "is this candidate the right orientation" are in use:

``angle_to_orbit``       used by the rank harnesses. Compares a candidate's
                         ``R_recovered`` against ``S_k @ R_true``.
``residual_rotation_deg`` used by pose_recovery for the pass/fail. Kabsch-
                         superposes the placed coordinates onto canonical and
                         takes the smallest angle to any symop.

``RotationPeak`` rotations are ``R_recovered``, which maps the SEARCH-MODEL frame
onto the crystal frame -- the rotation applied to the coordinates is its
transpose. If the orbit comparison omits that transpose it is comparing a
rotation with its own inverse's orbit, which is a different set unless the
rotation is an involution.

That matters beyond bookkeeping: the rank harness said ranking by the
translation correlation would beat the analytic R by 33/40 to 23/40, and end to
end it lost 31/40 to 36/40. A truth label that is wrong makes every rank in that
harness meaningless, so this checks it directly rather than by inference.
"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_grad_enabled(False)
from lab import rotated_case, seed_for, symmetry_orbit  # noqa: E402
from lab.truth import angle_to_orbit  # noqa: E402

pdb = sys.argv[1] if len(sys.argv) > 1 else "2DQ6"
trial = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n_cand = 25

from torchref.experimental.alignment.frf.rotation_utils import (  # noqa: E402
    rotation_angular_distance_deg, rotation_matrix_from_edmonds_euler)
from torchref.experimental.alignment.pipeline import (  # noqa: E402
    MolecularReplacementPipeline)
from torchref.experimental.alignment.rotation_search import (  # noqa: E402
    prepare_frf_inputs)

seed = seed_for(pdb, trial)
model, data, R_true = rotated_case(pdb, seed)
pipe = MolecularReplacementPipeline(data, model, verbose=0, n_rotation_peaks=200,
                                    n_rotation_candidates=n_cand)
frf = prepare_frf_inputs(model, data, d_min=pipe.d_min, d_max=pipe.d_max,
                         n_shells=pipe.n_shells, verbose=0)
pipe._frf = frf
peaks = pipe._rotation_candidates(frf)[:n_cand]

symops = data.spacegroup.matrices.to(torch.float64).cpu()
rb = data.cell.reciprocal_basis_matrix.to(torch.float64).cpu()
orbit_l = symmetry_orbit(R_true, symops, side="left", frame="cart",
                         reciprocal_basis=rb)
orbit_r = symmetry_orbit(R_true, symops, side="right", frame="cart",
                         reciprocal_basis=rb)
R_t = R_true.to(torch.float64).cpu()

print(f"# {pdb} trial={trial} n_cand={len(peaks)}")
print(f"{'k':>3s} {'orbit side=left':>16s} {'orbit side=right':>17s} "
      f"{'coords (Kabsch form)':>21s}")
n_l = n_r = n_c = 0
for k, p in enumerate(peaks):
    R = rotation_matrix_from_edmonds_euler(p.alpha, p.beta, p.gamma).to(torch.float64)
    a = angle_to_orbit(R, orbit_l)
    b = angle_to_orbit(R, orbit_r)
    c = min(float(rotation_angular_distance_deg(R.T @ R_t, symops[i]))
            for i in range(symops.shape[0]))
    n_l += a <= 8.0; n_r += b <= 8.0; n_c += c <= 8.0
    print(f"{k:3d} {a:16.2f} {b:17.2f} {c:21.2f}")
print(f"within 8 deg: side=left {n_l}/{len(peaks)}, side=right {n_r}/{len(peaks)}, "
      f"coords {n_c}/{len(peaks)}")
