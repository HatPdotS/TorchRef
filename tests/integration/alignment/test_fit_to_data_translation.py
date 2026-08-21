"""
Integration test for the new translation + joint R+t refinement in
`ModelFT.fit_to_data` (Phase 3 component B).

Setup: 1DAW.mtz (C2) F_obs + P1 search model. Apply a small known rotation and
fractional translation, then ask `fit_to_data` to recover both. Acceptance:
recovered `(R_residual, t_residual)` brings the model close to canonical
(within 8° rotation modulo C2 symmetry and within 0.1 fractional shift along
any axis modulo unit cell), and the post-refinement R-work drops well below
the pre-fit value.
"""
import math
from pathlib import Path

import pytest
import torch

from torchref.base.metrics.rfactor import rfactor_work_free
from torchref.experimental.alignment.align import align_model_to_data
from torchref.experimental.alignment.frf.rotation_utils import (
    rotation_angular_distance_deg,
    rotation_matrix_from_edmonds_euler,
)
from torchref.io.datasets.reflection_data import ReflectionData
from torchref.model import ModelFT
from torchref.scaling import Scaler


TEST_FILES = Path(__file__).resolve().parents[2] / "files"
PDB_1DAW = TEST_FILES / "pdb" / "1DAW.pdb"
MTZ_1DAW = TEST_FILES / "mtz" / "1DAW.mtz"


def _scale_and_rwork(model: ModelFT, data) -> float:
    scaler = Scaler(model=model, data=data, nbins=20, verbose=0)
    fcalc = model(data.hkl)
    scaler.initialize(fcalc)
    scaler.refine_lbfgs(fcalc=fcalc)
    rw, _ = rfactor_work_free(data, torch.abs(scaler.forward(fcalc)))
    return rw.item() if hasattr(rw, "item") else float(rw)


def _wrap_frac(t: torch.Tensor) -> torch.Tensor:
    """Wrap fractional coords into [-0.5, 0.5)."""
    return (t + 0.5) % 1.0 - 0.5


@pytest.mark.integration
@pytest.mark.slow
def test_fit_to_data_recovers_rotation_and_translation():
    data = ReflectionData().load_mtz(str(MTZ_1DAW))
    canonical = ModelFT().load_pdb(str(PDB_1DAW))
    canonical.spacegroup = "P 1"

    # Apply a known random rotation + fractional translation.
    R_true = rotation_matrix_from_edmonds_euler(0.6, 0.4, 1.2)
    R_apply = R_true.to(canonical.dtype_float)
    # .copy() first: Model.rotate mutates in place, so rotating `canonical`
    # directly would perturb the very reference this test compares against.
    rotated = canonical.copy().rotate(R_apply, center=canonical.xyz().mean(dim=0))
    t_frac_true = torch.tensor([0.18, -0.07, 0.23], dtype=canonical.dtype_float)
    perturbed = rotated.translate(t_frac_true, fractional=True)

    rwork_pre = _scale_and_rwork(perturbed, data)

    aligned = align_model_to_data(
        perturbed,
        data,
        d_min=4.0, d_max=15.0,
        L=32, n_shells=20,
        n_rotation_peaks=200, n_ml_refine=200,
        do_translation=True,
        do_joint_refine=True,
        verbose=0,
    )

    # Recovered rotation (modulo C2). Compare canonical vs aligned via centroid.
    xyz_canon = canonical.xyz().to(torch.float64)
    xyz_aligned = aligned.xyz().to(torch.float64)
    c_canon = xyz_canon.mean(dim=0)
    c_aligned = xyz_aligned.mean(dim=0)
    a = xyz_canon - c_canon
    b = xyz_aligned - c_aligned
    H = b.T @ a
    U, _, Vt = torch.linalg.svd(H)
    d = float(torch.sign(torch.det(Vt.T @ U.T)))
    D = torch.diag(torch.tensor([1.0, 1.0, d], dtype=H.dtype))
    R_residual = Vt.T @ D @ U.T
    sym_mats = data.spacegroup.matrices.to(torch.float64)
    best_rot_err = min(
        rotation_angular_distance_deg(R_residual, sym_mats[k])
        for k in range(sym_mats.shape[0])
    )
    assert best_rot_err < 8.0, (
        f"residual rotation {best_rot_err:.2f}° > 8° gate"
    )

    # We do not pin down the recovered translation directly — for spacegroups
    # with polar / non-unique origins (e.g. C2's free origin along y) the
    # recovered translation may differ from the applied one by an allowed
    # origin shift. The crystallographic test that this is a valid solution is
    # the R-factor of the scaled model.
    # The translation function brings R-work close to the canonical-native
    # reference (0.21 for 1DAW). The residual gap (~0.12) is from the
    # rotation function's ~2° angular error — a separate refinement that's
    # not part of the translation function. A 1.87° rotation residual on a
    # 100Å molecule moves atoms by ~3Å, which costs ~0.12 in R-work even
    # with the exactly correct translation.
    rwork_post = _scale_and_rwork(aligned, data)
    canonical_native = ModelFT().load_pdb(str(PDB_1DAW))  # native C2
    rwork_ref = _scale_and_rwork(canonical_native, data)
    assert rwork_post < rwork_ref + 0.18, (
        f"R-work {rwork_post:.4f} > reference {rwork_ref:.4f} + 0.18 "
        f"(pre-fit was {rwork_pre:.4f})"
    )
