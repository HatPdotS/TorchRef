"""
Integration test for ``align_model_to_data`` with the translation search on.

Setup: 1DAW.mtz (C2) F_obs + P1 search model. Apply a known rotation and
fractional translation, then ask the pipeline to recover both. Acceptance: the
returned model is within 8 degrees of canonical modulo the C2 symmetry, its
position is within 3 A of a symmetry image of canonical (modulo lattice
translations and the group's allowed origin shifts; y is polar and free), and
its scaled R-work is close to the deposited model's.
"""
from pathlib import Path

import pytest
import torch

from torchref.base.metrics.rfactor import rfactor_work_free
from torchref.experimental.alignment import align_model_to_data
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


def _cartesian_symops(data) -> torch.Tensor:
    """Point-group rotations as Cartesian matrices, ``B S B^-1``.

    ``spacegroup.matrices`` act on fractional coordinates. A Kabsch rotation is
    Cartesian, and comparing the two directly is only right when the cell is
    orthogonal and the operator diagonal -- in a trigonal cell two of the six
    mates of a correct placement read as 30 and 21 degrees.
    """
    B = data.cell.fractional_matrix.to(torch.float64)
    S = data.spacegroup.matrices.to(torch.float64)
    return B @ S @ torch.linalg.inv(B)


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
        n_shells=20,
        n_rotation_peaks=200,
        do_translation=True,
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
    sym_cart = _cartesian_symops(data)
    errs = [rotation_angular_distance_deg(R_residual, sym_cart[k])
            for k in range(sym_cart.shape[0])]
    k_best = min(range(len(errs)), key=errs.__getitem__)
    best_rot_err = errs[k_best]
    assert best_rot_err < 8.0, (
        f"residual rotation {best_rot_err:.2f}° > 8° gate"
    )

    # The translation, against the symmetry image whose rotation matched. In
    # C2 the origin is free along y, the centring makes (1/2, 1/2, 0) a lattice
    # vector, and (0, *, 1/2) is an allowed origin shift -- so x and z are each
    # determined only modulo 1/2 and y not at all. A placement at the right
    # orientation and 40 A from the true position used to pass this test.
    B = data.cell.fractional_matrix.to(torch.float64)
    Binv = torch.linalg.inv(B)
    S = data.spacegroup.matrices.to(torch.float64)
    T = data.spacegroup.translations.to(torch.float64)
    c_a = Binv @ c_aligned
    c_c = S[k_best] @ (Binv @ c_canon) + T[k_best]
    delta = c_a - c_c
    delta_xz = (delta + 0.25) % 0.5 - 0.25
    delta_xz[1] = 0.0
    trans_A = float((B @ delta_xz).norm())
    assert trans_A < 3.0, f"placed {trans_A:.1f} A from a symmetry image of canonical"

    # The crystallographic check that this is a valid solution is the R-factor
    # of the scaled model.
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
