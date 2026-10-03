"""Cell metric pinned to gemmi: direct basis, reciprocal basis, volume, d-spacings.

Every cell quantity derives from ``get_fractional_matrix``. The triple-cosine term of
the metric vanishes whenever one cell angle is 90 degrees, so the triclinic and
rhombohedral (R-setting) cells here are the ones that exercise it.
"""

import gemmi
import numpy as np
import pytest
import torch

from torchref.base.coordinates.transforms_torch import get_fractional_matrix
from torchref.base.reciprocal.basis import reciprocal_basis_matrix
from torchref.base.reciprocal.hkl import get_d_spacing
from torchref.config import get_float_dtype
from torchref.symmetry import Cell

pytestmark = pytest.mark.unit

CELLS = {
    "triclinic_5BOV": (44.199, 81.866, 89.925, 100.997, 106.903, 100.84),
    "triclinic_skewed": (30.0, 40.0, 50.0, 70.0, 80.0, 60.0),
    "rhombohedral_R": (60.0, 60.0, 60.0, 80.0, 80.0, 80.0),
    "rhombohedral_acute": (40.0, 40.0, 40.0, 60.0, 60.0, 60.0),
    "monoclinic": (50.0, 60.0, 70.0, 90.0, 105.0, 90.0),
    "hexagonal": (80.0, 80.0, 120.0, 90.0, 90.0, 120.0),
    "orthorhombic": (40.0, 50.0, 60.0, 90.0, 90.0, 90.0),
}

_RANGE = np.arange(-6, 7)
HKL = np.array(
    [(h, k, l) for h in _RANGE for k in _RANGE for l in _RANGE if (h, k, l) != (0, 0, 0)]
)


def _gemmi_matrices(params):
    cell = gemmi.UnitCell(*params)
    return np.array(cell.orth.mat.tolist()), np.array(cell.frac.mat.tolist())


@pytest.mark.parametrize("params", CELLS.values(), ids=CELLS.keys())
def test_d_spacing_matches_gemmi(params):
    cell = torch.tensor(params, dtype=get_float_dtype())
    d = get_d_spacing(torch.from_numpy(HKL), cell).double().numpy()
    uc = gemmi.UnitCell(*params)
    d_ref = np.array([uc.calculate_d(list(map(int, hkl))) for hkl in HKL])
    np.testing.assert_allclose(d, d_ref, rtol=2e-6)


@pytest.mark.parametrize("params", CELLS.values(), ids=CELLS.keys())
def test_direct_and_reciprocal_bases_match_gemmi(params):
    orth_ref, frac_ref = _gemmi_matrices(params)
    cell = torch.tensor(params, dtype=torch.float64)  # dtype-ok: reference precision
    np.testing.assert_allclose(get_fractional_matrix(cell).numpy(), orth_ref, atol=1e-10)
    np.testing.assert_allclose(reciprocal_basis_matrix(cell).numpy(), frac_ref, atol=1e-12)


@pytest.mark.parametrize("params", CELLS.values(), ids=CELLS.keys())
def test_cell_object_is_consistent_with_gemmi(params):
    _, frac_ref = _gemmi_matrices(params)
    # dtype-ok: reference precision, on CPU because MPS has no float64.
    cell = Cell(list(params), dtype=torch.float64, device="cpu")
    np.testing.assert_allclose(
        cell.reciprocal_basis_matrix.numpy(), frac_ref, atol=1e-12
    )
    np.testing.assert_allclose(
        (cell.reciprocal_basis_matrix @ cell.fractional_matrix).numpy(),
        np.eye(3),
        atol=1e-12,
    )
    assert float(cell.volume) == pytest.approx(gemmi.UnitCell(*params).volume, rel=1e-12)


def test_fractional_matrix_is_differentiable_in_the_cell():
    cell = torch.tensor(  # dtype-ok: gradcheck needs double precision
        CELLS["triclinic_skewed"], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(get_fractional_matrix, (cell,))
