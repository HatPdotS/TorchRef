"""Regression tests for fractional->Cartesian translation (cart = frac @ B.T).

Four call sites converted a fractional translation vector as ``frac @ B``
(missing the transpose) instead of the codebase convention ``frac @ B.T``
(== ``Cell.fractional_to_cartesian``). For non-orthogonal cells ``B != B.T``,
so the translation was applied wrong; orthogonal test cells (B diagonal) hid
it. See TORCHREF_AUDIT.md cluster 2.
"""

import pytest
import torch

from torchref.symmetry.cell import Cell

TRICLINIC = [40.0, 50.0, 60.0, 70.0, 80.0, 85.0]
ORTHO = [40.0, 50.0, 60.0, 90.0, 90.0, 90.0]


@pytest.mark.unit
def test_convention_uses_transpose_for_triclinic():
    """cart = frac @ B.T, and for a triclinic cell that differs from frac @ B."""
    cell = Cell(TRICLINIC)
    t = torch.tensor([0.13, -0.07, 0.21], dtype=cell.dtype, device=cell.device)
    B = cell.fractional_matrix

    expected = cell.fractional_to_cartesian(t)
    assert torch.allclose(expected, t @ B.T, atol=1e-6)
    # The buggy expression genuinely differs for a non-orthogonal cell.
    assert not torch.allclose(t @ B, t @ B.T, atol=1e-4)


@pytest.mark.unit
def test_orthogonal_cell_masks_the_bug():
    """For an orthogonal cell B is diagonal, so B == B.T and the bug is invisible."""
    cell = Cell(ORTHO)
    t = torch.tensor([0.13, -0.07, 0.21], dtype=cell.dtype, device=cell.device)
    B = cell.fractional_matrix
    assert torch.allclose(t @ B, t @ B.T, atol=1e-6)


@pytest.mark.integration
def test_model_translate_fractional_triclinic(sample_structure_pair):
    """Model.translate(fractional=True) must apply frac @ B.T on a triclinic cell."""
    from torchref.model.model_ft import ModelFT

    model = ModelFT()
    model.load_cif(str(sample_structure_pair["model"]))

    # Force a triclinic cell so the transpose matters.
    model.cell = Cell(TRICLINIC, dtype=model.dtype_float, device=model.device)

    xyz0 = model.xyz().clone()
    t = torch.tensor([0.13, -0.07, 0.21], dtype=model.dtype_float, device=model.device)

    model.translate(t, fractional=True)

    disp = model.xyz() - xyz0
    expected = model.cell.fractional_to_cartesian(t)

    # Every atom shifts by the same Cartesian vector = frac @ B.T.
    assert torch.allclose(disp[0], expected, atol=1e-4)
    assert torch.allclose(disp, expected.unsqueeze(0).expand_as(disp), atol=1e-4)
    # The old buggy expression (frac @ B) would have differed.
    assert not torch.allclose(disp[0], t @ model.cell.fractional_matrix, atol=1e-3)
