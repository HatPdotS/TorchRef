"""
Unit tests for Model.translate (now returns a copy, mirroring Model.rotate).
"""
from pathlib import Path

import pytest
import torch

from torchref.model import ModelFT


TEST_PDB = Path(__file__).resolve().parents[2] / "files" / "pdb" / "1DAW.pdb"


@pytest.fixture
def model():
    return ModelFT().load_pdb(str(TEST_PDB))


@pytest.mark.unit
def test_translate_returns_copy_not_inplace(model):
    """Original model coordinates must be unchanged after translate()."""
    xyz_before = model.xyz().detach().clone()
    t = torch.tensor([5.0, -3.0, 1.0], dtype=model.dtype_float)
    translated = model.translate(t)
    xyz_after = model.xyz().detach()
    assert torch.allclose(xyz_before, xyz_after), \
        "translate() must not mutate the source model"
    expected = xyz_before + t.to(xyz_before.dtype)
    assert torch.allclose(translated.xyz().detach(), expected, atol=1e-6)


@pytest.mark.unit
def test_translate_returns_different_object(model):
    translated = model.translate(torch.tensor([1.0, 0.0, 0.0], dtype=model.dtype_float))
    assert translated is not model
    # Two independent storages
    assert translated.xyz().data_ptr() != model.xyz().data_ptr()


@pytest.mark.unit
def test_translate_fractional_matches_cartesian(model):
    """`translate(t_frac, fractional=True)` agrees with the Cartesian form via
    the cell's `fractional_to_cartesian` helper (the canonical conversion)."""
    t_frac = torch.tensor([0.25, 0.10, -0.30], dtype=model.dtype_float)
    t_cart = model.cell.fractional_to_cartesian(t_frac)
    translated_frac = model.translate(t_frac, fractional=True)
    translated_cart = model.translate(t_cart, fractional=False)
    assert torch.allclose(
        translated_frac.xyz().detach(),
        translated_cart.xyz().detach(),
        atol=1e-5,
    )


@pytest.mark.unit
def test_translate_preserves_b_and_occupancy(model):
    """ADP and occupancy values must be unchanged by translation."""
    adp_before = model.adp().detach().clone()
    occ_before = model.occupancy().detach().clone()
    translated = model.translate(
        torch.tensor([2.5, 0.0, 0.0], dtype=model.dtype_float),
    )
    assert torch.allclose(translated.adp().detach(), adp_before)
    assert torch.allclose(translated.occupancy().detach(), occ_before)


@pytest.mark.unit
def test_translate_zero_is_noop_on_coords(model):
    """Translation by 0 returns a fresh copy with the same coordinates."""
    zero = torch.zeros(3, dtype=model.dtype_float)
    translated = model.translate(zero)
    assert translated is not model
    assert torch.allclose(translated.xyz().detach(), model.xyz().detach())
