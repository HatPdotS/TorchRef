"""What MixedModel accepts at construction: populations and the models' crystal."""

import pytest

from torchref.model import MixedModel, ModelFT
from torchref.symmetry import Cell


@pytest.fixture(scope="module")
def model(pdb_dir):
    path = str(pdb_dir / "1DAW.pdb")
    return ModelFT(max_res=3.0, verbose=0, device="cpu").load_pdb(path)


@pytest.mark.unit
@pytest.mark.parametrize("fractions", [[1.5, -0.5], [-0.5, 1.5]])
def test_fractions_must_be_non_negative(model, fractions):
    """A negative population that still sums to 1 is refused, not clamped."""
    with pytest.raises(ValueError, match="non-negative"):
        MixedModel([model, model.copy()], initial_fractions=fractions)


@pytest.mark.unit
def test_models_in_different_unit_cells_are_refused(model):
    """Structure factors of two crystals do not mix; the same cell does."""
    other = model.copy()
    other.cell = Cell(
        [214.7, 60.0, 70.0, 90.0, 100.0, 90.0],
        dtype=other.dtype_float,
        device=other.device,
    )
    with pytest.raises(ValueError, match="unit cell"):
        MixedModel([model, other])
    assert len(MixedModel([model, model.copy()]).models) == 2
