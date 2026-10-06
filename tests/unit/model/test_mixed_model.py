"""What MixedModel accepts at construction: populations and the models' crystal."""

import pytest

from torchref.model import MixedModel, ModelFT


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
