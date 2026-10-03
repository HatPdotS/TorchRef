"""Map units: the electrons-per-cubic-Angstrom synthesis.

Pinned: ``units="electrons"`` is the ``1/N``-normalised FFT rescaled by ``N / V``, i.e.
``(1/V) sum_h F(h) exp(-2 pi i h.x)``; the default is unchanged; an unknown unit is
rejected; a ``DifferenceMap`` accepts a per-reflection scale and the same units.
"""

import pytest
import torch

from torchref.io import ReflectionData
from torchref.maps import DifferenceMap, Map
from torchref.model.model_ft import ModelFT


@pytest.fixture(scope="module")
def model_ft_and_data(sample_structure_pair):
    model = ModelFT()
    model.load_cif(str(sample_structure_pair["model"]))
    data = ReflectionData()
    data.load_mtz(str(sample_structure_pair["reflections"]))
    return model, data


@pytest.mark.unit
def test_electrons_is_the_volume_normalised_synthesis(model_ft_and_data):
    model, data = model_ft_and_data
    normalized = Map(data, model, map_type="Fcalc").calculate()
    electrons = Map(data, model, map_type="Fcalc", units="electrons").calculate()
    volume = data.cell.volume.to(normalized.dtype)
    assert torch.allclose(
        electrons, normalized * (normalized.numel() / volume), rtol=1e-5, atol=1e-6
    )


@pytest.mark.unit
def test_unknown_units_are_rejected(model_ft_and_data):
    model, data = model_ft_and_data
    with pytest.raises(ValueError, match="units must be one of"):
        Map(data, model, units="e/A3")


@pytest.mark.unit
def test_difference_map_scale_and_units(model_ft_and_data):
    model, data = model_ft_and_data
    plain = DifferenceMap(data, data, model).calculate()
    scale = torch.full((len(data),), 2.0, dtype=plain.dtype, device=plain.device)
    scaled = DifferenceMap(data, data, model, scale=scale, units="electrons")
    out = scaled.calculate()
    assert out.shape == plain.shape and torch.isfinite(out).all()
    assert scaled.units == "electrons" and scaled.scale is scale
