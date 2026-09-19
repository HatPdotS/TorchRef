"""The scaler's observed-to-model factor without the bulk-solvent term.

Pinned: an uninitialised scaler reports ones; after initialisation the factor
reproduces ``forward`` exactly once the additive solvent term is removed, so dividing
observed amplitudes by it returns them to the model's absolute scale.
"""

import pytest
import torch

from torchref.io import ReflectionData
from torchref.model.model_ft import ModelFT
from torchref.scaling.scaler import Scaler


@pytest.fixture
def scaler(sample_structure_pair):
    model = ModelFT()
    model.load_cif(str(sample_structure_pair["model"]))
    data = ReflectionData(verbose=0)
    data.load_mtz(str(sample_structure_pair["reflections"]))
    return Scaler(model=model, data=data, nbins=10, verbose=0)


@pytest.mark.unit
def test_uninitialised_scaler_reports_ones(scaler):
    factor = scaler.multiplicative_scale()
    assert factor.shape == (int(scaler.bins.numel()),)
    assert torch.equal(factor, torch.ones_like(factor))
    assert factor.device == scaler.device


@pytest.mark.integration
def test_factor_reproduces_forward_without_solvent(scaler):
    scaler.initialize()
    fcalc = scaler.compute_fcalc()
    factor = scaler.multiplicative_scale()
    assert (factor > 0).all() and torch.isfinite(factor).all()
    assert not factor.requires_grad
    with torch.no_grad():
        scaled = scaler(fcalc, f_sol_override=torch.zeros_like(fcalc))
    assert torch.allclose(scaled, factor.to(scaled.dtype) * fcalc, rtol=1e-5, atol=1e-6)
