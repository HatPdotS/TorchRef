"""Tests for the activation-fraction floor diagnostic."""

import pytest
import torch


@pytest.mark.integration
def test_density_peaks_at_atom_sites(sample_structure_pair):
    """The Fourier-summed density is higher at atom sites than at random points."""
    from torchref.experimental.targets import OccupancyFloorDiagnostic
    from torchref.io import ReflectionData
    from torchref.model.model_ft import ModelFT

    model = ModelFT(max_res=2.5, verbose=0)
    model.load_cif(str(sample_structure_pair["model"]))
    data = ReflectionData()
    data.load_mtz(str(sample_structure_pair["reflections"]))

    generator = torch.Generator().manual_seed(0)
    frac = model.cell.cartesian_to_fractional(model.xyz()).detach()
    sites = frac[torch.randperm(frac.shape[0], generator=generator)[:300]]
    random_points = torch.rand(300, 3, generator=generator, dtype=frac.dtype)

    diagnostic = OccupancyFloorDiagnostic(model_dark=model, model_light=model)
    rho_sites = diagnostic.compute_density_at_positions(model, sites, data.hkl)
    rho_random = diagnostic.compute_density_at_positions(
        model, random_points.to(frac.device), data.hkl
    )

    assert rho_sites.mean() > rho_random.mean() + 5 * rho_random.std() / 300**0.5
    assert (rho_sites < 0).float().mean() < 0.1
