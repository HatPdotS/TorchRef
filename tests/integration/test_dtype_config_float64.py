"""Regression tests for float64-config dtype consistency.

Several code paths used to hardcode float32/complex128 (or call ``.float()``),
which raised a dtype mismatch in ``scatter_add``/``matmul`` when the library
was configured for float64 (``torchref.config.dtypes.float = torch.float64``).
These tests run the affected paths under a float64 config and assert they
neither raise nor silently downcast. See TORCHREF_AUDIT.md cluster 1.
"""

import pytest
import torch



@pytest.mark.unit
def test_translation_phases_complex_dtype_float64(double_cpu):
    """Symmetry.phase_factors must honor the configured complex dtype."""
    from torchref.symmetry import SpaceGroup

    # P21 gives two operations, one carrying a half translation.
    sym = SpaceGroup("P 21")
    hkl = torch.tensor([[1, 0, 0], [2, 1, 0], [0, 0, 3]])

    phases = sym.phase_factors(hkl)

    # Must not narrow to complex64 under a float64 configuration.
    assert phases.dtype == torch.complex128
    assert phases.shape == (2, 3)
    assert torch.isfinite(phases.real).all()


@pytest.mark.integration
def test_scaler_binwise_mean_intensity_float64(double_cpu, sample_structure_pair):
    """Scaler.get_binwise_mean_intensity used to crash in scatter_add under float64."""
    from torchref.io import ReflectionData
    from torchref.model.model_ft import ModelFT
    from torchref.scaling.scaler import Scaler

    model = ModelFT()
    model.load_cif(str(sample_structure_pair["model"]))

    data = ReflectionData()
    data.load_mtz(str(sample_structure_pair["reflections"]))

    scaler = Scaler(model=model, data=data, nbins=10, verbose=0)

    hkl = data()[0]
    fcalc = model(hkl)
    assert fcalc.dtype == torch.complex128

    # Pre-fix this raised: scatter_add float32 accumulator vs float64 source.
    mean_obs, mean_calc, mean_res = scaler.get_binwise_mean_intensity(fcalc)

    assert mean_obs.dtype == torch.float64
    assert mean_calc.dtype == torch.float64
    assert torch.isfinite(mean_obs).all()


@pytest.mark.integration
def test_occupancy_floor_density_matmul_float64(double_cpu, sample_structure_pair):
    """compute_density_at_positions hardcoded hkl.T.float(); matmul raised under float64."""
    from torchref.io import ReflectionData
    from torchref.model.model_ft import ModelFT
    from torchref.experimental.targets.occupancy_floor_diagnostic import (
        OccupancyFloorDiagnostic,
    )

    model = ModelFT()
    model.load_cif(str(sample_structure_pair["model"]))

    data = ReflectionData()
    data.load_mtz(str(sample_structure_pair["reflections"]))

    # Fractional positions in the configured (float64) dtype.
    positions = model.cell.cartesian_to_fractional(model.xyz())
    assert positions.dtype == torch.float64

    hkl = data()[0]

    diagnostic = OccupancyFloorDiagnostic(model_dark=model, model_light=model)
    # Pre-fix this raised: float64 positions @ float32 hkl.T.
    density = diagnostic.compute_density_at_positions(model, positions, hkl)

    assert density.dtype == torch.float64
    assert density.shape[0] == positions.shape[0]
    assert torch.isfinite(density).all()
