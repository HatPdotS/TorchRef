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
def test_occupancy_floor_density_matmul_float64(double_cpu, sample_structure_pair):
    """compute_density_at_positions hardcoded hkl.T.float(); matmul raised under float64."""
    from torchref.experimental.targets.occupancy_floor_diagnostic import (
        OccupancyFloorDiagnostic,
    )
    from torchref.io import ReflectionData
    from torchref.model.model_ft import ModelFT

    model = ModelFT()
    model.load_cif(str(sample_structure_pair["model"]))

    data = ReflectionData()
    data.load_mtz(str(sample_structure_pair["reflections"]))

    # Fractional positions in the configured (float64) dtype.
    positions = model.cell.cartesian_to_fractional(model.xyz())
    assert positions.dtype == torch.float64

    hkl = data.hkl

    diagnostic = OccupancyFloorDiagnostic(model_dark=model, model_light=model)
    # Pre-fix this raised: float64 positions @ float32 hkl.T.
    density = diagnostic.compute_density_at_positions(model, positions, hkl)

    assert density.dtype == torch.float64
    assert density.shape[0] == positions.shape[0]
    assert torch.isfinite(density).all()


@pytest.mark.integration
def test_disulfide_values_keep_float64(double_cpu, pdb_dir):
    """Disulfide targets reach the float64 restraints unrounded."""
    from torchref.model.model import Model

    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "3A5V.pdb"))
    entries = model.restraints.restraints

    for edge_type, reference, sigma in (
        ("bond", 2.031, 0.020),
        ("angle", 103.8, 1.8),
        ("torsion", 90.0, 10.0),
    ):
        group = entries[edge_type]["disulfide"]
        assert group["references"].dtype == torch.float64
        assert float((group["references"] - reference).abs().max()) < 1e-12
        assert float((group["sigmas"] - sigma).abs().max()) < 1e-12


@pytest.mark.integration
def test_torsion_wrap_keeps_float64(double_cpu, pdb_dir):
    """An n-fold torsion deviation folds by 2π/n in float64, not float32."""
    from torchref.model.model import Model

    model = Model(verbose=0)
    model.load_pdb(str(pdb_dir / "3A5V.pdb"))
    restraints = model.restraints
    xyz = model.xyz().detach()
    group = restraints.restraints["torsion"]["all"]

    deviations, _ = restraints.torsion_deviations_with_sigmas(xyz)
    calculated = restraints.torsions(group["indices"], xyz)
    diff = (calculated - group["references"]) * (torch.pi / 180.0)
    periodic = group["periods"] > 1
    half_step = torch.pi / group["periods"][periodic].to(diff.dtype)
    folded = torch.remainder(diff[periodic] + half_step, 2 * half_step) - half_step

    assert periodic.any() and deviations.dtype == torch.float64
    assert float((deviations[periodic] - folded).abs().max()) < 1e-12
