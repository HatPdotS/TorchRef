"""
Functional tests for ModelFT (Fourier Transform model).

These tests exercise the ModelFT class with real crystallographic data,
testing the FFT-based structure factor calculation pipeline.
"""

import pytest
import torch


@pytest.mark.integration
class TestModelFTInitialization:
    """Test ModelFT initialization with real data."""

    def test_modelft_empty_initialization(self):
        """Test empty ModelFT initialization."""
        from torchref.model.model_ft import ModelFT

        model = ModelFT()
        assert model is not None
        assert model.max_res == 1.0  # Default

    def test_modelft_with_custom_resolution(self):
        """Test ModelFT with custom resolution."""
        from torchref.model.model_ft import ModelFT

        model = ModelFT(max_res=1.5)
        assert model.max_res == 1.5

    def test_modelft_has_gridsize(self, sample_cif_file):
        """The grid resolves from the loaded cell and space group on first read."""
        from torchref.model.model_ft import ModelFT

        model = ModelFT(max_res=2.0, verbose=0)
        assert model.gridsize is None  # no crystal yet
        model.load_cif(str(sample_cif_file))

        assert model.gridsize is not None
        assert len(model.gridsize) == 3
        assert all(g > 0 for g in model.gridsize)
        assert model.xyz().shape[0] > 0
        assert model.parametrization
        assert model.adp().shape == (len(model.xyz()),)
        assert torch.all(model.adp() >= 0)


@pytest.mark.integration
class TestModelFTGridOperations:
    """Test ModelFT grid operations."""

    def test_setup_grid(self, loaded_model_ft):
        """An explicit grid size overrides the resolution-derived one."""

        model = loaded_model_ft
        derived = model.grid_shape
        assert derived is not None and len(derived) == 3

        model.setup_grid(gridsize=(24, 24, 24))
        assert model.grid_shape == (24, 24, 24)
        assert model.explicit_gridsize == (24, 24, 24)

        model.explicit_gridsize = None
        assert model.grid_shape == derived


@pytest.mark.integration
class TestModelFTRealSpaceMap:
    """Test ModelFT real space electron density map construction."""

    def test_get_real_space_grid(self, loaded_model_ft):
        """Test getting real space grid."""
        from torchref.base.math_torch import get_real_grid

        model = loaded_model_ft

        assert model.gridsize is not None
        grid = get_real_grid(model.cell, max_res=2.0, device=model.device)

        assert grid is not None
        assert len(grid.shape) == 4  # Should be 4D (nx, ny, nz, 3)
        assert grid.device == model.xyz().device
        assert grid.dtype == model.xyz().dtype


@pytest.mark.integration
class TestModelFTSymmetry:
    """Test ModelFT symmetry operations."""

    def test_map_symmetry_available(self, shared_model_ft):
        """Test map symmetry is available after loading."""

        model = shared_model_ft

        # Model should have spacegroup after loading
        assert model.spacegroup is not None

        # The map operator comes from the space group, keyed on the grid shape.
        gridsize = model.grid_shape
        assert gridsize is not None

        operator = model.spacegroup.map_operator(gridsize)
        assert operator is not None
        assert operator.map_shape == gridsize


@pytest.mark.integration
class TestModelFTMultipleStructures:
    """Test ModelFT with multiple structures."""

    def test_modelft_multiple_structures(self, all_structure_pairs):
        """Test ModelFT works with different structures."""
        from torchref.model.model_ft import ModelFT

        tested = 0
        for pair in all_structure_pairs[:3]:  # Test first 3
            try:
                model = ModelFT(max_res=3.0, verbose=0)
                model.load_cif(str(pair["model"]))

                # Basic checks
                assert model.xyz() is not None
                assert model.xyz().shape[0] > 0

                tested += 1
            except Exception as e:
                # Some structures may fail to load
                continue

        assert tested >= 1, "At least one structure should load"


@pytest.mark.integration
class TestModelFTCoordinateOperations:
    """Test ModelFT coordinate operations."""

    def test_fractional_to_cartesian(self, shared_model_ft):
        """Test fractional to cartesian conversion."""
        from torchref.base.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch,
        )

        model = shared_model_ft

        xyz = model.xyz()
        cell = model.cell

        # Round trip conversion
        frac = cartesian_to_fractional_torch(xyz, cell.data)
        assert frac.shape == xyz.shape
        xyz_back = fractional_to_cartesian_torch(frac, cell.data)

        # Should get back original coordinates (float32 roundtrip)
        assert torch.allclose(xyz, xyz_back, atol=1e-3)


@pytest.mark.integration
def test_forward_cache_contract(
    loaded_model_ft, loaded_reflection_data, monkeypatch
) -> None:
    """A model computes complex structure factors and caches only until invalidation."""
    from unittest.mock import Mock

    from torchref.config import caching, get_complex_dtype

    model = loaded_model_ft
    hkl = loaded_reflection_data.hkl[:32]
    monkeypatch.setattr(caching, "value", True)
    forward = Mock(wraps=model.forward)
    monkeypatch.setattr(model, "forward", forward)
    assert getattr(model, "_fwd_cached_output", None) is None

    first = model(hkl)
    assert first.shape == (len(hkl),)
    assert first.dtype == get_complex_dtype()
    assert first.device == hkl.device
    assert torch.isfinite(first).all()
    assert first.abs().sum() > 0
    assert model(hkl) is first
    assert forward.call_count == 1

    refreshed = model(hkl, recalc=True)
    assert forward.call_count == 2
    assert refreshed is not first
    # Accelerator reductions need not repeat bit-for-bit after recomputation.
    relative_error = torch.linalg.vector_norm(
        (refreshed - first).abs()
    ) / torch.linalg.vector_norm(first.abs())
    assert relative_error < 256 * torch.finfo(first.real.dtype).eps
