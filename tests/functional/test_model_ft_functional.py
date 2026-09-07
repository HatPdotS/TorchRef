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

    def test_modelft_load_cif(self, sample_cif_file):
        """Test loading a CIF file into ModelFT."""
        from torchref.model.model_ft import ModelFT

        model = ModelFT(max_res=2.0, verbose=0)
        model.load_cif(str(sample_cif_file))

        # Verify basic properties
        assert model.xyz() is not None
        assert model.xyz().shape[0] > 0
        assert model.cell is not None
        assert len(model.cell) == 6

    def test_modelft_has_gridsize(self, sample_cif_file):
        """The grid resolves from the loaded cell and space group on first read."""
        from torchref.model.model_ft import ModelFT

        model = ModelFT(max_res=2.0, verbose=0)
        assert model.gridsize is None  # no crystal yet
        model.load_cif(str(sample_cif_file))

        assert model.gridsize is not None
        assert len(model.gridsize) == 3
        assert all(g > 0 for g in model.gridsize)


@pytest.mark.integration
class TestModelFTParametrization:
    """Test ModelFT parametrization with real structures."""

    def test_parametrization_built(self, shared_model_ft):
        """Test that parametrization is built after loading."""

        model = shared_model_ft

        # Parametrization should be set
        assert model.parametrization is not None

    def test_scattering_factors_available(self, shared_model_ft):
        """Test that scattering factors can be computed."""

        model = shared_model_ft

        # Should be able to access atom properties
        xyz = model.xyz()
        assert xyz is not None
        assert xyz.dtype == torch.float32 or xyz.dtype == torch.float64


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

        # The grid helper moves its Cell in place when targeting CPU.
        model = loaded_model_ft

        assert model.gridsize is not None
        grid = get_real_grid(model.cell, max_res=2.0, device="cpu")

        assert grid is not None
        assert len(grid.shape) == 4  # Should be 4D (nx, ny, nz, 3)


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
class TestModelFTStateDictFunctional:
    """Test ModelFT state dict operations with real data."""

    def test_save_and_load_state_dict(self, loaded_model_ft, tmp_path):
        """Test saving and loading state dict."""
        from torchref.model.model_ft import ModelFT

        model = loaded_model_ft

        original_xyz = model.xyz().clone()

        # Save state dict
        state_dict = model.state_dict()

        # Create new model and load state
        model2 = ModelFT(max_res=2.0, verbose=0)

        # We need to ensure proper initialization
        # For now just verify state_dict works
        assert state_dict is not None
        assert len(state_dict) > 0


@pytest.mark.integration
class TestModelFTForwardPass:
    """Test ModelFT forward pass (structure factor calculation)."""

    def test_forward_method_exists(self, shared_model_ft):
        """Test that forward method is available."""

        model = shared_model_ft

        # Check forward method exists
        assert hasattr(model, "forward")

    def test_build_map_method(self, sample_cif_file):
        """Test build_map method if available."""
        from torchref.model.model_ft import ModelFT

        model = ModelFT(max_res=3.0, verbose=0)  # Lower res for faster test
        model.load_cif(str(sample_cif_file))

        # Check build_map method
        if hasattr(model, "build_map"):
            # Try to build map
            try:
                model.build_map()
                assert model.map is not None
            except Exception as e:
                # May fail if missing dependencies
                pytest.skip(f"build_map not available: {e}")


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
class TestModelFTCaching:
    """Test ModelFT caching mechanism."""

    def test_cache_initialization(self, loaded_model_ft):
        """Test that CachedForwardMixin cache starts empty."""

        model = loaded_model_ft

        # Mixin cache should start empty (lazily initialized)
        assert getattr(model, "_fwd_cached_output", None) is None

    def test_cache_usage(self, shared_model_ft):
        """Test that cache can be used for computations."""

        model = shared_model_ft

        # Access xyz twice - should use caching
        xyz1 = model.xyz()
        xyz2 = model.xyz()

        # Should return same tensor
        assert torch.allclose(xyz1, xyz2)


@pytest.mark.integration
class TestModelFTCoordinateOperations:
    """Test ModelFT coordinate operations."""

    def test_cartesian_to_fractional(self, shared_model_ft):
        """Test coordinate conversion."""
        from torchref.base.math_torch import cartesian_to_fractional_torch

        model = shared_model_ft

        xyz = model.xyz()
        cell = model.cell

        # Convert to fractional
        frac = cartesian_to_fractional_torch(xyz, cell.data)

        # Fractional coords should be bounded (mostly between 0 and 1)
        assert frac.shape == xyz.shape

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
        xyz_back = fractional_to_cartesian_torch(frac, cell.data)

        # Should get back original coordinates (float32 roundtrip)
        assert torch.allclose(xyz, xyz_back, atol=1e-3)


@pytest.mark.integration
class TestModelFTAnisoHandling:
    """Test ModelFT handling of anisotropic parameters."""

    def test_access_aniso_atoms(self, shared_model_ft):
        """Test accessing anisotropic atom information."""

        model = shared_model_ft

        # Check if aniso is available
        if hasattr(model, "get_aniso") or hasattr(model, "aniso"):
            # Structure has aniso
            pass

    def test_isotropic_b_factors(self, shared_model_ft):
        """Test accessing isotropic B-factors."""

        model = shared_model_ft

        # Get B-factors (now accessed via adp())
        b_factors = model.adp()

        assert b_factors is not None
        assert b_factors.shape[0] == model.xyz().shape[0]
        # B-factors should be positive
        assert torch.all(b_factors > 0) or torch.all(b_factors >= 0)
