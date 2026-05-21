"""
Functional tests for math functions with real crystallographic data.

These tests exercise math functions from math_torch.py with realistic inputs.
"""
import pytest
import torch
import numpy as np


@pytest.mark.integration
class TestStructureFactorCalculations:
    """Test structure factor calculation functions."""

    def test_get_scattering_vectors(self, model_and_data):
        """Test scattering vector calculation."""
        from torchref.base.math_torch import get_scattering_vectors
        
        data = model_and_data["data"]
        model = model_and_data["model"]
        
        hkl = data.hkl
        cell = model.cell
        
        # Cell is a dataclass wrapping a tensor; use cell.data to get the
        # underlying tensor while preserving its device.
        cell_double = cell.data.double()
        
        s_vectors = get_scattering_vectors(hkl.double(), cell_double)
        
        assert s_vectors.shape[0] == hkl.shape[0]
        assert s_vectors.shape[1] == 3

    def test_get_d_spacing(self, model_and_data):
        """Test d-spacing calculation."""
        from torchref.base.math_torch import get_d_spacing
        
        data = model_and_data["data"]
        model = model_and_data["model"]
        
        hkl = data.hkl
        cell = model.cell
        
        cell_double = cell.data.double()

        d = get_d_spacing(hkl.double(), cell_double)
        
        # d-spacing should be positive
        assert torch.all(d > 0)
        
        # d should be less than cell size
        max_cell = cell_double[:3].max()
        assert torch.all(d <= max_cell + 1e-6)

    def test_reciprocal_basis_matrix(self, model_and_data):
        """Test reciprocal basis matrix calculation."""
        from torchref.base.math_torch import reciprocal_basis_matrix
        
        model = model_and_data["model"]
        cell = model.cell
        
        cell_double = cell.data.double()

        recB = reciprocal_basis_matrix(cell_double)
        
        assert recB.shape == (3, 3)
        # Should be non-singular
        det = torch.det(recB)
        assert torch.abs(det) > 1e-10


@pytest.mark.integration
class TestRfactorFunctions:
    """Test R-factor calculation functions."""

    def test_rfactor_basic(self):
        """Test basic R-factor calculation."""
        from torchref.base.math_torch import get_rfactor_torch
        
        fobs = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        fcalc = torch.tensor([1.1, 2.0, 2.9, 4.1], dtype=torch.float32)
        
        r = get_rfactor_torch(fobs, fcalc)
        
        assert r >= 0
        assert r <= 1.0

    def test_get_rfactors(self, model_and_data):
        """Test R-factor calculation with real data."""
        from torchref.base.math_torch import get_rfactors
        
        data = model_and_data["data"]
        
        fobs = data.F
        # Simulate fcalc close to fobs
        fcalc = fobs * (1 + 0.1 * torch.randn_like(fobs))
        fcalc = torch.abs(fcalc)
        
        # Create rfree mask
        rfree_mask = torch.rand(len(fobs)) > 0.05  # 5% test set
        
        r_work, r_free = get_rfactors(fobs, fcalc, rfree_mask)
        
        assert r_work >= 0
        assert r_free >= 0


@pytest.mark.integration
class TestCoordinateTransformations:
    """Test coordinate transformation functions."""

    def test_coordinate_roundtrip(self, model_and_data):
        """Test coordinate transformation roundtrip."""
        from torchref.base.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch
        )
        
        model = model_and_data["model"]
        xyz = model.xyz().double()
        cell = model.cell
        
        cell_double = cell.data.double()

        # Convert to fractional and back
        frac = cartesian_to_fractional_torch(xyz, cell_double)
        xyz_back = fractional_to_cartesian_torch(frac, cell_double)
        
        # Should match original
        assert torch.allclose(xyz, xyz_back, atol=1e-4)


@pytest.mark.integration
class TestNLLFunctions:
    """Test NLL functions from math_torch."""

    def test_nll_xray_basic(self):
        """Test basic NLL X-ray calculation."""
        from torchref.base.math_torch import nll_xray
        
        fobs = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        fcalc = torch.tensor([11.0, 19.0, 31.0], dtype=torch.float32)
        sigma = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        
        nll = nll_xray(fobs, fcalc, sigma)
        
        assert torch.all(torch.isfinite(nll))

    def test_nll_xray_sum(self):
        """Test NLL X-ray sum."""
        from torchref.base.math_torch import nll_xray_sum
        
        fobs = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        fcalc = torch.tensor([11.0, 19.0, 31.0], dtype=torch.float32)
        sigma = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        
        nll_total = nll_xray_sum(fobs, fcalc, sigma)
        
        assert torch.isfinite(nll_total)

    def test_log_loss(self):
        """Test log loss function."""
        from torchref.base.math_torch import log_loss
        
        fobs = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        fcalc = torch.tensor([11.0, 19.0, 31.0], dtype=torch.float32)
        sigma = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        
        loss = log_loss(fobs, fcalc, sigma)
        
        assert torch.all(torch.isfinite(loss))


@pytest.mark.integration
class TestFrenchWilson:
    """Test French-Wilson conversion functions."""

    def test_french_wilson_basic(self):
        """Test French-Wilson conversion with mock data."""
        from torchref.base.math_torch import french_wilson_conversion
        
        # Create mock intensity data
        Iobs = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32)
        sigma_I = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float32)
        
        # This may return F and sigma_F
        result = french_wilson_conversion(Iobs, sigma_I)
        
        # Should return something
        assert result is not None


@pytest.mark.integration
class TestGridFunctions:
    """Test grid-related functions."""

    def test_find_grid_size(self, model_and_data):
        """Test grid size calculation."""
        from torchref.base import find_grid_size
        
        model = model_and_data["model"]
        cell = model.cell
        
        gridsize = find_grid_size(cell, max_res=2.0)
        
        assert len(gridsize) == 3
        assert all(g > 0 for g in gridsize)

    def test_get_real_grid(self, model_and_data):
        """Test real space grid creation."""
        from torchref.base.math_torch import get_real_grid
        
        model = model_and_data["model"]
        cell = model.cell
        
        grid = get_real_grid(cell, max_res=3.0, device='cpu')
        
        assert grid is not None
        # Should be 4D (nx, ny, nz, 3)
        assert len(grid.shape) == 4
        assert grid.shape[-1] == 3


@pytest.mark.integration
class TestFFTFunctions:
    """Test FFT-related functions."""

    def test_fft_basic(self):
        """Test FFT function."""
        from torchref.base.math_torch import fft
        
        # Create simple 3D grid (real-valued)
        grid = torch.randn(8, 8, 8, dtype=torch.float32)
        
        result = fft(grid)
        
        assert result.shape == grid.shape

    def test_ifft_basic(self):
        """Test inverse FFT function."""
        from torchref.base.math_torch import ifft
        
        # Create simple 3D grid
        grid = torch.randn(8, 8, 8, dtype=torch.complex64)
        
        result = ifft(grid)
        
        assert result.shape == grid.shape

    def test_fft_output_valid(self):
        """Test FFT output is valid."""
        from torchref.base.math_torch import fft
        
        # Create test grid
        grid = torch.randn(8, 8, 8, dtype=torch.float32)
        
        # FFT should produce finite values
        result = fft(grid)
        assert torch.all(torch.isfinite(result))


@pytest.mark.integration
class TestMiscMathFunctions:
    """Test miscellaneous math functions."""

    def test_smallest_diff(self):
        """Test smallest difference calculation."""
        from torchref.base.math_torch import smallest_diff
        
        # Create difference vectors
        diff = torch.tensor([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=torch.float64)
        
        # Create simple identity matrices
        inv_frac = torch.eye(3, dtype=torch.float64)
        frac = torch.eye(3, dtype=torch.float64)
        
        result = smallest_diff(diff, inv_frac, frac)
        
        # Should return something valid
        assert result is not None
        assert torch.all(torch.isfinite(result))

    def test_rotation_function(self):
        """Test coordinate rotation."""
        from torchref.base.math_torch import rotate_coords_torch
        
        coords = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        phi = torch.tensor(0.0)  # No rotation (tensor form)
        rho = torch.tensor(0.0)
        
        rotated = rotate_coords_torch(coords, phi, rho)
        
        # Should be same as original for no rotation
        assert torch.allclose(coords, rotated, atol=1e-6)


