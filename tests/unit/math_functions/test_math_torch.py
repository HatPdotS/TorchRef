"""
Unit tests for torchref.math_functions.math_torch

Tests coordinate transformations, structure factor calculations, and other
mathematical operations in PyTorch.
"""

import pytest
import torch
import numpy as np


class TestCoordinateTransformations:
    """Tests for coordinate transformation functions."""

    @pytest.mark.unit
    def test_cartesian_to_fractional_orthorhombic(self, mock_unit_cell, random_coordinates):
        """Test Cartesian to fractional conversion for orthorhombic cell."""
        from torchref.math_functions.math_torch import cartesian_to_fractional_torch
        
        coords = random_coordinates(n_atoms=5)
        cell = mock_unit_cell
        
        frac = cartesian_to_fractional_torch(coords, cell)
        
        assert frac.shape == coords.shape
        assert torch.all(torch.isfinite(frac))
        # For orthorhombic cell, fractional = cartesian / cell_length
        expected_frac = coords / cell[:3]
        assert torch.allclose(frac, expected_frac, rtol=1e-5)

    @pytest.mark.unit
    def test_fractional_to_cartesian_orthorhombic(self, mock_unit_cell, random_fractional_coordinates):
        """Test fractional to Cartesian conversion for orthorhombic cell."""
        from torchref.math_functions.math_torch import fractional_to_cartesian_torch
        
        frac = random_fractional_coordinates(n_atoms=5)
        cell = mock_unit_cell
        
        cart = fractional_to_cartesian_torch(frac, cell)
        
        assert cart.shape == frac.shape
        assert torch.all(torch.isfinite(cart))
        # For orthorhombic cell, cartesian = fractional * cell_length
        expected_cart = frac * cell[:3]
        assert torch.allclose(cart, expected_cart, rtol=1e-5)

    @pytest.mark.unit
    def test_coordinate_roundtrip(self, mock_unit_cell, random_coordinates):
        """Test that cart->frac->cart gives back original coordinates."""
        from torchref.math_functions.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch
        )
        
        coords = random_coordinates(n_atoms=10)
        cell = mock_unit_cell
        
        frac = cartesian_to_fractional_torch(coords, cell)
        cart_back = fractional_to_cartesian_torch(frac, cell)
        
        assert torch.allclose(coords, cart_back, rtol=1e-6)

    @pytest.mark.unit
    def test_coordinate_roundtrip_triclinic(self, mock_unit_cell_triclinic, random_coordinates):
        """Test roundtrip for non-orthorhombic (triclinic) cell."""
        from torchref.math_functions.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch
        )
        
        coords = random_coordinates(n_atoms=10).to(torch.float64)
        cell = mock_unit_cell_triclinic
        
        frac = cartesian_to_fractional_torch(coords, cell)
        cart_back = fractional_to_cartesian_torch(frac, cell)
        
        assert torch.allclose(coords, cart_back, rtol=1e-6)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_coordinate_transforms_gpu(self, mock_unit_cell, random_coordinates, gpu_device):
        """Test coordinate transformations on GPU."""
        from torchref.math_functions.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch
        )
        
        coords = random_coordinates(n_atoms=100).to(gpu_device)
        cell = mock_unit_cell.to(gpu_device)
        
        frac = cartesian_to_fractional_torch(coords, cell)
        cart_back = fractional_to_cartesian_torch(frac, cell)
        
        assert frac.device.type == "cuda"
        assert cart_back.device.type == "cuda"


class TestRFactorCalculations:
    """Tests for R-factor calculation."""

    @pytest.mark.unit
    def test_rfactor_identical(self, mock_fobs):
        """R-factor should be 0 for identical Fobs and Fcalc."""
        from torchref.math_functions.math_torch import get_rfactor_torch
        
        fobs = mock_fobs(n_reflections=100)
        fcalc = fobs.clone()
        
        rfactor = get_rfactor_torch(fobs, fcalc)
        
        assert torch.isclose(rfactor, torch.tensor(0.0, dtype=rfactor.dtype), atol=1e-6)

    @pytest.mark.unit
    def test_rfactor_scaled(self, mock_fobs):
        """R-factor calculation with scaled Fcalc."""
        from torchref.math_functions.math_torch import get_rfactor_torch
        
        fobs = mock_fobs(n_reflections=100)
        # Fcalc is 10% smaller than Fobs
        fcalc = fobs * 0.9
        
        rfactor = get_rfactor_torch(fobs, fcalc)
        
        # R = sum(|Fobs - Fcalc|) / sum(Fobs) = sum(0.1*Fobs) / sum(Fobs) = 0.1
        assert torch.isclose(rfactor, torch.tensor(0.1, dtype=rfactor.dtype), rtol=1e-5)

    @pytest.mark.unit
    def test_rfactor_complex(self, mock_structure_factors):
        """R-factor with complex structure factors (uses absolute values)."""
        from torchref.math_functions.math_torch import get_rfactor_torch
        
        fcalc = mock_structure_factors(n_reflections=100)
        fobs = torch.abs(fcalc)  # Use |Fcalc| as Fobs
        
        rfactor = get_rfactor_torch(fobs, fcalc)
        
        assert torch.isclose(rfactor, torch.tensor(0.0, dtype=rfactor.dtype), atol=1e-6)

    @pytest.mark.unit
    def test_rfactor_positive(self, mock_fobs):
        """R-factor should always be non-negative."""
        from torchref.math_functions.math_torch import get_rfactor_torch
        
        fobs = mock_fobs(n_reflections=100)
        fcalc = mock_fobs(n_reflections=100, seed=123)  # Different values
        
        rfactor = get_rfactor_torch(fobs, fcalc)
        
        assert rfactor >= 0


class TestOutlierDetection:
    """Tests for outlier detection functions."""

    @pytest.mark.unit
    def test_calc_outliers_no_outliers(self, mock_fobs):
        """No outliers when Fobs equals Fcalc."""
        from torchref.math_functions.math_torch import calc_outliers
        
        fobs = mock_fobs(n_reflections=100)
        fcalc = fobs.clone()
        
        outliers = calc_outliers(fobs, fcalc, z=3.0)
        
        assert outliers.sum() == 0

    @pytest.mark.unit
    def test_calc_outliers_with_extreme(self, mock_fobs):
        """Detect extreme outliers."""
        from torchref.math_functions.math_torch import calc_outliers
        
        fobs = mock_fobs(n_reflections=100)
        fcalc = fobs.clone()
        # Add extreme outlier
        fcalc[0] = fobs[0] * 10  # 10x different
        
        outliers = calc_outliers(fobs, fcalc, z=3.0)
        
        # Should detect at least the extreme outlier
        assert outliers[0] == True


class TestGridFunctions:
    """Tests for grid-related functions."""

    @pytest.mark.unit
    def test_get_real_grid_shape(self, mock_unit_cell):
        """Test real grid generation has correct shape."""
        from torchref.math_functions.math_torch import get_real_grid
        
        cell = mock_unit_cell
        grid = get_real_grid(cell, max_res=2.0)
        
        # Should be 3D grid with xyz in last dimension
        assert len(grid.shape) == 4
        assert grid.shape[-1] == 3

    @pytest.mark.unit
    def test_get_real_grid_custom_size(self, mock_unit_cell):
        """Test real grid with custom grid size."""
        from torchref.math_functions.math_torch import get_real_grid
        
        cell = mock_unit_cell
        gridsize = [10, 12, 14]
        grid = get_real_grid(cell, gridsize=gridsize)
        
        assert grid.shape[0] == 10
        assert grid.shape[1] == 12
        assert grid.shape[2] == 14
        assert grid.shape[3] == 3

    @pytest.mark.unit
    def test_find_grid_size(self, mock_unit_cell):
        """Test automatic grid size calculation."""
        from torchref.math_functions.math_torch import find_grid_size
        
        cell = mock_unit_cell
        grid_size = find_grid_size(cell, max_res=1.0)
        
        assert grid_size.shape == (3,)
        assert torch.all(grid_size > 0)


class TestTransformationMatrices:
    """Tests for transformation matrix operations."""

    @pytest.mark.unit
    def test_apply_transformation_identity(self, random_coordinates):
        """Identity transformation should not change coordinates."""
        from torchref.math_functions.math_torch import apply_transformation
        
        coords = random_coordinates(n_atoms=10)
        identity = torch.eye(3, 4, dtype=coords.dtype)  # 3x4 matrix with identity rotation, zero translation
        
        transformed = apply_transformation(coords, identity)
        
        assert torch.allclose(coords, transformed, rtol=1e-5)

    @pytest.mark.unit
    def test_apply_transformation_translation(self, random_coordinates):
        """Test pure translation."""
        from torchref.math_functions.math_torch import apply_transformation
        
        coords = random_coordinates(n_atoms=10)
        translation = torch.tensor([1.0, 2.0, 3.0], dtype=coords.dtype)
        transform = torch.eye(3, 4, dtype=coords.dtype)
        transform[:, 3] = translation
        
        transformed = apply_transformation(coords, transform)
        
        expected = coords + translation
        assert torch.allclose(transformed, expected, rtol=1e-5)


class TestAlignment:
    """Tests for structure alignment functions."""

    @pytest.mark.unit
    def test_align_identical(self, random_coordinates):
        """Aligning identical structures should give RMSD ~0."""
        from torchref.math_functions.math_torch import align_torch
        
        coords1 = random_coordinates(n_atoms=20).to(torch.float64)
        coords2 = coords1.clone()
        
        aligned = align_torch(coords1, coords2)
        
        rmsd = torch.sqrt(torch.mean(torch.sum((coords1 - aligned) ** 2, dim=1)))
        assert rmsd < 1e-6

    @pytest.mark.unit
    def test_align_translated(self, random_coordinates):
        """Alignment should handle pure translation."""
        from torchref.math_functions.math_torch import align_torch
        
        coords1 = random_coordinates(n_atoms=20).to(torch.float64)
        translation = torch.tensor([5.0, -3.0, 2.0], dtype=torch.float64)
        coords2 = coords1 + translation
        
        aligned = align_torch(coords1, coords2)
        
        rmsd = torch.sqrt(torch.mean(torch.sum((coords1 - aligned) ** 2, dim=1)))
        assert rmsd < 1e-5


class TestSmallestDiff:
    """Tests for periodic boundary difference calculations."""

    @pytest.mark.unit
    def test_smallest_diff_no_wrap(self, mock_unit_cell):
        """Test smallest difference without wrapping."""
        from torchref.math_functions.math_torch import smallest_diff
        from torchref.math_functions.math_numpy import get_inv_fractional_matrix, get_fractional_matrix
        
        cell = mock_unit_cell.numpy()
        inv_frac = torch.tensor(get_inv_fractional_matrix(cell), dtype=torch.float64)
        frac = torch.tensor(get_fractional_matrix(cell), dtype=torch.float64)
        
        # Small differences that don't need wrapping
        diff = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float64)
        
        result = smallest_diff(diff, inv_frac, frac)
        
        # Should return squared distances
        expected = torch.sum(diff ** 2, dim=-1)
        assert torch.allclose(result, expected, rtol=1e-5)


class TestRotation:
    """Tests for rotation functions."""

    @pytest.mark.unit
    def test_rotate_coords_identity(self, random_coordinates):
        """Zero rotation should not change coordinates."""
        from torchref.math_functions.math_torch import rotate_coords_torch
        
        coords = random_coordinates(n_atoms=10)
        
        # Pass tensors for phi and rho
        phi = torch.tensor(0.0, dtype=coords.dtype)
        rho = torch.tensor(0.0, dtype=coords.dtype)
        rotated = rotate_coords_torch(coords, phi=phi, rho=rho)
        
        assert torch.allclose(coords, rotated, rtol=1e-5)

    @pytest.mark.unit
    def test_rotate_coords_preserves_distances(self, random_coordinates):
        """Rotation should preserve pairwise distances."""
        from torchref.math_functions.math_torch import rotate_coords_torch
        
        coords = random_coordinates(n_atoms=10)
        
        # Calculate original pairwise distances
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)
        orig_dist = torch.sqrt(torch.sum(diff ** 2, dim=-1))
        
        # Rotate with tensor angles
        phi = torch.tensor(45.0, dtype=coords.dtype)
        rho = torch.tensor(30.0, dtype=coords.dtype)
        rotated = rotate_coords_torch(coords, phi=phi, rho=rho)
        
        # Calculate rotated pairwise distances
        diff_rot = rotated.unsqueeze(0) - rotated.unsqueeze(1)
        rot_dist = torch.sqrt(torch.sum(diff_rot ** 2, dim=-1))
        
        assert torch.allclose(orig_dist, rot_dist, rtol=1e-5)
