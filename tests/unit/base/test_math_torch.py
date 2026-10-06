"""
Unit tests for torchref.base.math_torch

Tests coordinate transformations, structure factor calculations, and other
mathematical operations in PyTorch.
"""

import pytest
import torch
import numpy as np


class TestCoordinateTransformations:
    """Tests for coordinate transformation functions."""

    @pytest.mark.unit
    def test_cartesian_to_fractional_orthorhombic(self, mock_cell, random_coordinates):
        """Test Cartesian to fractional conversion for orthorhombic cell."""
        from torchref.base.math_torch import cartesian_to_fractional_torch

        coords = random_coordinates(n_atoms=5)
        cell = mock_cell
        
        frac = cartesian_to_fractional_torch(coords, cell)
        
        assert frac.shape == coords.shape
        assert torch.all(torch.isfinite(frac))
        # For orthorhombic cell, fractional = cartesian / cell_length
        expected_frac = coords / cell[:3]
        assert torch.allclose(frac, expected_frac, rtol=1e-5)

    @pytest.mark.unit
    def test_fractional_to_cartesian_orthorhombic(self, mock_cell, random_fractional_coordinates):
        """Test fractional to Cartesian conversion for orthorhombic cell."""
        from torchref.base.math_torch import fractional_to_cartesian_torch

        frac = random_fractional_coordinates(n_atoms=5)
        cell = mock_cell
        
        cart = fractional_to_cartesian_torch(frac, cell)
        
        assert cart.shape == frac.shape
        assert torch.all(torch.isfinite(cart))
        # For orthorhombic cell, cartesian = fractional * cell_length
        expected_cart = frac * cell[:3]
        assert torch.allclose(cart, expected_cart, rtol=1e-5)

    @pytest.mark.unit
    def test_coordinate_roundtrip(self, mock_cell, random_coordinates):
        """Test that cart->frac->cart gives back original coordinates."""
        from torchref.base.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch
        )

        coords = random_coordinates(n_atoms=10)
        cell = mock_cell
        
        frac = cartesian_to_fractional_torch(coords, cell)
        cart_back = fractional_to_cartesian_torch(frac, cell)
        
        assert torch.allclose(coords, cart_back, rtol=1e-6)

    @pytest.mark.unit
    def test_coordinate_roundtrip_triclinic(self, mock_cell_triclinic, random_coordinates):
        """Test roundtrip for non-orthorhombic (triclinic) cell."""
        from torchref.base.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch
        )

        coords = random_coordinates(n_atoms=10)
        cell = mock_cell_triclinic
        # rtol relaxed slightly because default dtype may be float32.
        frac = cartesian_to_fractional_torch(coords, cell)
        cart_back = fractional_to_cartesian_torch(frac, cell)

        assert torch.allclose(coords, cart_back, rtol=1e-5, atol=1e-5)

    @pytest.mark.unit
    @pytest.mark.gpu
    def test_coordinate_transforms_gpu(self, mock_cell, random_coordinates, gpu_device):
        """Test coordinate transformations on GPU."""
        from torchref.base.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch
        )

        coords = random_coordinates(n_atoms=100).to(gpu_device)
        cell = mock_cell.to(gpu_device)
        
        frac = cartesian_to_fractional_torch(coords, cell)
        cart_back = fractional_to_cartesian_torch(frac, cell)
        
        assert frac.device.type == gpu_device.type
        assert cart_back.device.type == gpu_device.type


class TestGridFunctions:
    """Tests for grid-related functions."""

    @pytest.mark.unit
    def test_get_real_grid_shape(self, mock_cell):
        """Test real grid generation has correct shape."""
        from torchref.base.math_torch import get_real_grid
        from torchref.symmetry import Cell

        cell = mock_cell
        grid = get_real_grid(cell, gridsize=Cell(cell).compute_grid_size(2.0))
        
        # Should be 3D grid with xyz in last dimension
        assert len(grid.shape) == 4
        assert grid.shape[-1] == 3

    @pytest.mark.unit
    def test_get_real_grid_custom_size(self, mock_cell):
        """Test real grid with custom grid size."""
        from torchref.base.math_torch import get_real_grid

        cell = mock_cell
        gridsize = [10, 12, 14]
        grid = get_real_grid(cell, gridsize=gridsize)
        
        assert grid.shape[0] == 10
        assert grid.shape[1] == 12
        assert grid.shape[2] == 14
        assert grid.shape[3] == 3

    @pytest.mark.unit
    def test_place_on_grid_adds_only_missing_friedel_mates(self):
        """A mate already in the input keeps its own value; a lone h gets conj at -h."""
        from torchref.base.reciprocal.grid_operations import place_on_grid

        F = torch.tensor(2.0 + 1.0j)
        hkl = torch.tensor([[1, 0, 0], [-1, 0, 0]])
        pair = place_on_grid(hkl, torch.stack([F, F.conj()]), (8, 8, 8))
        assert pair[1, 0, 0] == F and pair[-1, 0, 0] == F.conj()

        lone = place_on_grid(hkl[:1], F.reshape(1), (8, 8, 8))
        assert lone[1, 0, 0] == F and lone[-1, 0, 0] == F.conj()


class TestFourierTransforms:
    """Tests for the crystallographic fft/ifft pair."""

    @pytest.mark.unit
    def test_volume_scale_ignores_the_batch_axis(self):
        """Each map of a (B, Nx, Ny, Nz) batch is scaled as if transformed alone."""
        from torchref.base.fourier import fft, ifft

        rho = torch.randn(3, 8, 10, 12, generator=torch.Generator().manual_seed(0))
        F = ifft(rho, 1000.0)
        assert torch.allclose(F[0], ifft(rho[0], 1000.0))
        assert torch.allclose(fft(F, 1000.0)[0], fft(F[0], 1000.0))


class TestTransformationMatrices:
    """Tests for transformation matrix operations."""

    @pytest.mark.unit
    def test_apply_transformation_identity(self, random_coordinates):
        """Identity transformation should not change coordinates."""
        from torchref.base.math_torch import apply_transformation
        
        coords = random_coordinates(n_atoms=10)
        identity = torch.eye(3, 4, dtype=coords.dtype)  # 3x4 matrix with identity rotation, zero translation
        
        transformed = apply_transformation(coords, identity)
        
        assert torch.allclose(coords, transformed, rtol=1e-5)

    @pytest.mark.unit
    def test_apply_transformation_translation(self, random_coordinates):
        """Test pure translation."""
        from torchref.base.math_torch import apply_transformation
        
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
        from torchref.base.math_torch import align_torch
        
        coords1 = random_coordinates(n_atoms=20).to(torch.float64)
        coords2 = coords1.clone()
        
        aligned = align_torch(coords1, coords2)
        
        rmsd = torch.sqrt(torch.mean(torch.sum((coords1 - aligned) ** 2, dim=1)))
        assert rmsd < 1e-6

    @pytest.mark.unit
    def test_align_translated(self, random_coordinates):
        """Alignment should handle pure translation."""
        from torchref.base.math_torch import align_torch
        
        coords1 = random_coordinates(n_atoms=20).to(torch.float64)
        translation = torch.tensor([5.0, -3.0, 2.0], dtype=torch.float64)
        coords2 = coords1 + translation
        
        aligned = align_torch(coords1, coords2)
        
        rmsd = torch.sqrt(torch.mean(torch.sum((coords1 - aligned) ** 2, dim=1)))
        assert rmsd < 1e-5


class TestSmallestDiff:
    """Tests for periodic boundary difference calculations."""

    @pytest.mark.unit
    def test_smallest_diff_aniso_leaves_its_input_alone(self):
        """(9, 0, 0) in a 10 Å cube maps to (-1, 0, 0); the input is not modified."""
        from torchref.base.coordinates import (
            get_fractional_matrix,
            get_inv_fractional_matrix_torch,
            smallest_diff,
            smallest_diff_aniso,
        )

        cell = torch.tensor([10.0, 10.0, 10.0, 90.0, 90.0, 90.0])
        inv_frac = get_inv_fractional_matrix_torch(cell)
        frac = get_fractional_matrix(cell)
        diff = torch.tensor([[9.0, 0.0, 0.0]])

        image = smallest_diff_aniso(diff, inv_frac, frac)
        assert torch.allclose(image, torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-5)
        assert torch.equal(diff, torch.tensor([[9.0, 0.0, 0.0]]))
        assert torch.allclose(smallest_diff(diff, inv_frac, frac), torch.ones(1))

    @pytest.mark.unit
    def test_smallest_diff_no_wrap(self, mock_cell):
        """Test smallest difference without wrapping."""
        from torchref.base.math_torch import smallest_diff
        from torchref.base.coordinates import (
            get_inv_fractional_matrix_torch,
            get_fractional_matrix,
        )

        cell = mock_cell.double()
        inv_frac = get_inv_fractional_matrix_torch(cell)
        frac = get_fractional_matrix(cell)
        
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
        from torchref.base.math_torch import rotate_coords_torch
        
        coords = random_coordinates(n_atoms=10)
        
        # Pass tensors for phi and rho
        phi = torch.tensor(0.0, dtype=coords.dtype)
        rho = torch.tensor(0.0, dtype=coords.dtype)
        rotated = rotate_coords_torch(coords, phi=phi, rho=rho)
        
        assert torch.allclose(coords, rotated, rtol=1e-5)

    @pytest.mark.unit
    def test_rotate_coords_preserves_distances(self, random_coordinates):
        """Rotation should preserve pairwise distances."""
        from torchref.base.math_torch import rotate_coords_torch
        
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
