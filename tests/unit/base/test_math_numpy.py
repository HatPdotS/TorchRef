"""
Unit tests for torchref.base.math_numpy

Tests coordinate transformations and other mathematical operations in NumPy.
"""

import pytest
import numpy as np
import torch


class TestCoordinateTransformations:
    """Tests for coordinate transformation functions."""

    @pytest.mark.unit
    def test_get_fractional_matrix_orthorhombic(self, mock_cell):
        """Test fractional matrix for orthorhombic cell."""
        from torchref.base.math_numpy import get_fractional_matrix

        cell = mock_cell.numpy()
        B = get_fractional_matrix(cell)
        
        assert B.shape == (3, 3)
        # For orthorhombic, B should be diagonal
        assert np.isclose(B[0, 0], cell[0])  # a
        assert np.isclose(B[1, 1], cell[1])  # b
        assert np.isclose(B[2, 2], cell[2])  # c
        # Off-diagonal should be ~0 (tolerance accommodates float32 default dtype)
        assert np.allclose(B[0, 1:], 0, atol=1e-5)
        assert np.isclose(B[1, 0], 0, atol=1e-5)

    @pytest.mark.unit
    def test_get_inv_fractional_matrix(self, mock_cell):
        """Test inverse fractional matrix."""
        from torchref.base.math_numpy import (
            get_fractional_matrix,
            get_inv_fractional_matrix
        )

        cell = mock_cell.numpy()
        B = get_fractional_matrix(cell)
        B_inv = get_inv_fractional_matrix(cell)
        
        # B @ B_inv should be identity
        identity = B @ B_inv
        assert np.allclose(identity, np.eye(3), rtol=1e-6)

    @pytest.mark.unit
    def test_cartesian_to_fractional(self, mock_cell, random_coordinates):
        """Test Cartesian to fractional conversion."""
        from torchref.base.math_numpy import cartesian_to_fractional

        cell = mock_cell.numpy()
        coords = random_coordinates(n_atoms=10).numpy()
        
        frac = cartesian_to_fractional(coords, cell)
        
        assert frac.shape == coords.shape

    @pytest.mark.unit
    def test_fractional_to_cartesian(self, mock_cell, random_fractional_coordinates):
        """Test fractional to Cartesian conversion."""
        from torchref.base.math_numpy import fractional_to_cartesian

        cell = mock_cell.numpy()
        frac = random_fractional_coordinates(n_atoms=10).numpy()
        
        cart = fractional_to_cartesian(frac, cell)
        
        assert cart.shape == frac.shape

    @pytest.mark.unit
    def test_coordinate_roundtrip(self, mock_cell, random_coordinates):
        """Test roundtrip conversion cart->frac->cart."""
        from torchref.base.math_numpy import (
            cartesian_to_fractional,
            fractional_to_cartesian
        )

        cell = mock_cell.numpy()
        coords = random_coordinates(n_atoms=10).numpy()
        
        frac = cartesian_to_fractional(coords, cell)
        cart_back = fractional_to_cartesian(frac, cell)
        
        assert np.allclose(coords, cart_back, rtol=1e-6)


class TestScatteringVectors:
    """Tests for scattering vector calculations."""

    @pytest.mark.unit
    def test_reciprocal_basis_matrix_orthorhombic(self, mock_cell):
        """Test reciprocal basis matrix for orthorhombic cell."""
        from torchref.base.math_numpy import reciprocal_basis_matrix

        cell = mock_cell.numpy()
        recB = reciprocal_basis_matrix(cell)
        
        assert recB.shape == (3, 3)
        # For orthorhombic: a* = 1/a, etc.
        assert np.isclose(recB[0, 0], 1.0 / cell[0], rtol=1e-6)
        assert np.isclose(recB[1, 1], 1.0 / cell[1], rtol=1e-6)
        assert np.isclose(recB[2, 2], 1.0 / cell[2], rtol=1e-6)

    @pytest.mark.unit
    def test_get_scattering_vectors(self, mock_cell, mock_hkl_indices):
        """Test scattering vector calculation."""
        from torchref.base.math_numpy import get_scattering_vectors

        cell = mock_cell.numpy()
        hkl = mock_hkl_indices(n_reflections=50).numpy()
        
        s = get_scattering_vectors(hkl, cell)
        
        assert s.shape == hkl.shape
        # (0,0,0) reflection excluded in fixture, so all should have |s| > 0
        s_lengths = np.sqrt(np.sum(s**2, axis=1))
        assert np.all(s_lengths > 0)

    @pytest.mark.unit
    def test_get_s(self, mock_cell, mock_hkl_indices):
        """Test s (|S|) calculation."""
        from torchref.base.math_numpy import get_s, get_scattering_vectors

        cell = mock_cell.numpy()
        hkl = mock_hkl_indices(n_reflections=50).numpy()
        
        s = get_s(hkl, cell)
        
        # Should equal norm of scattering vectors
        s_vectors = get_scattering_vectors(hkl, cell)
        expected = np.sqrt(np.sum(s_vectors**2, axis=1))
        
        assert np.allclose(s, expected, rtol=1e-6)


class TestRFactorCalculations:
    """Tests for R-factor calculation."""

    @pytest.mark.unit
    def test_get_rfactor_identical(self, mock_F_obs):
        """R-factor should be 0 for identical arrays."""
        from torchref.base.math_numpy import get_rfactor

        fobs = mock_F_obs(n_reflections=100).numpy()
        fcalc = fobs.copy()
        
        rfactor = get_rfactor(fobs, fcalc)
        
        assert np.isclose(rfactor, 0.0, atol=1e-10)

    @pytest.mark.unit
    def test_get_rfactor_scaled(self, mock_F_obs):
        """R-factor with scaled Fcalc."""
        from torchref.base.math_numpy import get_rfactor

        fobs = mock_F_obs(n_reflections=100).numpy()
        fcalc = fobs * 0.9
        
        rfactor = get_rfactor(fobs, fcalc)
        
        assert np.isclose(rfactor, 0.1, rtol=1e-5)


class TestRotation:
    """Tests for rotation functions."""

    @pytest.mark.unit
    def test_rotate_coords_identity(self, random_coordinates):
        """Zero rotation should not change coordinates."""
        from torchref.base.math_numpy import rotate_coords_numpy
        
        coords = random_coordinates(n_atoms=10).numpy()
        
        rotated = rotate_coords_numpy(coords, phi=0.0, rho=0.0)
        
        assert np.allclose(coords, rotated, rtol=1e-10)

    @pytest.mark.unit
    def test_rotate_coords_preserves_distances(self, random_coordinates):
        """Rotation should preserve pairwise distances."""
        from torchref.base.math_numpy import rotate_coords_numpy
        
        coords = random_coordinates(n_atoms=10).numpy()
        
        # Calculate original pairwise distances
        diff = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]
        orig_dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        
        # Rotate
        rotated = rotate_coords_numpy(coords, phi=45.0, rho=30.0)
        
        # Calculate rotated pairwise distances
        diff_rot = rotated[np.newaxis, :, :] - rotated[:, np.newaxis, :]
        rot_dist = np.sqrt(np.sum(diff_rot ** 2, axis=-1))
        
        assert np.allclose(orig_dist, rot_dist, rtol=1e-5)


class TestOutlierDetection:
    """Tests for outlier detection."""

    @pytest.mark.unit
    def test_calc_outliers_no_outliers(self, mock_F_obs):
        """No outliers when Fobs equals Fcalc."""
        from torchref.base.math_numpy import calc_outliers

        fobs = mock_F_obs(n_reflections=100).numpy()
        fcalc = fobs.copy()
        
        outliers = calc_outliers(fobs, fcalc, z=3.0)
        
        assert outliers.sum() == 0
