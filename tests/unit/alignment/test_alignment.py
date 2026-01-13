"""
Unit tests for Patterson alignment module.

Tests rotation utilities, Patterson calculation, sampling, and alignment recovery.
"""

import pytest
import torch
import numpy as np


class TestRotationFunctions:
    """Test rotation matrix utilities."""

    def test_axis_angle_roundtrip(self):
        """Test that axis-angle <-> matrix conversion is reversible."""
        from torchref.math_functions.math_torch import (
            axis_angle_to_rotation_matrix,
            rotation_matrix_to_axis_angle,
        )

        # Create random axis-angle vector
        original = torch.randn(3, dtype=torch.float64)

        # Convert to matrix and back
        R = axis_angle_to_rotation_matrix(original)
        recovered = rotation_matrix_to_axis_angle(R)

        # Should recover original (up to sign ambiguity for angle > pi)
        angle_orig = torch.norm(original)
        angle_recov = torch.norm(recovered)

        if angle_orig < np.pi:
            assert torch.allclose(original, recovered, atol=1e-6), \
                f"Roundtrip failed: {original} -> {recovered}"
        else:
            # For large angles, we might get equivalent representation
            assert torch.allclose(angle_orig % (2*np.pi), angle_recov % (2*np.pi), atol=1e-6)

    def test_rotation_matrix_orthogonal(self):
        """Test that generated rotation matrices are orthogonal."""
        from torchref.math_functions.math_torch import random_rotation_uniform

        for _ in range(5):
            R = random_rotation_uniform(1, dtype=torch.float64)

            # R @ R.T should be identity
            identity = torch.eye(3, dtype=torch.float64)
            assert torch.allclose(R @ R.T, identity, atol=1e-10), \
                "R @ R.T is not identity"

            # det(R) should be 1
            det = torch.det(R)
            assert torch.allclose(det, torch.tensor(1.0, dtype=torch.float64), atol=1e-10), \
                f"det(R) = {det.item()}, expected 1.0"

    def test_quaternion_to_rotation(self):
        """Test quaternion to rotation matrix conversion."""
        from torchref.math_functions.math_torch import quaternion_to_rotation_matrix

        # Identity quaternion [1, 0, 0, 0] should give identity matrix
        q_id = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        R_id = quaternion_to_rotation_matrix(q_id)
        assert torch.allclose(R_id, torch.eye(3, dtype=torch.float64), atol=1e-10)

        # 180 degree rotation around z-axis: q = [0, 0, 0, 1]
        q_z180 = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        R_z180 = quaternion_to_rotation_matrix(q_z180)
        expected = torch.tensor([
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float64)
        assert torch.allclose(R_z180, expected, atol=1e-10)


class TestTrilinearInterpolation:
    """Test trilinear interpolation."""

    def test_interpolate_at_grid_points(self):
        """Test that interpolation at grid points returns exact values."""
        from torchref.math_functions.math_torch import trilinear_interpolate

        # Create a simple 3x3x3 grid with known values
        grid = torch.arange(27, dtype=torch.float64).reshape(3, 3, 3)

        # Query at exact grid points (in fractional coords [0, 1))
        # Grid point (1, 1, 1) has value 13
        point = torch.tensor([[1/3, 1/3, 1/3]], dtype=torch.float64)
        value = trilinear_interpolate(grid, point)

        # Value at (1,1,1) = 1*9 + 1*3 + 1 = 13
        assert torch.allclose(value, torch.tensor([13.0], dtype=torch.float64), atol=1e-6)

    def test_interpolate_midpoint(self):
        """Test interpolation at midpoints between grid points."""
        from torchref.math_functions.math_torch import trilinear_interpolate

        # Create grid where value = x + y + z (in grid coords)
        grid = torch.zeros(4, 4, 4, dtype=torch.float64)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    grid[i, j, k] = i + j + k

        # Midpoint between (0,0,0) and (1,1,1) in fractional: (0.125, 0.125, 0.125)
        # This maps to grid coords (0.5, 0.5, 0.5)
        # Interpolated value should be 0.5 + 0.5 + 0.5 = 1.5
        point = torch.tensor([[0.125, 0.125, 0.125]], dtype=torch.float64)
        value = trilinear_interpolate(grid, point)
        assert torch.allclose(value, torch.tensor([1.5], dtype=torch.float64), atol=1e-6)


class TestCoordinateConversion:
    """Test coordinate conversion utilities."""

    def test_coordinate_conversion_preserves_dtype(self):
        """Test that coordinate conversion preserves tensor dtype."""
        from torchref.math_functions.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch
        )

        cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float64)

        # Test float64
        coords64 = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float64)
        frac64 = cartesian_to_fractional_torch(coords64, cell)
        assert frac64.dtype == torch.float64, f"float64 not preserved: {frac64.dtype}"

        # Test float32
        coords32 = coords64.to(torch.float32)
        cell32 = cell.to(torch.float32)
        frac32 = cartesian_to_fractional_torch(coords32, cell32)
        assert frac32.dtype == torch.float32, f"float32 not preserved: {frac32.dtype}"

    def test_coordinate_conversion_triclinic(self):
        """Test coordinate conversion roundtrip for triclinic cell."""
        from torchref.math_functions.math_torch import (
            cartesian_to_fractional_torch,
            fractional_to_cartesian_torch
        )

        # Triclinic cell with all angles != 90
        cell = torch.tensor([30.0, 40.0, 50.0, 70.0, 80.0, 85.0], dtype=torch.float64)

        # Random fractional coordinates
        frac_orig = torch.tensor([[0.15, 0.25, 0.35]], dtype=torch.float64)

        # Convert to Cartesian and back
        cart = fractional_to_cartesian_torch(frac_orig, cell)
        frac_back = cartesian_to_fractional_torch(cart, cell)

        assert torch.allclose(frac_orig, frac_back, atol=1e-10), \
            f"Triclinic roundtrip failed: {frac_orig} -> {cart} -> {frac_back}"


class TestModuleImports:
    """Test that all alignment module components import correctly."""

    def test_import_main_classes(self):
        """Test importing main alignment classes."""
        from torchref.alignment import PattersonAligner, AlignmentResult
        assert PattersonAligner is not None
        assert AlignmentResult is not None

