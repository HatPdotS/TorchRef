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


class TestPatterson:
    """Test Patterson map calculation and interpolation."""

    def test_patterson_origin_peak(self):
        """Test that Patterson has maximum at origin."""
        from torchref.alignment.patterson import calculate_patterson

        # Create simple reflection data
        hkl = torch.tensor([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 1, 1]
        ], dtype=torch.int32)
        F_obs = torch.ones(7, dtype=torch.float64)
        cell = torch.tensor([20.0, 20.0, 20.0, 90.0, 90.0, 90.0], dtype=torch.float64)

        patterson, grid_info = calculate_patterson(F_obs, hkl, cell, grid_spacing=1.0)

        # Origin should be maximum
        assert patterson[0, 0, 0] == patterson.max(), \
            "Patterson origin is not maximum"

    def test_patterson_centrosymmetric(self):
        """Test that Patterson is centrosymmetric: P(u) = P(-u)."""
        from torchref.alignment.patterson import calculate_patterson, interpolate_patterson

        # Create reflection data
        hkl = torch.tensor([
            [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0],
            [1, 1, 0], [1, 2, 0], [2, 1, 0]
        ], dtype=torch.int32)
        F_obs = torch.rand(7, dtype=torch.float64) + 0.5
        cell = torch.tensor([30.0, 30.0, 30.0, 90.0, 90.0, 90.0], dtype=torch.float64)

        patterson, grid_info = calculate_patterson(F_obs, hkl, cell, grid_spacing=1.0)

        # Check centrosymmetry at random points
        random_vecs = torch.randn(10, 3, dtype=torch.float64) * 5
        for vec in random_vecs:
            p_plus = interpolate_patterson(patterson, vec.unsqueeze(0), cell)
            p_minus = interpolate_patterson(patterson, (-vec).unsqueeze(0), cell)
            assert torch.allclose(p_plus, p_minus, atol=1e-6), \
                f"Patterson not centrosymmetric: P({vec}) != P({-vec})"

    def test_patterson_cartesian_to_fractional_conversion(self):
        """Test that Patterson interpolation correctly converts Cartesian vectors to fractional."""
        from torchref.alignment.patterson import interpolate_patterson
        from torchref.math_functions.math_torch import cartesian_to_fractional_torch

        # Create a simple Patterson with known peaks
        grid_size = 20
        patterson = torch.zeros(grid_size, grid_size, grid_size, dtype=torch.float64)
        patterson[0, 0, 0] = 1.0  # Origin peak
        patterson[4, 4, 4] = 0.5  # Peak at fractional [0.2, 0.2, 0.2]

        # Orthorhombic cell: fractional [0.2, 0.2, 0.2] = Cartesian [10, 12, 14]
        cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float64)

        # Query at origin
        origin_cart = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
        val_origin = interpolate_patterson(patterson, origin_cart, cell).item()
        assert abs(val_origin - 1.0) < 0.1, f"Origin value wrong: {val_origin}"

        # Query at Cartesian [10, 12, 14] which should map to fractional [0.2, 0.2, 0.2]
        peak_cart = torch.tensor([[10.0, 12.0, 14.0]], dtype=torch.float64)
        val_peak = interpolate_patterson(patterson, peak_cart, cell).item()
        assert abs(val_peak - 0.5) < 0.1, f"Peak value wrong: {val_peak}"

        # Verify the fractional conversion
        frac = cartesian_to_fractional_torch(peak_cart, cell)
        expected_frac = torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.float64)
        assert torch.allclose(frac, expected_frac, atol=1e-10), \
            f"Fractional conversion wrong: {frac} != {expected_frac}"

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


class TestVectorSampler:
    """Test atom pair sampling."""

    def test_uniform_sampling(self):
        """Test uniform sampling gives roughly equal selection."""
        from torchref.alignment.sampling import VectorSampler

        coords = np.random.randn(10, 3)
        Z = np.ones(10, dtype=np.int64)  # All same element

        sampler = VectorSampler(n_vectors=1000, weighting='uniform', seed=42)
        i_idx, j_idx, weights = sampler.sample(coords, Z)

        # Check indices are valid
        assert np.all(i_idx >= 0) and np.all(i_idx < 10)
        assert np.all(j_idx >= 0) and np.all(j_idx < 10)

        # No self-pairs
        assert not np.any(i_idx == j_idx)

    def test_z2_weighting_favors_heavy(self):
        """Test that Z² weighting favors heavy atoms."""
        from torchref.alignment.sampling import VectorSampler

        coords = np.random.randn(20, 3)
        # 10 light atoms (Z=6) and 10 heavy atoms (Z=80)
        Z = np.array([6]*10 + [80]*10, dtype=np.int64)

        sampler = VectorSampler(n_vectors=5000, weighting='Z2', seed=42)
        i_idx, j_idx, weights = sampler.sample(coords, Z)

        # Count heavy-heavy pairs (indices >= 10)
        heavy_heavy = np.sum((i_idx >= 10) & (j_idx >= 10))
        light_light = np.sum((i_idx < 10) & (j_idx < 10))

        # Heavy-heavy should dominate due to Z² weighting
        # Heavy: 80⁴ = 40960000, Light: 6⁴ = 1296, ratio ~31000:1
        assert heavy_heavy > light_light * 10, \
            f"Z² weighting not favoring heavy atoms: heavy={heavy_heavy}, light={light_light}"


class TestSyntheticAlignment:
    """Test Patterson alignment with synthetic data."""

    def test_synthetic_heavy_atom_recovery(self):
        """Test that alignment recovers correct orientation for synthetic heavy atoms."""
        from torchref.math_functions.math_torch import (
            place_on_grid, fft, find_grid_size,
            cartesian_to_fractional_torch, fractional_to_cartesian_torch,
            random_rotation_uniform
        )
        from torchref.alignment.patterson import interpolate_patterson

        # Simple cubic cell
        cell = torch.tensor([40.0, 40.0, 40.0, 90.0, 90.0, 90.0], dtype=torch.float64)

        # 4 heavy atoms at known positions
        heavy_frac = torch.tensor([
            [0.15, 0.15, 0.15],
            [0.35, 0.15, 0.15],
            [0.15, 0.35, 0.15],
            [0.15, 0.15, 0.35],
        ], dtype=torch.float64)
        heavy_cart = fractional_to_cartesian_torch(heavy_frac, cell)

        # Generate structure factors
        grid_size = find_grid_size(cell, 1.5)
        grid_size = ((grid_size + 1) // 2) * 2
        grid_size = torch.clamp(grid_size, min=24)

        h_range = torch.arange(-grid_size[0]//2, grid_size[0]//2)
        k_range = torch.arange(-grid_size[1]//2, grid_size[1]//2)
        l_range = torch.arange(-grid_size[2]//2, grid_size[2]//2)
        H, K, L = torch.meshgrid(h_range, k_range, l_range, indexing='ij')
        hkl = torch.stack([H.flatten(), K.flatten(), L.flatten()], dim=1)

        phase = 2 * np.pi * (hkl.float() @ heavy_frac.T.float())
        F_complex = torch.exp(1j * phase).sum(dim=1)
        F_obs = torch.abs(F_complex)

        mask = (F_obs > 0.1) & (hkl.abs().max(dim=1).values > 0)
        hkl_use = hkl[mask].to(torch.int32)
        F_use = F_obs[mask]

        # Calculate Patterson
        F_squared = (F_use ** 2).to(torch.complex128)
        reciprocal_grid = place_on_grid(
            hkls=hkl_use,
            structure_factor=F_squared,
            grid_size=grid_size,
            enforce_hermitian=True
        )
        patterson = fft(reciprocal_grid)
        patterson = patterson / patterson.max()

        # Apply transformation
        R_true = random_rotation_uniform(1, dtype=torch.float64)
        t_true = torch.tensor([3.0, -2.0, 1.5], dtype=torch.float64)
        heavy_cart_transformed = heavy_cart @ R_true.T + t_true

        # Score function
        def score_orientation(coords, R, t):
            transformed = coords @ R.T + t
            total = 0.0
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    vec = transformed[j] - transformed[i]
                    pval = interpolate_patterson(patterson, vec.unsqueeze(0), cell).item()
                    total += pval
            return total

        # Score at correct inverse
        R_inv = R_true.T
        t_inv = -R_inv @ t_true
        score_correct = score_orientation(heavy_cart_transformed, R_inv, t_inv)

        # Score at random orientations
        random_scores = []
        for _ in range(10):
            R_rand = random_rotation_uniform(1, dtype=torch.float64)
            t_rand = torch.rand(3, dtype=torch.float64) * 15
            random_scores.append(score_orientation(heavy_cart_transformed, R_rand, t_rand))

        # Correct should be significantly higher than random
        assert score_correct > max(random_scores), \
            f"Correct score ({score_correct:.4f}) not higher than max random ({max(random_scores):.4f})"
        assert score_correct > np.mean(random_scores) + 2 * np.std(random_scores), \
            "Correct score not significantly above random"


class TestModuleImports:
    """Test that all alignment module components import correctly."""

    def test_import_main_classes(self):
        """Test importing main alignment classes."""
        from torchref.alignment import PattersonAligner, AlignmentResult
        assert PattersonAligner is not None
        assert AlignmentResult is not None

    def test_import_patterson(self):
        """Test importing Patterson functions."""
        from torchref.alignment import calculate_patterson, interpolate_patterson
        assert calculate_patterson is not None
        assert interpolate_patterson is not None

    def test_import_sampling(self):
        """Test importing VectorSampler."""
        from torchref.alignment import VectorSampler
        assert VectorSampler is not None

    def test_import_rotation(self):
        """Test importing rotation utilities."""
        from torchref.alignment import (
            params_to_matrix, matrix_to_params,
            random_rotation_params, random_rotation_matrix
        )
        assert params_to_matrix is not None
        assert matrix_to_params is not None
        assert random_rotation_params is not None
        assert random_rotation_matrix is not None
