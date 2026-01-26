"""Tests for vectorized_add_to_map with automatic compilation."""
import pytest
import torch
import numpy as np
import time

from torchref.base.kernels import (
    vectorized_add_to_map,
    compute_metric_tensor,
    precompute_fractional_coords,
    warmup,
)
from torchref.base.math_torch import (
    vectorized_add_to_map as original_add_to_map,
)


class TestValidityAgainstOriginal:
    """Compare compiled version against original math_torch implementation."""

    @pytest.fixture
    def setup_tensors(self):
        """Create test tensors matching both function signatures."""
        def _setup(n_atoms=100, n_voxels=500, grid_shape=(64, 64, 64),
                   device="cpu", dtype=torch.float64, seed=42):
            torch.manual_seed(seed)

            # Cell parameters (orthorhombic for simplicity)
            cell = torch.tensor(
                [50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=dtype, device=device
            )
            frac_matrix = torch.diag(cell[:3])
            inv_frac_matrix = torch.diag(1.0 / cell[:3])

            # Atom parameters
            xyz = torch.rand(n_atoms, 3, device=device, dtype=dtype) * 30 + 10
            b = torch.rand(n_atoms, device=device, dtype=dtype) * 40 + 10
            A = torch.rand(n_atoms, 5, device=device, dtype=dtype) * 5 + 1
            B = torch.rand(n_atoms, 5, device=device, dtype=dtype) * 10 + 2
            occ = torch.rand(n_atoms, device=device, dtype=dtype) * 0.5 + 0.5

            # Voxel coordinates (Cartesian, around atoms)
            surrounding_coords = (
                xyz[:, None, :]
                + torch.randn(n_atoms, n_voxels, 3, device=device, dtype=dtype) * 2
            )
            voxel_indices = torch.randint(
                0, min(grid_shape), (n_atoms, n_voxels, 3), device=device
            )

            return {
                "surrounding_coords": surrounding_coords,
                "voxel_indices": voxel_indices,
                "grid_shape": grid_shape,
                "xyz": xyz,
                "b": b,
                "frac_matrix": frac_matrix,
                "inv_frac_matrix": inv_frac_matrix,
                "A": A,
                "B": B,
                "occ": occ,
            }
        return _setup

    def test_output_correlation_with_original(self, setup_tensors):
        """Compiled output correlates highly with original.

        Note: The compiled version uses softplus for smooth gradients while
        the original uses hard clamp. This gives slightly different B_total
        values and thus different outputs, but they should be highly correlated.
        """
        data = setup_tensors(n_atoms=50, n_voxels=200)

        # Original
        map_orig = torch.zeros(data["grid_shape"], dtype=data["xyz"].dtype)
        map_orig = original_add_to_map(
            data["surrounding_coords"], data["voxel_indices"], map_orig,
            data["xyz"], data["b"], data["inv_frac_matrix"], data["frac_matrix"],
            data["A"], data["B"], data["occ"]
        )

        # Auto-compiled version (same API now)
        map_compiled = torch.zeros(data["grid_shape"], dtype=data["xyz"].dtype)
        map_compiled = vectorized_add_to_map(
            data["surrounding_coords"], data["voxel_indices"], map_compiled,
            data["xyz"], data["b"], data["inv_frac_matrix"], data["frac_matrix"],
            data["A"], data["B"], data["occ"]
        )

        # Check correlation instead of exact match (softplus vs clamp causes differences)
        orig_flat = map_orig.flatten()
        compiled_flat = map_compiled.flatten()

        # Remove zeros for correlation
        mask = (orig_flat != 0) | (compiled_flat != 0)
        if mask.sum() > 0:
            correlation = torch.corrcoef(
                torch.stack([orig_flat[mask], compiled_flat[mask]])
            )[0, 1]
            assert correlation > 0.999, f"Correlation too low: {correlation}"

        # Check max relative difference is reasonable (within 1%)
        max_rel_diff = (
            (map_orig - map_compiled).abs().max() / (map_orig.abs().max() + 1e-10)
        )
        assert max_rel_diff < 0.01, f"Max relative difference too high: {max_rel_diff}"

    @pytest.mark.parametrize("n_atoms,n_voxels", [
        (10, 100),
        (100, 500),
        (500, 1000)
    ])
    def test_dynamic_shapes(self, setup_tensors, n_atoms, n_voxels):
        """Different sizes should work without errors."""
        data = setup_tensors(n_atoms=n_atoms, n_voxels=n_voxels)

        map_out = torch.zeros(data["grid_shape"], dtype=data["xyz"].dtype)
        result = vectorized_add_to_map(
            data["surrounding_coords"], data["voxel_indices"], map_out,
            data["xyz"], data["b"], data["inv_frac_matrix"], data["frac_matrix"],
            data["A"], data["B"], data["occ"]
        )

        assert result.shape == data["grid_shape"]
        assert torch.isfinite(result).all()

    def test_output_in_place(self, setup_tensors):
        """Verify that the function modifies the map in-place (like the original)."""
        data = setup_tensors(n_atoms=20, n_voxels=100)

        map_in = torch.zeros(data["grid_shape"], dtype=data["xyz"].dtype)

        result = vectorized_add_to_map(
            data["surrounding_coords"], data["voxel_indices"], map_in,
            data["xyz"], data["b"], data["inv_frac_matrix"], data["frac_matrix"],
            data["A"], data["B"], data["occ"]
        )

        # Function modifies in-place and returns same tensor (like original scatter_add_nd)
        assert result is map_in
        # Map should now contain density values
        assert result.abs().sum() > 0


class TestGradientValidity:
    """Test gradient flow through compiled function."""

    @pytest.fixture
    def grad_tensors(self):
        """Create tensors with requires_grad=True."""
        def _setup(n_atoms=50, n_voxels=200, grid_shape=(32, 32, 32), seed=42):
            torch.manual_seed(seed)
            dtype = torch.float64
            device = "cpu"

            frac_matrix = torch.diag(
                torch.tensor([50.0, 60.0, 70.0], dtype=dtype, device=device)
            )
            inv_frac_matrix = torch.diag(
                1.0 / torch.tensor([50.0, 60.0, 70.0], dtype=dtype, device=device)
            )

            xyz = torch.rand(n_atoms, 3, dtype=dtype, device=device).requires_grad_(True)
            b = (torch.rand(n_atoms, dtype=dtype, device=device) * 40 + 10).requires_grad_(True)
            A = torch.rand(n_atoms, 5, dtype=dtype, device=device).requires_grad_(True)
            B = (torch.rand(n_atoms, 5, dtype=dtype, device=device) * 10 + 2).requires_grad_(True)
            occ = torch.rand(n_atoms, dtype=dtype, device=device).requires_grad_(True)

            surrounding_coords = (
                xyz.detach()[:, None, :]
                + torch.randn(n_atoms, n_voxels, 3, dtype=dtype, device=device) * 2
            )
            voxel_indices = torch.randint(
                0, min(grid_shape), (n_atoms, n_voxels, 3), device=device
            )

            return {
                "surrounding_coords": surrounding_coords,
                "voxel_indices": voxel_indices,
                "grid_shape": grid_shape,
                "xyz": xyz,
                "b": b,
                "frac_matrix": frac_matrix,
                "inv_frac_matrix": inv_frac_matrix,
                "A": A,
                "B": B,
                "occ": occ,
            }
        return _setup

    def test_gradients_exist(self, grad_tensors):
        """All gradient-requiring params should have gradients after backward."""
        data = grad_tensors()

        density_map = torch.zeros(data["grid_shape"], dtype=data["xyz"].dtype)
        result = vectorized_add_to_map(
            data["surrounding_coords"], data["voxel_indices"], density_map,
            data["xyz"], data["b"], data["inv_frac_matrix"], data["frac_matrix"],
            data["A"], data["B"], data["occ"]
        )

        loss = result.sum()
        loss.backward()

        # Check all gradients exist and are finite
        for name in ["xyz", "b", "A", "B", "occ"]:
            grad = data[name].grad
            assert grad is not None, f"Gradient for {name} is None"
            assert torch.isfinite(grad).all(), (
                f"Gradient for {name} has non-finite values"
            )
            assert grad.abs().sum() > 0, f"Gradient for {name} is all zeros"

    def test_gradients_match_original(self, grad_tensors):
        """Gradients from compiled version match original implementation."""
        data = grad_tensors(n_atoms=30, n_voxels=100)

        # Clone tensors for original
        xyz_orig = data["xyz"].detach().clone().requires_grad_(True)
        b_orig = data["b"].detach().clone().requires_grad_(True)
        A_orig = data["A"].detach().clone().requires_grad_(True)
        B_orig = data["B"].detach().clone().requires_grad_(True)
        occ_orig = data["occ"].detach().clone().requires_grad_(True)

        # Create same surrounding coords for both
        torch.manual_seed(123)
        surrounding_coords = (
            xyz_orig.detach()[:, None, :]
            + torch.randn(30, 100, 3, dtype=data["xyz"].dtype) * 2
        )

        # Original backward
        map_orig = torch.zeros(data["grid_shape"], dtype=xyz_orig.dtype)
        map_orig = original_add_to_map(
            surrounding_coords, data["voxel_indices"], map_orig,
            xyz_orig, b_orig, data["inv_frac_matrix"], data["frac_matrix"],
            A_orig, B_orig, occ_orig
        )
        loss_orig = map_orig.sum()
        loss_orig.backward()

        # Compiled backward
        map_compiled = torch.zeros(data["grid_shape"], dtype=data["xyz"].dtype)
        map_compiled = vectorized_add_to_map(
            surrounding_coords, data["voxel_indices"], map_compiled,
            data["xyz"], data["b"], data["inv_frac_matrix"], data["frac_matrix"],
            data["A"], data["B"], data["occ"]
        )
        loss_compiled = map_compiled.sum()
        loss_compiled.backward()

        # Compare gradients - looser tolerance due to softplus vs clamp differences
        assert torch.allclose(
            data["A"].grad, A_orig.grad, rtol=5e-2, atol=1e-3
        ), "A gradients don't match"
        assert torch.allclose(
            data["B"].grad, B_orig.grad, rtol=5e-2, atol=1e-3
        ), "B gradients don't match"
        assert torch.allclose(
            data["occ"].grad, occ_orig.grad, rtol=5e-2, atol=1e-3
        ), "occ gradients don't match"

    def test_gradients_finite_differences(self, grad_tensors):
        """Verify gradients match finite differences for a simple case."""
        data = grad_tensors(n_atoms=5, n_voxels=20, grid_shape=(8, 8, 8))

        def func(A):
            density_map = torch.zeros(data["grid_shape"], dtype=A.dtype)
            result = vectorized_add_to_map(
                data["surrounding_coords"], data["voxel_indices"], density_map,
                data["xyz"].detach(), data["b"].detach(),
                data["inv_frac_matrix"], data["frac_matrix"],
                A, data["B"].detach(), data["occ"].detach()
            )
            return result.sum()

        # Check gradients for A using finite differences
        A_test = data["A"].detach().clone().requires_grad_(True)
        eps = 1e-5

        # Compute analytical gradient
        loss = func(A_test)
        loss.backward()
        analytical_grad = A_test.grad.clone()

        # Compute numerical gradient for a few elements
        for i in range(min(3, A_test.shape[0])):
            for j in range(min(2, A_test.shape[1])):
                A_plus = A_test.detach().clone()
                A_minus = A_test.detach().clone()
                A_plus[i, j] += eps
                A_minus[i, j] -= eps

                numerical_grad = (func(A_plus) - func(A_minus)) / (2 * eps)
                relative_error = abs(
                    numerical_grad - analytical_grad[i, j]
                ) / (abs(numerical_grad) + 1e-8)

                assert relative_error < 0.01, (
                    f"Gradient mismatch at [{i},{j}]: "
                    f"numerical={numerical_grad:.6f}, "
                    f"analytical={analytical_grad[i,j]:.6f}"
                )


class TestPerformance:
    """Performance benchmarks."""

    @pytest.fixture
    def setup_tensors(self):
        """Create test tensors for benchmarking."""
        def _setup(n_atoms=100, n_voxels=500, grid_shape=(64, 64, 64),
                   device="cpu", dtype=torch.float64, seed=42):
            torch.manual_seed(seed)

            cell = torch.tensor(
                [50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=dtype, device=device
            )
            frac_matrix = torch.diag(cell[:3])
            inv_frac_matrix = torch.diag(1.0 / cell[:3])

            xyz = torch.rand(n_atoms, 3, device=device, dtype=dtype) * 30 + 10
            b = torch.rand(n_atoms, device=device, dtype=dtype) * 40 + 10
            A = torch.rand(n_atoms, 5, device=device, dtype=dtype) * 5 + 1
            B = torch.rand(n_atoms, 5, device=device, dtype=dtype) * 10 + 2
            occ = torch.rand(n_atoms, device=device, dtype=dtype) * 0.5 + 0.5

            surrounding_coords = (
                xyz[:, None, :]
                + torch.randn(n_atoms, n_voxels, 3, device=device, dtype=dtype) * 2
            )
            voxel_indices = torch.randint(
                0, min(grid_shape), (n_atoms, n_voxels, 3), device=device
            )

            return {
                "surrounding_coords": surrounding_coords,
                "voxel_indices": voxel_indices,
                "grid_shape": grid_shape,
                "xyz": xyz,
                "b": b,
                "frac_matrix": frac_matrix,
                "inv_frac_matrix": inv_frac_matrix,
                "A": A,
                "B": B,
                "occ": occ,
            }
        return _setup

    @pytest.mark.slow
    def test_performance_improvement(self, setup_tensors):
        """Auto-compiled version should be faster after warmup."""
        data = setup_tensors(
            n_atoms=1000, n_voxels=2000, grid_shape=(128, 128, 128)
        )

        n_warmup = 3
        n_trials = 10

        # Warmup both
        for _ in range(n_warmup):
            map_orig = torch.zeros(data["grid_shape"], dtype=data["xyz"].dtype)
            original_add_to_map(
                data["surrounding_coords"], data["voxel_indices"], map_orig,
                data["xyz"], data["b"], data["inv_frac_matrix"],
                data["frac_matrix"], data["A"], data["B"], data["occ"]
            )

            map_compiled = torch.zeros(data["grid_shape"], dtype=data["xyz"].dtype)
            vectorized_add_to_map(
                data["surrounding_coords"], data["voxel_indices"], map_compiled,
                data["xyz"], data["b"], data["inv_frac_matrix"], data["frac_matrix"],
                data["A"], data["B"], data["occ"]
            )

        # Benchmark original
        times_orig = []
        for _ in range(n_trials):
            map_orig = torch.zeros(data["grid_shape"], dtype=data["xyz"].dtype)
            t0 = time.perf_counter()
            original_add_to_map(
                data["surrounding_coords"], data["voxel_indices"], map_orig,
                data["xyz"], data["b"], data["inv_frac_matrix"],
                data["frac_matrix"], data["A"], data["B"], data["occ"]
            )
            times_orig.append(time.perf_counter() - t0)

        # Benchmark auto-compiled
        times_compiled = []
        for _ in range(n_trials):
            map_compiled = torch.zeros(data["grid_shape"], dtype=data["xyz"].dtype)
            t0 = time.perf_counter()
            vectorized_add_to_map(
                data["surrounding_coords"], data["voxel_indices"], map_compiled,
                data["xyz"], data["b"], data["inv_frac_matrix"], data["frac_matrix"],
                data["A"], data["B"], data["occ"]
            )
            times_compiled.append(time.perf_counter() - t0)

        mean_orig = np.mean(times_orig)
        mean_compiled = np.mean(times_compiled)
        speedup = mean_orig / mean_compiled

        print(f"\nOriginal: {mean_orig*1000:.2f} ms")
        print(f"Auto-compiled: {mean_compiled*1000:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")

        # Assert some speedup (at least not significantly slower)
        assert speedup > 0.8, f"Compiled version too slow: {speedup:.2f}x"


class TestWarmup:
    """Test warmup function."""

    def test_warmup_no_error(self):
        """warmup should not raise errors."""
        warmup(device="cpu")

    def test_warmup_enables_fast_execution(self):
        """After warmup, function should work correctly."""
        # Warmup
        warmup(device="cpu")

        # Create test data and verify function works
        n_atoms, n_voxels = 10, 50
        grid_shape = (16, 16, 16)
        dtype = torch.float32

        surrounding_coords = torch.randn(n_atoms, n_voxels, 3, dtype=dtype)
        voxel_indices = torch.randint(0, 16, (n_atoms, n_voxels, 3))
        density_map = torch.zeros(grid_shape, dtype=dtype)
        xyz = torch.randn(n_atoms, 3, dtype=dtype)
        b = torch.rand(n_atoms, dtype=dtype) * 50 + 10
        inv_frac_matrix = torch.eye(3, dtype=dtype) * 0.02
        frac_matrix = torch.eye(3, dtype=dtype) * 50
        A = torch.rand(n_atoms, 5, dtype=dtype)
        B = torch.rand(n_atoms, 5, dtype=dtype) * 10 + 1
        occ = torch.ones(n_atoms, dtype=dtype)

        result = vectorized_add_to_map(
            surrounding_coords, voxel_indices, density_map, xyz, b,
            inv_frac_matrix, frac_matrix, A, B, occ
        )

        assert result.shape == grid_shape
        assert torch.isfinite(result).all()
