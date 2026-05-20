"""Tests for C++ parallel scatter_add (cpu_scatter.py).

Verifies correctness and gradient accuracy of structured_scatter_add
against PyTorch's native scatter_add_ for all valid use cases.
"""

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pytorch_scatter_reference(density_cube, wa, wbwc, map_size):
    """Reference implementation using PyTorch scatter_add_."""
    C, nx, ny, nz = density_cube.shape
    idx_flat = wa[:, :, None, None] + wbwc[:, None, :, :]  # (C, nx, ny, nz)
    result = torch.zeros(map_size, dtype=density_cube.dtype)
    result.scatter_add_(0, idx_flat.reshape(-1), density_cube.reshape(-1))
    return result


def _make_valid_indices(C, nx, ny, nz, grid_shape, seed=42):
    """Generate valid structured indices that stay within [0, map_size).

    Mimics how production code computes wa and wbwc from center_idx
    with modular arithmetic for periodic boundary conditions.
    """
    torch.manual_seed(seed)
    gx, gy, gz = grid_shape
    ny_nz = gy * gz

    # Random center voxels within the grid
    center_x = torch.randint(0, gx, (C,))
    center_y = torch.randint(0, gy, (C,))
    center_z = torch.randint(0, gz, (C,))

    # Axis offsets (symmetric around center, like real code)
    half = nx // 2
    offsets = torch.arange(-half, -half + nx)

    # wa: (C, nx) — x-axis indices with PBC wrapping
    wa = ((center_x[:, None] + offsets[None, :]) % gx) * ny_nz

    # wbwc: (C, ny, nz) — yz-plane indices with PBC wrapping
    half_y = ny // 2
    half_z = nz // 2
    offsets_y = torch.arange(-half_y, -half_y + ny)
    offsets_z = torch.arange(-half_z, -half_z + nz)
    wb = ((center_y[:, None] + offsets_y[None, :]) % gy) * gz   # (C, ny)
    wc = ((center_z[:, None] + offsets_z[None, :]) % gz)         # (C, nz)
    wbwc = wb[:, :, None] + wc[:, None, :]                       # (C, ny, nz)

    return wa.long(), wbwc.long()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cpp_scatter():
    """Import structured_scatter_add (compiles C++ on first call)."""
    try:
        from torchref.base.kernels.cpu_scatter import structured_scatter_add
        return structured_scatter_add
    except Exception as e:
        pytest.skip(f"C++ scatter not available: {e}")


# ---------------------------------------------------------------------------
# Compile-availability test
# ---------------------------------------------------------------------------

class TestCompilation:
    """Verify the C++ extension can be compiled in the current environment.

    This test does NOT skip on failure — failure here is the actual signal
    we want from CI / SLURM jobs. The full diagnostic captured by
    ``_get_module()`` is included in the assertion message.
    """

    def test_module_compiles(self):
        """C++ cpu_scatter extension must build successfully."""
        from torchref.base.kernels import cpu_scatter

        # Reset any cached failure from a previous test in the same process so
        # we get a fresh attempt with up-to-date diagnostics.
        cpu_scatter._module_failed = False
        cpu_scatter._module_error = None

        mod = cpu_scatter._get_module()

        if mod is None:
            err_summary, err_tb = cpu_scatter._module_error or ("unknown", "")
            # Surface environment context that commonly differs on SLURM nodes.
            import os
            import shutil
            import sys
            env_info = (
                f"  python:    {sys.executable}\n"
                f"  ninja:     {shutil.which('ninja')}\n"
                f"  CXX env:   {os.environ.get('CXX', '<unset>')}\n"
                f"  CC env:    {os.environ.get('CC', '<unset>')}\n"
                f"  PATH head: {os.environ.get('PATH', '').split(':')[:5]}\n"
                f"  HOME:      {os.environ.get('HOME', '<unset>')}\n"
                f"  TORCH_EXTENSIONS_DIR: "
                f"{os.environ.get('TORCH_EXTENSIONS_DIR', '<unset>')}\n"
            )
            pytest.fail(
                "C++ cpu_scatter compilation failed.\n"
                f"Error: {err_summary}\n\n"
                f"Environment:\n{env_info}\n"
                f"Full traceback:\n{err_tb}"
            )

        # Sanity-check the built module exposes both entry points
        # (one binding per index dtype: int32 fast path + int64 fallback).
        for name in (
            "structured_scatter_add_i32",
            "structured_scatter_add_i64",
            "structured_gather_i32",
            "structured_gather_i64",
        ):
            assert hasattr(mod, name), f"compiled module is missing {name}"


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

class TestCorrectnessVsPyTorch:
    """Verify C++ scatter matches PyTorch scatter_add_ for all valid cases."""

    def test_basic_small(self, cpp_scatter):
        """Small problem: 4 atoms, 5x5x5 cube, 20^3 grid."""
        C, nx, ny, nz = 4, 5, 5, 5
        grid_shape = (20, 20, 20)
        map_size = 20 * 20 * 20

        torch.manual_seed(0)
        density_cube = torch.randn(C, nx, ny, nz)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=0)

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert got.shape == (map_size,)
        assert torch.allclose(ref, got, atol=1e-6), \
            f"max err = {(ref - got).abs().max().item():.2e}"

    def test_single_atom(self, cpp_scatter):
        """Edge case: single atom (C=1)."""
        C, nx, ny, nz = 1, 7, 7, 7
        grid_shape = (30, 30, 30)
        map_size = 30 * 30 * 30

        torch.manual_seed(1)
        density_cube = torch.randn(C, nx, ny, nz)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=1)

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert torch.allclose(ref, got, atol=1e-6)

    def test_large_chunk(self, cpp_scatter):
        """Typical production chunk: 1024 atoms, 17^3 cube."""
        C, nx, ny, nz = 1024, 17, 17, 17
        grid_shape = (216, 90, 72)
        map_size = 216 * 90 * 72

        torch.manual_seed(2)
        density_cube = torch.randn(C, nx, ny, nz)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=2)

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert torch.allclose(ref, got, atol=1e-5), \
            f"max err = {(ref - got).abs().max().item():.2e}"

    def test_asymmetric_cube(self, cpp_scatter):
        """Non-cubic ROI: nx != ny != nz."""
        C, nx, ny, nz = 50, 9, 13, 7
        grid_shape = (64, 64, 64)
        map_size = 64 ** 3

        torch.manual_seed(3)
        density_cube = torch.randn(C, nx, ny, nz)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=3)

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert torch.allclose(ref, got, atol=1e-6)

    def test_asymmetric_grid(self, cpp_scatter):
        """Non-cubic grid (common in crystallography: a >> b,c)."""
        C, nx, ny, nz = 100, 11, 11, 11
        grid_shape = (300, 60, 48)
        map_size = 300 * 60 * 48

        torch.manual_seed(4)
        density_cube = torch.randn(C, nx, ny, nz)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=4)

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert torch.allclose(ref, got, atol=1e-5)

    def test_boundary_wrapping(self, cpp_scatter):
        """Atoms near grid boundaries where indices wrap around via PBC."""
        C, nx, ny, nz = 20, 17, 17, 17
        grid_shape = (32, 32, 32)
        map_size = 32 ** 3

        torch.manual_seed(5)
        density_cube = torch.randn(C, nx, ny, nz)

        # Force atoms near corners of the grid
        gx, gy, gz = grid_shape
        ny_nz = gy * gz
        half = nx // 2
        offsets = torch.arange(-half, -half + nx)
        offsets_y = torch.arange(-half, -half + ny)
        offsets_z = torch.arange(-half, -half + nz)

        # Place atoms at grid boundaries (0, last, near-last)
        corners = torch.tensor([0, 1, gx - 1, gx - 2, gx // 2])
        center_x = corners.repeat(C // len(corners) + 1)[:C]
        center_y = corners.repeat(C // len(corners) + 1)[:C]
        center_z = corners.repeat(C // len(corners) + 1)[:C]

        wa = ((center_x[:, None] + offsets[None, :]) % gx) * ny_nz
        wb = ((center_y[:, None] + offsets_y[None, :]) % gy) * gz
        wc = ((center_z[:, None] + offsets_z[None, :]) % gz)
        wbwc = wb[:, :, None] + wc[:, None, :]

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert torch.allclose(ref, got, atol=1e-6), \
            f"Boundary wrapping failed: max err = {(ref - got).abs().max().item():.2e}"

    def test_overlapping_rois(self, cpp_scatter):
        """Multiple atoms with heavily overlapping scatter regions."""
        C, nx, ny, nz = 50, 17, 17, 17
        grid_shape = (40, 40, 40)
        map_size = 40 ** 3

        torch.manual_seed(6)
        density_cube = torch.randn(C, nx, ny, nz)

        # All atoms at the same center → maximal overlap
        gx, gy, gz = grid_shape
        ny_nz = gy * gz
        half = nx // 2
        offsets = torch.arange(-half, -half + nx)
        offsets_y = torch.arange(-half, -half + ny)
        offsets_z = torch.arange(-half, -half + nz)

        center_x = torch.full((C,), gx // 2, dtype=torch.long)
        center_y = torch.full((C,), gy // 2, dtype=torch.long)
        center_z = torch.full((C,), gz // 2, dtype=torch.long)

        wa = ((center_x[:, None] + offsets[None, :]) % gx) * ny_nz
        wb = ((center_y[:, None] + offsets_y[None, :]) % gy) * gz
        wc = ((center_z[:, None] + offsets_z[None, :]) % gz)
        wbwc = wb[:, :, None] + wc[:, None, :]

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert torch.allclose(ref, got, atol=1e-5), \
            f"Overlapping ROIs failed: max err = {(ref - got).abs().max().item():.2e}"

    def test_all_positive_values(self, cpp_scatter):
        """All density values positive (physical electron density)."""
        C, nx, ny, nz = 100, 11, 11, 11
        grid_shape = (64, 64, 64)
        map_size = 64 ** 3

        torch.manual_seed(7)
        density_cube = torch.rand(C, nx, ny, nz) * 10.0  # strictly positive
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=7)

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert torch.allclose(ref, got, atol=1e-5)
        # Result should also be non-negative
        assert got.min() >= -1e-6

    def test_zero_density(self, cpp_scatter):
        """All-zero density cube (should produce zero map)."""
        C, nx, ny, nz = 10, 5, 5, 5
        grid_shape = (20, 20, 20)
        map_size = 20 ** 3

        density_cube = torch.zeros(C, nx, ny, nz)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=8)

        got = cpp_scatter(density_cube, wa, wbwc, map_size)
        assert torch.all(got == 0.0)

    def test_sparse_density(self, cpp_scatter):
        """Mostly-zero density with a few nonzero entries."""
        C, nx, ny, nz = 32, 9, 9, 9
        grid_shape = (50, 50, 50)
        map_size = 50 ** 3

        torch.manual_seed(9)
        density_cube = torch.zeros(C, nx, ny, nz)
        # Set only ~1% of entries to nonzero
        mask = torch.rand_like(density_cube) < 0.01
        density_cube[mask] = torch.randn(mask.sum())

        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=9)

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert torch.allclose(ref, got, atol=1e-6)

    def test_large_values(self, cpp_scatter):
        """Large magnitude values (stress test float32 accumulation)."""
        C, nx, ny, nz = 200, 11, 11, 11
        grid_shape = (80, 80, 80)
        map_size = 80 ** 3

        torch.manual_seed(10)
        density_cube = torch.randn(C, nx, ny, nz) * 1e4
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=10)

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        # Looser tolerance for large values
        assert torch.allclose(ref, got, rtol=1e-5, atol=1e-2), \
            f"Large values: max err = {(ref - got).abs().max().item():.2e}"

    def test_sorted_atoms(self, cpp_scatter):
        """Atoms sorted by 1D center (production configuration)."""
        C, nx, ny, nz = 512, 17, 17, 17
        grid_shape = (216, 90, 72)
        map_size = 216 * 90 * 72

        torch.manual_seed(11)
        density_cube = torch.randn(C, nx, ny, nz)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=11)

        # Sort by first wa entry (approximates sorting by 1D center)
        order = torch.argsort(wa[:, wa.shape[1] // 2])
        density_cube = density_cube[order]
        wa = wa[order]
        wbwc = wbwc[order]

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert torch.allclose(ref, got, atol=1e-5)

    def test_remainder_chunk(self, cpp_scatter):
        """Non-power-of-2 atom count (tests remainder handling)."""
        C, nx, ny, nz = 37, 9, 9, 9
        grid_shape = (50, 50, 50)
        map_size = 50 ** 3

        torch.manual_seed(12)
        density_cube = torch.randn(C, nx, ny, nz)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=12)

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)
        got = cpp_scatter(density_cube, wa, wbwc, map_size)

        assert torch.allclose(ref, got, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestGradients:
    """Verify gradient correctness of structured_scatter_add."""

    def test_gradcheck_small(self, cpp_scatter):
        """Numerical gradient check via finite differences (float32).

        Uses a small problem and generous tolerances appropriate for float32.
        """
        C, nx, ny, nz = 3, 5, 5, 5
        grid_shape = (20, 20, 20)
        map_size = 20 ** 3

        torch.manual_seed(20)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=20)

        # Compute analytical gradient
        dc = torch.randn(C, nx, ny, nz, requires_grad=True)
        grad_output = torch.randn(map_size)
        result = cpp_scatter(dc, wa, wbwc, map_size)
        result.backward(grad_output)
        analytical = dc.grad.clone()

        # Compute numerical gradient via finite differences
        eps = 1e-4
        numerical = torch.zeros_like(dc)
        dc_flat = dc.data.view(-1)
        for i in range(dc_flat.numel()):
            dc_flat[i] += eps
            f_plus = cpp_scatter(dc.data.view(C, nx, ny, nz), wa, wbwc, map_size)
            dc_flat[i] -= 2 * eps
            f_minus = cpp_scatter(dc.data.view(C, nx, ny, nz), wa, wbwc, map_size)
            dc_flat[i] += eps  # restore
            numerical.view(-1)[i] = ((f_plus - f_minus) * grad_output).sum() / (2 * eps)

        assert torch.allclose(analytical, numerical, atol=1e-3, rtol=1e-3), \
            f"Gradcheck failed: max err = {(analytical - numerical).abs().max().item():.2e}"

    def test_gradient_vs_pytorch(self, cpp_scatter):
        """Compare gradients from C++ scatter vs PyTorch scatter."""
        C, nx, ny, nz = 64, 11, 11, 11
        grid_shape = (64, 64, 64)
        map_size = 64 ** 3

        torch.manual_seed(21)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=21)

        # Random upstream gradient
        grad_output = torch.randn(map_size)

        # PyTorch path
        dc_pt = torch.randn(C, nx, ny, nz, requires_grad=True)
        ref = _pytorch_scatter_reference(dc_pt, wa, wbwc, map_size)
        ref.backward(grad_output)
        grad_pt = dc_pt.grad.clone()

        # C++ path
        dc_cpp = dc_pt.data.clone().requires_grad_(True)
        got = cpp_scatter(dc_cpp, wa, wbwc, map_size)
        got.backward(grad_output)
        grad_cpp = dc_cpp.grad.clone()

        assert torch.allclose(grad_pt, grad_cpp, atol=1e-6), \
            f"Gradient mismatch: max err = {(grad_pt - grad_cpp).abs().max().item():.2e}"

    def test_gradient_accumulation(self, cpp_scatter):
        """Gradients accumulate correctly across multiple scatter calls (like chunk loop)."""
        C, nx, ny, nz = 32, 9, 9, 9
        grid_shape = (40, 40, 40)
        map_size = 40 ** 3

        torch.manual_seed(22)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=22)

        # Two chunks summed (simulates the chunk loop)
        dc1 = torch.randn(C, nx, ny, nz, requires_grad=True)
        dc2 = torch.randn(C, nx, ny, nz, requires_grad=True)

        result = cpp_scatter(dc1, wa, wbwc, map_size) + \
                 cpp_scatter(dc2, wa, wbwc, map_size)
        loss = result.sum()
        loss.backward()

        # Reference: expected gradient of sum() through gather is all-ones gathered
        # grad_dc[c, ix, iy, iz] = 1.0 for all entries (since loss = sum of all)
        expected_grad = torch.ones(C, nx, ny, nz)

        assert torch.allclose(dc1.grad, expected_grad, atol=1e-6), \
            f"dc1 grad err: {(dc1.grad - expected_grad).abs().max().item():.2e}"
        assert torch.allclose(dc2.grad, expected_grad, atol=1e-6), \
            f"dc2 grad err: {(dc2.grad - expected_grad).abs().max().item():.2e}"

    def test_gradient_with_downstream_loss(self, cpp_scatter):
        """Gradient through a realistic loss: MSE between scattered result and target."""
        C, nx, ny, nz = 16, 7, 7, 7
        grid_shape = (32, 32, 32)
        map_size = 32 ** 3

        torch.manual_seed(23)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=23)
        target = torch.randn(map_size)

        # C++ path
        dc_cpp = torch.randn(C, nx, ny, nz, requires_grad=True)
        result_cpp = cpp_scatter(dc_cpp, wa, wbwc, map_size)
        loss_cpp = ((result_cpp - target) ** 2).mean()
        loss_cpp.backward()
        grad_cpp = dc_cpp.grad.clone()

        # PyTorch reference
        dc_pt = dc_cpp.data.clone().requires_grad_(True)
        idx_flat = wa[:, :, None, None] + wbwc[:, None, :, :]
        result_pt = torch.zeros(map_size)
        result_pt.scatter_add_(0, idx_flat.reshape(-1), dc_pt.reshape(-1))
        loss_pt = ((result_pt - target) ** 2).mean()
        loss_pt.backward()
        grad_pt = dc_pt.grad.clone()

        assert abs(loss_cpp.item() - loss_pt.item()) < 1e-5, \
            f"Loss mismatch: {loss_cpp.item():.6f} vs {loss_pt.item():.6f}"
        assert torch.allclose(grad_cpp, grad_pt, atol=1e-5), \
            f"Gradient mismatch: max err = {(grad_cpp - grad_pt).abs().max().item():.2e}"

    def test_gradient_overlapping_indices(self, cpp_scatter):
        """Gradient correct when multiple atoms scatter to the same voxel."""
        C, nx, ny, nz = 20, 11, 11, 11
        grid_shape = (30, 30, 30)
        map_size = 30 ** 3

        torch.manual_seed(24)

        # All atoms at the same center → all scatter to same voxels
        gx, gy, gz = grid_shape
        ny_nz = gy * gz
        half = nx // 2
        offsets = torch.arange(-half, -half + nx)
        offsets_y = torch.arange(-half, -half + ny)
        offsets_z = torch.arange(-half, -half + nz)

        center = torch.full((C,), gx // 2, dtype=torch.long)
        wa = ((center[:, None] + offsets[None, :]) % gx) * ny_nz
        wb = ((center[:, None] + offsets_y[None, :]) % gy) * gz
        wc = ((center[:, None] + offsets_z[None, :]) % gz)
        wbwc = wb[:, :, None] + wc[:, None, :]

        grad_output = torch.randn(map_size)

        # C++ path
        dc_cpp = torch.randn(C, nx, ny, nz, requires_grad=True)
        result_cpp = cpp_scatter(dc_cpp, wa, wbwc, map_size)
        result_cpp.backward(grad_output)
        grad_cpp = dc_cpp.grad.clone()

        # PyTorch reference
        dc_pt = dc_cpp.data.clone().requires_grad_(True)
        ref = _pytorch_scatter_reference(dc_pt, wa, wbwc, map_size)
        ref.backward(grad_output)
        grad_pt = dc_pt.grad.clone()

        assert torch.allclose(grad_cpp, grad_pt, atol=1e-6), \
            f"Overlapping gradient err: {(grad_cpp - grad_pt).abs().max().item():.2e}"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Verify results are independent of thread count."""

    @pytest.mark.parametrize("n_threads", [1, 2, 4, 8])
    def test_thread_count_invariance(self, cpp_scatter, n_threads):
        """Result must be identical regardless of OMP thread count."""
        C, nx, ny, nz = 200, 13, 13, 13
        grid_shape = (100, 80, 60)
        map_size = 100 * 80 * 60

        torch.manual_seed(30)
        density_cube = torch.randn(C, nx, ny, nz)
        wa, wbwc = _make_valid_indices(C, nx, ny, nz, grid_shape, seed=30)

        ref = _pytorch_scatter_reference(density_cube, wa, wbwc, map_size)

        prev_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(n_threads)
            got = cpp_scatter(density_cube, wa, wbwc, map_size)
            assert torch.allclose(ref, got, atol=1e-5), \
                f"{n_threads} threads: max err = {(ref - got).abs().max().item():.2e}"
        finally:
            torch.set_num_threads(prev_threads)
