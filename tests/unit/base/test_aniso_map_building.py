"""Unit tests for anisotropic electron-density map building (CPU).

Certifies the eager ``vectorized_add_to_map_aniso`` reference that the Triton
kernel is validated against (iso-reduction identity + float64 gradcheck), and
the optimized CPU box-splat kernel ``_add_anisotropic_cpu`` (iso-reduction +
forward/gradient parity against the eager two-step).

Note: ``_add_anisotropic_cpu`` goes through ``_do_structured_scatter`` (the
float32-only C++ scatter), so its parity with the float64 eager reference is at
float32 precision. Its gradients are validated by parity against the eager
reference (which is gradcheck-certified here); a direct gradcheck on the box
kernel is unreliable (float32 scatter + the discontinuous ``round`` voxel-center
index), exactly as for the isotropic separable kernel.
"""

import math

import pytest
import torch

from torchref.base.electron_density.map_building import vectorized_add_to_map_aniso
from torchref.base.electron_density.voxel_utils import find_relevant_voxels
from torchref.base.kernels import vectorized_add_to_map
from torchref.model.sf_fft import SfFFT
from torchref.symmetry import Cell

pytestmark = pytest.mark.unit

_EIGHT_PI_SQ = 8.0 * math.pi**2


def _grid(dtype=torch.float64):
    cell = Cell([12.0, 13.0, 14.0, 90.0, 90.0, 90.0])
    fft = SfFFT(cell, spacegroup="P1", max_res=3.0,
                dtype_float=dtype, device="cpu")
    fft.setup_grid(max_res=3.0)
    return (
        fft.real_space_grid.to(dtype),
        fft.inv_fractional_matrix.to(dtype),
        fft.fractional_matrix.to(dtype),
        3.0,  # splat radius for the legacy fixed-radius reference kernels
    )


def _atoms(dtype=torch.float64):
    # fractional positions chosen away from voxel-center half-boundaries
    xyz = torch.tensor([[4.1, 4.3, 5.2], [7.7, 8.9, 9.1]], dtype=dtype)
    A = torch.rand(2, 5, dtype=dtype) + 0.5
    B = torch.rand(2, 5, dtype=dtype) + 1.0
    occ = torch.rand(2, dtype=dtype) + 0.5
    b = torch.tensor([8.0, 14.0], dtype=dtype)  # isotropic B-factors
    return xyz, A, B, occ, b


def test_aniso_reduces_to_isotropic():
    """U = b/(8π²)·I (diagonal, isotropic) must reproduce the iso splat."""
    grid, inv_frac, frac, rad = _grid()
    xyz, A, B, occ, b = _atoms()

    surr, idx = find_relevant_voxels(grid, xyz, radius_angstrom=rad, inv_frac_matrix=inv_frac)
    iso_map = vectorized_add_to_map(
        surr, idx, torch.zeros(grid.shape[:3], dtype=torch.float64),
        xyz, b, inv_frac, frac, A, B, occ,
    )

    u_iso = torch.zeros(2, 6, dtype=torch.float64)
    u_iso[:, 0] = u_iso[:, 1] = u_iso[:, 2] = b / _EIGHT_PI_SQ
    aniso_map = vectorized_add_to_map_aniso(
        surr, idx, torch.zeros(grid.shape[:3], dtype=torch.float64),
        xyz, u_iso, inv_frac, frac, A, B, occ,
    )
    assert torch.allclose(iso_map, aniso_map, atol=1e-8, rtol=1e-6)


def test_aniso_map_gradcheck():
    grid, inv_frac, frac, rad = _grid()
    xyz0, A, B, occ0, _ = _atoms()
    # genuinely anisotropic U (positive-definite)
    u0 = torch.tensor(
        [[0.12, 0.10, 0.14, 0.02, 0.01, 0.015],
         [0.09, 0.13, 0.11, -0.01, 0.02, 0.005]],
        dtype=torch.float64,
    )
    w = torch.randn(grid.shape[:3], dtype=torch.float64)

    xyz = xyz0.clone().requires_grad_()
    u = u0.clone().requires_grad_()
    occ = occ0.clone().requires_grad_()

    def f(xyz, u, occ):
        surr, idx = find_relevant_voxels(
            grid, xyz, radius_angstrom=rad, inv_frac_matrix=inv_frac
        )
        dm = vectorized_add_to_map_aniso(
            surr, idx, torch.zeros(grid.shape[:3], dtype=torch.float64),
            xyz, u, inv_frac, frac, A, B, occ,
        )
        return (dm * w).sum()

    assert torch.autograd.gradcheck(f, (xyz, u, occ), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# Optimized CPU box-splat kernel: _add_anisotropic_cpu
# ---------------------------------------------------------------------------
def _grid_full(dtype=torch.float64):
    """Like _grid but also returns real_space_grid + voxel_size for the CPU kernel."""
    cell = Cell([12.0, 13.0, 14.0, 90.0, 90.0, 90.0])
    fft = SfFFT(cell, spacegroup="P1", max_res=3.0,
                dtype_float=dtype, device="cpu")
    fft.setup_grid(max_res=3.0)
    return fft


_ANISO_U = torch.tensor(
    [[0.12, 0.10, 0.14, 0.02, 0.01, 0.015],
     [0.09, 0.13, 0.11, -0.01, 0.02, 0.005]],
    dtype=torch.float64,
)


def test_cpu_aniso_reduces_to_isotropic():
    """_add_anisotropic_cpu with U=b/(8π²)·I == _add_isotropic_cpu_separable(b)."""
    import torchref.base.electron_density.main as ed

    fft = _grid_full()
    grid = fft.real_space_grid
    inv_frac, frac = fft.inv_fractional_matrix, fft.fractional_matrix
    vox, rad = fft.voxel_size, 3.0
    shape = grid.shape[:3]
    xyz, A, B, occ, b = _atoms()

    iso = ed._add_isotropic_cpu_separable(
        torch.zeros(shape, dtype=torch.float64), xyz, b, occ, A, B,
        inv_frac, frac, shape, vox, rad,
    )
    u_iso = torch.zeros(2, 6, dtype=torch.float64)
    u_iso[:, 0] = u_iso[:, 1] = u_iso[:, 2] = b / _EIGHT_PI_SQ
    ani = ed._add_anisotropic_cpu(
        grid, torch.zeros(shape, dtype=torch.float64), xyz, u_iso, occ, A, B,
        inv_frac, frac, rad, vox,
    )
    assert torch.allclose(iso, ani, atol=1e-4, rtol=1e-3)


def test_cpu_aniso_matches_eager():
    """Forward map + xyz/u/occ gradients match the eager two-step reference."""
    import torchref.base.electron_density.main as ed

    fft = _grid_full()
    grid = fft.real_space_grid
    inv_frac, frac = fft.inv_fractional_matrix, fft.fractional_matrix
    vox, rad = fft.voxel_size, 3.0
    shape = grid.shape[:3]
    xyz0, A, B, occ0, _ = _atoms()
    u0 = _ANISO_U
    w = torch.randn(shape, dtype=torch.float64)

    def run(kind):
        xyz = xyz0.clone().requires_grad_()
        u = u0.clone().requires_grad_()
        occ = occ0.clone().requires_grad_()
        if kind == "cpu":
            dm = ed._add_anisotropic_cpu(
                grid, torch.zeros(shape, dtype=torch.float64),
                xyz, u, occ, A, B, inv_frac, frac, rad, vox,
            )
        else:
            dm = ed._add_anisotropic_original(
                grid, torch.zeros(shape, dtype=torch.float64),
                xyz, u, occ, A, B, inv_frac, frac, rad,
            )
        (dm * w).sum().backward()
        return dm.detach(), xyz.grad, u.grad, occ.grad

    c = run("cpu")
    e = run("orig")
    # float32-precision parity (C++ structured scatter)
    assert torch.allclose(c[0], e[0], atol=1e-4, rtol=1e-3)   # density map
    assert torch.allclose(c[1], e[1], atol=1e-3, rtol=1e-2)   # grad xyz
    assert torch.allclose(c[2], e[2], atol=1e-3, rtol=1e-2)   # grad U
    assert torch.allclose(c[3], e[3], atol=1e-3, rtol=1e-2)   # grad occ
