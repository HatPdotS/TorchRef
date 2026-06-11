"""
Optimized vectorized_add_to_map with automatic CPU/GPU path selection.

This module provides optimized implementations for adding atoms to a density map
using ITC92 Gaussian parameterization. It automatically selects the best
implementation based on the device (CPU or GPU) and uses JIT-compiled kernels
for optimal performance.

Architecture:
- CPU: JIT-scripted kernel using einsum with metric tensor (efficient for CPU)
- GPU: JIT-scripted kernel using batch matmul (efficient for GPU)

Both implementations are fully differentiable and compile on import for
minimal first-call overhead.

Usage:
    from torchref.base.kernels import vectorized_add_to_map

    # Automatically selects CPU or GPU implementation based on tensor device
    density_map = vectorized_add_to_map(
        surrounding_coords, voxel_indices, density_map,
        xyz, b, inv_frac_matrix, frac_matrix, A, B, occ
    )
"""

import os
import torch

from torchref.utils.triton_dispatch import should_use_triton

# =============================================================================
# Cache directory for JIT kernels
# =============================================================================

_CACHE_DIR = os.environ.get(
    "TORCHREF_COMPILE_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "torchref", "inductor"),
)
os.makedirs(_CACHE_DIR, exist_ok=True)

__all__ = [
    "vectorized_add_to_map",
    "build_electron_density",
    "compute_metric_tensor",
    "precompute_fractional_coords",
    "warmup",
    "get_cache_dir",
    "clear_cache",
]

# =============================================================================
# Kernel state - compiled on import
# =============================================================================

_jit_cpu_kernel = None
_jit_gpu_kernel = None

# Triton kernel (lazy import, with fallback)
_triton_kernel = None
_triton_available = None  # None = not checked yet


def _get_triton_kernel():
    """Get the Triton fused kernel, or None if unavailable."""
    global _triton_kernel, _triton_available
    if _triton_available is None:
        try:
            from torchref.base.kernels.triton_kernel import fused_add_to_map_gpu

            _triton_kernel = fused_add_to_map_gpu
            _triton_available = True
        except ImportError:
            _triton_available = False
    return _triton_kernel


# =============================================================================
# Helper functions
# =============================================================================


def compute_metric_tensor(frac_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the metric tensor for calculating r² in fractional coordinates.

    The metric tensor G allows computing squared distances in Cartesian space
    from fractional coordinate differences:
        r² = diff_frac @ G @ diff_frac.T

    Parameters
    ----------
    frac_matrix : torch.Tensor
        Fractionalization matrix, shape (3, 3).

    Returns
    -------
    torch.Tensor
        Metric tensor G = frac_matrix.T @ frac_matrix, shape (3, 3).
    """
    return frac_matrix.T @ frac_matrix


def precompute_fractional_coords(
    coords_cart: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Convert Cartesian voxel coordinates to fractional coordinates.

    Parameters
    ----------
    coords_cart : torch.Tensor
        Cartesian coordinates, shape (N_atoms, N_voxels, 3).
    inv_frac_matrix : torch.Tensor
        Inverse fractionalization matrix, shape (3, 3).

    Returns
    -------
    torch.Tensor
        Fractional coordinates, shape (N_atoms, N_voxels, 3).
    """
    N_atoms, N_voxels = coords_cart.shape[:2]
    coords_flat = coords_cart.reshape(-1, 3)
    coords_frac_flat = coords_flat @ inv_frac_matrix.T
    return coords_frac_flat.reshape(N_atoms, N_voxels, 3)


# =============================================================================
# CPU JIT kernel - uses einsum with metric tensor
# =============================================================================

_JIT_CPU_CACHE_PATH = os.path.join(_CACHE_DIR, "jit_cpu_kernel.pt")


class _CpuDensityKernel(torch.nn.Module):
    """JIT-scriptable CPU density computation kernel."""

    def forward(
        self,
        coords_frac: torch.Tensor,
        voxel_indices: torch.Tensor,
        density_map: torch.Tensor,
        xyz: torch.Tensor,
        b: torch.Tensor,
        inv_frac_matrix: torch.Tensor,
        G: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        occ: torch.Tensor,
    ) -> torch.Tensor:
        # Convert xyz to fractional
        xyz_frac = xyz @ inv_frac_matrix.T

        # Compute B_total with clamp (matches original implementation)
        B_total = ((B + b[:, None]) * 0.25).clamp(min=0.1)

        # Normalization = (π / B_total)^1.5
        pi: float = 3.141592653589793
        pi_1p5: float = pi * 1.7724538509055159  # sqrt(pi)
        A_norm = A * occ[:, None] * pi_1p5 / (B_total * torch.sqrt(B_total))

        # PBC wrapping in fractional space
        diff_frac = coords_frac - xyz_frac[:, None, :]
        diff_wrapped = diff_frac - torch.round(diff_frac)

        # r² via metric tensor (efficient on CPU with einsum)
        r_squared = torch.einsum("avi,ij,avj->av", diff_wrapped, G, diff_wrapped)

        # Gaussian computation
        pi_sq: float = pi * pi
        exponents = -pi_sq * r_squared.unsqueeze(2) / B_total.unsqueeze(1)
        gaussian_terms = torch.exp(exponents)
        density = torch.einsum("ag,avg->av", A_norm, gaussian_terms)

        # Scatter add to density map
        ny: int = density_map.shape[1]
        nz: int = density_map.shape[2]
        strides = torch.tensor(
            [ny * nz, nz, 1], device=voxel_indices.device, dtype=torch.long
        )
        index_flat = torch.sum(voxel_indices.to(torch.long) * strides, dim=-1).view(-1)

        density_map.view(-1).scatter_add_(0, index_flat, density.reshape(-1))
        return density_map


def _get_jit_cpu_kernel():
    """Get or create the JIT-scripted CPU kernel."""
    global _jit_cpu_kernel

    if _jit_cpu_kernel is not None:
        return _jit_cpu_kernel

    # Try loading from cache
    if os.path.exists(_JIT_CPU_CACHE_PATH):
        try:
            _jit_cpu_kernel = torch.jit.load(_JIT_CPU_CACHE_PATH)
            return _jit_cpu_kernel
        except Exception:
            pass  # Cache corrupted, will recreate

    # Create and script the kernel
    kernel = _CpuDensityKernel()
    _jit_cpu_kernel = torch.jit.script(kernel)

    # Save to cache
    try:
        os.makedirs(os.path.dirname(_JIT_CPU_CACHE_PATH), exist_ok=True)
        torch.jit.save(_jit_cpu_kernel, _JIT_CPU_CACHE_PATH)
    except Exception:
        pass

    return _jit_cpu_kernel


# =============================================================================
# GPU JIT kernel - uses batch matmul (more efficient on GPU than einsum)
# =============================================================================

_JIT_GPU_CACHE_PATH = os.path.join(_CACHE_DIR, "jit_gpu_kernel.pt")


class _GpuDensityKernel(torch.nn.Module):
    """JIT-scriptable GPU density computation kernel."""

    def forward(
        self,
        surrounding_coords: torch.Tensor,
        voxel_indices: torch.Tensor,
        density_map: torch.Tensor,
        xyz: torch.Tensor,
        b: torch.Tensor,
        inv_frac_matrix: torch.Tensor,
        frac_matrix: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        occ: torch.Tensor,
    ) -> torch.Tensor:
        # Compute diff in Cartesian space
        diff = surrounding_coords - xyz[:, None, :]

        # Apply PBC using batch matmul (efficient on GPU)
        diff_frac = torch.matmul(diff, inv_frac_matrix.T)
        translation = torch.round(diff_frac)
        correction = torch.matmul(translation, frac_matrix.T)
        diff_wrapped = diff - correction

        # Compute r²
        r_squared = (diff_wrapped * diff_wrapped).sum(dim=-1)

        # Compute B_total with clamp
        B_total = ((B + b[:, None]) * 0.25).clamp(min=0.1)

        # Normalization = (π / B_total)^1.5
        pi: float = 3.141592653589793
        pi_sq: float = pi * pi
        pi_1p5: float = pi * 1.7724538509055159  # sqrt(pi)
        normalization = pi_1p5 / (B_total * torch.sqrt(B_total))

        # A_normalized
        A_normalized = A * occ[:, None] * normalization

        # Gaussian terms
        exponents = -pi_sq * r_squared[:, :, None] / B_total[:, None, :]
        gaussian_terms = torch.exp(exponents)

        # Density
        density = (A_normalized[:, None, :] * gaussian_terms).sum(dim=-1)

        # Scatter add to density map
        ny: int = density_map.shape[1]
        nz: int = density_map.shape[2]
        index_flat = (
            voxel_indices[:, :, 0].to(torch.int64) * (ny * nz)
            + voxel_indices[:, :, 1].to(torch.int64) * nz
            + voxel_indices[:, :, 2].to(torch.int64)
        ).flatten()

        density_map.view(-1).scatter_add_(0, index_flat, density.flatten())
        return density_map


def _get_jit_gpu_kernel():
    """Get or create the JIT-scripted GPU kernel."""
    global _jit_gpu_kernel

    if _jit_gpu_kernel is not None:
        return _jit_gpu_kernel

    # Try loading from cache
    if os.path.exists(_JIT_GPU_CACHE_PATH):
        try:
            _jit_gpu_kernel = torch.jit.load(_JIT_GPU_CACHE_PATH)
            return _jit_gpu_kernel
        except Exception:
            pass  # Cache corrupted, will recreate

    # Create and script the kernel
    kernel = _GpuDensityKernel()
    _jit_gpu_kernel = torch.jit.script(kernel)

    # Save to cache
    try:
        os.makedirs(os.path.dirname(_JIT_GPU_CACHE_PATH), exist_ok=True)
        torch.jit.save(_jit_gpu_kernel, _JIT_GPU_CACHE_PATH)
    except Exception:
        pass

    return _jit_gpu_kernel


# =============================================================================
# GPU simple implementation (fallback, no JIT)
# =============================================================================


def _add_to_map_gpu_simple(
    surrounding_coords: torch.Tensor,
    voxel_indices: torch.Tensor,
    density_map: torch.Tensor,
    xyz: torch.Tensor,
    b: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
    frac_matrix: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    occ: torch.Tensor,
) -> torch.Tensor:
    """Simple GPU implementation without JIT (for debugging)."""
    import numpy as np

    diff = surrounding_coords - xyz[:, None, :]
    diff_frac = torch.matmul(diff, inv_frac_matrix.T)
    translation = torch.round(diff_frac)
    correction = torch.matmul(translation, frac_matrix.T)
    diff_wrapped = diff - correction

    r_squared = (diff_wrapped * diff_wrapped).sum(dim=-1)

    B_total = ((B + b[:, None]) / 4).clamp(min=0.1)
    normalization = (np.pi / B_total) ** 1.5
    A_normalized = A * occ[:, None] * normalization

    exponents = -(np.pi**2) * r_squared[:, :, None] / B_total[:, None, :]
    gaussian_terms = torch.exp(exponents)
    density = (A_normalized[:, None, :] * gaussian_terms).sum(dim=-1)

    ny, nz = density_map.shape[1], density_map.shape[2]
    index_flat = (
        voxel_indices[:, :, 0].to(torch.int64) * (ny * nz)
        + voxel_indices[:, :, 1].to(torch.int64) * nz
        + voxel_indices[:, :, 2].to(torch.int64)
    ).flatten()

    density_map.view(-1).scatter_add_(0, index_flat, density.flatten())
    return density_map


# =============================================================================
# Main entry point
# =============================================================================


def vectorized_add_to_map(
    surrounding_coords: torch.Tensor,
    voxel_indices: torch.Tensor,
    density_map: torch.Tensor,
    xyz: torch.Tensor,
    b: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
    frac_matrix: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    occ: torch.Tensor,
) -> torch.Tensor:
    """
    Add atoms to density map using ITC92 Gaussian parameterization.

    Backend is chosen by the shared ``Engine`` via ``should_use_triton``: on
    GPU it uses the Triton fused kernel when Triton is permitted (CUDA+float32,
    engine AUTO/TRITON) and the pure-torch ``_add_to_map_gpu_simple`` otherwise
    (Engine.EAGER, float64, or Triton unavailable). CPU uses the JIT kernel.

    Parameters
    ----------
    surrounding_coords : torch.Tensor
        Cartesian coordinates of voxels, shape (N_atoms, N_voxels, 3).
    voxel_indices : torch.Tensor
        Indices of voxels in the map, shape (N_atoms, N_voxels, 3).
    density_map : torch.Tensor
        Electron density map to update, shape (nx, ny, nz).
    xyz : torch.Tensor
        Atom positions in Cartesian coordinates, shape (N_atoms, 3).
    b : torch.Tensor
        Isotropic B-factors, shape (N_atoms,).
    inv_frac_matrix : torch.Tensor
        Inverse fractionalization matrix, shape (3, 3).
    frac_matrix : torch.Tensor
        Fractionalization matrix, shape (3, 3).
    A : torch.Tensor
        ITC92 amplitude coefficients, shape (N_atoms, 5).
    B : torch.Tensor
        ITC92 width coefficients, shape (N_atoms, 5).
    occ : torch.Tensor
        Atomic occupancies, shape (N_atoms,).

    Returns
    -------
    torch.Tensor
        Updated electron density map (modified in-place).
    """
    if density_map.device.type == "cuda":
        # The shared Engine is the only switch: use the Triton kernel when it
        # permits (CUDA + float32, engine AUTO/TRITON); otherwise — Engine.EAGER,
        # float64, or Triton unavailable — the pure-torch, double-differentiable
        # ``_add_to_map_gpu_simple``.
        if should_use_triton(xyz):
            triton_fn = _get_triton_kernel()
            if triton_fn is not None:
                return triton_fn(
                    surrounding_coords,
                    voxel_indices,
                    density_map,
                    xyz,
                    b,
                    inv_frac_matrix,
                    frac_matrix,
                    A,
                    B,
                    occ,
                )
        return _add_to_map_gpu_simple(
            surrounding_coords,
            voxel_indices,
            density_map,
            xyz,
            b,
            inv_frac_matrix,
            frac_matrix,
            A,
            B,
            occ,
        )
    else:
        # CPU: Convert to fractional coords and use metric tensor
        coords_frac = precompute_fractional_coords(surrounding_coords, inv_frac_matrix)
        G = compute_metric_tensor(frac_matrix)
        kernel = _get_jit_cpu_kernel()
        return kernel(
            coords_frac,
            voxel_indices,
            density_map,
            xyz,
            b,
            inv_frac_matrix,
            G,
            A,
            B,
            occ,
        )


def build_electron_density(
    surrounding_coords: torch.Tensor,
    voxel_indices: torch.Tensor,
    density_map: torch.Tensor,
    xyz: torch.Tensor,
    b: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
    frac_matrix: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    occ: torch.Tensor,
) -> torch.Tensor:
    """
    Build electron density map from atomic parameters.

    This is an alias for vectorized_add_to_map for semantic clarity.

    Parameters
    ----------
    surrounding_coords : torch.Tensor
        Cartesian coordinates of voxels, shape (N_atoms, N_voxels, 3).
    voxel_indices : torch.Tensor
        Indices of voxels in the map, shape (N_atoms, N_voxels, 3).
    density_map : torch.Tensor
        Electron density map to update, shape (nx, ny, nz).
    xyz : torch.Tensor
        Atom positions in Cartesian coordinates, shape (N_atoms, 3).
    b : torch.Tensor
        Isotropic B-factors, shape (N_atoms,).
    inv_frac_matrix : torch.Tensor
        Inverse fractionalization matrix, shape (3, 3).
    frac_matrix : torch.Tensor
        Fractionalization matrix, shape (3, 3).
    A : torch.Tensor
        ITC92 amplitude coefficients, shape (N_atoms, 5).
    B : torch.Tensor
        ITC92 width coefficients, shape (N_atoms, 5).
    occ : torch.Tensor
        Atomic occupancies, shape (N_atoms,).

    Returns
    -------
    torch.Tensor
        Updated electron density map.
    """
    return vectorized_add_to_map(
        surrounding_coords,
        voxel_indices,
        density_map,
        xyz,
        b,
        inv_frac_matrix,
        frac_matrix,
        A,
        B,
        occ,
    )


# =============================================================================
# Utilities
# =============================================================================


def warmup(device: str = "auto") -> None:
    """
    Pre-compile kernels to avoid compilation overhead during first use.

    Parameters
    ----------
    device : str
        Device to warmup: "cpu", "cuda", or "auto" (default).
    """
    devices = []
    if device == "auto":
        devices.append("cpu")
        if torch.cuda.is_available():
            devices.append("cuda")
    else:
        devices.append(device)

    n_atoms, n_voxels = 256, 1000
    grid_shape = (64, 64, 64)

    for dev in devices:
        torch_device = torch.device(dev)
        surrounding_coords = torch.randn(n_atoms, n_voxels, 3, device=torch_device)
        voxel_indices = torch.randint(
            0, 64, (n_atoms, n_voxels, 3), device=torch_device
        )
        density_map = torch.zeros(grid_shape, device=torch_device)
        xyz = torch.randn(n_atoms, 3, device=torch_device)
        b = torch.rand(n_atoms, device=torch_device) * 50 + 10
        inv_frac_matrix = torch.eye(3, device=torch_device) * 0.02
        frac_matrix = torch.eye(3, device=torch_device) * 50
        A = torch.rand(n_atoms, 5, device=torch_device)
        B = torch.rand(n_atoms, 5, device=torch_device) * 10 + 1
        occ = torch.ones(n_atoms, device=torch_device)

        _ = vectorized_add_to_map(
            surrounding_coords,
            voxel_indices,
            density_map,
            xyz,
            b,
            inv_frac_matrix,
            frac_matrix,
            A,
            B,
            occ,
        )


def get_cache_dir() -> str:
    """Return the path to the JIT kernel cache directory."""
    return _CACHE_DIR


def clear_cache() -> None:
    """Clear the JIT kernel cache."""
    import shutil

    global _jit_cpu_kernel, _jit_gpu_kernel
    _jit_cpu_kernel = None
    _jit_gpu_kernel = None

    if os.path.exists(_CACHE_DIR):
        shutil.rmtree(_CACHE_DIR)
        os.makedirs(_CACHE_DIR, exist_ok=True)


# =============================================================================
# Compile kernels on import
# =============================================================================

# CPU kernel always compiles (fast, ~0.1s)
_get_jit_cpu_kernel()

# GPU kernel compiles if CUDA is available
if torch.cuda.is_available():
    _get_jit_gpu_kernel()
