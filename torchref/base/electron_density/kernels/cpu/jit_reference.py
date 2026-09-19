"""``vectorized_add_to_map`` with automatic CPU/GPU path selection.

Adds atoms to a density map under the ITC92 5-Gaussian parameterization, choosing the
implementation from the tensor device: CPU uses a JIT-scripted einsum kernel with a metric
tensor; on GPU, when the shared targets gate permits Triton (CUDA + float32, dispatch
AUTO/TRITON), the fused Triton branch is selected, otherwise the pure-torch,
double-differentiable ``_add_to_map_gpu_simple``. The CPU JIT and simple GPU paths are
fully differentiable and compile on import.
"""

import os
import torch

from torchref.base.targets._dispatch import use_triton

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
            from torchref.base.electron_density.kernels.cuda.fused import fused_add_to_map_gpu

            _triton_kernel = fused_add_to_map_gpu
            _triton_available = True
        except ImportError:
            _triton_available = False
    return _triton_kernel


# =============================================================================
# Helper functions
# =============================================================================


def compute_metric_tensor(frac_matrix: torch.Tensor) -> torch.Tensor:
    """Metric tensor ``G = frac_matrix.T @ frac_matrix``, ``(3, 3)``, so that
    ``r^2 = diff_frac @ G @ diff_frac.T`` gives Cartesian squared distances from fractional
    coordinate differences.
    """
    return frac_matrix.T @ frac_matrix


def precompute_fractional_coords(
    coords_cart: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
) -> torch.Tensor:
    """Cartesian ``(N_atoms, N_voxels, 3)`` coordinates converted to fractional via
    ``inv_frac_matrix`` ``(3, 3)``.
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
            [ny * nz, nz, 1], device=voxel_indices.device, dtype=torch.long  # dtype-ok: CPU-kernel strides for flat voxel index arithmetic; indexing requires long
        )
        index_flat = torch.sum(voxel_indices.to(torch.long) * strides, dim=-1).view(-1)  # dtype-ok: voxel indices flattened for scatter; indexing requires long

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
            voxel_indices[:, :, 0].to(torch.int64) * (ny * nz)  # dtype-ok: voxel-index flat-arithmetic term for scatter; requires int64
            + voxel_indices[:, :, 1].to(torch.int64) * nz  # dtype-ok: voxel-index flat-arithmetic term for scatter; requires int64
            + voxel_indices[:, :, 2].to(torch.int64)  # dtype-ok: voxel-index flat-arithmetic term for scatter; requires int64
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
        voxel_indices[:, :, 0].to(torch.int64) * (ny * nz)  # dtype-ok: voxel-index flat-arithmetic term for scatter; requires int64
        + voxel_indices[:, :, 1].to(torch.int64) * nz  # dtype-ok: voxel-index flat-arithmetic term for scatter; requires int64
        + voxel_indices[:, :, 2].to(torch.int64)  # dtype-ok: voxel-index flat-arithmetic term for scatter; requires int64
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
    """Add atoms to a density map using the ITC92 5-Gaussian parameterization.

    The backend follows the shared targets gate: the Triton fused kernel where Triton is
    permitted (CUDA + float32, engine AUTO/TRITON), the pure-torch
    ``_add_to_map_gpu_simple`` otherwise (force_portable, float64, no Triton), and the JIT
    kernel on CPU.

    Parameters
    ----------
    surrounding_coords, voxel_indices : torch.Tensor
        Cartesian coordinates and map indices of the voxels, ``(N_atoms, N_voxels, 3)``.
    density_map : torch.Tensor
        Map to update, ``(nx, ny, nz)``.
    xyz, b, occ : torch.Tensor
        Positions ``(N_atoms, 3)``, isotropic B-factors and occupancies ``(N_atoms,)``.
    inv_frac_matrix, frac_matrix : torch.Tensor
        Fractionalization matrix and its inverse, ``(3, 3)``.
    A, B : torch.Tensor
        ITC92 amplitudes and widths, ``(N_atoms, 5)`` each.

    Returns
    -------
    torch.Tensor
        The updated map. **In-place mutation is not guaranteed** -- the CPU/JIT and simple
        GPU branches mutate ``density_map``, while the Triton branch returns a new clone and
        leaves the input unchanged, so callers must always use the returned value.
    """
    if density_map.device.type == "cuda":
        # The shared targets gate is the only switch: use the Triton kernel when it
        # permits (CUDA + float32); otherwise — force_portable,
        # float64, or Triton unavailable — the pure-torch, double-differentiable
        # ``_add_to_map_gpu_simple``.
        if use_triton(xyz):
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
    """Alias for :func:`vectorized_add_to_map`, taking the same *voxel-level* arguments.

    Distinct from :func:`torchref.base.electron_density.main.build_electron_density`, which
    takes *atomic* parameters and performs the full table-based variable-radius dispatch.
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
    """Pre-compile the kernels for ``device`` ("cpu", "cuda" or "auto") so the first
    real call pays no compilation cost.
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
