"""``vectorized_add_to_map``: the voxel-list density splat, chosen by device.

Adds atoms to a density map under the ITC92 5-Gaussian parameterization. On CUDA the
``triton`` row of :data:`torchref.base.targets._dispatch.TARGET_BACKENDS` decides (CUDA
float32, off under ``force_portable``): the fused Triton kernel where it matches, the
eager, double-differentiable ``_add_to_map_gpu_simple`` otherwise. Every other device
runs a TorchScript einsum kernel with a metric tensor, scripted on first use or by
:func:`warmup` and cached under :func:`get_cache_dir`.
"""

import glob
import hashlib
import inspect
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

__all__ = [
    "vectorized_add_to_map",
    "compute_metric_tensor",
    "precompute_fractional_coords",
    "warmup",
    "get_cache_dir",
    "clear_cache",
]

# =============================================================================
# Kernel state - built on first use
# =============================================================================

_jit_cpu_kernel = None

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
        # Compiled by TorchScript: no Tensor.new_tensor, no config dtype getters.
        strides = torch.tensor(
            [ny * nz, nz, 1],
            device=voxel_indices.device,
            # dtype-ok: int64 strides make the flat voxel index int64; scatter_add_ requires int64 on torch < 2.8
            dtype=torch.long,
        )
        index_flat = torch.sum(voxel_indices.to(torch.long) * strides, dim=-1).view(-1)  # dtype-ok: voxel indices flattened for scatter; indexing requires long

        density_map.view(-1).scatter_add_(0, index_flat, density.reshape(-1))
        return density_map


def _jit_cpu_cache_path() -> str:
    """Cache file of the scripted CPU kernel.

    Named after the torch version and a hash of the kernel source, so a file written by
    another torch build or for an edited kernel is never loaded.
    """
    source = inspect.getsource(_CpuDensityKernel).encode()
    digest = hashlib.sha256(source).hexdigest()[:16]
    name = f"jit_cpu_kernel-{torch.__version__}-{digest}.pt"
    return os.path.join(_CACHE_DIR, name)


def _get_jit_cpu_kernel():
    """Get or create the JIT-scripted CPU kernel."""
    global _jit_cpu_kernel

    if _jit_cpu_kernel is not None:
        return _jit_cpu_kernel

    cache_path = _jit_cpu_cache_path()
    if os.path.exists(cache_path):
        try:
            _jit_cpu_kernel = torch.jit.load(cache_path)
            return _jit_cpu_kernel
        except Exception:
            pass  # Cache corrupted, will recreate

    # Create and script the kernel
    kernel = _CpuDensityKernel()
    _jit_cpu_kernel = torch.jit.script(kernel)

    # Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.jit.save(_jit_cpu_kernel, cache_path)
    except Exception:
        pass

    return _jit_cpu_kernel


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
    """Splat eagerly with Cartesian wrapping; the CUDA path when Triton is not used."""
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

    On CUDA the ``triton`` row of ``TARGET_BACKENDS`` picks the fused Triton kernel
    (float32, Triton importable, ``force_portable`` off), else the eager
    ``_add_to_map_gpu_simple``; other devices run the TorchScript kernel.

    Parameters
    ----------
    surrounding_coords, voxel_indices : torch.Tensor
        Cartesian coordinates and map indices of the voxels, ``(N_atoms, N_voxels, 3)``.
    density_map : torch.Tensor
        Map to update, ``(nx, ny, nz)``.
    xyz, b, occ : torch.Tensor
        Positions ``(N_atoms, 3)``, isotropic B-factors and occupancies ``(N_atoms,)``.
    inv_frac_matrix, frac_matrix : torch.Tensor
        Cartesian-to-fractional and fractional-to-Cartesian, ``(3, 3)``.
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
        # TARGET_BACKENDS probes the target Triton kernels, not the fused splat, so a
        # splat that fails to import still lands on the eager path.
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
    """Drop the scripted kernel from memory and delete its cache files.

    Only the ``jit_cpu_kernel*.pt`` files in :func:`get_cache_dir` are removed; the
    directory and anything else in it are kept, since ``TORCHREF_COMPILE_CACHE`` may
    name a directory shared with other caches.
    """
    global _jit_cpu_kernel
    _jit_cpu_kernel = None

    for path in glob.glob(os.path.join(_CACHE_DIR, "jit_cpu_kernel*.pt")):
        os.remove(path)
