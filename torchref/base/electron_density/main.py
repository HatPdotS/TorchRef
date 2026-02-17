"""
Central electron density building with automatic backend selection.

Dispatches to the optimal implementation based on device and available backends:

- CUDA + Triton: Fused kernel (fastest, skips find_relevant_voxels entirely)
- CUDA + Triton: Two-step Triton kernel (find_relevant_voxels + Triton density)
- CUDA + JIT: Two-step JIT kernel
- CPU: JIT CPU kernel with metric tensor

Override with env var TORCHREF_ATOM_PLACEMENT_GPU_MODE:
  "triton" (default) — fused Triton > two-step Triton > JIT
  "jit"              — JIT kernel only
  "simple"           — simple GPU kernel (debugging)
"""

import os
from typing import Optional

import torch

_GPU_MODE = os.environ.get("TORCHREF_ATOM_PLACEMENT_GPU_MODE", "triton")

# Lazy-loaded fused Triton backend
_fused_fn = None
_fused_checked = False


def _get_fused_triton():
    """Return the fused Triton kernel, or None if unavailable."""
    global _fused_fn, _fused_checked
    if not _fused_checked:
        try:
            from torchref.base.kernels.triton_kernel import fused_find_and_place_atoms
            _fused_fn = fused_find_and_place_atoms
        except ImportError:
            pass
        _fused_checked = True
    return _fused_fn


def build_electron_density(
    real_space_grid: torch.Tensor,
    xyz_iso: torch.Tensor,
    adp_iso: torch.Tensor,
    occ_iso: torch.Tensor,
    A_iso: torch.Tensor,
    B_iso: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
    frac_matrix: torch.Tensor,
    radius_angstrom: float,
    voxel_size: torch.Tensor,
    xyz_aniso: Optional[torch.Tensor] = None,
    u_aniso: Optional[torch.Tensor] = None,
    occ_aniso: Optional[torch.Tensor] = None,
    A_aniso: Optional[torch.Tensor] = None,
    B_aniso: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build an electron density map from atomic parameters.

    Selects the fastest available backend automatically. On CUDA, tries
    the fused Triton kernel first (eliminates find_relevant_voxels), then
    falls back to two-step Triton or JIT. On CPU, uses the JIT kernel.

    Parameters
    ----------
    real_space_grid : torch.Tensor
        Coordinate grid, shape (nx, ny, nz, 3).
    xyz_iso : torch.Tensor
        Isotropic atom positions, shape (n_iso, 3).
    adp_iso : torch.Tensor
        Isotropic B-factors, shape (n_iso,).
    occ_iso : torch.Tensor
        Isotropic occupancies, shape (n_iso,).
    A_iso, B_iso : torch.Tensor
        ITC92 coefficients, shape (n_iso, 5).
    inv_frac_matrix : torch.Tensor
        Cartesian-to-fractional matrix, shape (3, 3).
    frac_matrix : torch.Tensor
        Fractional-to-Cartesian matrix, shape (3, 3).
    radius_angstrom : float
        Radius around each atom in Angstroms.
    voxel_size : torch.Tensor
        Voxel dimensions, shape (3,).
    xyz_aniso : torch.Tensor, optional
        Anisotropic atom positions, shape (n_aniso, 3).
    u_aniso : torch.Tensor, optional
        Anisotropic U parameters, shape (n_aniso, 6).
    occ_aniso : torch.Tensor, optional
        Anisotropic occupancies, shape (n_aniso,).
    A_aniso, B_aniso : torch.Tensor, optional
        ITC92 coefficients for anisotropic atoms, shape (n_aniso, 5).
    dtype : torch.dtype, optional
        Float dtype for the density map. Default torch.float32.

    Returns
    -------
    torch.Tensor
        Electron density map, shape (nx, ny, nz).
    """
    device = real_space_grid.device
    density_map = torch.zeros(
        real_space_grid.shape[:-1], dtype=dtype, device=device,
    )

    # --- isotropic atoms ---
    if len(xyz_iso) > 0:
        density_map = _add_isotropic(
            real_space_grid, density_map,
            xyz_iso, adp_iso, occ_iso, A_iso, B_iso,
            inv_frac_matrix, frac_matrix,
            radius_angstrom, voxel_size,
        )

    # --- anisotropic atoms ---
    if xyz_aniso is not None and len(xyz_aniso) > 0:
        density_map = _add_anisotropic(
            real_space_grid, density_map,
            xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso,
            inv_frac_matrix, frac_matrix,
            radius_angstrom,
        )

    return density_map


# =========================================================================
# Internal dispatch helpers
# =========================================================================

def _add_isotropic(
    real_space_grid, density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
):
    """Add isotropic atoms using the best available backend."""
    is_cuda = density_map.device.type == "cuda"

    # Tier 1: fused Triton (single kernel, no find_relevant_voxels)
    if is_cuda and _GPU_MODE not in ("jit", "simple"):
        fused = _get_fused_triton()
        if fused is not None:
            try:
                return fused(
                    real_space_grid, density_map, xyz, adp,
                    inv_frac_matrix, frac_matrix, A, B, occ,
                    radius_angstrom, voxel_size,
                )
            except Exception:
                pass  # fall through to two-step

    # Tier 2+: find_relevant_voxels + vectorized_add_to_map
    # (vectorized_add_to_map has its own Triton/JIT/simple dispatch)
    from torchref.base.electron_density.voxel_utils import find_relevant_voxels
    from torchref.base.kernels import vectorized_add_to_map

    surrounding_coords, voxel_indices = find_relevant_voxels(
        real_space_grid, xyz,
        radius_angstrom=radius_angstrom,
        inv_frac_matrix=inv_frac_matrix,
    )
    return vectorized_add_to_map(
        surrounding_coords, voxel_indices, density_map,
        xyz, adp, inv_frac_matrix, frac_matrix, A, B, occ,
    )


def _add_anisotropic(
    real_space_grid, density_map, xyz, u, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom,
):
    """Add anisotropic atoms (always two-step, no Triton kernel yet)."""
    from torchref.base.electron_density.voxel_utils import find_relevant_voxels
    from torchref.base.electron_density.map_building import vectorized_add_to_map_aniso

    surrounding_coords, voxel_indices = find_relevant_voxels(
        real_space_grid, xyz,
        radius_angstrom=radius_angstrom,
        inv_frac_matrix=inv_frac_matrix,
    )
    return vectorized_add_to_map_aniso(
        surrounding_coords, voxel_indices, density_map,
        xyz, u, inv_frac_matrix, frac_matrix, A, B, occ,
    )
