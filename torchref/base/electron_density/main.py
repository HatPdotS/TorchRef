"""
Central electron density building with automatic backend selection.

Dispatches to the optimal implementation based on device and available backends:

GPU backends (``ISO_MAP_ENGINE_GPU``):
  "separable_triton" — Tier 0, separable 1-D table lookups (default)
  "fused_triton"     — Tier 1, fused Triton kernel
  "original"         — Tier 2, find_relevant_voxels + vectorized_add_to_map
  "auto"             — try separable → fused → original (legacy default)

CPU backends (``ISO_MAP_ENGINE_CPU``):
  "separable"          — separable Gaussian splatting (default)
  "separable_compiled" — torch.compile'd separable
  "fused"              — fused fractional-space kernel
  "original"           — find_relevant_voxels + vectorized_add_to_map (same as GPU Tier 2)

Override at import time via environment variables::

    TORCHREF_ISO_MAP_ENGINE_GPU=auto
    TORCHREF_ISO_MAP_ENGINE_CPU=fused

or at runtime by setting the module-level variables directly::

    import torchref.base.electron_density.main as ed
    ed.ISO_MAP_ENGINE_GPU = "original"
    ed.ISO_MAP_ENGINE_CPU = "fused"
"""

import math
import os
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Engine selection — set via env vars or overwrite at runtime
# ---------------------------------------------------------------------------
ISO_MAP_ENGINE_GPU: str = os.environ.get("TORCHREF_ISO_MAP_ENGINE_GPU", "auto")
ISO_MAP_ENGINE_CPU: str = os.environ.get("TORCHREF_ISO_MAP_ENGINE_CPU", "separable")

# Legacy env var (backward compat) — only consulted when GPU engine is "auto"
_GPU_MODE = os.environ.get("TORCHREF_ATOM_PLACEMENT_GPU_MODE", "triton")

# Lazy-loaded Triton backends
_fused_fn = None
_fused_checked = False
_separable_fn = None
_separable_checked = False


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


def _get_separable_triton():
    """Return the separable Triton kernel, or None if unavailable."""
    global _separable_fn, _separable_checked
    if not _separable_checked:
        try:
            from torchref.base.kernels.separable_triton_kernel import (
                separable_density_gpu,
            )
            _separable_fn = separable_density_gpu
        except ImportError:
            pass
        _separable_checked = True
    return _separable_fn


# Lazy-loaded C++ parallel scatter for CPU
_cpp_scatter_fn = None
_cpp_scatter_checked = False


def _get_cpp_scatter():
    """Return the C++ parallel scatter_add, or None if unavailable.

    Eagerly triggers the C++ compilation so that failures (missing ninja,
    unsupported compiler flags, etc.) are caught here rather than mid-calculation.
    """
    global _cpp_scatter_fn, _cpp_scatter_checked
    if not _cpp_scatter_checked:
        try:
            from torchref.base.kernels.cpu_scatter import structured_scatter_add, _get_module
            # Trigger compilation now — _get_module returns None on failure
            if _get_module() is not None:
                _cpp_scatter_fn = structured_scatter_add
        except Exception:
            pass
        _cpp_scatter_checked = True
    return _cpp_scatter_fn


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
    """Add isotropic atoms using the backend selected by ISO_MAP_ENGINE_*."""
    is_cuda = density_map.device.type == "cuda"

    if is_cuda:
        return _add_isotropic_gpu(
            real_space_grid, density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
        )
    else:
        return _add_isotropic_cpu(
            real_space_grid, density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
        )


def _add_isotropic_gpu(
    real_space_grid, density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
):
    """GPU dispatch for isotropic atoms, controlled by ISO_MAP_ENGINE_GPU."""
    engine = ISO_MAP_ENGINE_GPU

    if engine == "separable_triton":
        fn = _get_separable_triton()
        if fn is None:
            raise RuntimeError("Separable Triton kernel not available")
        return fn(
            density_map, xyz, adp,
            inv_frac_matrix, frac_matrix, A, B, occ,
            radius_angstrom,
        )

    if engine == "fused_triton":
        fn = _get_fused_triton()
        if fn is None:
            raise RuntimeError("Fused Triton kernel not available")
        return fn(
            real_space_grid, density_map, xyz, adp,
            inv_frac_matrix, frac_matrix, A, B, occ,
            radius_angstrom, voxel_size,
        )

    if engine == "original":
        return _add_isotropic_original(
            real_space_grid, density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_angstrom,
        )

    if engine == "auto":
        # Legacy fallback: try separable → fused → original
        if _GPU_MODE not in ("jit", "simple"):
            separable = _get_separable_triton()
            if separable is not None:
                try:
                    return separable(
                        density_map, xyz, adp,
                        inv_frac_matrix, frac_matrix, A, B, occ,
                        radius_angstrom,
                    )
                except Exception:
                    pass

        if _GPU_MODE not in ("jit", "simple"):
            fused = _get_fused_triton()
            if fused is not None:
                try:
                    return fused(
                        real_space_grid, density_map, xyz, adp,
                        inv_frac_matrix, frac_matrix, A, B, occ,
                        radius_angstrom, voxel_size,
                    )
                except Exception:
                    pass

        return _add_isotropic_original(
            real_space_grid, density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_angstrom,
        )

    raise ValueError(
        f"Unknown ISO_MAP_ENGINE_GPU={engine!r}. "
        f"Choose from: auto, separable_triton, fused_triton, original"
    )


def _add_isotropic_original(
    real_space_grid, density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom,
):
    """Tier 2+: find_relevant_voxels + vectorized_add_to_map."""
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


def _add_isotropic_cpu(
    real_space_grid, density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
):
    """CPU dispatch for isotropic atoms, controlled by ISO_MAP_ENGINE_CPU."""
    engine = ISO_MAP_ENGINE_CPU
    grid_shape_tuple = real_space_grid.shape[:3]

    if engine == "separable":
        return _add_isotropic_cpu_separable(
            density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix,
            grid_shape_tuple, voxel_size, radius_angstrom,
        )

    if engine == "separable_compiled":
        return _add_isotropic_cpu_separable_compiled(
            density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix,
            grid_shape_tuple, voxel_size, radius_angstrom,
        )

    if engine == "fused":
        return _add_isotropic_cpu_fused(
            density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix,
            grid_shape_tuple, voxel_size, radius_angstrom,
        )

    if engine == "original":
        return _add_isotropic_original(
            real_space_grid, density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_angstrom,
        )

    raise ValueError(
        f"Unknown ISO_MAP_ENGINE_CPU={engine!r}. "
        f"Choose from: separable, separable_compiled, fused, original"
    )


# =========================================================================
# Fused CPU implementation
# =========================================================================

# Cached radius mask to avoid recomputing every call
_cached_radius_offsets = None
_cached_radius_key = None


def _get_radius_offsets(voxel_size, radius_angstrom, device):
    """Get or compute the integer offsets within a spherical radius.

    Cached across calls since the result depends only on voxel_size and radius.
    """
    global _cached_radius_offsets, _cached_radius_key

    key = (tuple(voxel_size.tolist()), float(radius_angstrom), device)
    if _cached_radius_key == key and _cached_radius_offsets is not None:
        return _cached_radius_offsets

    min_voxelsize = voxel_size.min()
    box_radius = int(math.ceil(radius_angstrom / min_voxelsize.item()))
    r = torch.arange(-box_radius, box_radius + 1, device=device)
    gx, gy, gz = torch.meshgrid(r, r, r, indexing="ij")
    offsets_all = torch.stack((gx, gy, gz), dim=-1)
    dist = torch.sqrt(((offsets_all.float() * voxel_size) ** 2).sum(-1))
    local_offsets = offsets_all[dist <= radius_angstrom]  # (R, 3) int

    _cached_radius_offsets = local_offsets
    _cached_radius_key = key
    return local_offsets


# =========================================================================
# Separable Gaussian splatting
# =========================================================================

# Cached box radius for separable approach
_cached_separable_data = None
_cached_separable_key = None


def _get_box_radius(voxel_size, radius_angstrom, device):
    """Get axis offsets and box size for separable Gaussian splatting (cached).

    Returns
    -------
    axis_offsets : torch.Tensor
        Integer offsets [-box_radius, ..., box_radius], shape (n_axis,).
    n_axis : int
        Cube side length (2 * box_radius + 1).
    """
    global _cached_separable_data, _cached_separable_key

    key = (tuple(voxel_size.tolist()), float(radius_angstrom), device)
    if _cached_separable_key == key and _cached_separable_data is not None:
        return _cached_separable_data

    min_voxelsize = voxel_size.min()
    box_radius = int(math.ceil(radius_angstrom / min_voxelsize.item()))
    n_axis = 2 * box_radius + 1
    r = torch.arange(-box_radius, box_radius + 1, device=device)

    result = (r, n_axis)
    _cached_separable_data = result
    _cached_separable_key = key
    return result


def _separable_density(
    d_frac: torch.Tensor,
    alpha: torch.Tensor,
    A_norm: torch.Tensor,
    G: torch.Tensor,
    has_ab: bool,
    has_ac: bool,
    has_bc: bool,
) -> torch.Tensor:
    """Separable Gaussian density evaluation.

    Factorizes exp(-alpha * r^T G r) into 1D Gaussians per axis with 2D
    cross-term corrections for non-orthogonal cells. Batches all corrections
    across the 5 ITC92 components and uses einsum where possible.

    For non-orthogonal cells, cross-term exponents are combined with the
    relevant diagonal exponents before taking exp() to avoid float32 overflow
    (exp(-big) * exp(+big) = 0 * inf = NaN).  Each combined 2D block
    exponent corresponds to a principal sub-matrix of G (positive definite),
    guaranteeing the exponent is always <= 0 and exp() is in (0, 1].

    Dispatch by crystal system for optimal performance:
    - Orthogonal (no cross terms): separable 1D products + einsum
    - Hexagonal  (ab only):  combined ab exponent + einsum with e_c
    - Monoclinic (ac only):  combined ac exponent + einsum with e_b
    - General    (bc, or multiple cross terms): full 3D exponent per component

    Parameters
    ----------
    d_frac : (C, 3, n_axis) — fractional distances per axis, PBC-wrapped.
    alpha : (C, N_comp) — pi^2 / B_total.
    A_norm : (C, N_comp) — weighted amplitudes.
    G : (3, 3) — metric tensor frac_matrix.T @ frac_matrix.
    has_ab : bool — whether G[0,1] cross-term is significant.
    has_ac : bool — whether G[0,2] cross-term is significant.
    has_bc : bool — whether G[1,2] cross-term is significant.

    Returns
    -------
    (C, n_axis, n_axis, n_axis) density cube.
    """
    # --- Convert fractional → Cartesian per-axis ---
    cell_lengths = torch.sqrt(torch.diagonal(G))
    d_cart = d_frac * cell_lengths[None, :, None]  # (C, 3, n)

    # --- 1D exponents (always <= 0) ---
    da2 = d_cart[:, 0, :] ** 2
    db2 = d_cart[:, 1, :] ** 2
    dc2 = d_cart[:, 2, :] ** 2

    log_a = -alpha.unsqueeze(2) * da2.unsqueeze(1)  # (C, Nc, n)
    log_b = -alpha.unsqueeze(2) * db2.unsqueeze(1)
    log_c = -alpha.unsqueeze(2) * dc2.unsqueeze(1)

    if not (has_ab or has_ac or has_bc):
        # ---- Orthogonal cells: pure separable, all exp() args <= 0 ----
        e_a = torch.exp(log_a)
        e_b = torch.exp(log_b)
        e_c = torch.exp(log_c)
        e_ab = e_a.unsqueeze(3) * e_b.unsqueeze(2)
        return torch.einsum("cg,cgij,cgk->cijk", A_norm, e_ab, e_c)

    # --- Cross-term coefficients ---
    cos_gamma = G[0, 1] / (cell_lengths[0] * cell_lengths[1])
    cos_beta = G[0, 2] / (cell_lengths[0] * cell_lengths[2])
    cos_alpha = G[1, 2] / (cell_lengths[1] * cell_lengths[2])

    da = d_cart[:, 0, :]
    db = d_cart[:, 1, :]
    dc = d_cart[:, 2, :]
    alpha_4d = alpha[:, :, None, None]  # (C, Nc, 1, 1)

    if has_ab and not has_ac and not has_bc:
        # ---- Hexagonal / trigonal: only ab cross-term ----
        # Combined 2D exponent: -alpha*(da2 + db2 + 2*cos_gamma*da*db)
        # = -alpha * d_ab^T G_ab d_ab <= 0 (G_ab positive definite)
        prod_ab = da.unsqueeze(2) * db.unsqueeze(1)
        log_ab = (log_a[:, :, :, None] + log_b[:, :, None, :]
                  + (-2.0 * alpha_4d * cos_gamma * prod_ab[:, None, :, :]))
        slice_ab = torch.exp(log_ab)  # (C, Nc, n, n), all in (0, 1]
        e_c = torch.exp(log_c)
        return torch.einsum("cg,cgij,cgk->cijk", A_norm, slice_ab, e_c)

    if has_ac and not has_ab and not has_bc:
        # ---- Monoclinic (beta != 90): only ac cross-term ----
        # Combined 2D exponent: -alpha*(da2 + dc2 + 2*cos_beta*da*dc)
        # = -alpha * d_ac^T G_ac d_ac <= 0 (G_ac positive definite)
        prod_ac = da.unsqueeze(2) * dc.unsqueeze(1)
        log_ac = (log_a[:, :, :, None] + log_c[:, :, None, :]
                  + (-2.0 * alpha_4d * cos_beta * prod_ac[:, None, :, :]))
        e_ac = torch.exp(log_ac)  # (C, Nc, n_a, n_c), all in (0, 1]
        e_b = torch.exp(log_b)
        return torch.einsum("cg,cgj,cgik->cijk", A_norm, e_b, e_ac)

    # ---- General path (triclinic, or multiple cross-terms) ----
    # Combine ALL exponents into a single 3D value per voxel per component
    # to guarantee no overflow.  Component loop keeps memory at O(C*n^3).
    prod_ab = da.unsqueeze(2) * db.unsqueeze(1) if has_ab else None
    prod_ac = da.unsqueeze(2) * dc.unsqueeze(1) if has_ac else None
    prod_bc = db.unsqueeze(2) * dc.unsqueeze(1) if has_bc else None

    C = d_frac.shape[0]
    n = d_frac.shape[2]
    density_cube = d_frac.new_zeros(C, n, n, n)
    for g in range(alpha.shape[1]):
        # Full 3D exponent: -alpha * r^T G r  (always <= 0)
        exp_3d = (log_a[:, g, :, None, None]
                  + log_b[:, g, None, :, None]
                  + log_c[:, g, None, None, :])
        if has_ab:
            exp_3d = exp_3d + (
                -2.0 * alpha[:, g, None, None] * cos_gamma
                * prod_ab
            ).unsqueeze(3)  # broadcast (C, n_a, n_b, 1)
        if has_ac:
            exp_3d = exp_3d + (
                -2.0 * alpha[:, g, None, None] * cos_beta
                * prod_ac
            ).unsqueeze(2)  # broadcast (C, n_a, 1, n_c)
        if has_bc:
            exp_3d = exp_3d + (
                -2.0 * alpha[:, g, None, None] * cos_alpha
                * prod_bc
            ).unsqueeze(1)  # broadcast (C, 1, n_b, n_c)
        density_cube += A_norm[:, g, None, None, None] * torch.exp(exp_3d)

    return density_cube


_compiled_separable_density = None


def _get_compiled_separable_density():
    """Return a torch.compile'd version of _separable_density (lazy, cached)."""
    global _compiled_separable_density
    if _compiled_separable_density is None:
        _compiled_separable_density = torch.compile(_separable_density)
    return _compiled_separable_density


_CHUNK_SIZES = (4096, 2048, 1024, 512)


def _add_isotropic_cpu_separable(
    density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix,
    grid_shape_tuple, voxel_size, radius_angstrom,
):
    """Separable Gaussian splatting for isotropic atoms.

    Factorizes the 3D Gaussian into 1D components along each fractional axis,
    with 2D cross-term corrections for non-zero off-diagonal elements of the
    metric tensor. Uses a component loop for non-orthogonal cells to keep peak
    memory low (~1.7 MB vs ~22 MB for the full 5D intermediate).

    Reduces exp() calls from O(r^3) to O(r) per atom for orthogonal cells,
    and O(r^2) for monoclinic/hexagonal cells. Handles all cell geometries.
    """
    device = density_map.device
    grid_shape = torch.tensor(grid_shape_tuple, device=device)
    grid_shape_float = grid_shape.float()

    # --- Box radius (cached) ---
    axis_offsets, n_axis = _get_box_radius(voxel_size, radius_angstrom, device)

    # --- Constants ---
    pi = math.pi
    pi_sq = pi * pi
    pi_sqrt = math.sqrt(pi)
    pi_1p5 = pi * pi_sqrt
    G = frac_matrix.T @ frac_matrix  # metric tensor
    inv_grid = 1.0 / grid_shape_float

    nx_val = int(grid_shape[0])
    ny_val = int(grid_shape[1])
    nz_val = int(grid_shape[2])
    ny_nz = ny_val * nz_val

    # --- Atom fractional coords & center indices ---
    xyz_frac = xyz @ inv_frac_matrix.T  # (N, 3) — unwrapped, preserves gradients
    xyz_frac_wrapped = xyz_frac % 1.0  # only used for index computation
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).long()  # (N, 3)

    # --- B_total, normalized amplitudes, and exponent coefficients ---
    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)  # (N, 5)
    A_norm = A * occ[:, None] * pi_1p5 / (B_total * torch.sqrt(B_total))  # (N, 5)
    alpha = pi_sq / B_total  # (N, 5)

    # --- Cross-term flags (computed once) ---
    tol = 1e-3 * torch.norm(torch.diagonal(G))
    has_ab = bool(torch.abs(G[0, 1]) > tol)
    has_ac = bool(torch.abs(G[0, 2]) > tol)
    has_bc = bool(torch.abs(G[1, 2]) > tol)

    # --- Precompute fractional axis offsets (shared across chunks) ---
    # axis_offsets_frac[dim, i] = axis_offsets[i] / grid_shape[dim]
    axis_offsets_frac = axis_offsets.float().unsqueeze(0) * inv_grid.unsqueeze(1)  # (3, n_axis)

    # --- Sort atoms by 1D voxel center for cache-friendly scatter ---
    center_1d = (center_idx[:, 0] * ny_nz
                 + center_idx[:, 1] * nz_val
                 + center_idx[:, 2])
    atom_order = torch.argsort(center_1d)
    xyz_frac = xyz_frac[atom_order]
    center_idx = center_idx[atom_order]
    alpha = alpha[atom_order]
    A_norm = A_norm[atom_order]

    # --- Precompute 1D scatter indices for ALL atoms ---
    # (N, n_axis) each, ~0.4 MB per axis for 3k atoms — avoids recomputing per chunk
    all_wa = (center_idx[:, 0:1] + axis_offsets.unsqueeze(0)) % nx_val * ny_nz
    all_wb = (center_idx[:, 1:2] + axis_offsets.unsqueeze(0)) % ny_val * nz_val
    all_wc = (center_idx[:, 2:3] + axis_offsets.unsqueeze(0)) % nz_val
    # 2D yz-plane index: (N, n_axis, n_axis) — cuts outer sum from 2 adds to 1
    all_wbwc = all_wb.unsqueeze(2) + all_wc.unsqueeze(1)

    # --- Process in chunks ---
    N = xyz.shape[0]
    CHUNK = 1024
    map_size = density_map.numel()
    density_flat = density_map.view(-1)

    # Try C++ parallel scatter; fall back to PyTorch scatter_add_
    use_cpp = _get_cpp_scatter() is not None

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)

        # Sub-grid offset: fractional displacement from atom to nearest grid point
        center_frac = center_idx[start:end].float() * inv_grid  # (C, 3)
        sub_grid_offset = xyz_frac[start:end] - center_frac  # (C, 3)

        # 1D fractional distances along each axis: (C, 3, n_axis)
        d_frac = axis_offsets_frac.unsqueeze(0) - sub_grid_offset.unsqueeze(2)
        d_frac = d_frac - torch.round(d_frac)  # PBC

        # Density computation → (C, n_axis, n_axis, n_axis)
        density_cube = _separable_density(
            d_frac,
            alpha[start:end],
            A_norm[start:end],
            G,
            has_ab, has_ac, has_bc,
        )

        if use_cpp:
            # Parallel partitioned scatter — no idx_flat materialized
            chunk_result = _get_cpp_scatter()(
                density_cube,
                all_wa[start:end],
                all_wbwc[start:end],
                map_size,
            )
            density_flat = density_flat + chunk_result
        else:
            # Fallback: standard PyTorch scatter
            idx_flat = (all_wa[start:end, :, None, None]
                        + all_wbwc[start:end, None, :, :])
            density_flat.scatter_add_(
                0, idx_flat.reshape(-1), density_cube.reshape(-1))

    return density_flat.view(density_map.shape)


def _add_isotropic_cpu_separable_compiled(
    density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix,
    grid_shape_tuple, voxel_size, radius_angstrom,
):
    """Compiled variant of separable Gaussian splatting.

    Same algorithm as _add_isotropic_cpu_separable but uses torch.compile on
    _separable_density with decreasing fixed chunk sizes (4096, 2048, 1024, 512)
    to keep compiled shapes stable across different proteins. The small remainder
    runs eagerly to avoid extra recompilation.
    """
    device = density_map.device
    grid_shape = torch.tensor(grid_shape_tuple, device=device)
    grid_shape_float = grid_shape.float()

    # --- Box radius (cached) ---
    axis_offsets, n_axis = _get_box_radius(voxel_size, radius_angstrom, device)

    # --- Constants ---
    pi = math.pi
    pi_sq = pi * pi
    pi_sqrt = math.sqrt(pi)
    pi_1p5 = pi * pi_sqrt
    G = frac_matrix.T @ frac_matrix  # metric tensor
    inv_grid = 1.0 / grid_shape_float

    nx_val = int(grid_shape[0])
    ny_val = int(grid_shape[1])
    nz_val = int(grid_shape[2])
    ny_nz = ny_val * nz_val

    # --- Atom fractional coords & center indices ---
    xyz_frac = xyz @ inv_frac_matrix.T  # (N, 3) — unwrapped, preserves gradients
    xyz_frac_wrapped = xyz_frac % 1.0  # only used for index computation
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).long()  # (N, 3)

    # --- B_total, normalized amplitudes, and exponent coefficients ---
    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)  # (N, 5)
    A_norm = A * occ[:, None] * pi_1p5 / (B_total * torch.sqrt(B_total))  # (N, 5)
    alpha = pi_sq / B_total  # (N, 5)

    # --- Cross-term flags (computed once, passed as compile-time constants) ---
    tol = 1e-3 * torch.norm(torch.diagonal(G))
    has_ab = bool(torch.abs(G[0, 1]) > tol)
    has_ac = bool(torch.abs(G[0, 2]) > tol)
    has_bc = bool(torch.abs(G[1, 2]) > tol)

    # --- Precompute fractional axis offsets (shared across chunks) ---
    axis_offsets_frac = axis_offsets.float().unsqueeze(0) * inv_grid.unsqueeze(1)  # (3, n_axis)

    # --- Process with decreasing fixed chunk sizes for stable compiled shapes ---
    N = xyz.shape[0]
    compiled_fn = _get_compiled_separable_density()
    density_flat = density_map.view(-1)
    offset = 0
    remaining = N

    for chunk_size in _CHUNK_SIZES:
        while remaining >= chunk_size:
            end = offset + chunk_size
            _splat_chunk(
                offset, end, center_idx, xyz_frac, axis_offsets_frac, inv_grid,
                alpha, A_norm, G, has_ab, has_ac, has_bc,
                axis_offsets, nx_val, ny_val, nz_val, ny_nz,
                density_flat, compiled_fn,
            )
            offset = end
            remaining -= chunk_size

    # --- Eager remainder (no recompilation for the tail) ---
    if remaining > 0:
        _splat_chunk(
            offset, offset + remaining, center_idx, xyz_frac,
            axis_offsets_frac, inv_grid, alpha, A_norm, G,
            has_ab, has_ac, has_bc, axis_offsets,
            nx_val, ny_val, nz_val, ny_nz,
            density_flat, _separable_density,
        )

    return density_map


def _splat_chunk(
    start, end, center_idx, xyz_frac, axis_offsets_frac, inv_grid,
    alpha, A_norm, G, has_ab, has_ac, has_bc,
    axis_offsets, nx_val, ny_val, nz_val, ny_nz,
    density_flat, density_fn,
):
    """Compute separable density for one chunk and scatter into the map."""
    # Sub-grid offset: fractional displacement from atom to nearest grid point
    center_frac = center_idx[start:end].float() * inv_grid  # (C, 3)
    sub_grid_offset = xyz_frac[start:end] - center_frac  # (C, 3)

    # 1D fractional distances along each axis: (C, 3, n_axis)
    d_frac = axis_offsets_frac.unsqueeze(0) - sub_grid_offset.unsqueeze(2)
    d_frac = d_frac - torch.round(d_frac)  # PBC

    # Density computation → (C, n_axis, n_axis, n_axis)
    density_cube = density_fn(
        d_frac, alpha[start:end], A_norm[start:end],
        G, has_ab, has_ac, has_bc,
    )

    # Full-cube scatter: 1D wrapped indices combined via outer sum
    chunk_center = center_idx[start:end]
    wa = (chunk_center[:, 0:1] + axis_offsets.unsqueeze(0)) % nx_val * ny_nz  # (C, n)
    wb = (chunk_center[:, 1:2] + axis_offsets.unsqueeze(0)) % ny_val * nz_val
    wc = (chunk_center[:, 2:3] + axis_offsets.unsqueeze(0)) % nz_val
    idx_flat = (wa.unsqueeze(2).unsqueeze(3)
                + wb.unsqueeze(1).unsqueeze(3)
                + wc.unsqueeze(1).unsqueeze(2))  # (C, n, n, n)
    density_flat.scatter_add_(0, idx_flat.reshape(-1), density_cube.reshape(-1))


def _add_isotropic_cpu_fused(
    density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix,
    grid_shape_tuple, voxel_size, radius_angstrom,
):
    """Fused CPU path for isotropic atoms.

    Avoids the expensive Cartesian↔Fractional round-trip by computing voxel
    fractional coordinates directly from integer indices. Processes atoms in
    chunks to keep intermediates in L3 cache.
    """
    device = density_map.device
    grid_shape = torch.tensor(grid_shape_tuple, device=device)
    grid_shape_float = grid_shape.float()

    # --- Radius mask (cached) ---
    local_offsets = _get_radius_offsets(voxel_size, radius_angstrom, device)

    # --- Constants ---
    pi = math.pi
    pi_sq = pi * pi
    pi_sqrt = math.sqrt(pi)
    pi_1p5 = pi * pi_sqrt
    G = frac_matrix.T @ frac_matrix  # metric tensor
    inv_grid = 1.0 / grid_shape_float

    ny_nz = int(grid_shape[1]) * int(grid_shape[2])
    nz_val = int(grid_shape[2])
    strides = torch.tensor([ny_nz, nz_val, 1], device=device, dtype=torch.long)

    # --- Atom fractional coords & center indices ---
    xyz_frac = xyz @ inv_frac_matrix.T  # (N, 3) — unwrapped, preserves gradients
    xyz_frac_wrapped = xyz_frac % 1.0  # only used for index computation
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).long()  # (N, 3)

    # --- B_total and normalized amplitudes (small, atom-level tensors) ---
    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)  # (N, 5)
    A_norm = A * occ[:, None] * pi_1p5 / (B_total * torch.sqrt(B_total))  # (N, 5)

    # --- Process in chunks for cache efficiency ---
    N = xyz.shape[0]
    CHUNK = 1024

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)

        # Voxel indices (wrapped for scatter and frac coord computation)
        vi = (center_idx[start:end].unsqueeze(1) + local_offsets.unsqueeze(0)) % grid_shape
        # shape: (C, R, 3)

        # Fractional voxel positions — direct from integer indices
        voxel_frac = vi.float() * inv_grid  # (C, R, 3)

        # Fractional diff with PBC — use unwrapped xyz_frac to preserve gradients
        diff_frac = voxel_frac - xyz_frac[start:end].unsqueeze(1)
        diff_frac = diff_frac - torch.round(diff_frac)

        # r² via metric tensor: exact for any cell geometry
        r_sq = torch.einsum("avi,ij,avj->av", diff_frac, G, diff_frac)

        # Gaussian density
        chunk_B = B_total[start:end]
        exponents = -pi_sq * r_sq.unsqueeze(2) / chunk_B.unsqueeze(1)
        density = torch.einsum(
            "ag,avg->av", A_norm[start:end], torch.exp(exponents)
        )

        # Scatter add to map
        idx_flat = (vi.to(torch.long) * strides).sum(-1).view(-1)
        density_map.view(-1).scatter_add_(0, idx_flat, density.reshape(-1))

    return density_map


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
