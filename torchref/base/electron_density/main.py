"""
Central electron density building with automatic backend selection.

The coarse Triton-vs-eager decision is made by the shared, capability-based
``Engine`` in :mod:`torchref.utils.triton_dispatch` (CUDA+float32+Triton ->
Triton; otherwise eager). There is **no environment-variable dispatch** — force
a path with ``with use_engine(Engine.EAGER): ...`` or ``set_engine(...)``.

Which specific implementation runs within each coarse path is a local *tier*
choice, set in code via the module-level constants below (no env vars):

GPU Triton tiers (``GPU_TIER``):
  "auto"             — try fused → separable → original fallback (default)
  "fused_triton"     — fused Triton kernel
  "separable_triton" — separable 1-D table lookups

CPU eager tiers (``CPU_TIER``):
  "separable"          — separable Gaussian splatting (default)
  "separable_compiled" — torch.compile'd separable
  "fused"              — fused fractional-space kernel
  "original"           — find_relevant_voxels + vectorized_add_to_map

MPS eager tiers (``MPS_TIER``): "single" (default), "separable_compiled", "separable".

Override a tier at runtime by setting the module-level variable directly::

    import torchref.base.electron_density.main as ed
    ed.GPU_TIER = "fused_triton"
    ed.CPU_TIER = "fused"
"""

import math
from typing import Optional

import torch

from torchref.config import dtypes, get_float_dtype
from torchref.utils.triton_dispatch import Engine, get_engine, should_use_triton

# ---------------------------------------------------------------------------
# Local tier selection — the coarse Triton/eager gate is the shared Engine
# (see torchref.utils.triton_dispatch). These pick the specific impl within a
# path and are code-level only (no env vars). Override at runtime via
# ``ed.GPU_TIER = ...`` etc.
# ---------------------------------------------------------------------------
GPU_TIER: str = "auto"        # triton tier: auto | fused_triton | separable_triton
CPU_TIER: str = "separable"   # eager: separable | separable_compiled | fused | original
# MPS tiers:
#   "single"             — single-pass (no chunking), one math + one scatter call.
#                          Default — avoids per-chunk autograd.Function overhead
#                          and is required for the Metal scatter to win over
#                          PyTorch scatter_add_ (chunking dilutes that win on
#                          Apple Silicon, see profiling_mps/ANALYSIS.md).
#   "separable_compiled" — CPU separable_compiled (multi-chunk, compiled math).
#   "separable"          — CPU separable (multi-chunk, eager math).
MPS_TIER: str = "single"

# Tier-2 ("original") GPU implementation mode: "triton" | "jit" | "simple".
GPU_ORIGINAL_MODE: str = "triton"

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


_aniso_fused_fn = None
_aniso_fused_checked = False


def _get_aniso_fused_triton():
    """Return the fused anisotropic Triton kernel, or None if unavailable."""
    global _aniso_fused_fn, _aniso_fused_checked
    if not _aniso_fused_checked:
        try:
            from torchref.base.kernels.triton_kernel import (
                aniso_fused_find_and_place_atoms,
            )
            _aniso_fused_fn = aniso_fused_find_and_place_atoms
        except ImportError:
            pass
        _aniso_fused_checked = True
    return _aniso_fused_fn


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


def _do_structured_scatter(
    density_cube: torch.Tensor,
    wa: torch.Tensor,
    wbwc: torch.Tensor,
    density_flat: torch.Tensor,
    map_size: int,
) -> torch.Tensor:
    """Pick the fastest structured scatter for the device.

    On CPU, dispatches to the custom C++ kernel (``cpu_scatter``) when
    available — partitioned, no atomics, ~2× faster than PyTorch's stock
    ``scatter_add_``. On every other device (MPS, CUDA, CPU without the
    extension built) falls back to PyTorch ``scatter_add_`` with int64
    indices.

    Returns the resulting flat density tensor. The C++ path accumulates
    out-of-place; the ``scatter_add_`` fallback mutates ``density_flat``
    in place. Both return the up-to-date tensor.
    """
    if density_cube.device.type == "cpu":
        cpp_fn = _get_cpp_scatter()
        if cpp_fn is not None:
            return density_flat + cpp_fn(density_cube, wa, wbwc, map_size)
    # Fallback: PyTorch scatter_add_ requires int64 indices.
    idx_flat = wa[:, :, None, None] + wbwc[:, None, :, :]
    density_flat.scatter_add_(
        0,
        idx_flat.reshape(-1).to(torch.int64),
        density_cube.reshape(-1),
    )
    return density_flat


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
    dtype: torch.dtype = get_float_dtype(),
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
            radius_angstrom, voxel_size,
        )

    return density_map


# =========================================================================
# Internal dispatch helpers
# =========================================================================

def _add_isotropic(
    real_space_grid, density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
):
    """Add isotropic atoms, gated by the shared Engine then the local tier.

    The coarse Triton-vs-eager choice is the shared ``should_use_triton``
    gate (CUDA+float32+Triton+engine). When it permits Triton we take the GPU
    Triton tier ladder (``GPU_TIER``); otherwise we fall to the device-eager
    path. ``Engine.EAGER`` or float64 inputs route a CUDA tensor to the eager
    ``_add_isotropic_original`` rather than the CPU kernels.
    """
    device_type = density_map.device.type

    if device_type == "cuda":
        if should_use_triton(xyz):
            return _add_isotropic_gpu(
                real_space_grid, density_map, xyz, adp, occ, A, B,
                inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
            )
        # eager on GPU (Engine.EAGER, no Triton, or non-float32 inputs)
        return _add_isotropic_original(
            real_space_grid, density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_angstrom,
        )
    if device_type == "mps":
        return _add_isotropic_mps(
            real_space_grid, density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
        )
    return _add_isotropic_cpu(
        real_space_grid, density_map, xyz, adp, occ, A, B,
        inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
    )


def _add_isotropic_mps(
    real_space_grid, density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
):
    """MPS dispatch — see ``MPS_TIER`` above for tier choices."""
    engine = MPS_TIER
    grid_shape_tuple = real_space_grid.shape[:3]

    if engine == "single":
        return _add_isotropic_mps_single(
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
    if engine == "separable":
        return _add_isotropic_cpu_separable(
            density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix,
            grid_shape_tuple, voxel_size, radius_angstrom,
        )
    raise ValueError(
        f"Unknown MPS_TIER={engine!r}. "
        f"Choose from: single, separable_compiled, separable"
    )


def _add_isotropic_mps_single(
    density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix,
    grid_shape_tuple, voxel_size, radius_angstrom,
):
    """Single-pass MPS splat: one math call, one scatter call.

    The multi-chunk strategy in ``_add_isotropic_cpu_separable_compiled``
    exists so torch.compile sees a small set of fixed shapes across
    different protein sizes. Per refinement loop the atom count is
    constant — one compile suffices. Eliminating the per-chunk PyTorch
    op overhead saves ~10 ms / iter at 1DAW scale on MPS — and that's
    the only thing that needs to be different from the CPU path here.
    See profiling_mps/ANALYSIS.md for the breakdown.
    """
    device = density_map.device
    grid_shape = torch.tensor(grid_shape_tuple, device=device)
    grid_shape_float = grid_shape.float()

    axis_offsets, n_axis = _get_box_radius(voxel_size, radius_angstrom, device)
    axis_offsets = axis_offsets.to(dtypes.int)

    pi = math.pi
    pi_sq = pi * pi
    pi_sqrt = math.sqrt(pi)
    pi_1p5 = pi * pi_sqrt
    G = frac_matrix.T @ frac_matrix
    inv_grid = 1.0 / grid_shape_float

    nx_val = grid_shape_tuple[0]
    ny_val = grid_shape_tuple[1]
    nz_val = grid_shape_tuple[2]
    ny_nz = ny_val * nz_val
    map_size = density_map.numel()

    xyz_frac = xyz @ inv_frac_matrix.T
    xyz_frac_wrapped = xyz_frac % 1.0
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).to(dtypes.int)

    B_total = ((B + adp[:, None]) * 0.25).clamp(min=0.1)
    A_norm = A * occ[:, None] * pi_1p5 / (B_total * torch.sqrt(B_total))
    alpha = pi_sq / B_total

    tol = 1e-3 * torch.norm(torch.diagonal(G))
    has_ab = bool(torch.abs(G[0, 1]) > tol)
    has_ac = bool(torch.abs(G[0, 2]) > tol)
    has_bc = bool(torch.abs(G[1, 2]) > tol)

    axis_offsets_frac = axis_offsets.float().unsqueeze(0) * inv_grid.unsqueeze(1)

    # Math for ALL atoms at once
    center_frac = center_idx.float() * inv_grid
    sub_grid_offset = xyz_frac - center_frac
    d_frac = axis_offsets_frac.unsqueeze(0) - sub_grid_offset.unsqueeze(2)
    d_frac = d_frac - torch.round(d_frac)

    # Compiled separable density — one shape per structure, one compile
    density_fn = _get_compiled_separable_density()
    density_cube = density_fn(d_frac, alpha, A_norm, G, has_ab, has_ac, has_bc)

    # Structured indices for all atoms
    all_wa = (center_idx[:, 0:1] + axis_offsets.unsqueeze(0)) % nx_val * ny_nz
    all_wb = (center_idx[:, 1:2] + axis_offsets.unsqueeze(0)) % ny_val * nz_val
    all_wc = (center_idx[:, 2:3] + axis_offsets.unsqueeze(0)) % nz_val
    all_wbwc = all_wb.unsqueeze(2) + all_wc.unsqueeze(1)

    # Single scatter call into a flat view of density_map (zero-initialised
    # by the caller in build_electron_density).
    density_flat = density_map.view(-1)
    density_flat = _do_structured_scatter(
        density_cube, all_wa, all_wbwc, density_flat, map_size,
    )

    return density_flat.view(density_map.shape)


def _add_isotropic_gpu(
    real_space_grid, density_map, xyz, adp, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
):
    """GPU Triton dispatch for isotropic atoms, controlled by ``GPU_TIER``.

    Only reached when the shared ``should_use_triton`` gate has already
    permitted Triton; this picks which Triton tier to run.
    """
    engine = GPU_TIER

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

    if engine == "auto":
        # Try fused → separable → original. Fused was ~0.26 ms faster fwd+bw
        # than separable on A100/1DAW in profile_model_sf benchmarking
        # because its larger per-launch kernel cost is more than offset by
        # reduced downstream index_put traffic. Separable is kept as a
        # robustness fallback for grid configurations where fused trips.
        if GPU_ORIGINAL_MODE not in ("jit", "simple"):
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

        if GPU_ORIGINAL_MODE not in ("jit", "simple"):
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

        return _add_isotropic_original(
            real_space_grid, density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_angstrom,
        )

    raise ValueError(
        f"Unknown GPU_TIER={engine!r}. "
        f"Choose from: auto, separable_triton, fused_triton"
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
    """CPU dispatch for isotropic atoms, controlled by ``CPU_TIER``."""
    engine = CPU_TIER
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
        f"Unknown CPU_TIER={engine!r}. "
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
    # int32 indices: the cpu_scatter C++ kernel takes int32, and for any
    # realistic crystallographic grid (nx*ny*nz < 2**31) all scatter indices
    # fit in int32. Halving index bandwidth speeds up the inner loop.
    axis_offsets = axis_offsets.to(dtypes.int)

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
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).to(dtypes.int)  # (N, 3) int32

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

    # --- Precompute 1D scatter indices for ALL atoms (int32) ---
    # (N, n_axis) each, ~0.2 MB per axis for 3k atoms — avoids recomputing per chunk
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

        density_flat = _do_structured_scatter(
            density_cube,
            all_wa[start:end],
            all_wbwc[start:end],
            density_flat,
            map_size,
        )

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
    # int32 indices to match _do_structured_scatter's MPS / CPU C++ kernels;
    # the PyTorch scatter_add_ fallback casts to int64 inside the helper.
    axis_offsets = axis_offsets.to(dtypes.int)

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
    map_size = density_map.numel()

    # --- Atom fractional coords & center indices ---
    xyz_frac = xyz @ inv_frac_matrix.T  # (N, 3) — unwrapped, preserves gradients
    xyz_frac_wrapped = xyz_frac % 1.0  # only used for index computation
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).to(dtypes.int)  # (N, 3) int32

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
            density_flat = _splat_chunk(
                offset, end, center_idx, xyz_frac, axis_offsets_frac, inv_grid,
                alpha, A_norm, G, has_ab, has_ac, has_bc,
                axis_offsets, nx_val, ny_val, nz_val, ny_nz, map_size,
                density_flat, compiled_fn,
            )
            offset = end
            remaining -= chunk_size

    # --- Eager remainder (no recompilation for the tail) ---
    if remaining > 0:
        density_flat = _splat_chunk(
            offset, offset + remaining, center_idx, xyz_frac,
            axis_offsets_frac, inv_grid, alpha, A_norm, G,
            has_ab, has_ac, has_bc, axis_offsets,
            nx_val, ny_val, nz_val, ny_nz, map_size,
            density_flat, _separable_density,
        )

    return density_flat.view(density_map.shape)


def _splat_chunk(
    start, end, center_idx, xyz_frac, axis_offsets_frac, inv_grid,
    alpha, A_norm, G, has_ab, has_ac, has_bc,
    axis_offsets, nx_val, ny_val, nz_val, ny_nz, map_size,
    density_flat, density_fn,
):
    """Compute separable density for one chunk and scatter into the map.

    Returns the (possibly new) flat density tensor — the C++ and MPS
    structured-scatter backends produce out-of-place results that we have
    to rebind through the chunk loop.
    """
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

    # Structured (wa, wbwc) indices — int32; cast to int64 happens inside
    # the helper's PyTorch fallback.
    chunk_center = center_idx[start:end]
    wa = (chunk_center[:, 0:1] + axis_offsets.unsqueeze(0)) % nx_val * ny_nz  # (C, n)
    wb = (chunk_center[:, 1:2] + axis_offsets.unsqueeze(0)) % ny_val * nz_val
    wc = (chunk_center[:, 2:3] + axis_offsets.unsqueeze(0)) % nz_val
    wbwc = wb.unsqueeze(2) + wc.unsqueeze(1)  # (C, n, n)

    return _do_structured_scatter(
        density_cube, wa, wbwc, density_flat, map_size,
    )


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


def _aniso_density_cube(d_frac, frac_matrix, Minv, A_norm):
    """Anisotropic Gaussian density over the local voxel cube.

    The aniso analogue of :func:`_separable_density` — but the 3D Gaussian
    does NOT factorize across axes (cross-terms U12/U13/U23), so it builds the
    full Cartesian offset cube and evaluates the quadratic form directly. The
    component loop keeps peak memory at O(C*n^3).

    Parameters
    ----------
    d_frac : (C, 3, n) — PBC-wrapped fractional offsets per axis.
    frac_matrix : (3, 3) — fractional -> Cartesian.
    Minv : (C, 5, 3, 3) — inverse of M_g = (B_g*I + 8*pi^2*U)/4.
    A_norm : (C, 5) — A * occ * pi^1.5 / sqrt(det M_g).

    Returns
    -------
    (C, n, n, n) density cube.
    """
    pi_sq = math.pi * math.pi
    da = d_frac[:, 0, :][:, :, None, None]  # (C, n, 1, 1)
    db = d_frac[:, 1, :][:, None, :, None]  # (C, 1, n, 1)
    dc = d_frac[:, 2, :][:, None, None, :]  # (C, 1, 1, n)
    fm = frac_matrix
    # Cartesian offset cube r = frac_matrix @ d_frac  (each (C, n, n, n))
    cx = fm[0, 0] * da + fm[0, 1] * db + fm[0, 2] * dc
    cy = fm[1, 0] * da + fm[1, 1] * db + fm[1, 2] * dc
    cz = fm[2, 0] * da + fm[2, 1] * db + fm[2, 2] * dc

    C = d_frac.shape[0]
    n = d_frac.shape[2]
    density_cube = d_frac.new_zeros(C, n, n, n)
    for g in range(Minv.shape[1]):
        m00 = Minv[:, g, 0, 0][:, None, None, None]
        m11 = Minv[:, g, 1, 1][:, None, None, None]
        m22 = Minv[:, g, 2, 2][:, None, None, None]
        m01 = Minv[:, g, 0, 1][:, None, None, None]
        m02 = Minv[:, g, 0, 2][:, None, None, None]
        m12 = Minv[:, g, 1, 2][:, None, None, None]
        q = (
            m00 * cx * cx + m11 * cy * cy + m22 * cz * cz
            + 2.0 * (m01 * cx * cy + m02 * cx * cz + m12 * cy * cz)
        )
        density_cube = density_cube + A_norm[:, g, None, None, None] * torch.exp(-pi_sq * q)
    return density_cube


def _add_anisotropic_cpu(
    real_space_grid, density_map, xyz, u, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
):
    """Optimized box-splat for anisotropic atoms (CPU/MPS).

    Mirrors :func:`_add_isotropic_cpu_separable` (center index -> local voxel
    cube -> per-axis ``d_frac`` -> structured scatter) but evaluates the full
    3D anisotropic Gaussian over the cube (no separable 1D factorization).
    """
    device = density_map.device
    grid_shape_tuple = real_space_grid.shape[:3]
    grid_shape = torch.tensor(grid_shape_tuple, device=device)
    grid_shape_float = grid_shape.float()

    axis_offsets, n_axis = _get_box_radius(voxel_size, radius_angstrom, device)
    axis_offsets = axis_offsets.to(dtypes.int)

    pi = math.pi
    pi_1p5 = pi * math.sqrt(pi)
    eight_pi_sq = 8.0 * pi * pi
    inv_grid = 1.0 / grid_shape_float

    nx_val = int(grid_shape[0])
    ny_val = int(grid_shape[1])
    nz_val = int(grid_shape[2])
    ny_nz = ny_val * nz_val

    xyz_frac = xyz @ inv_frac_matrix.T  # (N, 3) — unwrapped, preserves gradients
    xyz_frac_wrapped = xyz_frac % 1.0
    center_idx = torch.round(xyz_frac_wrapped * grid_shape_float).to(dtypes.int)

    # Per-atom, per-Gaussian 3x3 M = (B_g*I + 8*pi^2*U)/4  -> Minv, det, A_norm
    N = xyz.shape[0]
    eye = torch.eye(3, dtype=xyz.dtype, device=device)
    U3 = xyz.new_zeros(N, 3, 3)
    U3[:, 0, 0] = u[:, 0]; U3[:, 1, 1] = u[:, 1]; U3[:, 2, 2] = u[:, 2]
    U3[:, 0, 1] = U3[:, 1, 0] = u[:, 3]
    U3[:, 0, 2] = U3[:, 2, 0] = u[:, 4]
    U3[:, 1, 2] = U3[:, 2, 1] = u[:, 5]
    M = (B[:, :, None, None] * eye + eight_pi_sq * U3[:, None, :, :]) / 4.0  # (N,5,3,3)
    Minv = torch.linalg.inv(M)
    det = torch.linalg.det(M).clamp(min=1e-10)
    A_norm = A * occ[:, None] * pi_1p5 / torch.sqrt(det)  # (N, 5)

    axis_offsets_frac = axis_offsets.float().unsqueeze(0) * inv_grid.unsqueeze(1)

    # Sort atoms by 1D voxel center for cache-friendly scatter
    center_1d = (center_idx[:, 0] * ny_nz + center_idx[:, 1] * nz_val + center_idx[:, 2])
    atom_order = torch.argsort(center_1d)
    xyz_frac = xyz_frac[atom_order]
    center_idx = center_idx[atom_order]
    Minv = Minv[atom_order]
    A_norm = A_norm[atom_order]

    all_wa = (center_idx[:, 0:1] + axis_offsets.unsqueeze(0)) % nx_val * ny_nz
    all_wb = (center_idx[:, 1:2] + axis_offsets.unsqueeze(0)) % ny_val * nz_val
    all_wc = (center_idx[:, 2:3] + axis_offsets.unsqueeze(0)) % nz_val
    all_wbwc = all_wb.unsqueeze(2) + all_wc.unsqueeze(1)

    CHUNK = 1024
    map_size = density_map.numel()
    density_flat = density_map.view(-1)

    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        center_frac = center_idx[start:end].float() * inv_grid
        sub_grid_offset = xyz_frac[start:end] - center_frac
        d_frac = axis_offsets_frac.unsqueeze(0) - sub_grid_offset.unsqueeze(2)
        d_frac = d_frac - torch.round(d_frac)  # PBC

        density_cube = _aniso_density_cube(
            d_frac, frac_matrix, Minv[start:end], A_norm[start:end]
        )
        density_flat = _do_structured_scatter(
            density_cube, all_wa[start:end], all_wbwc[start:end],
            density_flat, map_size,
        )

    return density_flat.view(density_map.shape)


def _add_anisotropic_original(
    real_space_grid, density_map, xyz, u, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom,
):
    """Eager two-step anisotropic splat (find_relevant_voxels + scatter).

    The reference implementation; also the CUDA-eager fallback (float64 /
    ``Engine.EAGER``).
    """
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


def _add_anisotropic(
    real_space_grid, density_map, xyz, u, occ, A, B,
    inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
):
    """Add anisotropic atoms (mirrors the isotropic device dispatch).

    CUDA+float32 (engine permitting) -> fused Triton kernel (voxels in-kernel,
    no large intermediate). CUDA eager (float64 / ``Engine.EAGER``) -> the
    two-step reference. CPU/MPS -> the optimized box-splat ``_add_anisotropic_cpu``.
    """
    if should_use_triton(xyz):
        fn = _get_aniso_fused_triton()
        if fn is not None:
            try:
                return fn(
                    real_space_grid, density_map, xyz, u,
                    inv_frac_matrix, frac_matrix, A, B, occ,
                    radius_angstrom, voxel_size,
                )
            except Exception:
                if get_engine() is Engine.TRITON:
                    raise
                # AUTO: fall back to eager
        elif get_engine() is Engine.TRITON:
            raise RuntimeError("Anisotropic fused Triton kernel is unavailable")

    if density_map.device.type == "cuda":
        return _add_anisotropic_original(
            real_space_grid, density_map, xyz, u, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_angstrom,
        )

    return _add_anisotropic_cpu(
        real_space_grid, density_map, xyz, u, occ, A, B,
        inv_frac_matrix, frac_matrix, radius_angstrom, voxel_size,
    )
