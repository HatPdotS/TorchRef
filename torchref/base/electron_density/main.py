"""
Central electron density building, dispatched solely by the shared ``Engine``.

The capability-based ``Engine`` in :mod:`torchref.utils.triton_dispatch`
(AUTO/TRITON/EAGER) is the *only* switch — there is no environment-variable
dispatch and no parallel "tier" knobs:

- ``Engine.AUTO`` — fastest available per device: CUDA+float32 -> the fused
  Triton kernel; CUDA+float64 -> pure-torch; CPU -> the C++-scatter fast path;
  MPS -> single-pass. (Falls back to the pure-torch splat if a Triton kernel
  fails.)
- ``Engine.EAGER`` — the pure-PyTorch (``scatter_add``) reference on every
  device. Double-differentiable; use it for Hessians / debugging. Force it with
  ``with use_engine(Engine.EAGER): ...``.
- ``Engine.TRITON`` — force the fused Triton kernel (raises if not CUDA+float32).

The individual splat implementations now live one-per-file under
:mod:`torchref.base.electron_density.kernels` (``cpu/separable``, ``cpu_fused``, ``cpu_aniso``,
``mps_separable``, ``eager_reference``, plus the Triton kernels and the
``offsets`` / ``scatter_dispatch`` helpers). They are re-imported here so the
historical ``torchref.base.electron_density.main`` namespace is unchanged and
they remain callable directly for benchmarking. This module keeps only the
``Engine``-based dispatch.
"""

from typing import Optional

import torch

from torchref.config import get_float_dtype, get_sigma_cutoff_ed
from torchref.utils.triton_dispatch import Engine, get_engine, should_use_triton

# --- Moved splat implementations (re-imported to preserve this namespace) ---
from torchref.base.electron_density.kernels.offsets import _get_box_radius, _get_radius_offsets
from torchref.base.electron_density.kernels.cpu.scatter_dispatch import (
    _do_structured_scatter,
    _get_cpp_scatter,
)
from torchref.base.electron_density.kernels.cpu.separable import (
    _CHUNK_SIZES,
    _add_isotropic_cpu_separable,
    _add_isotropic_cpu_separable_compiled,
    _get_compiled_separable_density,
    _separable_density,
    _splat_chunk,
)
from torchref.base.electron_density.kernels.cpu.fused import _add_isotropic_cpu_fused
from torchref.base.electron_density.kernels.cpu.aniso import _add_anisotropic_cpu, _aniso_density_cube
from torchref.base.electron_density.kernels.mps.separable import _add_isotropic_mps_single
from torchref.base.electron_density.kernels.cpu.eager_reference import (
    _add_anisotropic_original,
    _add_isotropic_original,
)

# --- Per-atom variable-radius density path ---
# The splat radius is no longer a single scalar; each atom is truncated at its own
# N_sigma * sigma_eff radius (N_sigma = torchref.sigma_cutoff_ed). CUDA+float32 uses
# the variable-radius Triton kernels (work plan + WorkQueueGridDensity{,Aniso});
# CPU uses the grouped-separable / fused / aniso splats. The pure-torch reference
# and MPS have no variable-radius kernel, so they receive the per-structure max of
# the per-atom radii (correct, just not work-optimized).
from torchref.base.electron_density.radius_policy import (
    per_atom_radius_aniso,
    per_atom_radius_iso,
)
from torchref.base.electron_density.kernels.cuda.variable_radius import (
    WorkQueueGridDensity,
    WorkQueueGridDensityAniso,
)
from torchref.base.electron_density.kernels.cpu.variable_radius import (
    add_anisotropic_cpu_var,
    add_anisotropic_plain_var,
    add_isotropic_cpu_separable_var,
    add_isotropic_plain_var,
)


def build_electron_density(
    real_space_grid: torch.Tensor,
    xyz_iso: torch.Tensor,
    adp_iso: torch.Tensor,
    occ_iso: torch.Tensor,
    A_iso: torch.Tensor,
    B_iso: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
    frac_matrix: torch.Tensor,
    voxel_size: torch.Tensor,
    xyz_aniso: Optional[torch.Tensor] = None,
    u_aniso: Optional[torch.Tensor] = None,
    occ_aniso: Optional[torch.Tensor] = None,
    A_aniso: Optional[torch.Tensor] = None,
    B_aniso: Optional[torch.Tensor] = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Build an electron density map from atomic parameters.

    Dispatches the isotropic and anisotropic splats through the shared
    ``Engine`` (see the module docstring): the fused Triton kernel on
    CUDA+float32 under ``Engine.AUTO``/``Engine.TRITON``, and otherwise the
    pure-torch / C++-scatter splat appropriate for the device.

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
    voxel_size : torch.Tensor
        Voxel dimensions, shape (3,). The per-atom truncation radius is derived
        internally from each atom's B/U and ``torchref.sigma_cutoff_ed``.
    xyz_aniso : torch.Tensor, optional
        Anisotropic atom positions, shape (n_aniso, 3).
    u_aniso : torch.Tensor, optional
        Anisotropic U parameters, shape (n_aniso, 6).
    occ_aniso : torch.Tensor, optional
        Anisotropic occupancies, shape (n_aniso,).
    A_aniso, B_aniso : torch.Tensor, optional
        ITC92 coefficients for anisotropic atoms, shape (n_aniso, 5).
    dtype : torch.dtype, optional
        Float dtype for the density map. Defaults to the configured float
        dtype (``get_float_dtype()``), which may be float64.

    Returns
    -------
    torch.Tensor
        Electron density map, shape (nx, ny, nz).
    """
    if dtype is None:
        dtype = get_float_dtype()
    device = real_space_grid.device
    density_map = torch.zeros(
        real_space_grid.shape[:-1],
        dtype=dtype,
        device=device,
    )

    # --- isotropic atoms ---
    if len(xyz_iso) > 0:
        density_map = _add_isotropic(
            real_space_grid,
            density_map,
            xyz_iso,
            adp_iso,
            occ_iso,
            A_iso,
            B_iso,
            inv_frac_matrix,
            frac_matrix,
            voxel_size,
        )

    # --- anisotropic atoms ---
    if xyz_aniso is not None and len(xyz_aniso) > 0:
        density_map = _add_anisotropic(
            real_space_grid,
            density_map,
            xyz_aniso,
            u_aniso,
            occ_aniso,
            A_aniso,
            B_aniso,
            inv_frac_matrix,
            frac_matrix,
            voxel_size,
        )

    return density_map


# =========================================================================
# Internal dispatch helpers
# =========================================================================


def _add_isotropic(
    real_space_grid,
    density_map,
    xyz,
    adp,
    occ,
    A,
    B,
    inv_frac_matrix,
    frac_matrix,
    voxel_size,
):
    """Add isotropic atoms with a per-atom variable radius. ``Engine`` is the switch.

    The per-atom radius is ``clamp(ceil(N_sigma * sigma_eff), [2,7])`` with
    ``N_sigma = torchref.sigma_cutoff_ed``.

    - ``should_use_triton`` (CUDA + float32 + Triton, engine AUTO/TRITON) -> the
      variable-radius Triton kernel (``WorkQueueGridDensity``). On kernel failure
      under AUTO it falls through to the portable splat; under ``Engine.TRITON``
      it raises (never silently degrade).
    - CPU + AUTO -> the fast C++-scatter grouped-separable splat.
    - Everything else (``Engine.EAGER`` on any device, CUDA float64, MPS) -> the
      portable plain-``scatter_add`` grouped splat: identical per-atom radius,
      double-differentiable, float64-capable, device-agnostic.

    Every path truncates each atom at its own ``N_sigma * sigma_eff`` radius.
    """
    n_sigma = get_sigma_cutoff_ed()
    radius_per_atom = per_atom_radius_iso(adp, B, n_sigma=n_sigma)

    if should_use_triton(xyz):
        try:
            r2cut = radius_per_atom * radius_per_atom
            coeff_mask = torch.ones(xyz.shape[0], 5, dtype=xyz.dtype, device=xyz.device)
            return density_map + WorkQueueGridDensity.apply(
                real_space_grid,
                xyz,
                adp,
                occ,
                A,
                B,
                r2cut,
                coeff_mask,
                inv_frac_matrix,
                frac_matrix,
            )
        except Exception:
            if get_engine() is Engine.TRITON:
                raise
            # AUTO: fall through to the portable splat

    grid_shape_tuple = real_space_grid.shape[:3]
    if get_engine() is Engine.AUTO and density_map.device.type == "cpu":
        return add_isotropic_cpu_separable_var(
            density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, grid_shape_tuple, voxel_size, radius_per_atom,
        )
    return add_isotropic_plain_var(
        density_map, xyz, adp, occ, A, B,
        inv_frac_matrix, frac_matrix, grid_shape_tuple, voxel_size, radius_per_atom,
    )


def _add_anisotropic(
    real_space_grid,
    density_map,
    xyz,
    u,
    occ,
    A,
    B,
    inv_frac_matrix,
    frac_matrix,
    voxel_size,
):
    """Add anisotropic atoms with a per-atom variable radius (mirrors the iso path).

    The per-atom radius is the isotropic bounding radius of the ellipsoid
    (largest principal axis, ``per_atom_radius_aniso``).

    CUDA+float32 (engine permitting) -> ``WorkQueueGridDensityAniso``. CPU + AUTO
    -> the fast C++-scatter grouped box-splat ``add_anisotropic_cpu_var``.
    Everything else (``Engine.EAGER`` on any device, CUDA float64, MPS) -> the
    portable plain-``scatter_add`` box-splat ``add_anisotropic_plain_var``
    (double-diff, float64-capable, device-agnostic). All paths use the per-atom
    radius (isotropic bounding sphere of the ellipsoid).
    """
    n_sigma = get_sigma_cutoff_ed()
    radius_per_atom = per_atom_radius_aniso(B, u, n_sigma=n_sigma)

    if should_use_triton(xyz):
        try:
            r2cut = radius_per_atom * radius_per_atom
            coeff_mask = torch.ones(xyz.shape[0], 5, dtype=xyz.dtype, device=xyz.device)
            return density_map + WorkQueueGridDensityAniso.apply(
                real_space_grid,
                xyz,
                u,
                occ,
                A,
                B,
                r2cut,
                coeff_mask,
                inv_frac_matrix,
                frac_matrix,
            )
        except Exception:
            if get_engine() is Engine.TRITON:
                raise
            # AUTO: fall back to the portable splat

    if get_engine() is Engine.AUTO and density_map.device.type == "cpu":
        return add_anisotropic_cpu_var(
            real_space_grid, density_map, xyz, u, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_per_atom, voxel_size,
        )
    return add_anisotropic_plain_var(
        real_space_grid, density_map, xyz, u, occ, A, B,
        inv_frac_matrix, frac_matrix, radius_per_atom, voxel_size,
    )
