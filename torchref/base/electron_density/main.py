"""
Central electron density building, dispatched from one declarative table.

**One truncation contract, every backend**, so ``sigma_cutoff_ed`` means the same thing
on every device: voxel v receives atom i's density iff ``||w||^2 <= r_i^2``, where ``w``
is the minimum-image **Cartesian atom->voxel** vector (sphere centred on the atom, not
on its nearest grid node) and ``r_i`` is the raw
:mod:`~torchref.base.electron_density.radius_policy` radius, enumerated over the
triclinic-correct per-axis box ``ceil(r_i * n_axis * ||inv_frac row_axis||)``. There is
no global splat radius, no grid-dependent requantization, no diagonal metric, no cube.

Which kernel runs, and whether a runtime failure may degrade, is read from
:data:`._backends.DENSITY_BACKENDS` -- the only place those criteria are written down.
Selection needs no configuration and there are no env-var knobs: the fastest kernel for
the device and dtype wins, and a missing or throwing accelerator degrades to the
portable splat with a warning. The one override is ``force_portable`` (per call, or
``with use_portable(): ...``), for the single failure automatic fallback cannot detect
-- a kernel that runs and returns *wrong numbers* rather than raising.
"""

from typing import Optional

import torch

from torchref.config import get_float_dtype, get_sigma_cutoff_ed
from torchref.utils.backends import run_or_degrade, select

from torchref.base.electron_density._backends import DENSITY_BACKENDS

# Re-imported to preserve this namespace: ``scaling/solvent.py`` imports
# ``_get_radius_offsets`` from here, not from its defining module.
from torchref.base.electron_density.kernels.offsets import _get_radius_offsets
from torchref.base.electron_density.radius_policy import (
    per_atom_radius_aniso,
    per_atom_radius_iso,
)


def build_electron_density(
    grid_shape,
    device: torch.device,
    xyz_iso: torch.Tensor,
    adp_iso: torch.Tensor,
    occ_iso: torch.Tensor,
    A_iso: torch.Tensor,
    B_iso: torch.Tensor,
    inv_frac_matrix: torch.Tensor,
    frac_matrix: torch.Tensor,
    xyz_aniso: Optional[torch.Tensor] = None,
    u_aniso: Optional[torch.Tensor] = None,
    occ_aniso: Optional[torch.Tensor] = None,
    A_aniso: Optional[torch.Tensor] = None,
    B_aniso: Optional[torch.Tensor] = None,
    dtype: torch.dtype = None,
    force_portable: Optional[bool] = None,
) -> torch.Tensor:
    """Build an electron density map, shape ``(nx, ny, nz)``, from atomic parameters.

    Each atom is splatted at its own truncation radius by whichever kernel
    ``DENSITY_BACKENDS`` selects; see the module docstring.

    Parameters
    ----------
    grid_shape : tuple of int
        Map dimensions ``(nx, ny, nz)``. No coordinate grid is needed or built: every
        splat derives a voxel's Cartesian position arithmetically from its index and
        ``inv_frac_matrix``.
    device : torch.device
        Device to allocate the map on.
    xyz_iso, adp_iso, occ_iso : torch.Tensor
        Isotropic positions ``(n_iso, 3)``, B-factors and occupancies ``(n_iso,)``.
    A_iso, B_iso : torch.Tensor
        ITC92 coefficients, shape ``(n_iso, 5)``.
    inv_frac_matrix, frac_matrix : torch.Tensor
        Cartesian-to-fractional and fractional-to-Cartesian, shape ``(3, 3)``.
    xyz_aniso, u_aniso, occ_aniso : torch.Tensor, optional
        Anisotropic positions ``(n_aniso, 3)``, U ``(n_aniso, 6)``, occupancies.
    A_aniso, B_aniso : torch.Tensor, optional
        ITC92 coefficients for the anisotropic atoms, shape ``(n_aniso, 5)``.
    dtype : torch.dtype, optional
        Float dtype of the map; defaults to ``get_float_dtype()``, which may be float64.
    force_portable : bool, optional
        Pin the portable reference splat. ``None`` defers to the process-wide setting
        (``torchref.utils.use_portable``). Use it to check whether an accelerator kernel
        is producing wrong numbers -- the one failure automatic fallback cannot detect.
    """
    if dtype is None:
        dtype = get_float_dtype()
    density_map = torch.zeros(tuple(grid_shape), dtype=dtype, device=device)

    # --- isotropic atoms ---
    if len(xyz_iso) > 0:
        density_map = _add_isotropic(
            density_map,
            xyz_iso,
            adp_iso,
            occ_iso,
            A_iso,
            B_iso,
            inv_frac_matrix,
            frac_matrix,
            force_portable=force_portable,
        )

    # --- anisotropic atoms ---
    if xyz_aniso is not None and len(xyz_aniso) > 0:
        density_map = _add_anisotropic(
            density_map,
            xyz_aniso,
            u_aniso,
            occ_aniso,
            A_aniso,
            B_aniso,
            inv_frac_matrix,
            frac_matrix,
            force_portable=force_portable,
        )

    return density_map


# =========================================================================
# Internal dispatch helpers
# =========================================================================


def _add_isotropic(
    density_map,
    xyz,
    adp,
    occ,
    A,
    B,
    inv_frac_matrix,
    frac_matrix,
    force_portable=None,
):
    """Add isotropic atoms at per-atom radius ``clamp(ceil(N_sigma * sigma_eff), [2,7])``.

    Every path applies the identical spherical cutoff, so the backend affects speed, not
    result. Only the first six arguments carry the device/dtype contract.
    """
    radius_per_atom = per_atom_radius_iso(adp, B, n_sigma=get_sigma_cutoff_ed())
    args = (density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_per_atom)
    backend = select(DENSITY_BACKENDS, args, force_portable=force_portable)
    return run_or_degrade(DENSITY_BACKENDS, backend, False, *args)


def _add_anisotropic(
    density_map,
    xyz,
    u,
    occ,
    A,
    B,
    inv_frac_matrix,
    frac_matrix,
    force_portable=None,
):
    """Add anisotropic atoms, radius = the ellipsoid's largest principal axis.

    Every path culls on the Euclidean sphere at that radius and evaluates the
    Mahalanobis form inside it; otherwise identical to the isotropic pass.
    """
    radius_per_atom = per_atom_radius_aniso(B, u, n_sigma=get_sigma_cutoff_ed())
    args = (density_map, xyz, u, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_per_atom)
    backend = select(DENSITY_BACKENDS, args, force_portable=force_portable)
    return run_or_degrade(DENSITY_BACKENDS, backend, True, *args)
