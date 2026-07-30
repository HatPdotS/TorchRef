"""
Central electron density building, dispatched solely by the shared ``Engine``.

One truncation contract, every backend
-------------------------------------
Every atom is splatted at its own per-atom truncation radius
(``N_sigma * sigma_eff``, with ``N_sigma = torchref.sigma_cutoff_ed``); there is
no single global splat radius. Crucially, every production kernel now applies the
*same* cutoff, so ``sigma_cutoff_ed`` means the same thing on every device:

    voxel v receives atom i's density iff ``||w||^2 <= r_i^2``, where ``w`` is the
    minimum-image **Cartesian atom->voxel** vector (the sphere is centred on the
    atom, not on its nearest grid node) and ``r_i`` is the raw
    :mod:`~torchref.base.electron_density.radius_policy` radius,

enumerated over the triclinic-correct per-axis box
``ceil(r_i * n_axis * ||inv_frac row_axis||)``. No grid-dependent requantization of
the radius, no diagonal-metric approximation, no cube. Historically the backends
disagreed here -- Metal inflated the radius to a whole voxel, the portable CPU
splat used a node-centred diagonal metric, and the CPU fast path splatted a cube --
by amounts comparable to or larger than the truncation error itself.

Engine dispatch
---------------
Which kernel runs is decided from a table, not from an if/elif ladder: see
:data:`torchref.base.electron_density._backends.DENSITY_BACKENDS`. That table is the only
place the criteria are written down -- device, dtype, which engines admit each backend, how
availability is probed, and whether a runtime failure may degrade -- so it is also the only
place to look, or to edit when adding a backend. The mechanics of reading it live in
:mod:`torchref.utils.backends`.

The capability-based ``Engine`` (AUTO/TRITON/METAL/EAGER) is the *only* switch: no
environment-variable dispatch, no parallel "tier" knobs. ``AUTO`` picks the fastest
available path per device and degrades quietly if an accelerator kernel is missing or
throws; ``EAGER`` pins the portable splat everywhere, which is the double-differentiable
route to use for Hessians; ``TRITON`` and ``METAL`` force their kernel and raise rather than
degrade, so a benchmark or an A/B comparison cannot silently measure something else. Pass
``engine=`` per call, or scope it with ``with use_engine(...)``.

The production splats live in ``kernels/cuda/variable_radius.py``,
``kernels/cpu/sphere_splat.py``, ``kernels/mps/variable_radius.py`` and (portable)
``kernels/cpu/variable_radius.py``; the per-atom radius policy is in
:mod:`torchref.base.electron_density.radius_policy`.
"""

from typing import Optional

import torch

from torchref.config import get_float_dtype, get_sigma_cutoff_ed
from torchref.utils.backends import run_or_degrade, select
from torchref.utils.triton_dispatch import Engine

from torchref.base.electron_density._backends import DENSITY_BACKENDS

# Re-imported to preserve this namespace: ``scaling/solvent.py`` imports
# ``_get_radius_offsets`` from here, not from its defining module.
from torchref.base.electron_density.kernels.offsets import _get_radius_offsets
from torchref.base.electron_density.radius_policy import (
    per_atom_radius_aniso,
    per_atom_radius_iso,
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
    engine: Optional[Engine] = None,
) -> torch.Tensor:
    """
    Build an electron density map from atomic parameters.

    Each atom is splatted at its own per-atom truncation radius
    (``N_sigma * sigma_eff``, with ``N_sigma = torchref.sigma_cutoff_ed``). Which kernel
    does the splatting is decided from ``DENSITY_BACKENDS``; see the module docstring.

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
        Voxel dimensions, shape (3,). **Unused by every splat** -- the per-atom
        truncation radius comes from each atom's B/U and ``torchref.sigma_cutoff_ed``
        via :mod:`~torchref.base.electron_density.radius_policy`, and the enumeration
        box from ``inv_frac_matrix``. Retained because ``SfFFT`` passes it
        positionally; dropping it is a wider API change.
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
    engine : Engine, optional
        Per-call backend override; defaults to the process-wide engine. Prefer this over
        ``set_engine`` when you only mean to steer the density splat -- a process-wide
        ``Engine.METAL`` also sends every target math function down the eager path, which
        would skew a benchmark.

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
            density_map,
            xyz_iso,
            adp_iso,
            occ_iso,
            A_iso,
            B_iso,
            inv_frac_matrix,
            frac_matrix,
            engine=engine,
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
            engine=engine,
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
    engine=None,
):
    """Add isotropic atoms with a per-atom variable radius.

    The per-atom radius is ``clamp(ceil(N_sigma * sigma_eff), [2,7])`` with
    ``N_sigma = torchref.sigma_cutoff_ed``.

    Which kernel runs, and what happens if it fails, are read from ``DENSITY_BACKENDS``
    rather than restated here -- there is no branch in this function to keep in sync with
    the table. Every path applies the identical spherical cutoff, so the choice affects
    speed, not result.

    Only the first six arguments carry the device/dtype contract; the table names them.
    """
    radius_per_atom = per_atom_radius_iso(adp, B, n_sigma=get_sigma_cutoff_ed())
    args = (density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_per_atom)
    backend = select(DENSITY_BACKENDS, args, engine)
    return run_or_degrade(DENSITY_BACKENDS, backend, False, *args, engine=engine)


def _add_anisotropic(
    density_map,
    xyz,
    u,
    occ,
    A,
    B,
    inv_frac_matrix,
    frac_matrix,
    engine=None,
):
    """Add anisotropic atoms with a per-atom variable radius (mirrors the iso path).

    The per-atom radius is the isotropic bounding radius of the ellipsoid (largest
    principal axis, ``per_atom_radius_aniso``). Every path culls on the Euclidean sphere at
    that radius and evaluates the Mahalanobis form inside it -- one contract, as for the
    isotropic pass.

    Same table, same selection: the two passes differ only in the radius policy and in
    which variant of each kernel the table names.
    """
    radius_per_atom = per_atom_radius_aniso(B, u, n_sigma=get_sigma_cutoff_ed())
    args = (density_map, xyz, u, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_per_atom)
    backend = select(DENSITY_BACKENDS, args, engine)
    return run_or_degrade(DENSITY_BACKENDS, backend, True, *args, engine=engine)
