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
the device and dtype wins. An unavailable accelerator falls through to the next row
silently (CI catches it through ``expect_available``), a CUDA or Metal kernel that throws
degrades to the portable splat with a ``TorchRefDegradationWarning``, and the CPU
kernel's runtime errors propagate. The one override is ``force_portable`` (per call, or
``with use_portable(): ...``), for the single failure automatic fallback cannot detect
-- a kernel that runs and returns *wrong numbers* rather than raising.
"""

from typing import Optional

import torch

from torchref.config import get_float_dtype, get_sigma_cutoff_ed
from torchref.utils.backends import run_or_degrade, select

from torchref.base.electron_density._backends import DENSITY_BACKENDS
from torchref.base.electron_density.batching import dense_offsets, normalize_atoms
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
    batch_offsets_iso: Optional[torch.Tensor] = None,
    batch_offsets_aniso: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build an electron density map, shape ``(nx, ny, nz)``, from atomic parameters.

    Each atom is splatted at its own truncation radius by whichever kernel
    ``DENSITY_BACKENDS`` selects; see the module docstring.

    Optionally batched over models that share the cell and grid (see
    :mod:`~torchref.base.electron_density.batching`), returning ``(B, nx, ny, nz)``. Each
    atom set is batched on its own: dense, with a leading ``B`` on every atom tensor
    (``A``/``B`` may stay ``(n, 5)`` and be shared), or ragged, flat with
    ``batch_offsets_*``. An empty unbatched set beside a batched one contributes to no
    member; a non-empty one is refused as ambiguous.

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
    batch_offsets_iso, batch_offsets_aniso : torch.Tensor, optional
        ``(B + 1,)`` integer offsets assigning flat (ragged) isotropic / anisotropic atoms
        to members. Both sets must describe the same ``B``.

    Returns
    -------
    torch.Tensor
        ``(nx, ny, nz)``, or ``(B, nx, ny, nz)`` for a batched call.

    Raises
    ------
    ValueError
        If the two atom sets disagree on the batch, or the offsets do not cover the
        atoms.
    """
    if dtype is None:
        dtype = get_float_dtype()
    iso = normalize_atoms(xyz_iso, adp_iso, occ_iso, A_iso, B_iso, batch_offsets_iso)
    aniso = None
    if xyz_aniso is not None:
        aniso = normalize_atoms(
            xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso, batch_offsets_aniso
        )
    iso, aniso, n_members = _common_batch(iso, aniso)
    shape = tuple(grid_shape) if n_members is None else (n_members, *grid_shape)
    density_map = torch.zeros(shape, dtype=dtype, device=device)

    if iso[0].shape[0] > 0:
        density_map = _add_isotropic(
            density_map, *iso[:5], inv_frac_matrix, frac_matrix,
            force_portable=force_portable, batch_offsets=iso[5],
        )
    if aniso is not None and aniso[0].shape[0] > 0:
        density_map = _add_anisotropic(
            density_map, *aniso[:5], inv_frac_matrix, frac_matrix,
            force_portable=force_portable, batch_offsets=aniso[5],
        )
    return density_map


def _common_batch(iso, aniso):
    """Reconcile the batch of the two normalized atom sets.

    Returns both sets, with an empty unbatched set given all-empty offsets when the
    other is batched, and the member count (None when unbatched).
    """
    sets = [t for t in (iso, aniso) if t is not None]
    counts = {t[5].numel() - 1 for t in sets if t[5] is not None}
    if not counts:
        return iso, aniso, None
    if len(counts) > 1:
        raise ValueError(
            f"the isotropic and anisotropic atoms describe different batch sizes "
            f"{sorted(counts)}"
        )
    (n_members,) = counts

    def align(t):
        if t is None or t[5] is not None:
            return t
        if t[0].shape[0] > 0:
            raise ValueError(
                "one atom set is batched and the other is not; batch both, or pass the "
                "unbatched one empty"
            )
        return (*t[:5], dense_offsets(n_members, 0, t[0].device))

    return align(iso), align(aniso), n_members


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
    batch_offsets=None,
):
    """Add isotropic atoms at per-atom radius ``clamp(ceil(N_sigma * sigma_eff), [2,7])``.

    Every path applies the identical spherical cutoff, so the backend affects speed, not
    result. Only the first six arguments carry the device/dtype contract.
    """
    radius_per_atom = per_atom_radius_iso(adp, B, n_sigma=get_sigma_cutoff_ed())
    args = (density_map, xyz, adp, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_per_atom)
    backend = select(DENSITY_BACKENDS, args, force_portable=force_portable)
    return run_or_degrade(
        DENSITY_BACKENDS, backend, False, *args, batch_offsets=batch_offsets
    )


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
    batch_offsets=None,
):
    """Add anisotropic atoms, radius = the ellipsoid's largest principal axis.

    Every path culls on the Euclidean sphere at that radius and evaluates the
    Mahalanobis form inside it; otherwise identical to the isotropic pass.
    """
    radius_per_atom = per_atom_radius_aniso(B, u, n_sigma=get_sigma_cutoff_ed())
    args = (density_map, xyz, u, occ, A, B,
            inv_frac_matrix, frac_matrix, radius_per_atom)
    backend = select(DENSITY_BACKENDS, args, force_portable=force_portable)
    return run_or_degrade(
        DENSITY_BACKENDS, backend, True, *args, batch_offsets=batch_offsets
    )
