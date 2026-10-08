"""
Voxel utility functions for electron density calculations.

Functions for finding relevant voxels around atoms for map building, and the Cartesian
ball of voxel offsets they enumerate.
"""

from typing import Sequence

import torch

from torchref.base.electron_density.kernels.cpu.variable_radius import (
    _axis_half_widths,
    _box_offsets,
)
from torchref.config import get_int_dtype

#: Relative slack on ``radius**2`` in the membership test. Grid offsets land exactly on
#: the radius often (axis-aligned steps in an orthogonal cell), and float rounding of
#: ``dist_sq`` would put such a tie on either side at random. The slack resolves every
#: tie the same way in float32 and float64: inside without ``strict``, outside with it.
#: It is ~100 float32 ulps, while distinct grid distances are far further apart.
_BOUNDARY_RTOL = 1e-5


def voxel_offsets_within(
    radius: float,
    frac_matrix: torch.Tensor,
    grid_dims: Sequence[int],
    *,
    strict: bool = False,
) -> torch.Tensor:
    """Integer voxel offsets whose Cartesian displacement is within ``radius``.

    Offset ``o`` moves a point by ``frac_matrix @ (o / grid_dims)``, so the set is a
    true Cartesian ball in any cell, enumerated over the density kernels' triclinic
    per-axis box ``ceil(radius * n_axis * ||inv_frac row_axis||)``.

    Parameters
    ----------
    radius : float
        Ball radius in Å.
    frac_matrix : torch.Tensor
        Fractional-to-Cartesian matrix, shape ``(3, 3)``.
    grid_dims : sequence of int
        Grid dimensions ``(nx, ny, nz)``.
    strict : bool, default False
        Keep displacements shorter than ``radius`` rather than up to it. Offsets within
        a relative ``1e-5`` of ``radius**2`` count as exactly on it either way.

    Returns
    -------
    torch.Tensor
        Offsets, shape ``(R, 3)``, in the configured integer dtype, on the CPU
        whatever the device of ``frac_matrix``.
    """
    # The set depends only on the cell, grid and radius, so it is built once on the
    # CPU (callers cache it) and is the same set on every device. ``radius`` may arrive
    # as a 0-dim tensor on an accelerator; it must not meet the CPU tensors below.
    r_sq = float(radius) ** 2
    frac = frac_matrix.detach().cpu()
    half_widths = _axis_half_widths(float(radius), torch.linalg.inv(frac), grid_dims)
    offsets, off_cart = _box_offsets(half_widths, frac, grid_dims, "cpu", frac.dtype)
    dist_sq = (off_cart * off_cart).sum(-1)
    if strict:
        keep = dist_sq < r_sq * (1.0 - _BOUNDARY_RTOL)
    else:
        keep = dist_sq <= r_sq * (1.0 + _BOUNDARY_RTOL)
    return offsets[keep].to(get_int_dtype())


def half_voxel_diagonal(frac_matrix: torch.Tensor, grid_dims: Sequence[int]) -> float:
    """Half the longest body diagonal of one voxel, in Å.

    The farthest a point can lie from its nearest grid node, so a ball padded by it
    around that node holds every voxel within the unpadded radius of the point.

    Parameters
    ----------
    frac_matrix : torch.Tensor
        Fractional-to-Cartesian matrix, shape ``(3, 3)``.
    grid_dims : sequence of int
        Grid dimensions ``(nx, ny, nz)``.

    Returns
    -------
    float
        The half diagonal in Å.
    """
    frac = frac_matrix.detach().cpu()
    steps = frac / torch.tensor(grid_dims, dtype=frac.dtype)
    corners = torch.tensor(
        [[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]], dtype=frac.dtype
    )
    return 0.5 * float((corners @ steps.T).norm(dim=1).max())


def find_relevant_voxels(real_space_grid, xyz, radius_angstrom=4, inv_frac_matrix=None):
    """Voxels around each atom: a superset of those within ``radius_angstrom``.

    Each atom gets every voxel within ``radius_angstrom`` plus half the longest voxel
    diagonal of its nearest grid node, which covers every voxel within
    ``radius_angstrom`` of the atom itself. Callers that need the exact cutoff apply it
    (:func:`~torchref.base.electron_density.solvent_mask.add_to_solvent_mask` does).

    Parameters
    ----------
    real_space_grid : torch.Tensor
        Cartesian coordinate at each grid point, shape ``(nx, ny, nz, 3)``.
    xyz : torch.Tensor
        Cartesian atom coordinates, shape ``(N, 3)`` or ``(3,)``.
    radius_angstrom : float, optional
        Radius in Å, one value for every atom. The production splat
        (``main.build_electron_density``) does not use this helper; it derives a
        per-atom radius from each atom's B/U and ``torchref.sigma_cutoff_ed``.
    inv_frac_matrix : torch.Tensor, optional
        Cartesian-to-fractional, shape ``(3, 3)``. Required for non-orthogonal cells.

    Returns
    -------
    tuple
        ``(surrounding_coords, voxel_indices_wrapped)``, each ``(N, R, 3)`` for the ``R``
        candidate voxels. Only the voxel indices are wrapped -- atom coordinates are
        NOT, because ``smallest_diff()`` does the minimum-image work.
    """
    # Ensure xyz is 2D (N, 3)
    if xyz.ndim == 1:
        xyz = xyz.unsqueeze(0)

    grid_shape = torch.tensor(real_space_grid.shape[:3], device=xyz.device)

    # Get grid origin (first voxel corner)
    grid_origin = real_space_grid[0, 0, 0]

    # Convert atom positions to grid indices
    # For non-orthogonal cells, we must use fractional coordinates
    if inv_frac_matrix is not None:
        # Proper way: Cartesian -> Fractional -> Wrap to [0,1] -> Grid indices
        # This ensures atoms outside the unit cell are correctly wrapped
        xyz_frac = torch.matmul(inv_frac_matrix, xyz.T).T  # (N, 3)
        xyz_frac = xyz_frac % 1.0  # Wrap to [0, 1]
        center_idx = torch.round(xyz_frac * grid_shape.unsqueeze(0)).to(get_int_dtype())
    else:
        # Fallback for orthogonal cells (less accurate for non-orthogonal)
        voxelsize = real_space_grid[3, 3, 3] - real_space_grid[2, 2, 2]
        center_idx = torch.round(
            (xyz - grid_origin.unsqueeze(0)) / voxelsize.unsqueeze(0)
        ).to(get_int_dtype())

    voxel_indices_wrapped = excise_angstrom_radius_around_coord(
        real_space_grid, center_idx, radius_angstrom
    )

    # Extract coordinates from real_space_grid
    # For each atom, get all surrounding voxel coordinates
    surrounding_coords = real_space_grid[
        voxel_indices_wrapped[..., 0],
        voxel_indices_wrapped[..., 1],
        voxel_indices_wrapped[..., 2],
    ]

    return surrounding_coords, voxel_indices_wrapped


def excise_angstrom_radius_around_coord(
    real_space_grid, start_indices, radius_angstrom=4.0
):
    """Wrapped voxel indices ``(N, R, 3)`` around each of ``start_indices`` ``(N, 3)``
    on ``real_space_grid`` ``(nx, ny, nz, 3)``.

    The ``R`` offsets cover every voxel within ``radius_angstrom`` (Å) of any point
    whose nearest grid node is the start index: a Cartesian ball of ``radius_angstrom``
    plus :func:`half_voxel_diagonal` around the node, so the set is a superset and
    callers apply the exact cutoff. The cell is read from the grid's voxel steps.
    Indices are wrapped for periodic boundaries so they stay valid array indices.
    """
    if start_indices.ndim == 1:
        start_indices = start_indices.unsqueeze(0)
    grid_dims = tuple(real_space_grid.shape[:3])
    grid_shape = torch.tensor(grid_dims, device=start_indices.device)
    g = real_space_grid
    ends = torch.stack((g[1, 0, 0], g[0, 1, 0], g[0, 0, 1]), dim=1)
    steps = ends - g[0, 0, 0].unsqueeze(1)
    frac_matrix = steps * torch.tensor(grid_dims, dtype=g.dtype, device=g.device)

    radius = radius_angstrom + half_voxel_diagonal(frac_matrix, grid_dims)
    local_offsets = voxel_offsets_within(radius, frac_matrix, grid_dims)
    local_offsets = local_offsets.to(start_indices.device)

    voxel_indices = local_offsets.unsqueeze(0) + start_indices.unsqueeze(1)
    voxel_indices_wrapped = voxel_indices % grid_shape.unsqueeze(0).unsqueeze(0)
    return voxel_indices_wrapped
