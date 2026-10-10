"""Map symmetrization by trilinear interpolation, for grids that need it.

The fallback behind :func:`~torchref.symmetry.map_symmetry.build_map_operator` when a
grid does not satisfy the group's divisibility, so symmetry mates land between grid
points. Interpolating costs accuracy that exact indexing does not, which is why
:meth:`~torchref.symmetry.symmetry.Symmetry.suggest_grid_size` exists -- prefer fixing
the grid over landing here.

Reach this through :meth:`~torchref.symmetry.symmetry.Symmetry.symmetrize_map`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from torchref.symmetry.map_symmetry import _combine
from torchref.utils.device_mixin import DeviceMixin


class _MapSymmetryInterpolation(DeviceMixin):
    """Symmetrize maps by resampling with ``grid_sample``.

    Parameters
    ----------
    symmetry : Symmetry
        The group to apply.
    map_shape : tuple of int
        Density map dimensions ``(nx, ny, nz)``.

    Notes
    -----
    Precomputes one sampling grid per operation, shape
    ``(n_ops, nx, ny, nz, 3)`` -- hundreds of megabytes at production grid sizes. That
    is why :class:`~torchref.symmetry.symmetry.Symmetry` memoizes only the most recent
    shape and drops the cache on any device move.
    """

    def __init__(self, symmetry, map_shape: tuple):
        self.symmetry = symmetry
        self.map_shape = tuple(int(n) for n in map_shape)
        self.sampling_grids = self._build_sampling_grids()

    @property
    def n_ops(self) -> int:
        """Number of symmetry operations."""
        return self.symmetry.n_ops

    @property
    def device(self) -> torch.device:
        """Device the sampling grids live on."""
        return self.sampling_grids.device

    def _build_sampling_grids(self) -> torch.Tensor:
        """Precompute per-operation ``grid_sample`` coordinates in ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_ops, nx, ny, nz, 3)``.
        """
        nx, ny, nz = self.map_shape
        symmetry = self.symmetry
        dtype = symmetry.dtype
        device = symmetry.device

        # Voxels at fractional edges i/N, the CCTBX/gemmi convention.
        fx = torch.arange(nx, dtype=dtype, device=device) / nx
        fy = torch.arange(ny, dtype=dtype, device=device) / ny
        fz = torch.arange(nz, dtype=dtype, device=device) / nz
        gx, gy, gz = torch.meshgrid(fx, fy, fz, indexing="ij")
        grid_flat = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

        transformed = symmetry.expand_positions(grid_flat)  # (n_ops, N, 3)
        # Wrap into [0, 1) for periodic boundaries.
        transformed = transformed - torch.floor(transformed)

        # On the periodically padded map (N + 1 voxels per axis, see _resample) the
        # fraction f sits at grid coordinate -1 + 2 f.
        sampling = -1.0 + 2.0 * transformed
        sampling = sampling.reshape(self.n_ops, nx, ny, nz, 3)

        # grid_sample reads the last axis as [x, y, z] -> [W, H, D], i.e. the REVERSE
        # of our [fx, fy, fz] -> [D, H, W]. Dropping this reorder still interpolates,
        # silently against the wrong axes.
        return sampling[..., [2, 1, 0]].contiguous()

    def _check_shape(self, density_map: torch.Tensor) -> None:
        """Reject a map whose shape this operator was not built for."""
        if tuple(density_map.shape) != self.map_shape:
            raise ValueError(
                f"Map shape {tuple(density_map.shape)} does not match the operator's "
                f"{self.map_shape}"
            )

    def mate(self, density_map: torch.Tensor, op_index: int) -> torch.Tensor:
        """One symmetry mate of ``density_map``.

        Parameters
        ----------
        density_map : torch.Tensor
            Density, shape ``(nx, ny, nz)``.
        op_index : int
            Operation index in ``[0, n_ops)``.

        Returns
        -------
        torch.Tensor
            Shape ``(nx, ny, nz)``.
        """
        if op_index < 0 or op_index >= self.n_ops:
            raise ValueError(
                f"Operation index {op_index} out of range [0, {self.n_ops - 1}]"
            )
        self._check_shape(density_map)
        return self._resample(_pad_periodic(density_map), op_index)

    def _resample(self, padded: torch.Tensor, op_index: int) -> torch.Tensor:
        """Sample one operation's mate off a map from :func:`_pad_periodic`."""
        # align_corners=True maps -1 and +1 to the first and last voxel of the padded
        # map, fractions 0 and 1. The padding repeats index 0 at the end, so a mate
        # in the last interval interpolates toward index 0 rather than clamping to
        # N-1; 'border' only absorbs rounding at the edge.
        transformed = F.grid_sample(
            padded,
            self.sampling_grids[op_index].unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return transformed.squeeze(0).squeeze(0)

    def all_mates(self, density_map: torch.Tensor) -> torch.Tensor:
        """Every symmetry mate, stacked.

        Parameters
        ----------
        density_map : torch.Tensor
            Density, shape ``(nx, ny, nz)``.

        Returns
        -------
        torch.Tensor
            Shape ``(n_ops, nx, ny, nz)``.
        """
        self._check_shape(density_map)
        padded = _pad_periodic(density_map)
        return torch.stack(
            [self._resample(padded, i) for i in range(self.n_ops)], dim=0
        )

    def symmetrize(
        self, density_map: torch.Tensor, combine: str = "sum"
    ) -> torch.Tensor:
        """Apply every operation and reduce the mates.

        Parameters
        ----------
        density_map : torch.Tensor
            Density, shape ``(nx, ny, nz)``.
        combine : {'sum', 'max'}, default 'sum'
            Reduction across mates.

        Returns
        -------
        torch.Tensor
            Shape ``(nx, ny, nz)``.
        """
        return _combine(self.all_mates(density_map), combine)

    def __repr__(self) -> str:
        return (
            f"_MapSymmetryInterpolation(n_ops={self.n_ops}, "
            f"map_shape={self.map_shape})"
        )


def _pad_periodic(density_map: torch.Tensor) -> torch.Tensor:
    """``density_map`` as ``(1, 1, nx + 1, ny + 1, nz + 1)``, index 0 repeated last."""
    return F.pad(density_map[None, None], (0, 1, 0, 1, 0, 1), mode="circular")


__all__ = ["_MapSymmetryInterpolation"]
