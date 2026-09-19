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

        shape_t = torch.tensor([nx, ny, nz], dtype=dtype, device=device)
        # grid_coord = -1 + 2*N/(N-1) * frac, per dimension.
        sampling = -1.0 + 2.0 * shape_t / (shape_t - 1.0) * transformed
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

        # align_corners=True maps -1 to index 0 and +1 to index N-1, matching the
        # grid-edge convention above; padding_mode='border' is safe only because
        # _build_sampling_grids already wrapped the coordinates.
        transformed = F.grid_sample(
            density_map.unsqueeze(0).unsqueeze(0),
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
        return torch.stack(
            [self.mate(density_map, i) for i in range(self.n_ops)], dim=0
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


__all__ = ["_MapSymmetryInterpolation"]
