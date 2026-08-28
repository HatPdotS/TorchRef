"""Real-space map symmetrization, selected by grid compatibility.

Two operators apply a :class:`~torchref.symmetry.symmetry.Symmetry` to a density map:
:class:`_MapSymmetryDirect` indexes symmetry mates at exact integers, and
:class:`~torchref.symmetry.map_symmetry_interpolation._MapSymmetryInterpolation`
falls back to ``grid_sample`` when the grid does not admit that.
:func:`build_map_operator` picks between them.

Reach these through :meth:`~torchref.symmetry.symmetry.Symmetry.symmetrize_map` rather
than directly: it owns the caching, and it is the reason the choice of operator does
not leak into calling code. A grid that forces interpolation costs accuracy silently,
so ask :meth:`~torchref.symmetry.symmetry.Symmetry.can_index_directly` before
committing to a grid, and
:meth:`~torchref.symmetry.symmetry.Symmetry.suggest_grid_size` to fix one.

Neither operator needs the unit cell: symmetry acts on fractional coordinates, so the
cell metric never enters.
"""

from __future__ import annotations

import torch

from torchref.utils.device_mixin import DeviceMixin


def build_map_operator(symmetry, map_shape: tuple):
    """Build the operator suited to ``map_shape``.

    Parameters
    ----------
    symmetry : Symmetry
        The group to apply.
    map_shape : tuple of int
        Density map dimensions ``(nx, ny, nz)``.

    Returns
    -------
    _MapSymmetryDirect or _MapSymmetryInterpolation
        Direct integer indexing when the grid permits it, otherwise the interpolating
        fallback.
    """
    if symmetry.can_index_directly(map_shape):
        return _MapSymmetryDirect(symmetry, map_shape)

    # Imported here, not at module scope: the interpolation module imports this one for
    # the shared operator contract.
    from torchref.symmetry.map_symmetry_interpolation import (
        _MapSymmetryInterpolation,
    )

    return _MapSymmetryInterpolation(symmetry, map_shape)


def _combine(mates: torch.Tensor, combine: str) -> torch.Tensor:
    """Reduce stacked symmetry mates.

    Parameters
    ----------
    mates : torch.Tensor
        Stacked mates, shape ``(n_ops, nx, ny, nz)``.
    combine : {'sum', 'max'}
        ``'sum'`` for electron density, ``'max'`` for masks and boolean data.

    Returns
    -------
    torch.Tensor
        Shape ``(nx, ny, nz)``.

    Raises
    ------
    ValueError
        For an unknown mode.
    """
    if combine == "sum":
        return mates.sum(dim=0)
    if combine == "max":
        return mates.max(dim=0)[0]
    raise ValueError(f"Unknown combine mode: {combine}. Use 'sum' or 'max'.")


class _MapSymmetryDirect(DeviceMixin):
    """Symmetrize maps by exact integer indexing, one operation at a time.

    Valid only on a grid whose dimensions satisfy the group's divisibility, so every
    symmetry mate falls on a grid point. :func:`build_map_operator` enforces that.

    Parameters
    ----------
    symmetry : Symmetry
        The group to apply.
    map_shape : tuple of int
        Density map dimensions ``(nx, ny, nz)``.

    Notes
    -----
    Holds no precomputed grids. Index grids are recomputed per operation so peak memory
    stays at one index grid plus two density maps, rather than scaling with the number
    of operations.
    """

    def __init__(self, symmetry, map_shape: tuple):
        self.symmetry = symmetry
        self.map_shape = tuple(int(n) for n in map_shape)

    @property
    def n_ops(self) -> int:
        """Number of symmetry operations."""
        return self.symmetry.n_ops

    @property
    def device(self) -> torch.device:
        """Device the operations live on."""
        return self.symmetry.device

    def _index_grid(self, op_index: int) -> torch.Tensor:
        """Integer index grid for one operation.

        Parameters
        ----------
        op_index : int
            Operation index.

        Returns
        -------
        torch.Tensor
            Shape ``(nx, ny, nz, 3)``, dtype ``int64``.

        Notes
        -----
        Deliberately does not use the batched
        :meth:`~torchref.symmetry.symmetry.Symmetry.expand_positions`: that would
        transform every operation at once, which is exactly the O(n_ops * grid) memory
        this operator exists to avoid.
        """
        nx, ny, nz = self.map_shape
        symmetry = self.symmetry
        dtype = symmetry.dtype
        device = symmetry.device

        fx = torch.arange(nx, dtype=dtype, device=device) / nx
        fy = torch.arange(ny, dtype=dtype, device=device) / ny
        fz = torch.arange(nz, dtype=dtype, device=device) / nz
        gx, gy, gz = torch.meshgrid(fx, fy, fz, indexing="ij")
        grid_flat = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

        transformed = torch.matmul(symmetry.matrices[op_index], grid_flat.T).T
        transformed = transformed + symmetry.translations[op_index]
        transformed = transformed - torch.floor(transformed)

        shape_t = torch.tensor([nx, ny, nz], dtype=dtype, device=device)
        indices = torch.round(transformed * shape_t).to(torch.int64)
        indices[:, 0] %= nx
        indices[:, 1] %= ny
        indices[:, 2] %= nz

        return indices.reshape(nx, ny, nz, 3)

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
        ig = self._index_grid(op_index)
        return density_map[ig[..., 0], ig[..., 1], ig[..., 2]]

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

        Notes
        -----
        Accumulates one mate at a time instead of stacking, keeping peak memory
        independent of the operation count.
        """
        self._check_shape(density_map)
        result = self.mate(density_map, 0)
        for i in range(1, self.n_ops):
            mate = self.mate(density_map, i)
            if combine == "sum":
                result = result + mate
            elif combine == "max":
                result = torch.max(result, mate)
            else:
                raise ValueError(
                    f"Unknown combine mode: {combine}. Use 'sum' or 'max'."
                )
        return result

    def __repr__(self) -> str:
        return (
            f"_MapSymmetryDirect(n_ops={self.n_ops}, map_shape={self.map_shape})"
        )


__all__ = ["build_map_operator"]
