"""
Map-level symmetry operations for electron density maps.

This module provides efficient symmetry operations applied directly to density maps,
which is much faster than applying symmetry to individual atoms.

This module uses a factory pattern: calling MapSymmetry() will automatically
return either MapSymmetryDirect (fast, no interpolation) or MapSymmetryInterpolation
(fallback with interpolation) depending on grid compatibility.

Space groups can be specified as strings, integers (1-230), or gemmi.SpaceGroup objects.
"""

import torch
import torch.nn as nn

from torchref.symmetry.spacegroup import SpaceGroup, SpaceGroupLike
from torchref.utils.device_mixin import DeviceMixin


def MapSymmetry(
    space_group: SpaceGroupLike,
    map_shape,
    cell_params,
    dtype_float=torch.float32,
    verbose=1,
    device=torch.device("cpu"),
):
    """
    Factory function to create the appropriate MapSymmetry implementation.

    This function checks if the grid size is compatible with direct indexing
    (no interpolation needed). If compatible, returns MapSymmetryDirect for
    maximum performance. Otherwise, returns MapSymmetryInterpolation as fallback.

    Parameters
    ----------
    space_group : str, int, or gemmi.SpaceGroup
        Space group specification (e.g., 'P21', 4, gemmi.SpaceGroup('P 21')).
    map_shape : tuple of int
        Shape of the density map (nx, ny, nz).
    cell_params : torch.Tensor, shape (6,)
        Unit cell parameters [a, b, c, alpha, beta, gamma] in Å and degrees.
    dtype_float : torch.dtype, default torch.float32
        Floating point precision to use.
    verbose : int, default 1
        Verbosity level (0=silent, 1=info, 2=debug).
    device : torch.device, default torch.device('cpu')
        Device to use for computation.

    Returns
    -------
    MapSymmetryDirect or MapSymmetryInterpolation
        The appropriate implementation based on grid compatibility.
    """
    # Check grid compatibility
    symmetry = SpaceGroup(space_group, dtype=dtype_float, device=device)
    compat = symmetry.check_grid_compatibility(map_shape)

    if compat["can_use_direct_indexing"]:
        # Use fast direct indexing implementation
        if verbose > 0:
            print(
                f"MapSymmetry: Using direct indexing (no interpolation) for {space_group}"
            )
        return MapSymmetryDirect(
            space_group, map_shape, cell_params, dtype_float, verbose, device
        )
    else:
        # Use interpolation-based fallback
        if verbose > 0:
            print("MapSymmetry: Grid not compatible with direct indexing")
            print(f"  Using interpolation-based fallback for {space_group}")
            if compat["issues"]:
                for issue in compat["issues"]:
                    print(f"    - {issue}")
            suggested = symmetry.suggest_grid_size(map_shape, make_fft_friendly=True)
            print(f"    Suggested grid for direct indexing: {suggested}")

        # Import and return interpolation version
        from torchref.symmetry.map_symmetry_interpolation import (
            MapSymmetry as MapSymmetryInterpolation,
        )

        return MapSymmetryInterpolation(
            space_group, map_shape, cell_params, dtype_float, verbose, device
        )


class MapSymmetryDirect(DeviceMixin, nn.Module):
    """
    Fast direct-indexing implementation of crystallographic symmetry operations.

    Computes symmetry mates one operation at a time (streaming) so that
    memory usage is O(grid) regardless of the number of symmetry operations,
    rather than O(n_ops * grid) for storing precomputed index grids.

    NOTE: Do not instantiate this class directly. Use the MapSymmetry() factory
    function instead, which will automatically select the appropriate implementation.
    """

    def __init__(
        self,
        space_group,
        map_shape,
        cell_params,
        dtype_float=torch.float32,
        verbose=1,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.dtype_float = dtype_float
        self.space_group = space_group
        self.map_shape = tuple(map_shape)
        self.cell_params = cell_params
        self.verbose = verbose
        self.device = device

        self.symmetry = SpaceGroup(
            space_group, dtype=self.dtype_float, device=self.device
        )
        self.n_ops = self.symmetry.matrices.shape[0]
        self.can_use_direct_indexing = True

        if self.verbose > 0:
            print(f"MapSymmetryDirect initialized for {space_group}")
            print(f"  Number of symmetry operations: {self.n_ops}")
            print(f"  Map shape: {self.map_shape}")

    # ------------------------------------------------------------------
    # Core: compute index grid for a single symmetry operation
    # ------------------------------------------------------------------

    def _compute_index_grid(self, op_index: int) -> torch.Tensor:
        """Compute the integer index grid for one symmetry operation.

        Returns shape (nx, ny, nz, 3) int64.  The result is temporary
        and not stored as a buffer.
        """
        nx, ny, nz = self.map_shape
        device = self.symmetry.matrices.device

        fx = torch.arange(nx, dtype=self.dtype_float, device=device) / nx
        fy = torch.arange(ny, dtype=self.dtype_float, device=device) / ny
        fz = torch.arange(nz, dtype=self.dtype_float, device=device) / nz
        gx, gy, gz = torch.meshgrid(fx, fy, fz, indexing="ij")
        grid_flat = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

        transformed = torch.matmul(self.symmetry.matrices[op_index], grid_flat.T).T
        transformed = transformed + self.symmetry.translations[op_index]
        transformed = transformed - torch.floor(transformed)

        shape_t = torch.tensor([nx, ny, nz], dtype=self.dtype_float, device=device)
        indices = torch.round(transformed * shape_t).to(torch.int64)
        indices[:, 0] %= nx
        indices[:, 1] %= ny
        indices[:, 2] %= nz

        return indices.reshape(nx, ny, nz, 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_symmetry_mate(self, density_map, operation_index):
        """Apply a single symmetry operation via direct indexing."""
        if operation_index < 0 or operation_index >= self.n_ops:
            raise ValueError(
                f"Operation index {operation_index} out of range [0, {self.n_ops-1}]"
            )
        if density_map.shape != self.map_shape:
            raise ValueError(
                f"Map shape {density_map.shape} doesn't match expected {self.map_shape}"
            )
        ig = self._compute_index_grid(operation_index)
        return density_map[ig[..., 0], ig[..., 1], ig[..., 2]]

    def forward(self, density_map, apply_symmetry=True, combine_mode="sum"):
        """Apply symmetry operations to density map.

        Computes one symmetry mate at a time and accumulates into the
        result, so peak memory is only 1 index grid + 2 density maps
        regardless of the number of symmetry operations.
        """
        if not apply_symmetry or self.n_ops == 1:
            return density_map

        ig = self._compute_index_grid(0)
        if combine_mode == "sum":
            result = density_map[ig[..., 0], ig[..., 1], ig[..., 2]]
            for i in range(1, self.n_ops):
                ig = self._compute_index_grid(i)
                result = result + density_map[ig[..., 0], ig[..., 1], ig[..., 2]]
        elif combine_mode == "max":
            result = density_map[ig[..., 0], ig[..., 1], ig[..., 2]]
            for i in range(1, self.n_ops):
                ig = self._compute_index_grid(i)
                result = torch.max(
                    result, density_map[ig[..., 0], ig[..., 1], ig[..., 2]]
                )
        else:
            raise ValueError(
                f"Unknown combine_mode: {combine_mode}. Use 'sum' or 'max'."
            )

        return result

    def __call__(self, density_map, apply_symmetry=True, combine_mode="sum"):
        """Make the class callable like a PyTorch module."""
        return self.forward(
            density_map, apply_symmetry=apply_symmetry, combine_mode=combine_mode
        )

    def get_symmetry_info(self):
        """Get information about symmetry operations."""
        return {
            "space_group": self.space_group,
            "n_operations": self.n_ops,
            "matrices": self.symmetry.matrices,
            "translations": self.symmetry.translations,
        }

    def __repr__(self):
        return (
            f"MapSymmetryDirect(space_group='{self.space_group}', "
            f"n_ops={self.n_ops}, map_shape={self.map_shape})"
        )
