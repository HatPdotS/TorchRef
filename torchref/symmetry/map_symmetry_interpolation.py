"""
Map-level symmetry operations for electron density maps.

This module provides efficient symmetry operations applied directly to density maps,
which is much faster than applying symmetry to individual atoms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchref.config import get_float_dtype, normalize_device
from torchref.symmetry.spacegroup import SpaceGroup
from torchref.utils.device_mixin import DeviceMixin


class MapSymmetry(DeviceMixin, nn.Module):
    """
    Applies crystallographic symmetry operations to electron density maps.

    Takes an asymmetric-unit density map, applies each operation in fractional
    coordinates, interpolates via ``grid_sample`` and sums the mates -- cheaper
    than generating symmetry mates per atom and recalculating density. The
    per-operation sampling grids are precomputed in ``__init__`` for the fixed
    ``map_shape``.

    Attributes
    ----------
    space_group : str
        Space group name.
    map_shape : tuple of int
        Shape of the density map (nx, ny, nz).
    cell_params : numpy.ndarray
        Unit cell parameters.
    symmetry : Symmetry
        Symmetry operations handler.
    n_ops : int
        Number of symmetry operations.

    Examples
    --------
    ::

        map_sym = MapSymmetry(space_group='P21', map_shape=(64, 64, 64), cell_params=cell)
        asymmetric_map = model.build_density_map()
        symmetric_map = map_sym(asymmetric_map)
    """

    def __init__(
        self,
        space_group,
        map_shape,
        cell_params,
        dtype_float=None,
        verbose=1,
        device=None,
    ):
        """
        Initialize map symmetry operator.

        Parameters
        ----------
        space_group : str
            Space group name (e.g., 'P1', 'P21', 'P-1', etc.).
        map_shape : tuple of int
            Shape of the density map (nx, ny, nz).
        cell_params : array-like, shape (6,)
            Unit cell parameters [a, b, c, alpha, beta, gamma] in Å and degrees.
        dtype_float : torch.dtype, optional
            Floating point precision to use. Defaults to the configured
            ``dtypes.float`` (``get_float_dtype()``, float32 in production).
        verbose : int, default 1
            Verbosity level.
        device : torch.device, default: configured device.current
            Device to use for computation.
        """
        super().__init__()
        if dtype_float is None:
            dtype_float = get_float_dtype()
        device = normalize_device(device)
        self.dtype_float = dtype_float
        self.space_group = space_group
        self.map_shape = tuple(map_shape)
        self.cell_params = np.array(cell_params)
        self.verbose = verbose
        self.device = device
        self.symmetry = SpaceGroup(
            space_group, dtype=self.dtype_float, device=self.device
        )
        self.n_ops = self.symmetry.matrices.shape[0]
        if self.verbose > 0:
            print(f"MapSymmetry initialized for {space_group}")
            print(f"  Number of symmetry operations: {self.n_ops}")
            print(f"  Map shape: {self.map_shape}")

        self._setup_fractional_grid()
        self._setup_symmetry_grids()

    def _setup_fractional_grid(self):
        """Fractional grid with voxels at edges i/N (CCTBX/gemmi convention)."""
        nx, ny, nz = self.map_shape

        fx = torch.arange(nx, dtype=self.dtype_float, device=self.device) / nx
        fy = torch.arange(ny, dtype=self.dtype_float, device=self.device) / ny
        fz = torch.arange(nz, dtype=self.dtype_float, device=self.device) / nz

        # indexing='ij' so the result is (nx, ny, nz, 3) with the last dim [fx, fy, fz].
        grid_fx, grid_fy, grid_fz = torch.meshgrid(fx, fy, fz, indexing="ij")
        grid_frac = torch.stack([grid_fx, grid_fy, grid_fz], dim=-1)

        self.register_buffer("grid_frac", grid_frac)

    def _setup_symmetry_grids(self):
        """Precompute per-operation ``grid_sample`` coordinates in [-1, 1]."""
        nx, ny, nz = self.map_shape

        grid_flat = self.grid_frac.reshape(-1, 3)

        sampling_grids_list = []

        for i in range(self.n_ops):
            # R @ coords + t, on (3, nx*ny*nz)
            transformed = torch.matmul(self.symmetry.matrices[i], grid_flat.T)
            transformed = transformed.T  # (nx*ny*nz, 3)
            transformed = transformed + self.symmetry.translations[i]

            # Wrap to [0, 1) for periodic boundary conditions
            transformed = transformed - torch.floor(transformed)
            grid_shape_tensor = torch.tensor(
                [nx, ny, nz], dtype=self.dtype_float, device=transformed.device
            )
            # grid_coord = -1 + 2*N/(N-1) * frac, per dimension
            sampling_coords = (
                -1.0 + 2.0 * grid_shape_tensor / (grid_shape_tensor - 1.0) * transformed
            )

            sampling_grid = sampling_coords.reshape(nx, ny, nz, 3)

            # grid_sample reads the last axis as [x, y, z] -> [W, H, D], i.e. the
            # REVERSE of our [fx, fy, fz] -> [D, H, W]. Dropping this reorder still
            # interpolates, silently against the wrong axes.
            sampling_grid = sampling_grid[
                ..., [2, 1, 0]
            ]  # [fx, fy, fz] -> [fz, fy, fx]

            sampling_grids_list.append(sampling_grid)

        sampling_grids_stacked = torch.stack(sampling_grids_list, dim=0)

        self.register_buffer("sampling_grids", sampling_grids_stacked)

    def get_symmetry_mate(self, density_map, operation_index):
        """
        Apply a single symmetry operation to get one symmetry mate.

        Parameters
        ----------
        density_map : torch.Tensor, shape (nx, ny, nz)
            Electron density map (typically from asymmetric unit).
        operation_index : int
            Index of the symmetry operation to apply (0 to n_ops-1).

        Returns
        -------
        torch.Tensor, shape (nx, ny, nz)
            Density map after applying the symmetry operation.
        """
        if operation_index < 0 or operation_index >= self.n_ops:
            raise ValueError(
                f"Operation index {operation_index} out of range [0, {self.n_ops-1}]"
            )

        # Ensure map is correct shape
        if density_map.shape != self.map_shape:
            raise ValueError(
                f"Map shape {density_map.shape} doesn't match expected {self.map_shape}"
            )

        # Prepare for grid_sample
        map_5d = density_map.unsqueeze(0).unsqueeze(0)  # (1, 1, nx, ny, nz)

        # Get sampling grid for this operation
        sampling_grid = self.sampling_grids[operation_index]
        sampling_grid_batch = sampling_grid.unsqueeze(0)

        # Interpolate map at transformed coordinates
        # align_corners=True ensures that:
        #   -1 maps to index 0 (fractional coord 0)
        #   +1 maps to index N-1 (fractional coord (N-1)/N)
        # This matches the grid-edge convention (voxels at i/N)
        # padding_mode='border' handles periodic boundary conditions via the wrapping
        # we did in _setup_symmetry_grids
        transformed_map = F.grid_sample(
            map_5d,
            sampling_grid_batch,
            mode="bilinear",  # Trilinear interpolation for 3D
            padding_mode="border",  # Use border mode since we pre-wrapped coordinates
            align_corners=True,  # Critical: matches grid-edge convention
        )

        # Remove batch and channel dimensions
        transformed_map = transformed_map.squeeze(0).squeeze(0)

        return transformed_map

    def get_all_symmetry_mates(self, density_map):
        """
        Get all symmetry mates as a list.

        Parameters
        ----------
        density_map : torch.Tensor, shape (nx, ny, nz)
            Electron density map (typically from asymmetric unit).

        Returns
        -------
        list of torch.Tensor
            List of symmetry-related maps, one for each operation.
        """
        mates = []
        for i in range(self.n_ops):
            mates.append(self.get_symmetry_mate(density_map, i))
        return mates

    def forward(self, density_map, apply_symmetry=True, combine_mode="sum"):
        """
        Apply symmetry operations to density map.

        Parameters
        ----------
        density_map : torch.Tensor, shape (nx, ny, nz)
            Electron density map (typically from asymmetric unit).
        apply_symmetry : bool, default True
            If True, apply all symmetry operations and combine them.
            If False, return input map unchanged (useful for P1 or debugging).
        combine_mode : str, default 'sum'
            How to combine symmetry mates:

            - 'sum': Sum all symmetry mates (for electron density)
            - 'max': Take maximum across symmetry mates (for masks/boolean data)

        Returns
        -------
        torch.Tensor, shape (nx, ny, nz)
            Symmetry-expanded density map (combined symmetry mates).
        """
        if not apply_symmetry or self.n_ops == 1:
            # No symmetry or P1
            return density_map

        # Get all symmetry mates
        mates = self.get_all_symmetry_mates(density_map)
        mates_stacked = torch.stack(mates, dim=0)

        # Combine according to mode
        if combine_mode == "sum":
            symmetric_map = mates_stacked.sum(dim=0)
        elif combine_mode == "max":
            symmetric_map = mates_stacked.max(dim=0)[0]  # max returns (values, indices)
        else:
            raise ValueError(
                f"Unknown combine_mode: {combine_mode}. Use 'sum' or 'max'."
            )

        return symmetric_map

    def __call__(self, density_map, apply_symmetry=True, combine_mode="sum"):
        """Make the class callable like a PyTorch module."""
        return self.forward(
            density_map, apply_symmetry=apply_symmetry, combine_mode=combine_mode
        )

    def get_symmetry_info(self):
        """
        Get information about symmetry operations.

        Returns
        -------
        dict
            Dictionary with the following keys:

            - 'space_group' : str
            - 'n_operations' : int
            - 'matrices' : torch.Tensor, shape (n_ops, 3, 3)
            - 'translations' : torch.Tensor, shape (n_ops, 3)
        """
        return {
            "space_group": self.space_group,
            "n_operations": self.n_ops,
            "matrices": self.symmetry.matrices,
            "translations": self.symmetry.translations,
        }

    def __repr__(self):
        return (
            f"MapSymmetry(space_group='{self.space_group}', "
            f"n_ops={self.n_ops}, map_shape={self.map_shape})"
        )
