"""
Simple crystallographic model for direct tensor-based structure factor calculation.

This module provides a simplified interface for crystallographic refinement
where atomic parameters can be passed directly to the forward call without
needing to load from PDB/CIF files. Supports varying unit cells and space
groups per forward call.
"""

from typing import List, Optional, Tuple

import gemmi
import torch
import torch.nn as nn

import torchref.math_functions.get_scattering_factor_torch as gsf
import torchref.math_functions.math_numpy as mnp
import torchref.symmetry.symmetry as sym
from torchref.math_functions.math_torch import (
    extract_structure_factor_from_grid,
    find_grid_size,
    find_relevant_voxels,
    get_real_grid,
    ifft,
    vectorized_add_to_map,
)
from torchref.symmetry.map_symmetry import MapSymmetry


class SimpleModel(nn.Module):
    """
    A simplified crystallographic model that accepts atomic parameters directly.

    This module computes structure factors from atomic coordinates, B-factors,
    occupancies, and atom types. Unlike ModelFT, it accepts cell and spacegroup
    parameters in the forward call, allowing processing of multiple structures
    with different crystallographic parameters.

    Parameters
    ----------
    max_res : float, optional
        Maximum resolution for grid spacing in Angstroms. Default is 1.0.
    radius_angstrom : float, optional
        Radius in Angstroms for density calculation around each atom. Default is 4.0.
    dtype_float : torch.dtype, optional
        Data type for floating point tensors. Default is torch.float32.
    device : torch.device, optional
        Computation device. Default is torch.device('cpu').
    verbose : int, optional
        Verbosity level. Default is 0.

    Examples
    --------
    ::

        model = SimpleModel(max_res=1.5)
        xyz = torch.randn(100, 3, requires_grad=True)
        B = torch.ones(100) * 20.0
        occ = torch.ones(100)
        atom_types = ['C'] * 50 + ['N'] * 30 + ['O'] * 20
        cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])
        hkl = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sf = model(xyz, B, occ, atom_types, cell, 'P 21 21 21', hkl)
    """

    def __init__(
        self,
        max_res: float = 1.0,
        radius_angstrom: float = 4.0,
        dtype_float: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        verbose: int = 0,
    ):
        super().__init__()

        self.max_res = max_res
        self.radius_angstrom = radius_angstrom
        self.dtype_float = dtype_float
        self.device = device
        self.verbose = verbose

        # Cache for scattering parameters (element -> (A, B))
        self._scattering_cache = {}

        # Cache for symmetry operators (spacegroup -> (Symmetry, MapSymmetry or None))
        self._symmetry_cache = {}

        # Current grid/cell state (updated per forward call)
        self._current_cell = None
        self._current_spacegroup = None
        self._current_gridsize = None

    def _get_scattering_params(
        self, atom_types: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ITC92 scattering parameters for a list of atom types.

        Parameters
        ----------
        atom_types : list of str
            Element symbols for each atom (e.g., ['C', 'N', 'O', ...]).

        Returns
        -------
        A : torch.Tensor
            ITC92 A parameters with shape (n_atoms, 5).
        B : torch.Tensor
            ITC92 B parameters with shape (n_atoms, 5).
        """
        # Get unique elements and cache their parameters
        unique_elements = list(set(atom_types))
        for elem in unique_elements:
            if elem not in self._scattering_cache:
                params = gsf.get_parametrization_atom(0, elem)  # charge=0
                self._scattering_cache[elem] = (
                    params[0].to(device=self.device, dtype=self.dtype_float),
                    params[1].to(device=self.device, dtype=self.dtype_float),
                )

        # Build A and B tensors for all atoms
        A_list = []
        B_list = []
        for atom in atom_types:
            A, B = self._scattering_cache[atom]
            A_list.append(A)
            B_list.append(B)

        return torch.cat(A_list, dim=0), torch.cat(B_list, dim=0)

    def _get_symmetry_function(self, spacegroup: str) -> sym.Symmetry:
        """Get or create symmetry function for spacegroup."""
        if spacegroup not in self._symmetry_cache:
            sg_gemmi = gemmi.SpaceGroup(spacegroup.replace("  ", " "))
            sg_hm = sg_gemmi.hm
            self._symmetry_cache[spacegroup] = {
                "function": sym.Symmetry(sg_hm),
                "hm": sg_hm,
                "map_symmetry": {},  # grid_shape -> MapSymmetry
            }
        return self._symmetry_cache[spacegroup]["function"]

    def _get_map_symmetry(
        self, spacegroup: str, grid_shape: Tuple[int, ...], cell: torch.Tensor
    ) -> Optional[MapSymmetry]:
        """Get or create MapSymmetry for given spacegroup and grid shape."""
        sg_cache = self._symmetry_cache.get(spacegroup)
        if sg_cache is None:
            self._get_symmetry_function(spacegroup)
            sg_cache = self._symmetry_cache[spacegroup]

        # Check if we have cached MapSymmetry for this grid shape
        shape_key = tuple(grid_shape)
        if shape_key not in sg_cache["map_symmetry"]:
            sg_cache["map_symmetry"][shape_key] = MapSymmetry(
                space_group=sg_cache["hm"],
                map_shape=grid_shape,
                cell_params=cell,
                verbose=self.verbose,
                device=self.device,
            )
        return sg_cache["map_symmetry"][shape_key]

    def _compute_grid(
        self, cell: torch.Tensor, spacegroup: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute grid and transformation matrices for given cell and spacegroup.

        Returns
        -------
        real_space_grid : torch.Tensor
            Real-space coordinate grid with shape (nx, ny, nz, 3).
        gridsize : torch.Tensor
            Grid dimensions (nx, ny, nz).
        inv_frac_matrix : torch.Tensor
            Inverse fractional coordinate transformation matrix.
        frac_matrix : torch.Tensor
            Fractional coordinate transformation matrix.
        """
        cell_np = cell.detach().cpu().numpy()

        # Compute transformation matrices
        inv_frac_matrix = torch.tensor(
            mnp.get_inv_fractional_matrix(cell_np),
            dtype=self.dtype_float,
            device=self.device,
        )
        frac_matrix = torch.tensor(
            mnp.get_fractional_matrix(cell_np),
            dtype=self.dtype_float,
            device=self.device,
        )

        # Compute grid size
        gridsize_initial = find_grid_size(cell, self.max_res)
        gridsize_tuple = tuple(gridsize_initial.tolist())

        # Optimize grid size based on symmetry
        sym_func = self._get_symmetry_function(spacegroup)
        gridsize_optimized = sym_func.suggest_grid_size(gridsize_tuple)
        gridsize = torch.tensor(
            gridsize_optimized, dtype=torch.int32, device=self.device
        )

        # Compute real-space grid
        real_space_grid = get_real_grid(cell, gridsize=gridsize, device=self.device)

        if self.verbose > 1:
            print(f"Grid computed: shape={gridsize_optimized}, cell={cell_np}")

        return real_space_grid, gridsize, inv_frac_matrix, frac_matrix

    def _build_map(
        self,
        xyz: torch.Tensor,
        B: torch.Tensor,
        occ: torch.Tensor,
        A: torch.Tensor,
        B_scatter: torch.Tensor,
        real_space_grid: torch.Tensor,
        inv_frac_matrix: torch.Tensor,
        frac_matrix: torch.Tensor,
        map_symmetry: Optional[MapSymmetry] = None,
        apply_symmetry: bool = True,
    ) -> torch.Tensor:
        """
        Build electron density map from tensor inputs.

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates with shape (n_atoms, 3).
        B : torch.Tensor
            B-factors with shape (n_atoms,).
        occ : torch.Tensor
            Occupancies with shape (n_atoms,).
        A : torch.Tensor
            ITC92 A parameters with shape (n_atoms, 5).
        B_scatter : torch.Tensor
            ITC92 B parameters with shape (n_atoms, 5).
        real_space_grid : torch.Tensor
            Real-space coordinate grid.
        inv_frac_matrix : torch.Tensor
            Inverse fractional matrix.
        frac_matrix : torch.Tensor
            Fractional matrix.
        map_symmetry : MapSymmetry, optional
            Symmetry operator for the map.
        apply_symmetry : bool, optional
            Whether to apply space group symmetry. Default is True.

        Returns
        -------
        torch.Tensor
            Electron density map.
        """
        # Initialize empty map
        density_map = torch.zeros(
            real_space_grid.shape[:-1], dtype=self.dtype_float, device=self.device
        )

        # Find relevant voxels for each atom
        surrounding_coords, voxel_indices = find_relevant_voxels(
            real_space_grid,
            xyz,
            radius_angstrom=self.radius_angstrom,
            inv_frac_matrix=inv_frac_matrix,
        )

        # Add atom densities to map
        density_map = vectorized_add_to_map(
            surrounding_coords,
            voxel_indices,
            density_map,
            xyz,
            B,
            inv_frac_matrix,
            frac_matrix,
            A,
            B_scatter,
            occ,
        )

        # Apply symmetry if requested
        if apply_symmetry and map_symmetry is not None:
            density_map = map_symmetry(density_map)

        return density_map

    def forward(
        self,
        xyz: torch.Tensor,
        B: torch.Tensor,
        occ: torch.Tensor,
        atom_types: List[str],
        cell: torch.Tensor,
        spacegroup: str,
        hkl: torch.Tensor,
        apply_symmetry: bool = True,
    ) -> torch.Tensor:
        """
        Compute structure factors from atomic parameters.

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates in Angstroms with shape (n_atoms, 3).
        B : torch.Tensor
            Isotropic B-factors with shape (n_atoms,).
        occ : torch.Tensor
            Atomic occupancies with shape (n_atoms,).
        atom_types : list of str
            Element symbols for each atom (e.g., ['C', 'N', 'O', ...]).
        cell : torch.Tensor
            Unit cell parameters (a, b, c, alpha, beta, gamma) with shape (6,).
        spacegroup : str
            Space group symbol in Hermann-Mauguin notation (e.g., 'P 21 21 21').
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        apply_symmetry : bool, optional
            Whether to apply space group symmetry to the map. Default is True.

        Returns
        -------
        torch.Tensor
            Complex structure factors with shape (n_reflections,).

        Examples
        --------
        ::

            model = SimpleModel(max_res=1.5)
            xyz = torch.tensor([[10.0, 10.0, 10.0], [15.0, 15.0, 15.0]])
            B = torch.tensor([20.0, 25.0])
            occ = torch.tensor([1.0, 1.0])
            atom_types = ['C', 'N']
            cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0])
            hkl = torch.tensor([[1, 0, 0], [0, 1, 0]])
            sf = model(xyz, B, occ, atom_types, cell, 'P 1', hkl)
        """
        # Ensure tensors are on correct device and dtype
        xyz = xyz.to(device=self.device, dtype=self.dtype_float)
        B = B.to(device=self.device, dtype=self.dtype_float)
        occ = occ.to(device=self.device, dtype=self.dtype_float)
        cell = cell.to(device=self.device, dtype=self.dtype_float)
        hkl = hkl.to(device=self.device)

        # Compute grid and transformation matrices
        real_space_grid, gridsize, inv_frac_matrix, frac_matrix = self._compute_grid(
            cell, spacegroup
        )

        # Get map symmetry operator
        grid_shape = tuple(real_space_grid.shape[:-1])
        map_symmetry = (
            self._get_map_symmetry(spacegroup, grid_shape, cell)
            if apply_symmetry
            else None
        )

        # Get scattering parameters
        A, B_scatter = self._get_scattering_params(atom_types)

        # Build density map
        density_map = self._build_map(
            xyz,
            B,
            occ,
            A,
            B_scatter,
            real_space_grid,
            inv_frac_matrix,
            frac_matrix,
            map_symmetry,
            apply_symmetry,
        )

        # Compute structure factors via FFT
        reciprocal_space_grid = ifft(density_map)
        sf = extract_structure_factor_from_grid(reciprocal_space_grid, hkl)

        if self.verbose > 2:
            assert torch.all(
                torch.isfinite(sf)
            ), "Non-finite structure factors computed."

        return sf

    def compute_density_map(
        self,
        xyz: torch.Tensor,
        B: torch.Tensor,
        occ: torch.Tensor,
        atom_types: List[str],
        cell: torch.Tensor,
        spacegroup: str,
        apply_symmetry: bool = True,
    ) -> torch.Tensor:
        """
        Compute only the electron density map without structure factors.

        Parameters
        ----------
        xyz : torch.Tensor
            Atomic coordinates in Angstroms with shape (n_atoms, 3).
        B : torch.Tensor
            Isotropic B-factors with shape (n_atoms,).
        occ : torch.Tensor
            Atomic occupancies with shape (n_atoms,).
        atom_types : list of str
            Element symbols for each atom.
        cell : torch.Tensor
            Unit cell parameters (a, b, c, alpha, beta, gamma) with shape (6,).
        spacegroup : str
            Space group symbol in Hermann-Mauguin notation.
        apply_symmetry : bool, optional
            Whether to apply space group symmetry. Default is True.

        Returns
        -------
        torch.Tensor
            Electron density map.
        """
        xyz = xyz.to(device=self.device, dtype=self.dtype_float)
        B = B.to(device=self.device, dtype=self.dtype_float)
        occ = occ.to(device=self.device, dtype=self.dtype_float)
        cell = cell.to(device=self.device, dtype=self.dtype_float)

        # Compute grid
        real_space_grid, gridsize, inv_frac_matrix, frac_matrix = self._compute_grid(
            cell, spacegroup
        )

        # Get map symmetry
        grid_shape = tuple(real_space_grid.shape[:-1])
        map_symmetry = (
            self._get_map_symmetry(spacegroup, grid_shape, cell)
            if apply_symmetry
            else None
        )

        # Get scattering parameters
        A, B_scatter = self._get_scattering_params(atom_types)

        # Build and return map
        return self._build_map(
            xyz,
            B,
            occ,
            A,
            B_scatter,
            real_space_grid,
            inv_frac_matrix,
            frac_matrix,
            map_symmetry,
            apply_symmetry,
        )

    def clear_cache(self):
        """Clear cached scattering parameters and symmetry operations."""
        self._scattering_cache = {}
        self._symmetry_cache = {}

    def cuda(self, device=None):
        """Move model to GPU."""
        self.device = torch.device("cuda" if device is None else device)
        # Re-cache scattering parameters on new device
        new_cache = {}
        for elem, (A, B) in self._scattering_cache.items():
            new_cache[elem] = (A.cuda(device), B.cuda(device))
        self._scattering_cache = new_cache
        # Clear symmetry cache (will be rebuilt on demand)
        self._symmetry_cache = {}
        return self

    def cpu(self):
        """Move model to CPU."""
        self.device = torch.device("cpu")
        # Re-cache scattering parameters on CPU
        new_cache = {}
        for elem, (A, B) in self._scattering_cache.items():
            new_cache[elem] = (A.cpu(), B.cpu())
        self._scattering_cache = new_cache
        # Clear symmetry cache (will be rebuilt on demand)
        self._symmetry_cache = {}
        return self
