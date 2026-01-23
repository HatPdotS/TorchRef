from typing import Optional, Tuple

import gemmi
import numpy as np
import torch

from torchref.math_functions.math_torch import (
    fft,
    hash_tensors,
    ifft,
)
from torchref.model.fft import FFT
from torchref.model.model import Model
from torchref.symmetry import Symmetry
from torchref.symmetry.map_symmetry import MapSymmetry
from torchref.utils.utils import TensorDict


class ModelFT(Model):
    """
    Model subclass for Fourier Transform-based electron density and structure factor calculations.

    ModelFT extends the base Model class with capabilities for computing electron
    density maps in real space and structure factors via FFT. Uses ITC92
    parametrization for electron density calculations.

    Parameters
    ----------
    max_res : float, optional
        Maximum resolution for grid spacing in Angstroms. Default is 1.0.
    radius_angstrom : float, optional
        Radius in Angstroms for density calculation around each atom. Default is 4.0.
    gridsize : tuple of int, optional
        Explicit grid size (nx, ny, nz). If None, computed from cell and max_res.
    *args
        Additional positional arguments passed to parent Model class.
    **kwargs
        Additional keyword arguments passed to parent Model class.

    Attributes
    ----------
    max_res : float
        Maximum resolution for grid spacing.
    radius_angstrom : float
        Radius for density calculation.
    gridsize : torch.Tensor
        Grid dimensions (nx, ny, nz).
    real_space_grid : torch.Tensor
        Real-space coordinate grid with shape (nx, ny, nz, 3).
    map : torch.Tensor or None
        Computed electron density map.
    parametrization : dict
        ITC92 parametrization dictionary {element: (A, B, C)}.
    map_symmetry : MapSymmetry
        Symmetry operator for map calculations.

    Examples
    --------
    Empty initialization for state_dict loading::

        model = ModelFT()
        model.load_state_dict(torch.load('model.pt'))

    File-based initialization::

        model = ModelFT(max_res=1.5)
        model.load_pdb('structure.pdb')
    """

    def __init__(
        self,
        *args,
        max_res=1.0,
        radius_angstrom=4.0,
        gridsize: Optional[Tuple[int, int, int]] = None,
        **kwargs,
    ):
        """
        Initialize an empty ModelFT shell.

        Creates a model shell ready for file loading via load_pdb()/load_cif()
        or state restoration via load_state_dict().

        Parameters
        ----------
        max_res : float, optional
            Maximum resolution for grid spacing in Angstroms. Default is 1.0.
        radius_angstrom : float, optional
            Radius in Angstroms for density calculation. Default is 4.0.
        gridsize : tuple of int, optional
            Explicit grid size tuple (nx, ny, nz). If None, computed automatically.
        *args
            Passed to parent Model class.
        **kwargs
            Passed to parent Model class.
        """
        super().__init__(*args, **kwargs)

        # FT-specific configuration
        self.max_res = max_res
        self.radius_angstrom = radius_angstrom
        self._explicit_gridsize = gridsize
        self._cache = TensorDict()


    def load_pdb(self, filename):
        """
        Load a PDB file and initialize the model with FT-specific setup.

        Parameters
        ----------
        filename : str
            Path to the PDB file.

        Returns
        -------
        ModelFT
            Self, for method chaining.
        """
        super().load_pdb(filename)
        # Set cell and spacegroup on FFT submodule
        self._fft = FFT(self.cell, self.spacegroup, device=self.device, dtype_float=self.dtype_float)
        self.setup_grid()
        return self

    def select(self, selection):
        selection = super().select(selection)
        selection._build_parametrization()
        # Create FFT submodule for the selection
        selection._fft = FFT(selection.cell, selection.spacegroup, device=selection.device, dtype_float=selection.dtype_float)
        selection.setup_grid()
        return selection

    def load_cif(self, filename):
        """
        Load a CIF file and initialize the model with FT-specific setup.

        Parameters
        ----------
        filename : str
            Path to the CIF/mmCIF file.

        Returns
        -------
        ModelFT
            Self, for method chaining.
        """
        super().load_cif(filename)
        self._build_parametrization()
        # Create FFT submodule with cell and spacegroup
        self._fft = FFT(self.cell, self.spacegroup, device=self.device, dtype_float=self.dtype_float)
        self.setup_grid()
        return self

    def setup_gridsize(self, max_res=None):
        """
        Compute optimal grid dimensions.

        Delegates to FFT.compute_grid_size().

        Parameters
        ----------
        max_res : float, optional
            Maximum resolution in Angstroms. If None, uses self.max_res.

        Returns
        -------
        torch.Tensor
            Grid dimensions (nx, ny, nz) as int32 tensor.
        """
        if max_res is not None:
            self.max_res = max_res
            self._fft.max_res = max_res

        if self.verbose > 1:
            print(f"Defining grid size for ={self.max_res} Å")

        # Use Cell's compute_grid_size method
        gridsize = self.cell.compute_grid_size(self.max_res)
        return torch.tensor(gridsize, dtype=torch.int32, device=self.device)

    def _build_parametrization(self):
        """
        Build ITC92 parametrization for all atoms in the model.

        Delegates to parent Model class which handles the actual parametrization
        building. This method exists for API compatibility.
        """
        # Use parent's implementation
        return super()._build_parametrization()

    # =========================================================================
    # Backward-compatible properties for scattering parameters
    # =========================================================================

    @property
    def A(self) -> torch.Tensor:
        """
        ITC92 A parameters (amplitudes) for all atoms.

        Returns
        -------
        torch.Tensor
            A parameters with shape (n_atoms, 5).
        """
        self._build_parametrization()
        return self._A

    @property
    def B(self) -> torch.Tensor:
        """
        ITC92 B parameters (widths) for all atoms.

        Returns
        -------
        torch.Tensor
            B parameters with shape (n_atoms, 5).
        """
        self._build_parametrization()
        return self._B

    # =========================================================================
    # Backward-compatible properties for FFT grid attributes
    # =========================================================================

    @property
    def gridsize(self) -> Optional[torch.Tensor]:
        """Grid dimensions (nx, ny, nz)."""
        return self._fft.gridsize

    @gridsize.setter
    def gridsize(self, value):
        """Set grid size (for backward compatibility)."""
        self._fft.gridsize = value

    @property
    def real_space_grid(self) -> Optional[torch.Tensor]:
        """Real-space coordinate grid with shape (nx, ny, nz, 3)."""
        return self._fft.real_space_grid

    @real_space_grid.setter
    def real_space_grid(self, value):
        """Set real space grid (for backward compatibility)."""
        self._fft.real_space_grid = value

    @property
    def voxel_size(self) -> Optional[torch.Tensor]:
        """Voxel dimensions."""
        return self._fft.voxel_size

    @voxel_size.setter
    def voxel_size(self, value):
        """Set voxel size (for backward compatibility)."""
        self._fft.voxel_size = value

    @property
    def map_symmetry(self) -> Optional[MapSymmetry]:
        """Symmetry operator for map calculations."""
        return self._fft.map_symmetry

    @map_symmetry.setter
    def map_symmetry(self, value):
        """Set map symmetry (for backward compatibility)."""
        self._fft.map_symmetry = value

    def get_iso(self):
        """
        Get isotropic atoms with their ITC92 parameters.

        Returns
        -------
        xyz : torch.Tensor
            Atomic coordinates with shape (n_atoms, 3).
        adp : torch.Tensor
            Atomic displacement parameters (isotropic) with shape (n_atoms,).
        occupancy : torch.Tensor
            Occupancies with shape (n_atoms,).
        A : torch.Tensor
            ITC92 A parameters (amplitudes) with shape (n_atoms, 5).
        B : torch.Tensor
            ITC92 B parameters (widths) with shape (n_atoms, 5).
        """
        # Get base isotropic data from parent
        xyz, adp, occupancy = super().get_iso()

        # Get scattering parameters from parent
        A, B = self.get_scattering_params_iso()

        return xyz, adp, occupancy, A, B

    def get_aniso(self):
        """
        Get anisotropic atoms with their ITC92 parameters.

        Returns
        -------
        xyz : torch.Tensor
            Atomic coordinates with shape (n_atoms, 3).
        u : torch.Tensor
            Anisotropic U parameters with shape (n_atoms, 6).
        occupancy : torch.Tensor
            Occupancies with shape (n_atoms,).
        A : torch.Tensor
            ITC92 A parameters (amplitudes) with shape (n_atoms, 5).
        B : torch.Tensor
            ITC92 B parameters (widths) with shape (n_atoms, 5).
        """
        # Get base anisotropic data from parent
        xyz, u, occupancy = super().get_aniso()

        # Get scattering parameters from parent
        A, B = self.get_scattering_params_aniso()

        return xyz, u, occupancy, A, B

    def setup_grid(self, max_res=None, gridsize=None):
        """
        Setup real-space grid for electron density calculation.

        Delegates to FFT.setup_grid() using the stored cell and spacegroup.

        Parameters
        ----------
        max_res : float, optional
            Maximum resolution for grid spacing in Angstroms.
            If None, uses self.max_res.
        gridsize : tuple of int, optional
            Explicit grid size (nx, ny, nz). If None, computed automatically
            using Cell.compute_grid_size() and Symmetry.suggest_grid_size().
        """
        if max_res is not None:
            self.max_res = max_res
            self._fft.max_res = max_res

        if self.verbose > 1:
            print(f"Setting up grids with max_res={self.max_res} Å")

        # Determine grid size to use
        gridsize_to_use = gridsize or self._explicit_gridsize

        # Delegate to FFT submodule (which now uses stored cell/spacegroup)
        self._fft.setup_grid(
            gridsize=gridsize_to_use,
            max_res=self.max_res,
        )

        if self.verbose > 2:
            print(f"Grid shape: {self._fft.real_space_grid.shape[:-1]}")
            print(f"Voxel size: {self._fft.voxel_size}")

    def get_radius(self, min_radius_Angstrom: float = 4.0):
        """
        Get the radius in voxels used for density calculation around each atom.

        Parameters
        ----------
        min_radius_Angstrom : float, optional
            Minimum radius in Angstroms. Default is 4.0.

        Returns
        -------
        int
            Radius in voxels.
        """
        if not hasattr(self, "real_space_grid") or self.real_space_grid is None:
            self.setup_grid()
        voxel_size = self.real_space_grid[1, 1, 1] - self.real_space_grid[0, 0, 0]
        min_radius = (
            torch.ceil(min_radius_Angstrom / torch.min(voxel_size))
            .to(torch.int32)
            .item()
        )
        if self.verbose > 1:
            print(
                f"Calculated radius for density calculation: {min_radius} voxels (voxel size: {voxel_size}), this corresponds to at least {min_radius_Angstrom} Å"
            )
        return min_radius

    def build_complete_map(self, radius=None, apply_symmetry=True):
        """
        Build electron density map from all atoms.

        Uses get_iso() and get_aniso() to get atom data and constructs
        the complete electron density map.

        Parameters
        ----------
        radius : int, optional
            Radius in voxels around each atom to compute density.
            If None, uses self.radius.
        apply_symmetry : bool, optional
            If True and space group is not P1, apply symmetry operations
            to the map. Default is True.

        Returns
        -------
        torch.Tensor
            Electron density map with symmetry applied if requested.
        """
        self.map = self.build_initial_map(apply_symmetry=apply_symmetry)

        if self.verbose > 2:
            print(
                f"Density map built. Sum: {self.map.sum():.2f}, Max: {self.map.max():.4f}"
            )
        return self.map

    def build_initial_map(self, apply_symmetry=True):
        """
        Build electron density map from atomic parameters.

        Delegates to FFT.build_density_map() using the model's stored parameters.

        Parameters
        ----------
        apply_symmetry : bool, optional
            If True, apply crystallographic symmetry to the map. Default is True.

        Returns
        -------
        torch.Tensor
            Electron density map with shape (nx, ny, nz).
        """
        if self._fft.real_space_grid is None:
            self.setup_grid()

        if self.verbose > 2:
            print(
                f"Building density map with radius={self.radius_angstrom} angstrom..."
            )

        # Get isotropic atoms
        xyz_iso, adp_iso, occ_iso, A_iso, B_iso = self.get_iso()

        if self.verbose > 3:
            assert torch.all(
                torch.isfinite(A_iso)
            ), "Non-finite values found in A_iso during map building."
            assert torch.all(
                torch.isfinite(B_iso)
            ), "Non-finite values found in B_iso during map building."
            assert torch.all(
                torch.isfinite(xyz_iso)
            ), "Non-finite values found in xyz_iso during map building."
            assert torch.all(
                torch.isfinite(adp_iso)
            ), "Non-finite values found in adp_iso during map building."
            assert torch.all(
                torch.isfinite(occ_iso)
            ), "Non-finite values found in occ_iso during map building."

        # Get anisotropic atoms
        xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso = self.get_aniso()

        # Delegate to FFT submodule
        self.map = self._fft.build_density_map(
            xyz_iso=xyz_iso,
            adp_iso=adp_iso,
            occ_iso=occ_iso,
            A_iso=A_iso,
            B_iso=B_iso,
            xyz_aniso=xyz_aniso if len(xyz_aniso) > 0 else None,
            u_aniso=u_aniso if len(xyz_aniso) > 0 else None,
            occ_aniso=occ_aniso if len(xyz_aniso) > 0 else None,
            A_aniso=A_aniso if len(xyz_aniso) > 0 else None,
            B_aniso=B_aniso if len(xyz_aniso) > 0 else None,
            apply_symmetry=apply_symmetry,
        )

        if self.verbose > 3:
            assert torch.all(
                torch.isfinite(self.map)
            ), "Non-finite values found in map."

        return self.map

    def save_map(self, filename):
        """
        Save the electron density map to a CCP4 format file.

        Parameters
        ----------
        filename : str
            Output filename for the map.

        Raises
        ------
        ValueError
            If no map has been computed yet.
        """
        if self.map is None:
            raise ValueError("No map to save. Call build_density_map() first.")

        np_map = self.map.detach().cpu().numpy().astype(np.float32)
        cell = self.cell.tolist()
        if self.verbose > 0:
            print(f"Saving map to {filename}")
            print(f"  Map shape: {self.map.shape}")
            print(f"  Map sum: {self.map.sum():.2f}")
            print(f"  Map range: [{self.map.min():.4f}, {self.map.max():.4f}]")

        map_ccp = gemmi.Ccp4Map()
        map_ccp.grid = gemmi.FloatGrid(
            np_map, gemmi.UnitCell(*cell), gemmi.SpaceGroup("P1")
        )
        map_ccp.setup(0.0)
        map_ccp.update_ccp4_header()
        map_ccp.write_ccp4_map(filename)
        if self.verbose > 0:
            print("Map saved successfully")

    def get_map_statistics(self):
        """Get statistics about the current density map."""
        if self.map is None:
            return None

        stats = {
            "shape": self.map.shape,
            "sum": float(self.map.sum()),
            "mean": float(self.map.mean()),
            "std": float(self.map.std()),
            "min": float(self.map.min()),
            "max": float(self.map.max()),
            "n_positive": int((self.map > 0).sum()),
            "n_negative": int((self.map < 0).sum()),
        }
        return stats

    def rebuild_map(self, radius=None):
        """
        Rebuild the density map from scratch.

        Convenience method that clears and rebuilds everything.

        Parameters
        ----------
        radius : int, optional
            Radius in voxels around each atom. If None, uses self.radius.
            If specified, overrides self.radius.

        Returns
        -------
        torch.Tensor
            Rebuilt electron density map.
        """
        if self.verbose > 1:
            print("Rebuilding density map from scratch...")
        return self.build_density_map(radius=radius)

    def to(self, device=None, dtype=None):
        """
        Move model and FT-specific data to specified device and/or dtype.

        Parameters
        ----------
        device : torch.device or str, optional
            Target device.
        dtype : torch.dtype, optional
            Target data type.

        Returns
        -------
        ModelFT
            Self, for method chaining.
        """
        result = super().to(device=device, dtype=dtype)

        # Update FFT module's device/dtype tracking
        if device is not None:
            self._fft.device = torch.device(device)
        if dtype is not None:
            self._fft.dtype_float = dtype

        return result

    def cuda(self, device=None):
        """Move model and FT-specific data to GPU."""
        return self.to(device='cuda' if device is None else device)

    def cpu(self):
        """Move model and FT-specific data to CPU."""
        return self.to(device="cpu")

    def update_pdb(self):
        """
        Update PDB with current atomic parameters.
        """
        super().update_pdb()

    def reset_cache(self):
        self._cache = TensorDict()

    def get_structure_factor(self, hkl: torch.Tensor, recalc=True) -> torch.Tensor:
        """
        Get structure factors for given hkl reflections.

        Uses caching to avoid recomputation if parameters haven't changed.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        recalc : bool, optional
            If True, forces recalculation even if cached. Default is True.

        Returns
        -------
        torch.Tensor
            Complex structure factors with shape (n_reflections,).
        """
        # Compute current parameter hash
        params = (*self.parameters(), hkl)
        current_param_hash = hash_tensors(params)

        key = current_param_hash
        if not recalc and key in self._cache:
            if self.verbose > 2:
                print("Using cached structure factors")
            return self._cache[key]

        # Build map and compute structure factors using FFT module
        self.build_complete_map()
        self.reciprocal_space_grid = ifft(self.map)
        sf = self._fft.map_to_structure_factors(self.map, hkl)

        self._cache[key] = sf.detach()

        return sf

    def fft(self):
        """Perform FFT on the current reciprocal grid."""
        self.density = fft(self.reciprocal_space_grid)
        return self.density

    def forward(self, hkl, recalc=True) -> torch.Tensor:
        """
        Forward pass to compute structure factors for given hkl.

        Parameters
        ----------
        hkl : torch.Tensor
            Miller indices with shape (n_reflections, 3).
        recalc : bool, optional
            If True, forces recalculation. Default is True.

        Returns
        -------
        torch.Tensor
            Calculated complex structure factors with shape (n_reflections,).
        """
        f = self.get_structure_factor(hkl, recalc=recalc)
        if self.verbose > 2:
            assert torch.all(
                torch.isfinite(f)
            ), "Non-finite values found while calculating fcalc."

        return f

    def copy(self):
        """
        Create a deep copy of the ModelFT.

        Creates a complete independent copy including all Model base class data,
        FFT submodule state (gridsize, real_space_grid, voxel_size, map_symmetry),
        ITC92 parametrization, and scalar attributes.
        Cache is reset to empty.

        Returns
        -------
        ModelFT
            A new ModelFT instance with copied data.

        Examples
        --------
        ::

            model = ModelFT().load_pdb('structure.pdb')
            model_copy = model.copy()
            # model_copy is independent, changes won't affect model
        """
        if not self.initialized:
            raise RuntimeError("Cannot copy an uninitialized ModelFT. Load data first.")

        # Create new ModelFT instance with same configuration
        model_copy = ModelFT(
            dtype_float=self.dtype_float,
            verbose=self.verbose,
            device=self.device,
            strip_H=self.strip_H,
            max_res=self.max_res,
            radius_angstrom=self.radius_angstrom,
            gridsize=self._explicit_gridsize,
        )

        # Deep copy the PDB DataFrame
        model_copy.pdb = self.pdb.copy(deep=True)

        # Copy scalar attributes
        model_copy.spacegroup = self.spacegroup  # gemmi.SpaceGroup is immutable
        model_copy.symmetry = Symmetry(self.spacegroup) if self.spacegroup else None
        model_copy.initialized = True

        # Copy Cell object
        if self.cell is not None:
            model_copy.cell = self.cell.clone()

        # Copy all registered buffers using PyTorch's _buffers dict
        # (excluding FFT submodule buffers which are handled separately)
        for buffer_name, buffer_value in self._buffers.items():
            if buffer_value is not None:
                model_copy.register_buffer(buffer_name, buffer_value.clone())

        # Copy all modules (parameter wrappers) using their .copy() methods
        # Note: _fft is handled separately below
        for module_name, module in self._modules.items():
            if module_name == "_fft":
                continue  # Handled below
            if module is not None and hasattr(module, "copy"):
                setattr(model_copy, module_name, module.copy())

        # Copy alternative conformation pairs
        if hasattr(self, "altloc_pairs") and self.altloc_pairs:
            model_copy.altloc_pairs = [
                tuple(tensor.clone() for tensor in group) for group in self.altloc_pairs
            ]
        else:
            model_copy.altloc_pairs = []

        # Copy FT-specific attributes: _parametrization dict
        if hasattr(self, "_parametrization") and self._parametrization is not None:
            import copy as copy_module
            model_copy._parametrization = copy_module.deepcopy(self._parametrization)

        # Copy map if it exists
        if self.map is not None:
            model_copy.map = self.map.clone()

        # Copy FFT submodule state by setting up grid if it was set up
        if self._fft.real_space_grid is not None:
            model_copy.setup_grid(max_res=self.max_res)

        # Reset cache (don't copy cached structure factors)
        from torchref.utils.utils import TensorDict
        object.__setattr__(model_copy, "_cache", TensorDict())

        if self.verbose > 0:
            print(f"✓ ModelFT copied successfully ({len(model_copy.pdb)} atoms)")

        return model_copy

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Return a dictionary containing the complete state of the ModelFT.

        Extends parent Model.state_dict() with FT-specific parameters including
        max_res, radius_angstrom. Grid state is handled by the FFT submodule.

        Parameters
        ----------
        destination : dict, optional
            Optional dict to populate.
        prefix : str, optional
            Prefix for parameter names. Default is ''.
        keep_vars : bool, optional
            Whether to keep variables in computational graph. Default is False.

        Returns
        -------
        dict
            Complete state dictionary.
        """
        # Get parent Model state_dict (includes _A, _B buffers and FFT submodule)
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

        # Add ModelFT-specific state
        state[prefix + "max_res"] = self.max_res
        state[prefix + "radius_angstrom"] = self.radius_angstrom

        # Note: FFT submodule state (gridsize, real_space_grid, voxel_size) is
        # automatically included via PyTorch's module serialization with _fft. prefix
        # _parametrization dict is not saved as it can be rebuilt from _A, _B buffers
        # _cache is not saved as it should be rebuilt

        return state

    @classmethod
    def create_from_state_dict(
        cls,
        state_dict: dict,
        device: torch.device = torch.device("cpu"),
        verbose: int = 1,
        dtype_float: torch.dtype = torch.float32,
    ) -> "ModelFT":
        """
        Create a fully initialized ModelFT from a state dictionary.

        This is the recommended way to restore a ModelFT from a saved state.
        Creates an instance with properly initialized submodules, then loads the state.

        Parameters
        ----------
        state_dict : dict
            State dictionary from torch.save(model.state_dict(), ...).
        device : torch.device, optional
            Device to place tensors on. Default is torch.device('cpu').
        verbose : int, optional
            Verbosity level. Default is 1.
        dtype_float : torch.dtype, optional
            Float dtype for tensors. Default is torch.float32.

        Returns
        -------
        ModelFT
            Fully initialized instance with restored state.
        """
        from torchref.symmetry import SpaceGroup

        # Extract ModelFT-specific metadata
        max_res = state_dict.pop("max_res", 1.0)
        radius_angstrom = state_dict.pop("radius_angstrom", 4.0)

        # Extract Model metadata
        pdb = state_dict.pop("pdb", None)
        spacegroup_str = state_dict.pop("spacegroup", None)
        cell_tensor = state_dict.pop("cell", None)
        initialized = state_dict.pop("initialized", False)
        saved_dtype = state_dict.pop("dtype_float", dtype_float)
        state_dict.pop("device", None)  # Remove but don't use (use provided device)
        strip_H = state_dict.pop("strip_H", True)
        altloc_pairs = state_dict.pop("altloc_pairs", [])

        # Extract grid info from FFT submodule state
        # Note: FFT buffers are prefixed with "_fft."
        gridsize = state_dict.pop("_fft.gridsize", None)
        # Also try old-style keys for backward compatibility
        if gridsize is None:
            gridsize = state_dict.pop("gridsize", None)

        # Create instance with FT-specific params
        instance = cls(
            dtype_float=saved_dtype,
            verbose=verbose,
            device=device,
            strip_H=strip_H,
            max_res=max_res,
            radius_angstrom=radius_angstrom,
        )

        # Set metadata
        instance.pdb = pdb
        instance.initialized = initialized
        instance.altloc_pairs = altloc_pairs

        # Setup spacegroup if it exists
        if spacegroup_str is not None:
            instance.spacegroup = SpaceGroup(spacegroup_str)
            instance.symmetry = Symmetry(instance.spacegroup)
        else:
            instance.spacegroup = None
            instance.symmetry = None

        # Create Cell object from saved tensor data
        from torchref.symmetry import Cell
        if cell_tensor is not None:
            instance.cell = Cell(cell_tensor, dtype=saved_dtype, device=device)

        # If PDB exists, create the parameter wrappers with correct shapes
        if pdb is not None:
            from torchref.model.parameter_wrappers import (
                MixedTensor,
                OccupancyTensor,
                PositiveMixedTensor,
            )

            n_atoms = len(pdb)

            # Create MixedTensors
            xyz_mask = state_dict.get("xyz.refinable_mask")
            b_mask = state_dict.get("b.refinable_mask")
            u_mask = state_dict.get("u.refinable_mask")

            instance.xyz = MixedTensor(
                torch.tensor(pdb[["x", "y", "z"]].values, dtype=saved_dtype),
                refinable_mask=xyz_mask,
                name="xyz",
            )
            instance.b = PositiveMixedTensor(
                torch.tensor(pdb["tempfactor"].values, dtype=saved_dtype),
                refinable_mask=b_mask,
                name="b_factor",
            )
            instance.u = MixedTensor(
                torch.tensor(
                    pdb[["u11", "u22", "u33", "u12", "u13", "u23"]].values,
                    dtype=saved_dtype,
                ),
                refinable_mask=u_mask,
                name="aniso_U",
            )

            # Create OccupancyTensor
            initial_occ = torch.tensor(pdb["occupancy"].values, dtype=saved_dtype)
            sharing_groups, altloc_groups, refinable_mask = (
                instance._create_occupancy_groups(pdb, initial_occ)
            )

            saved_occ_mask = state_dict.get("occupancy.refinable_mask")
            if saved_occ_mask is not None:
                if saved_occ_mask.device != sharing_groups.device:
                    saved_occ_mask = saved_occ_mask.to(sharing_groups.device)
                refinable_mask = saved_occ_mask[sharing_groups]

            instance.occupancy = OccupancyTensor(
                initial_values=initial_occ,
                sharing_groups=sharing_groups,
                altloc_groups=altloc_groups,
                refinable_mask=refinable_mask,
                dtype=saved_dtype,
                device=device,
                name="occupancy",
            )

            # Register aniso_flag buffer
            if "aniso_flag" not in instance._buffers or instance.aniso_flag is None:
                instance.register_buffer(
                    "aniso_flag",
                    torch.tensor(pdb["anisou_flag"].values, dtype=torch.bool),
                )

            # Register mask buffers
            instance.register_buffer(
                "xyz_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )
            instance.register_buffer(
                "b_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )
            instance.register_buffer(
                "u_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )
            instance.register_buffer(
                "occupancy_mask", torch.ones(n_atoms, dtype=torch.bool, device=device)
            )

            # Register vdw_radii if present
            if "vdw_radii" in state_dict and state_dict["vdw_radii"] is not None:
                instance.register_buffer(
                    "vdw_radii", torch.zeros_like(state_dict["vdw_radii"], device=device)
                )

            # Handle _A and _B buffers (scattering parameters)
            # Check both old-style (A, B) and new-style (_A, _B) keys
            a_key = "_A" if "_A" in state_dict else "A" if "A" in state_dict else None
            b_key = "_B" if "_B" in state_dict else "B" if "B" in state_dict else None

            if a_key and state_dict[a_key] is not None:
                instance.register_buffer(
                    "_A", torch.zeros_like(state_dict[a_key], device=device)
                )
            if b_key and state_dict[b_key] is not None:
                instance.register_buffer(
                    "_B", torch.zeros_like(state_dict[b_key], device=device)
                )

        # Setup grid via FFT submodule
        if gridsize is not None and cell_tensor is not None:
            if isinstance(gridsize, torch.Tensor):
                gs_tuple = tuple(int(x) for x in gridsize.tolist())
            else:
                gs_tuple = tuple(int(x) for x in gridsize)

            instance.setup_grid(gridsize=gs_tuple)

        # Filter state_dict and load
        # Remap old-style A/B keys to new _A/_B keys
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not hasattr(v, 'shape') or v.shape[0] > 0:
                # Remap old keys to new keys
                if k == "A":
                    filtered_state_dict["_A"] = v
                elif k == "B":
                    filtered_state_dict["_B"] = v
                else:
                    filtered_state_dict[k] = v

        instance.load_state_dict(filtered_state_dict, strict=False)

        # Reset cache
        from torchref.utils.utils import TensorDict
        object.__setattr__(instance, "_cache", TensorDict())

        if verbose > 0:
            n_atoms = len(instance.pdb) if instance.pdb is not None else 0
            print(f"Created ModelFT from state_dict: {n_atoms} atoms")

        return instance
