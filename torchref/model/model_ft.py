from torchref.model.model import Model
import torch
import numpy as np
from torchref.math_functions.math_torch import find_relevant_voxels, vectorized_add_to_map, vectorized_add_to_map_aniso\
,ifft,extract_structure_factor_from_grid,fft,get_real_grid, find_grid_size,hash_tensors
import gemmi
import torchref.math_functions.get_scattering_factor_torch as gsf
from torchref.symmetrie.map_symmetry import MapSymmetry
import torchref.symmetrie.symmetrie as sym
from typing import Optional, Tuple
from torchref.utils.utils import TensorDict

class ModelFT(Model):
    """
    ModelFT is a purpose-built subclass of model for Fourier Transform (FT) based 
    electron density map calculations and structure factor refinement.
    
    Key differences from base model:
    - Uses ITC92 parametrization for electron density calculations
    - Builds electron density maps in real space
    - Computes structure factors via FFT
    - No residue-level caching - uses direct atom access via get_iso/get_aniso
    """

    def __init__(self, *args, max_res=1.0, radius_angstrom=4.0, gridsize: Optional[Tuple[int, int, int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.map = None  # Electron density map
        self.max_res = max_res 
        self.radius_angstrom = radius_angstrom  # Radius in Angstroms for density calc
        self._cache = TensorDict()  # Cache for structure factors
        if gridsize is not None:
            self.register_buffer("gridsize", torch.tensor(gridsize, dtype=torch.int32))
        else:
            self.register_buffer("gridsize", None)

    def load_pdb(self, filename):
        """
        Load a PDB file and initialize the model with FT-specific setup.
        """
        super().load_pdb(filename)
        self._build_parametrization()
        self.setup_grid()
        return self

    def load_cif(self, filename):
        """
        Load a CIF file and initialize the model with FT-specific setup.
        """
        super().load_cif(filename)
        self._build_parametrization()
        self.setup_grid()
        return self
    
    def setup_gridsize(self, max_res=None):
        if max_res is not None:
            self.max_res = max_res
        if self.verbose > 1: print(f"Defining grid size for ={self.max_res} Å")
        gridsize_initial = find_grid_size(self.cell, self.max_res)
        if hasattr(self, 'spacegroup') and self.spacegroup is not None:
            # Convert tensor to tuple of ints for suggest_grid_size
            gridsize_tuple = tuple(gridsize_initial.tolist())
            gridsize_optimized = sym.Symmetry(self.spacegroup).suggest_grid_size(gridsize_tuple)
            if self.verbose > 1:
                print(f"Optimized grid size from {gridsize_tuple} to {gridsize_optimized} based on symmetry, for cell {self.cell} and maxres {self.max_res}")
            return torch.tensor(gridsize_optimized, dtype=torch.int32)
        return torch.tensor(gridsize_initial, dtype=torch.int32)
    
    def _build_parametrization(self):
        """
        Build ITC92 parametrization for all atoms in the model.
        Stores the parametrization dictionary: {element: (A, B, C)}
        """
        if self.verbose > 1: print("Building ITC92 parametrization...")

        self.parametrization = gsf.get_parameterization_extended(self.pdb)
        if self.verbose > 0: print(f"Parametrization built for {len(self.parametrization)} unique atom types")
        if self.verbose > 1: print('Elements with parametrization:', list(self.parametrization.keys()))

        elements = self.pdb.element.tolist()

        self.register_buffer("A", torch.cat([self.parametrization[element][0] for element in elements],dim=0))
        self.register_buffer("B", torch.cat([self.parametrization[element][1] for element in elements],dim=0))

    def get_iso(self):
        """
        Get isotropic atoms with their ITC92 parameters.
        
        Returns:
        --------
        xyz : torch.Tensor (n_atoms, 3)
            Atomic coordinates
        b : torch.Tensor (n_atoms,)
            B-factors
        occupancy : torch.Tensor (n_atoms,)
            Occupancies
        A : torch.Tensor (n_atoms, 5)
            ITC92 A parameters (amplitudes)
        B : torch.Tensor (n_atoms, 5)
            ITC92 B parameters (widths)
        """
        # Get base isotropic data
        xyz, b, occupancy = super().get_iso()
        
        # Get elements for isotropic atoms
        iso_mask = ~self.aniso_flag
        A = self.A[iso_mask]
        B = self.B[iso_mask]
        
        return xyz, b, occupancy, A, B
    
    def get_aniso(self):
        """
        Get anisotropic atoms with their ITC92 parameters.
        
        Returns:
        --------
        xyz : torch.Tensor (n_atoms, 3)
            Atomic coordinates
        u : torch.Tensor (n_atoms, 6)
            Anisotropic U parameters
        occupancy : torch.Tensor (n_atoms,)
            Occupancies
        A : torch.Tensor (n_atoms, 5)
            ITC92 A parameters (amplitudes)
        B : torch.Tensor (n_atoms, 5)
            ITC92 B parameters (widths)
        """
        # Get base anisotropic data
        xyz, u, occupancy = super().get_aniso()
        
        # Get elements for anisotropic atoms
        aniso_mask = self.aniso_flag

        A = self.A[aniso_mask]  
        B = self.B[aniso_mask]
        
        return xyz, u, occupancy, A, B
    
    def setup_grid(self, max_res=None, gridsize=None):
        """
        Setup real-space grid for electron density calculation.
        
        Parameters:
        -----------
        max_res : float
            Maximum resolution for grid spacing (in Angstroms)
        gridsize : tuple, optional
            Explicit grid size (nx, ny, nz)
        """
        if max_res is not None:
            self.max_res = max_res
        if self.verbose > 1: print(f"Setting up grids with max_res={self.max_res} Å")
        if gridsize is not None:
            self.register_buffer("gridsize", torch.tensor(gridsize, dtype=torch.int32, device=self.device))
        else:
            self.register_buffer("gridsize", self.setup_gridsize(max_res=self.max_res).to(dtype=torch.int32).to(device=self.device))

        self.register_buffer("real_space_grid", get_real_grid(self.cell, gridsize=self.gridsize, device=self.device))
        self.register_buffer("voxel_size", self.real_space_grid[2, 2, 2] - self.real_space_grid[1, 1, 1])

        # Initialize map symmetry operator
        if hasattr(self, 'spacegroup') and self.spacegroup is not None:
            self.map_symmetry = MapSymmetry(
                space_group=self.spacegroup,
                map_shape=self.real_space_grid.shape[:-1],
                cell_params=self.cell, verbose=self.verbose, device=self.device
            )

        if self.verbose > 2: 
            print(f"Grid shape: {self.real_space_grid.shape[:-1]}")
            print(f"Voxel size: {self.voxel_size}")


    def get_radius(self, min_radius_Angstrom: float = 4.0):
        """
        Get the radius (in voxels) used for density calculation around each atom.
        """
        if not hasattr(self, 'real_space_grid') or self.real_space_grid is None:
            self.setup_grid(
            )
        voxel_size = self.real_space_grid[1, 1, 1] - self.real_space_grid[0, 0, 0]
        min_radius = torch.ceil(min_radius_Angstrom / torch.min(voxel_size)).to(torch.int32).item()
        if self.verbose > 1:
            print(f"Calculated radius for density calculation: {min_radius} voxels (voxel size: {voxel_size}), this corresponds to at least {min_radius_Angstrom} Å")
        return min_radius
    
    def build_complete_map(self, radius=None, apply_symmetry=True):
        """
        Build electron density map from all atoms.
        Uses get_iso() and get_aniso() to get atom data.
        
        Parameters:
        -----------
        radius : int or None
            Radius (in voxels) around each atom to compute density.
            If None, uses self.radius.
        apply_symmetry : bool, default True
            If True and space group is not P1, apply symmetry operations to the map
        
        Returns:
        --------
        map : torch.Tensor
            Electron density map (with symmetry applied if requested)
        """
        self.map = self.build_initial_map(apply_symmetry=apply_symmetry)

        if self.verbose > 2: print(f"Density map built. Sum: {self.map.sum():.2f}, Max: {self.map.max():.4f}")
        return self.map
    
    def build_initial_map(self, apply_symmetry=True):
        if not 'real_space_grid' in self._buffers:
            self.setup_grid()

        if self.verbose > 2: print(f"Building density map with radius={self.radius_angstrom} angstrom...")

        # Reset map
        self.map = torch.zeros(self.real_space_grid.shape[:-1], dtype=self.dtype_float, device=self.device)
        
        # Add isotropic atoms
        xyz_iso, b_iso, occ_iso, A_iso, B_iso = self.get_iso()
        if self.verbose > 3:
            assert torch.all(torch.isfinite(A_iso)), "Non-finite values found in A_iso during map building."
            assert torch.all(torch.isfinite(B_iso)), "Non-finite values found in B_iso during map building."
            assert torch.all(torch.isfinite(xyz_iso)), "Non-finite values found in xyz_iso during map building."
            assert torch.all(torch.isfinite(b_iso)), "Non-finite values found in b_iso during map building."
            assert torch.all(torch.isfinite(occ_iso)), "Non-finite values found in occ_iso during map building."

        if len(xyz_iso) > 0:
            if self.verbose > 3:
                print(xyz_iso.shape, b_iso.shape, occ_iso.shape, A_iso.shape, B_iso.shape)
            if self.verbose > 2:
                print(f"  Adding {len(xyz_iso)} isotropic atoms...")
            surrounding_coords, voxel_indices = find_relevant_voxels(
                self.real_space_grid, xyz_iso, radius_angstrom=self.radius_angstrom, inv_frac_matrix=self.inv_fractional_matrix
            )
            self.map = vectorized_add_to_map(
                surrounding_coords, voxel_indices, self.map,
                xyz_iso, b_iso,
                self.inv_fractional_matrix, self.fractional_matrix,
                A_iso, B_iso, occ_iso
            )
        if self.verbose > 3:
            assert torch.all(torch.isfinite(self.map)), "Non-finite values found in map after adding isotropic atoms."
        # Add anisotropic atoms
        xyz_aniso, u_aniso, occ_aniso, A_aniso, B_aniso = self.get_aniso()
        
        if len(xyz_aniso) > 0:
            if self.verbose > 2: print(f"  Adding {len(xyz_aniso)} anisotropic atoms...")
            surrounding_coords, voxel_indices = find_relevant_voxels(
                self.real_space_grid, xyz_aniso, radius_angstrom=self.radius_angstrom, inv_frac_matrix=self.inv_fractional_matrix
            )
            self.map = vectorized_add_to_map_aniso(
                surrounding_coords, voxel_indices, self.map,
                xyz_aniso, u_aniso,
                self.inv_fractional_matrix, self.fractional_matrix,
                A_aniso, B_aniso, occ_aniso
            )
        if self.verbose > 3:
            assert torch.all(torch.isfinite(self.map)), "Non-finite values found in map after adding anisotropic atoms."
        # Apply symmetry if requested
        if apply_symmetry and self.map_symmetry is not None:
            if self.verbose > 2: print(f"  Applying {self.map_symmetry.n_ops} symmetry operations...")
            self.map = self.map_symmetry(self.map)
            if self.verbose > 3:
                assert torch.all(torch.isfinite(self.map)), "Non-finite values found in map after applying symmetry."
        return self.map
    
    def save_map(self, filename):
        """
        Save the electron density map to a CCP4 format file.
        
        Parameters:
        -----------
        filename : str
            Output filename for the map
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
        map_ccp.grid = gemmi.FloatGrid(np_map, gemmi.UnitCell(*cell), gemmi.SpaceGroup('P1'))
        map_ccp.setup(0.0)
        map_ccp.update_ccp4_header()
        map_ccp.write_ccp4_map(filename)
        if self.verbose > 0: print(f"Map saved successfully")
    
    def get_map_statistics(self):
        """Get statistics about the current density map."""
        if self.map is None:
            return None
        
        stats = {
            'shape': self.map.shape,
            'sum': float(self.map.sum()),
            'mean': float(self.map.mean()),
            'std': float(self.map.std()),
            'min': float(self.map.min()),
            'max': float(self.map.max()),
            'n_positive': int((self.map > 0).sum()),
            'n_negative': int((self.map < 0).sum()),
        }
        return stats
    
    def rebuild_map(self, radius=None):
        """
        Rebuild the density map from scratch.
        Convenience method that clears and rebuilds everything.
        
        Parameters:
        -----------
        radius : int or None
            Radius (in voxels) around each atom.
            If None, uses self.radius. If specified, overrides self.radius.
        """
        if self.verbose > 1: print("Rebuilding density map from scratch...")
        return self.build_density_map(radius=radius)
    
    def cuda(self, device=None):
        """Move model and FT-specific data to GPU."""
        super().cuda(device)
        if self.map is not None:
            self.map = self.map.cuda(device)
        return self
    
    def cpu(self):
        """Move model and FT-specific data to CPU."""
        super().cpu()
        if self.map is not None:
            self.map = self.map.cpu()
        return self
    
    def update_pdb(self):
        """
        Update PDB with current atomic parameters.
        """
        super().update_pdb()
    
    def reset_cache(self):
        self._cache = TensorDict()

    def get_structure_factor(self, hkl: torch.Tensor, recalc=False) -> torch.Tensor:
        """
        Get structure factors for given hkl reflections.
        Uses caching to avoid recomputation if parameters haven't changed.
        
        Parameters:
        -----------
        hkl : torch.Tensor (n_reflections, 3)
            Miller indices
        recalc : bool
            If True, forces recalculation even if cached
            
        Returns:
        --------
        sf : torch.Tensor (n_reflections,)
            Complex structure factors
        """
        # Compute current parameter hash
        params = (*self.parameters(),hkl)
        current_param_hash = hash_tensors(params)
    
        key = current_param_hash
        if not recalc and key in self._cache:
            if self.verbose > 2:
                print("Using cached structure factors")
            return self._cache[key]
        
        # Build map and compute structure factors
        self.build_complete_map()
        self.reciprocal_space_grid = ifft(self.map)
        sf = extract_structure_factor_from_grid(self.reciprocal_space_grid, hkl)
        
        self._cache[key] = sf.detach()
        
        return sf
    
    def fft(self):
        """Perform FFT on the current reciprocal grid."""
        self.density = fft(self.reciprocal_space_grid)
        return self.density
    
    def forward(self, hkl, recalc=False) -> torch.Tensor:
        """
        Forward pass to compute structure factors for given hkl.
        
        Parameters:
        -----------
        hkl : torch.Tensor (n_reflections, 3)
            Miller indices
            
        Returns:
        --------
        F_calc : torch.Tensor (n_reflections,)
            Calculated complex structure factors
        """
        f = self.get_structure_factor(hkl,recalc=recalc)
        if self.verbose > 2:
            assert torch.all(torch.isfinite(f)), "Non-finite values found while calculating fcalc."

        return f
    
    def copy(self):
        """
        Create a deep copy of the ModelFT with all parameters, buffers, and FT-specific data.
        
        This method creates a complete independent copy including:
        - All Model base class data (via parent copy logic)
        - FT-specific buffers (gridsize, real_space_grid, voxel_size, A, B)
        - ITC92 parametrization dictionary
        - Map symmetry operator
        - Scalar attributes (max_res, radius_angstrom, map)
        - Cache (reset to empty)
        
        Returns:
            ModelFT: A new ModelFT instance with copied data
            
        Example:
            >>> model = ModelFT().load_pdb('structure.pdb')
            >>> model_copy = model.copy()
            >>> # model_copy is independent, changes won't affect model
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
            gridsize=None  # Will be copied from buffers
        )
        
        # Deep copy the PDB DataFrame
        model_copy.pdb = self.pdb.copy(deep=True)
        
        # Copy scalar attributes from Model
        model_copy.spacegroup = self.spacegroup
        model_copy.spacegroup_gemmi = self.spacegroup_gemmi
        model_copy.initialized = True
        
        # Copy spacegroup function
        model_copy.spacegroup_function = sym.Symmetry(self.spacegroup)
        
        # Copy all registered buffers using PyTorch's _buffers dict
        for buffer_name, buffer_value in self._buffers.items():
            if buffer_value is not None:
                model_copy.register_buffer(buffer_name, buffer_value.clone())
        
        # Copy all modules (parameter wrappers) using their .copy() methods
        for module_name, module in self._modules.items():
            if module is not None and hasattr(module, 'copy'):
                setattr(model_copy, module_name, module.copy())
        
        # Copy alternative conformation pairs
        if hasattr(self, 'altloc_pairs') and self.altloc_pairs:
            model_copy.altloc_pairs = [
                tuple(tensor.clone() for tensor in group) 
                for group in self.altloc_pairs
            ]
        else:
            model_copy.altloc_pairs = []
        
        # Copy FT-specific attributes
        if hasattr(self, 'parametrization') and self.parametrization is not None:
            # Deep copy the parametrization dictionary
            import copy as copy_module
            model_copy.parametrization = copy_module.deepcopy(self.parametrization)
        
        # Copy map if it exists
        if self.map is not None:
            model_copy.map = self.map.clone()
        
        # Copy map_symmetry if it exists
        if hasattr(self, 'map_symmetry') and self.map_symmetry is not None:
            from torchref.symmetrie.map_symmetry import MapSymmetry
            model_copy.map_symmetry = MapSymmetry(
                space_group=self.spacegroup,
                map_shape=model_copy.real_space_grid.shape[:-1],
                cell_params=model_copy.cell,
                verbose=self.verbose,
                device=self.device
            )
        
        # Reset cache (don't copy cached structure factors)
        # Use object.__setattr__ since _cache is special in nn.Module
        from torchref.utils.utils import TensorDict
        object.__setattr__(model_copy, '_cache', TensorDict())
        
        if self.verbose > 0:
            print(f"✓ ModelFT copied successfully ({len(model_copy.pdb)} atoms)")
        
        return model_copy

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Returns a dictionary containing the complete state of the ModelFT.
        
        Extends parent Model.state_dict() with FT-specific parameters:
        - max_res, radius_angstrom
        - gridsize
        - parametrization (ITC92 coefficients)
        
        Args:
            destination: Optional dict to populate
            prefix: Prefix for parameter names
            keep_vars: Whether to keep variables in computational graph
            
        Returns:
            dict: Complete state dictionary
        """
        # Get parent Model state_dict
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Add ModelFT-specific state
        state[prefix + 'max_res'] = self.max_res
        state[prefix + 'radius_angstrom'] = self.radius_angstrom
        
        # Note: parametrization dict contains tensors that are already in buffers A and B
        # gridsize is a registered buffer so it's already included
        # _cache is not saved as it should be rebuilt
        
        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Loads the ModelFT state from a dictionary.
        
        Args:
            state_dict: Dictionary containing model state
            strict: Whether to strictly enforce that keys match
        """
        # Extract ModelFT-specific state
        self.max_res = state_dict.pop('max_res', 1.0)
        self.radius_angstrom = state_dict.pop('radius_angstrom', 4.0)
        
        # If we are loading into an empty/uninitialized model, we need to register buffers
        # so that load_state_dict can load them.
        # We check if 'pdb' is in state_dict to determine if we need to setup buffers.
        # Note: keys might have prefixes if this is called from a parent module, 
        # but usually load_state_dict is called with a dict where keys match this module's parameters.
        # If called directly, keys are 'pdb', 'A', 'B', etc.
        
        pdb_key = 'pdb'
        if pdb_key in state_dict and state_dict[pdb_key] is not None:
            pdb_df = state_dict[pdb_key]
            n_atoms = len(pdb_df)
            
            # Register A and B buffers if missing
            if not hasattr(self, 'A'):
                # A has shape (n_atoms, 5)
                self.register_buffer('A', torch.zeros(n_atoms, 5, dtype=torch.float32, device=self.device))
            if not hasattr(self, 'B'):
                # B has shape (n_atoms, 5)
                self.register_buffer('B', torch.zeros(n_atoms, 5, dtype=torch.float32, device=self.device))
                
        # Register gridsize if missing
        if not hasattr(self, 'gridsize') or self.gridsize is None:
            # gridsize is (3,) int32
            self.register_buffer('gridsize', torch.zeros(3, dtype=torch.int32, device=self.device))

        # Ensure FT-specific buffers are registered before loading
        # This is necessary when loading into an uninitialized model
        
        # 1. Get gridsize and cell from state_dict to determine shapes
        gridsize = state_dict.get('gridsize')
        cell = state_dict.get('cell')
        spacegroup = state_dict.get('spacegroup')
        
        if gridsize is not None:
            # Ensure gridsize is on the correct device/type if we use it
            # But here we just need the values
            if isinstance(gridsize, torch.Tensor):
                gs_list = gridsize.tolist()
            else:
                gs_list = gridsize
                
            nx, ny, nz = gs_list
            
            if not hasattr(self, 'real_space_grid'):
                # Shape is (nx, ny, nz, 3)
                self.register_buffer('real_space_grid', torch.zeros(nx, ny, nz, 3, dtype=self.dtype_float, device=self.device))
            
            if not hasattr(self, 'voxel_size'):
                self.register_buffer('voxel_size', torch.tensor(0.0, dtype=self.dtype_float, device=self.device))
                
            # 2. Instantiate MapSymmetry if needed
            # Check if any map_symmetry keys are in state_dict
            has_map_sym = any(k.startswith('map_symmetry.') for k in state_dict.keys())
            
            if has_map_sym and (not hasattr(self, 'map_symmetry') or self.map_symmetry is None):
                if spacegroup is not None and cell is not None:
                    from torchref.symmetrie.map_symmetry import MapSymmetry
                    self.map_symmetry = MapSymmetry(
                        space_group=spacegroup,
                        map_shape=(nx, ny, nz),
                        cell_params=cell,
                        verbose=self.verbose,
                        device=self.device
                    )
        
        # Load parent Model state_dict
        result = super().load_state_dict(state_dict, strict=strict)
        
        # Rebuild parametrization dict from A and B buffers if PDB exists
        if hasattr(self, 'pdb') and self.pdb is not None and hasattr(self, 'A') and hasattr(self, 'B'):
            self.parametrization = {}
            elements = self.pdb.element.unique().tolist()
            idx = 0
            for element in elements:
                n_atoms = (self.pdb.element == element).sum()
                self.parametrization[element] = (self.A[idx:idx+5], self.B[idx:idx+5])
                idx += 5
        
        # Reset cache
        from torchref.utils.utils import TensorDict
        object.__setattr__(self, '_cache', TensorDict())
        
        return result



