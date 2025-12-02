
'''
A base model class for atomic structure models using PyTorch.
'''

import torch
import torch.nn as nn
from typing import Optional, Union
from torchref.io import file_writers
from torchref.utils.utils import sanitize_pdb_dataframe
import torchref.symmetrie.symmetrie as sym
import torchref.math_functions.math_numpy as mnp
from torchref.math_functions import math_torch
from torchref.model.parameter_wrappers import MixedTensor, OccupancyTensor, PositiveMixedTensor
from torchref.io import cif_readers, legacy_format_readers
from torchref.utils.debug_utils import DebugMixin
import gemmi

class Model(DebugMixin, nn.Module):
    """
    Base model class for atomic structure models using PyTorch.

    This class provides the foundation for managing atomic structure data
    including coordinates, B-factors, anisotropic displacement parameters,
    and occupancies. It supports both empty initialization for state_dict
    loading and file-based initialization from PDB/CIF files.

    Parameters
    ----------
    dtype_float : torch.dtype, optional
        Data type for floating point tensors. Default is torch.float32.
    verbose : int, optional
        Verbosity level for logging. Default is 1.
    device : torch.device, optional
        Computation device. Default is torch.device('cpu').
    strip_H : bool, optional
        Whether to strip hydrogen atoms when loading. Default is True.

    Attributes
    ----------
    xyz : MixedTensor
        Atomic coordinates tensor with shape (n_atoms, 3).
    b : PositiveMixedTensor
        Isotropic B-factors tensor with shape (n_atoms,).
    u : MixedTensor
        Anisotropic displacement parameters with shape (n_atoms, 6).
    occupancy : OccupancyTensor
        Atomic occupancies with values in [0, 1].
    pdb : pandas.DataFrame
        DataFrame containing atomic model data.
    cell : torch.Tensor
        Unit cell parameters [a, b, c, alpha, beta, gamma].
    spacegroup : str
        Space group symbol in Hermann-Mauguin notation.
    initialized : bool
        Whether the model has been initialized with data.

    Examples
    --------
    Empty initialization for state_dict loading:

    >>> model = Model()
    >>> model.load_state_dict(torch.load('model.pt'))

    File-based initialization:

    >>> model = Model()
    >>> model.load_pdb('structure.pdb')
    """
    
    def __init__(self, dtype_float=torch.float32, verbose=1, device=torch.device('cpu'), strip_H: bool = True):
        """
        Initialize an empty Model shell.

        Creates a model shell ready for file loading via load_pdb()/load_cif()
        or state restoration via load_state_dict().

        Parameters
        ----------
        dtype_float : torch.dtype, optional
            Data type for floating point tensors. Default is torch.float32.
        verbose : int, optional
            Verbosity level for logging. Default is 1.
        device : torch.device, optional
            Computation device. Default is torch.device('cpu').
        strip_H : bool, optional
            Whether to strip hydrogen atoms when loading. Default is True.
        """
        super().__init__()
        # Configuration
        self.dtype_float = dtype_float
        self.verbose = verbose
        self.device = device
        self.strip_H = strip_H
        
        # State tracking
        self.initialized = False
        self.altloc_pairs = []
        
        # These will be set during load() or load_state_dict()
        self.pdb = None
        self.spacegroup = None
        self.spacegroup_gemmi = None
        self.spacegroup_function = None
        
        # Submodules (created during load or load_state_dict)
        self.xyz = None
        self.b = None
        self.u = None
        self.occupancy = None
    
    def __bool__(self):
        """Return the initialization status when used in boolean context."""
        return self.initialized
    
    def load(self, reader):
        self.pdb, cell, self.spacegroup = reader()
        
        self.pdb = self.pdb.loc[self.pdb['element'] != 'H'].reset_index(drop=True) if self.strip_H else self.pdb
        self.pdb.dropna(subset=['x', 'y', 'z', 'tempfactor', 'occupancy'], inplace=True)
        self.pdb['index'] = self.pdb.index.to_numpy(dtype=int)
        
        self.register_buffer('cell',torch.tensor(cell,requires_grad=False,dtype=self.dtype_float,device=self.device))
        self.spacegroup_gemmi = gemmi.SpaceGroup(self.spacegroup.replace('  ',' '))
        self.spacegroup = self.spacegroup_gemmi.hm
        self.spacegroup_function = sym.Symmetry(self.spacegroup)


        # Register buffers for various matrices
        self.register_buffer('inv_fractional_matrix',torch.tensor(mnp.get_inv_fractional_matrix(self.cell),dtype=self.dtype_float,requires_grad=False))
        self.register_buffer('fractional_matrix',torch.tensor(mnp.get_fractional_matrix(self.cell),dtype=self.dtype_float,requires_grad=False))
        self.register_buffer('aniso_flag',torch.tensor(self.pdb['anisou_flag'].values,dtype=torch.bool))
        self.register_buffer('recB', math_torch.reciprocal_basis_matrix(self.cell).to(dtype=self.dtype_float).to(self.device))
        
        # Create MixedTensors for model parameters
        self.xyz = MixedTensor(torch.tensor(self.pdb[['x', 'y', 'z']].values,dtype=self.dtype_float), name='xyz')
        self.b = PositiveMixedTensor(torch.tensor(self.pdb['tempfactor'].values,dtype=self.dtype_float), name='b_factor')
        self.u = MixedTensor(torch.tensor(self.pdb[['u11', 'u22', 'u33', 'u12', 'u13', 'u23']].values,dtype=self.dtype_float), name='aniso_U')
        
        # Create OccupancyTensor with residue-level sharing and altloc support
        initial_occ = torch.tensor(self.pdb['occupancy'].values, dtype=self.dtype_float)
        sharing_groups, altloc_groups, refinable_mask = self._create_occupancy_groups(self.pdb, initial_occ)
        self.occupancy = OccupancyTensor(
            initial_values=initial_occ,
            sharing_groups=sharing_groups,
            altloc_groups=altloc_groups,
            refinable_mask=refinable_mask,
            dtype=self.dtype_float,
            device=self.device,
            name='occupancy'
        )

        self.set_default_masks()
        self.register_alternative_conformations()
        self.initialized = True
        return self

    def load_pdb(self, file):
        """
        Load atomic model from PDB file.

        Parameters
        ----------
        file : str
            Path to PDB file.

        Returns
        -------
        Model
            Self, for method chaining.
        """
        reader = legacy_format_readers.PDB(verbose=self.verbose).read(file)   
        return self.load(reader)
    
    def load_cif(self, file):
        """
        Load atomic model from mmCIF file.

        Parameters
        ----------
        file : str
            Path to CIF/mmCIF file.

        Returns
        -------
        Model
            Self, for method chaining.
        """
        if self.verbose > 0:
            print(f"Loading CIF file: {file}")
        
        # Read CIF file
        cif_reader = cif_readers.ModelCIFReader(file)

        return self.load(cif_reader)
    
    def _create_occupancy_groups(self, pdb_df, initial_occ):
        """
        Create sharing groups and altloc groups for occupancy.

        This method identifies atoms that should share occupancy values and
        groups alternative conformations for proper constraint handling.

        Logic:
        1. First identify alternative conformations (multiple altlocs per residue)
        2. For altloc groups: ALL atoms in each conformation share one collapsed index
        3. For non-altloc residues: group by similar occupancy (within 0.01 tolerance)
        4. Only refine occupancies that differ from 1.0

        Parameters
        ----------
        pdb_df : pandas.DataFrame
            PDB DataFrame with atom information.
        initial_occ : torch.Tensor
            Tensor of initial occupancy values with shape (n_atoms,).

        Returns
        -------
        sharing_groups_tensor : torch.Tensor
            Tensor of shape (n_atoms,) where each value is the collapsed index
            for that atom.
        altloc_groups : list of tuple
            List of tuples of atom index lists for alternative conformations.
        refinable_mask : torch.Tensor
            Boolean tensor indicating which atoms should be refined.
        """
        n_atoms = len(initial_occ)
        altloc_groups = []
        refinable_mask = torch.zeros(n_atoms, dtype=torch.bool)
        
        # Initialize sharing groups tensor - each atom maps to its own index initially
        sharing_groups_tensor = torch.arange(n_atoms, dtype=torch.long)
        collapsed_idx = 0
        
        # First pass: identify and process alternative conformations
        # For altloc atoms: ALL atoms in a conformation MUST share the same collapsed index
        # regardless of their individual occupancy values
        pdb_with_altlocs = pdb_df[pdb_df['altloc'] != '']
        altloc_residues = set()  # Track which residues have altlocs
        
        if len(pdb_with_altlocs) > 0:
            grouped_by_residue = pdb_with_altlocs.groupby(['resname', 'resseq', 'chainid'])
            
            for (resname, resseq, chainid), group in grouped_by_residue:
                unique_altlocs = sorted(group['altloc'].unique())
                
                # Only process if there are multiple conformations
                if len(unique_altlocs) > 1:
                    altloc_residues.add((resname, resseq, chainid))
                    conformation_atom_lists = []
                    
                    for altloc in unique_altlocs:
                        # Get all atoms for this specific altloc
                        altloc_atoms = group[group['altloc'] == altloc]
                        indices = altloc_atoms['index'].tolist()
                        
                        # Assign ALL atoms in this conformation to the same collapsed index
                        sharing_groups_tensor[indices] = collapsed_idx
                        
                        # Check if any atom in this conformation has occupancy != 1.0
                        for idx in indices:
                            if abs(initial_occ[idx].item() - 1.0) > 0.01:
                                refinable_mask[idx] = True
                        
                        conformation_atom_lists.append(indices)
                        collapsed_idx += 1
                    
                    # Add to altloc_groups
                    altloc_groups.append(tuple(conformation_atom_lists))
        
        # Second pass: process non-altloc residues
        # Group by residue, and create sharing groups based on occupancy similarity
        grouped = pdb_df.groupby(['resname', 'resseq', 'chainid', 'altloc'])
        
        for (resname, resseq, chainid, altloc), group in grouped:
            # Skip if this residue has alternative conformations (already processed)
            if (resname, resseq, chainid) in altloc_residues:
                continue
            
            indices = group['index'].tolist()
            
            if len(indices) == 0:
                continue
            
            # Get occupancies for this residue
            residue_occs = initial_occ[indices]
            
            # Check if all occupancies are within tolerance
            occ_min = residue_occs.min().item()
            occ_max = residue_occs.max().item()
            occ_mean = residue_occs.mean().item()
            
            if (occ_max - occ_min) <= 0.01:
                # All atoms in residue have similar occupancy - create sharing group
                sharing_groups_tensor[indices] = collapsed_idx
                collapsed_idx += 1
                
                # Only refine if mean occupancy differs from 1.0
                if abs(occ_mean - 1.0) > 0.01:
                    for idx in indices:
                        refinable_mask[idx] = True
            else:
                # Occupancies differ within residue - each atom independent
                # Refine those that differ from 1.0
                for idx in indices:
                    if abs(initial_occ[idx].item() - 1.0) > 0.01:
                        refinable_mask[idx] = True
        
        # Compact the indices - make them contiguous from 0 to n_collapsed-1
        unique_indices = torch.unique(sharing_groups_tensor, sorted=True)
        index_map = torch.zeros(n_atoms, dtype=torch.long)
        for new_idx, old_idx in enumerate(unique_indices):
            mask = (sharing_groups_tensor == old_idx)
            sharing_groups_tensor[mask] = new_idx
        
        n_collapsed = len(unique_indices)
        
        if self.verbose > 1:
            n_groups = n_collapsed
            n_independent = n_atoms - n_collapsed  # Atoms not sharing with others
            n_refinable = refinable_mask.sum().item()
            n_altloc_groups = len(altloc_groups)
            
            print(f"\nOccupancy Setup:")
            print(f"  Total atoms: {n_atoms}")
            print(f"  Collapsed indices: {n_collapsed}")
            print(f"  Alternative conformation groups: {n_altloc_groups}")
            print(f"  Refinable atoms: {n_refinable}")
            print(f"  Compression ratio: {n_atoms / n_collapsed:.2f}x")
        
        return sharing_groups_tensor, altloc_groups, refinable_mask

    def update_pdb(self):
        self.pdb.loc[:, ['x', 'y', 'z']] = self.xyz().cpu().detach().numpy()
        self.pdb.loc[:, ['u11', 'u22', 'u33', 'u12', 'u13', 'u23']] = self.u().cpu().detach().numpy()
        self.pdb.loc[:, 'tempfactor'] = self.b().cpu().detach().numpy()
        self.pdb.loc[:, 'occupancy'] = self.occupancy().cpu().detach().numpy()
        return self.pdb
    
    def get_vdw_radii(self):
        """
        Get van der Waals radii for all atoms based on their elements.

        Caches the result in self.vdw_radii for future calls.

        Returns
        -------
        torch.Tensor
            Van der Waals radii for each atom with shape (n_atoms,).
        """
        import os
        import pandas as pd
        if hasattr(self, 'vdw_radii'):
            return self.vdw_radii
        elements = self.pdb.loc[:, 'element']
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'caching/files/atomic_vdw_radii.csv')
        vdw_df = pd.read_csv(path, comment='#')   
        vdw_df['element'] = vdw_df['element'].str.strip().str.capitalize()
        elements = elements.str.strip().str.capitalize()
        elements_not_in = elements[~elements.isin(vdw_df['element'])]
        if len(elements_not_in) > 0:
            # Add missing elements with default vdW radius 1.9 Å
            missing = sorted(set(e.strip().capitalize() for e in elements_not_in))
            if missing:
                add_df = pd.DataFrame({'element': missing,
                                       'vdW_Radius_Angstrom': [1.9] * len(missing)})
                vdw_df = pd.concat([vdw_df, add_df], ignore_index=True)


        vdw_radii = vdw_df.set_index('element').loc[elements]['vdW_Radius_Angstrom'].values
        self.register_buffer('vdw_radii', torch.tensor(vdw_radii, dtype=self.dtype_float, device=self.device))
        assert len(self.vdw_radii) == len(self.pdb), f"vdW radii length mismatch with number of atoms {len(self.vdw_radii)} != {len(self.pdb)}"
        return self.vdw_radii

    def cuda(self, device: Optional[Union[int, torch.device]] = None):
        super().cuda(device)
        if self.altloc_pairs:
            self.altloc_pairs = [tuple(tensor.cuda(device) for tensor in group) for group in self.altloc_pairs]
        self.device = torch.device('cuda')
        print(f"Model moved to device: {self.device}")
        return self
    
    def cpu(self):
        super().cpu()
        if self.altloc_pairs:
            self.altloc_pairs = [tuple(tensor.cpu() for tensor in group) for group in self.altloc_pairs]
        self.device = torch.device('cpu')
        print(f"Model moved to device: {self.device}")
        return self
    
    def copy(self):
        """
        Create a deep copy of the Model.

        Creates a complete independent copy including all registered buffers,
        module parameters, PDB DataFrame, and spacegroup information.

        Returns
        -------
        Model
            A new Model instance with copied data.

        Examples
        --------
        >>> model = Model().load_pdb('structure.pdb')
        >>> model_copy = model.copy()
        >>> # model_copy is independent, changes won't affect model
        """
        if not self.initialized:
            raise RuntimeError("Cannot copy an uninitialized Model. Load data first.")
        
        # Create new model instance with same configuration
        model_copy = Model(
            dtype_float=self.dtype_float,
            verbose=self.verbose,
            device=self.device,
            strip_H=self.strip_H
        )
        
        # Deep copy the PDB DataFrame
        model_copy.pdb = self.pdb.copy(deep=True)
        
        # Copy scalar attributes
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
        
        if self.verbose > 0:
            print(f"✓ Model copied successfully ({len(model_copy.pdb)} atoms)")
        
        return model_copy
    
    def write_pdb(self, filename):
        self.update_pdb()
        self.pdb = sanitize_pdb_dataframe(self.pdb)
        self.pdb.attrs['spacegroup'] = self.spacegroup_gemmi.hm
        file_writers.write_file(self.pdb, filename)

    def get_iso(self):
        xyz = self.xyz()[~self.aniso_flag]
        b = self.b()[~self.aniso_flag]
        occupancy = self.occupancy()[~self.aniso_flag]
        return xyz, b, occupancy

    def set_default_masks(self):
        self.register_buffer("xyz_mask", torch.ones(len(self.pdb), dtype=torch.bool, device=self.device))
        self.xyz.update_refinable_mask(self.xyz_mask)
        self.register_buffer("b_mask", ~self.b().detach().isnan())
        self.b.update_refinable_mask(self.b_mask)
        self.register_buffer("u_mask", ~self.u().detach().isnan().any(dim=1))
        self.u.update_refinable_mask(self.u_mask)
        self.register_buffer("occupancy_mask", self.occupancy() < 0.999)
        self.occupancy.update_refinable_mask(self.occupancy_mask)

    def freeze(self, target: str):
        if target == 'xyz':
            self.xyz.fix_all()
        elif target == 'b':
            self.b.fix_all()
        elif target == 'u':
            self.u.fix_all()
        elif target == 'occupancy':
            self.occupancy.freeze_all()  # OccupancyTensor uses freeze_all() not fix_all()
    
    def freeze_all(self):
        self.freeze('xyz')
        self.freeze('b')
        self.freeze('u')
        self.freeze('occupancy')

    def unfreeze_all(self):
        self.unfreeze('xyz')
        self.unfreeze('b')
        self.unfreeze('u')
        self.unfreeze('occupancy')

    def unfreeze(self, target: str):
        if target == 'xyz':
            self.xyz.update_refinable_mask(self.xyz_mask)
        elif target == 'b':
            self.b.update_refinable_mask(self.b_mask)
        elif target == 'u':
            self.u.update_refinable_mask(self.u_mask)
        elif target == 'occupancy':
            # OccupancyTensor uses unfreeze_all() or update_refinable_mask() with full atom space mask
            self.occupancy.update_refinable_mask(self.occupancy_mask, in_compressed_space=False)

    def update_mask_from_selection(self, selection_string: str, target: str, 
                                   mode: str = 'set', freeze: bool = True):
        """
        Update the refinable mask for a parameter using Phenix-style selection syntax.

        This method updates the internal mask buffer (xyz_mask, b_mask, u_mask, or
        occupancy_mask) based on the selection. The updated mask is NOT automatically
        applied to the parameter tensors - use apply_mask_to_parameter() to apply it.

        Parameters
        ----------
        selection_string : str
            Phenix-style selection string (see parse_phenix_selection docs).
        target : str
            Parameter to update: 'xyz', 'b', 'u', or 'occupancy'.
        mode : str, optional
            How to combine with current mask:
            - 'set': Replace mask with selection (default)
            - 'add': Add selection to current mask
            - 'remove': Remove selection from current mask
        freeze : bool, optional
            If True (default), selected atoms will be frozen (mask=False).
            If False, selected atoms will be unfrozen (mask=True).

        Raises
        ------
        ValueError
            If target is not recognized or selection syntax is invalid.

        Examples
        --------
        >>> # Freeze chain A coordinates
        >>> model.update_mask_from_selection("chain A", "xyz", mode='set', freeze=True)
        >>> model.apply_mask_to_parameter("xyz")

        >>> # Unfreeze backbone atoms
        >>> model.update_mask_from_selection("name CA or name C or name N", "xyz", freeze=False)
        >>> model.apply_mask_to_parameter("xyz")
        """
        from torchref.utils.utils import create_selection_mask
        
        # Map target to the corresponding mask buffer
        mask_map = {
            'xyz': 'xyz_mask',
            'b': 'b_mask',
            'u': 'u_mask',
            'occupancy': 'occupancy_mask'
        }
        
        if target not in mask_map:
            raise ValueError(f"Invalid target: '{target}'. Must be one of: {list(mask_map.keys())}")
        
        mask_name = mask_map[target]
        current_mask = getattr(self, mask_name)
        
        # Get selection mask
        selection_mask = create_selection_mask(
            selection_string, 
            self.pdb, 
            current_mask=current_mask if mode != 'set' else None,
            mode=mode
        )
        
        # Invert selection if we're freezing (refinable_mask=False means frozen)
        if freeze:
            updated_mask = current_mask & ~selection_mask
        else:
            updated_mask = selection_mask
        
        # Update the buffer
        setattr(self, mask_name, updated_mask)
        
        if self.verbose > 0:
            n_selected = selection_mask.sum().item()
            n_refinable = updated_mask.sum().item()
            action = "frozen" if freeze else "unfrozen"
            print(f"Selection '{selection_string}' ({n_selected} atoms) {action} for {target}")
            print(f"  Total refinable atoms for {target}: {n_refinable}/{len(self.pdb)}")

    def apply_mask_to_parameter(self, target: str):
        """
        Apply the current mask buffer to the parameter tensor.

        Takes the current state of the mask buffer (xyz_mask, b_mask, etc.)
        and applies it to the corresponding parameter tensor's refinable mask.

        Parameters
        ----------
        target : str
            Parameter to update: 'xyz', 'b', 'u', or 'occupancy'.

        Raises
        ------
        ValueError
            If target is not recognized.

        Examples
        --------
        >>> model.update_mask_from_selection("chain A", "xyz", freeze=True)
        >>> model.apply_mask_to_parameter("xyz")
        """
        if target == 'xyz':
            self.xyz.update_refinable_mask(self.xyz_mask)
        elif target == 'b':
            self.b.update_refinable_mask(self.b_mask)
        elif target == 'u':
            self.u.update_refinable_mask(self.u_mask)
        elif target == 'occupancy':
            self.occupancy.update_refinable_mask(self.occupancy_mask, in_compressed_space=False)
        else:
            raise ValueError(f"Invalid target: '{target}'. Must be 'xyz', 'b', 'u', or 'occupancy'")
        
        if self.verbose > 0:
            n_refinable = getattr(self, f"{target}_mask").sum().item()
            print(f"  Applied mask to {target}: {n_refinable} atoms refinable")

    def freeze_selection(self, selection_string: str, targets: Union[str, list] = 'all'):
        """
        Freeze atoms matching a Phenix-style selection for specified parameters.

        Convenience method that combines update_mask_from_selection() and
        apply_mask_to_parameter() into a single call.

        Parameters
        ----------
        selection_string : str
            Phenix-style selection string.
        targets : str or list of str, optional
            Parameter(s) to freeze. Can be:
            - 'all': Freeze xyz, b, u, and occupancy (default)
            - str: Single parameter ('xyz', 'b', 'u', 'occupancy')
            - list: List of parameters, e.g., ['xyz', 'b']

        Examples
        --------
        >>> # Freeze all parameters for chain A
        >>> model.freeze_selection("chain A", targets='all')

        >>> # Freeze only coordinates for residues 10-20
        >>> model.freeze_selection("resseq 10:20", targets='xyz')
        """
        # Handle 'all' target
        if targets == 'all':
            targets = ['xyz', 'b', 'u', 'occupancy']
        elif isinstance(targets, str):
            targets = [targets]
        
        # Update and apply masks for each target
        for target in targets:
            self.update_mask_from_selection(selection_string, target, mode='set', freeze=True)
            self.apply_mask_to_parameter(target)

    def unfreeze_selection(self, selection_string: str, targets: Union[str, list] = 'all'):
        """
        Unfreeze atoms matching a Phenix-style selection for specified parameters.

        Convenience method that combines update_mask_from_selection() and
        apply_mask_to_parameter() into a single call.

        Parameters
        ----------
        selection_string : str
            Phenix-style selection string.
        targets : str or list of str, optional
            Parameter(s) to unfreeze. Can be:
            - 'all': Unfreeze xyz, b, u, and occupancy (default)
            - str: Single parameter ('xyz', 'b', 'u', 'occupancy')
            - list: List of parameters, e.g., ['xyz', 'b']

        Examples
        --------
        >>> # Unfreeze all parameters for chain A
        >>> model.unfreeze_selection("chain A", targets='all')

        >>> # Unfreeze only coordinates for backbone atoms
        >>> model.unfreeze_selection("name CA or name C or name N", targets='xyz')
        """
        # Handle 'all' target
        if targets == 'all':
            targets = ['xyz', 'b', 'u', 'occupancy']
        elif isinstance(targets, str):
            targets = [targets]
        
        # Update and apply masks for each target
        for target in targets:
            self.update_mask_from_selection(selection_string, target, mode='set', freeze=False)
            self.apply_mask_to_parameter(target)

    def get_aniso(self):
        xyz = self.xyz()[self.aniso_flag]
        u = self.u()[self.aniso_flag]
        occupancy = self.occupancy()[self.aniso_flag]
        return xyz, u, occupancy
    
    def parameters(self, recurse: bool = True):
        return (p for p in super().parameters(recurse) if p.numel() > 0)
    
    def named_mixed_tensors(self):
        """
        Iterate over all MixedTensor attributes with their names.
        
        Yields:
            Tuple of (name, MixedTensor)
        """
        for name, module in self.named_modules():
            if isinstance(module, MixedTensor) and module != self:
                yield name, module
    
    def print_parameters_info(self):
        """Print information about all MixedTensor parameters."""
        print("=" * 80)
        print("Model Parameters Summary")
        print("=" * 80)
        for attr_name, mixed_tensor in self.named_mixed_tensors():
            print(f"\n{attr_name}: {mixed_tensor}")
            if mixed_tensor.get_refinable_count() > 0:
                print(f"  Refinable values: min={mixed_tensor.refinable_params.min().item():.4f}, "
                      f"max={mixed_tensor.refinable_params.max().item():.4f}, "
                      f"mean={mixed_tensor.refinable_params.mean().item():.4f}")
        print("=" * 80)

    def register_alternative_conformations(self):
        """
        Identify and register all alternative conformation groups in the structure.

        For each residue that has alternative conformations (altloc A, B, C, etc.),
        this method identifies all atoms belonging to each conformation and stores
        their indices as tensors in a tuple.

        The result is stored in self.altloc_pairs as a list of tuples, where each
        tuple contains tensors of atom indices for each alternative conformation.

        Examples
        --------
        For a residue with conformations A and B:

        >>> # Conformation A has atoms at indices [100, 101, 102, ...]
        >>> # Conformation B has atoms at indices [110, 111, 112, ...]
        >>> # Result: [(tensor([100, 101, 102, ...]), tensor([110, 111, 112, ...])), ...]

        For a residue with conformations A, B, C:

        >>> # Result: [(tensor([200, 201, ...]), tensor([210, 211, ...]), tensor([220, 221, ...])), ...]
        """
        # Initialize the list to store alternative conformation groups
        self.altloc_pairs = []
        
        # Get all atoms with alternative conformations (non-empty altloc field)
        pdb_with_altlocs = self.pdb[self.pdb['altloc'] != '']
        
        if len(pdb_with_altlocs) == 0:
            # No alternative conformations in this structure
            return
        
        # Group by residue (resname, resseq, chainid) to find all residues
        # that have alternative conformations
        grouped = pdb_with_altlocs.groupby(['resname', 'resseq', 'chainid'])
        
        for (resname, resseq, chainid), group in grouped:
            # Get all unique altloc identifiers for this residue
            unique_altlocs = sorted(group['altloc'].unique())
            
            # Only register if there are actually multiple conformations
            if len(unique_altlocs) > 1:
                # For each altloc, collect all atom indices belonging to that conformation
                conformation_tensors = []
                for altloc in unique_altlocs:
                    # Get all atoms for this specific altloc
                    altloc_atoms = group[group['altloc'] == altloc]
                    # Get their indices and convert to tensor
                    indices = torch.tensor(altloc_atoms['index'].tolist(), dtype=torch.long)
                    conformation_tensors.append(indices)
                
                # Store as a tuple of tensors
                self.altloc_pairs.append(tuple(conformation_tensors))

    def shake_coords(self, stddev: float):
        """
        Apply random Gaussian noise to atomic coordinates.

        Perturbs the atomic coordinates by adding Gaussian noise with a
        specified standard deviation. The noise is applied to all atoms.

        Parameters
        ----------
        stddev : float
            Standard deviation of the Gaussian noise to be added, in Angstroms.
        """
        xyz = self.xyz().detach()
        new_xyz = xyz + torch.normal(mean=0.0, std=stddev, size=xyz.shape)
        self.xyz = MixedTensor(new_xyz, refinable_mask=self.xyz.refinable_mask, name='xyz')
   
    def shake_b_factors(self, stddev: float):
        """
        Apply random Gaussian noise to B-factors (temperature factors).

        Perturbs the B-factors by adding Gaussian noise with a specified
        standard deviation. The noise is applied to all atoms.

        Parameters
        ----------
        stddev : float
            Standard deviation of the Gaussian noise to be added, in 1/Angstrom^2.
        """
        b_factors = self.b().detach()
        new_b = b_factors + torch.normal(mean=0.0, std=stddev, size=b_factors.shape)
        self.b = PositiveMixedTensor(new_b, refinable_mask=self.b.refinable_mask, name='b_factor')

    def adp_loss(self):
        """
        Compute the ADP (B-factor) regularization loss.

        This loss encourages B-factors to have similar values across the
        structure, helping to prevent overfitting during refinement.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the ADP loss.
        """
        b_current = self.b()
        b_mean = torch.mean(b_current)
        loss = torch.mean((b_current - b_mean) ** 2)
        return loss
    
    def adp_nll_loss(self, target_log_std: float = 0.2):
        """
        Compute negative log-likelihood of ADPs assuming Gaussian distribution in log-space.

        This regularization penalizes B-factors that deviate from a target distribution
        with a FIXED standard deviation (hyperparameter), avoiding circular dependency
        on the current distribution's statistics.

        The NLL for a Gaussian distribution in log-space is::

            NLL = 0.5 * mean[(log_b - mu)^2 / sigma^2 + log(2*pi*sigma^2)]

        Where mu is the mean of log-space B-factors (computed from current data) and
        sigma is the FIXED target standard deviation (hyperparameter).

        Parameters
        ----------
        target_log_std : float, optional
            Target standard deviation in log-space. Default is 0.2.
            - 0.1 = very tight (B-factors within ~10% of mean)
            - 0.2 = moderate spread (B-factors within ~20% of mean) [RECOMMENDED]
            - 0.3 = looser spread (B-factors within ~30% of mean)

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the NLL. Lower values indicate the distribution
            is closer to the target Gaussian with fixed sigma.

        Examples
        --------
        >>> # During refinement
        >>> structure_factor_loss = compute_structure_factor_loss()
        >>> nll_reg = model.adp_nll_loss(target_log_std=0.2)
        >>> total_loss = structure_factor_loss + 0.01 * nll_reg
        >>> total_loss.backward()

        Notes
        -----
        Uses FIXED sigma (no circular dependency on current distribution).
        Smaller target_log_std = stronger regularization (tighter distribution).
        """
        # Access the internal log-space values directly from the PositiveMixedTensor
        # The parent MixedTensor.forward() returns log-space values before exp()
        log_b = super(PositiveMixedTensor, self.b).forward()
        
        # Compute mean in log-space (target center of distribution)
        mu = torch.mean(log_b).detach()
        
        # Use FIXED target_log_std (not computed from data)
        sigma = target_log_std
        
        # Compute NLL for Gaussian distribution
        # NLL = 0.5 * [(log_b - μ)² / σ² + log(2πσ²)]
        ln_2pi_sigma2 = torch.log(torch.tensor(2.0 * torch.pi * sigma**2, 
                                               dtype=self.dtype_float, 
                                               device=self.device))
        
        squared_deviations = (log_b - mu) ** 2
        nll_per_atom = 0.5 * (squared_deviations / (sigma**2) + ln_2pi_sigma2)
        
        # Return mean NLL across all atoms
        nll = torch.mean(nll_per_atom)
        
        return nll
    
    def adp_nll_loss_per_atom(self, target_log_std: float = 0.2):
        """
        Compute per-atom negative log-likelihood for B-factors in log-space.

        Returns the NLL contribution for each individual atom, useful for
        identifying outliers or applying atom-specific regularization weights.

        The per-atom NLL is::

            NLL_i = 0.5 * [(log_b_i - mu)^2 / sigma^2 + log(2*pi*sigma^2)]

        Parameters
        ----------
        target_log_std : float, optional
            Fixed target standard deviation in log-space. Default is 0.2.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_atoms,) with per-atom NLL values.
            Higher values indicate atoms farther from the mean.

        Examples
        --------
        >>> # Get per-atom NLL
        >>> atom_nll = model.adp_nll_loss_per_atom(target_log_std=0.2)
        >>> # Identify outlier atoms (high NLL)
        >>> threshold = atom_nll.mean() + 2 * atom_nll.std()
        >>> outliers = atom_nll > threshold
        """
        # Access the internal log-space values
        log_b = super(PositiveMixedTensor, self.b).forward()
        
        # Compute mean in log-space
        mu = torch.mean(log_b)
        
        # Use FIXED target_log_std
        sigma = target_log_std
        
        # Compute per-atom NLL
        ln_2pi_sigma2 = torch.log(torch.tensor(2.0 * torch.pi * sigma**2,
                                               dtype=self.dtype_float,
                                               device=self.device))
        
        squared_deviations = (log_b - mu) ** 2
        nll_per_atom = 0.5 * (squared_deviations / (sigma**2) + ln_2pi_sigma2)
        
        return nll_per_atom
    
    def adp_kl_divergence_loss(self, target_log_std: float = 0.2):
        """
        Compute KL divergence between log B-factor distribution and target Gaussian.

        Measures how different the current log B-factor distribution is from a
        target Gaussian distribution with the current mean of log B-factors and
        a fixed target standard deviation.

        KL divergence formula for two Gaussians with same mean::

            KL(q || p) = log(sigma_target/sigma_data) + sigma_data^2 / (2*sigma_target^2) - 0.5

        Parameters
        ----------
        target_log_std : float, optional
            Target standard deviation in log-space. Default is 0.2.
            Controls how tightly B-factors should cluster.

        Returns
        -------
        torch.Tensor
            Scalar KL divergence value (always >= 0).
            0 means distributions match perfectly.
            Higher values mean more deviation from target.

        Examples
        --------
        >>> # Use in loss function
        >>> loss = xray_loss + w_adp * model.adp_kl_divergence_loss(0.2)

        Notes
        -----
        Lower target_log_std = stronger regularization (tighter distribution).
        Mean is detached so it adapts to the natural scale of the data.
        """

        # Access the internal log-space values
        log_b = super(PositiveMixedTensor, self.b).forward()
        
        # Compute statistics of actual distribution
        mu_data = torch.mean(log_b).detach()  # Detached mean (adapts to data)
        sigma_data = torch.std(log_b)  # Current std (to be regularized)
        
        # Target distribution parameters
        mu_target = mu_data  # Same mean as data
        sigma_target = target_log_std  # Fixed target std
        
        # KL divergence: KL(actual || target) for Gaussians with same mean
        # KL = log(σ_target/σ_data) + σ_data² / (2σ_target²) - 0.5
        log_sigma_ratio = torch.log(torch.tensor(sigma_target, dtype=self.dtype_float, device=self.device) / sigma_data)
        variance_ratio = (sigma_data ** 2) / (2 * sigma_target ** 2)
        
        kl_divergence = log_sigma_ratio + variance_ratio - 0.5
        
        return kl_divergence

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Return a dictionary containing the complete state of the Model.

        Includes all registered buffers, model parameters (xyz, b, u, occupancy),
        PDB DataFrame, and metadata (spacegroup, device, dtype, etc.).

        Parameters
        ----------
        destination : dict, optional
            Optional dict to populate with state.
        prefix : str, optional
            Prefix for parameter names. Default is ''.
        keep_vars : bool, optional
            Whether to keep variables in computational graph. Default is False.

        Returns
        -------
        dict
            Complete state dictionary.
        """
        # Get parent class state_dict (includes all registered buffers)
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        # Add model-specific state
        state[prefix + 'pdb'] = self.pdb.copy() if hasattr(self, 'pdb') else None
        state[prefix + 'spacegroup'] = self.spacegroup if hasattr(self, 'spacegroup') else None
        state[prefix + 'initialized'] = self.initialized
        state[prefix + 'dtype_float'] = self.dtype_float
        state[prefix + 'device'] = self.device
        state[prefix + 'strip_H'] = self.strip_H
        state[prefix + 'altloc_pairs'] = self.altloc_pairs if hasattr(self, 'altloc_pairs') else []
        
        return state

    def save_state(self, path: str):
        """
        Save the complete state of the model to a file.

        Parameters
        ----------
        path : str
            Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved model state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load the complete state of the model from a file.

        Parameters
        ----------
        path : str
            Path to load the state dictionary from.
        strict : bool, optional
            Whether to strictly enforce that keys match. Default is True.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        loaded = type(self).create_from_state_dict(state_dict, device=self.device, verbose=self.verbose)
        # Copy loaded state to self
        self.__dict__.update(loaded.__dict__)
        if self.verbose > 0:
            print(f"Loaded model state from {path}")
    
    @classmethod
    def create_from_state_dict(cls, state_dict: dict, device: torch.device = torch.device('cpu'), 
                               verbose: int = 1, dtype_float: torch.dtype = torch.float32) -> 'Model':
        """
        Create a fully initialized Model from a state dictionary.

        This is the recommended way to restore a Model from a saved state.
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
        Model
            Fully initialized instance with restored state.
        """
        # Extract metadata (non-tensor data that we handle specially)
        pdb = state_dict.pop('pdb', None)
        spacegroup = state_dict.pop('spacegroup', None)
        initialized = state_dict.pop('initialized', False)
        saved_dtype = state_dict.pop('dtype_float', dtype_float)
        saved_device = state_dict.pop('device', device)
        strip_H = state_dict.pop('strip_H', True)
        altloc_pairs = state_dict.pop('altloc_pairs', [])
        
        # Create instance
        instance = cls(dtype_float=saved_dtype, verbose=verbose, device=device, strip_H=strip_H)
        
        # Set metadata
        instance.pdb = pdb
        instance.spacegroup = spacegroup
        instance.initialized = initialized
        instance.altloc_pairs = altloc_pairs
        
        # Setup spacegroup objects if spacegroup exists
        if spacegroup is not None:
            import gemmi
            instance.spacegroup_gemmi = gemmi.SpaceGroup(spacegroup.replace('  ', ' '))
            instance.spacegroup_function = sym.Symmetry(spacegroup)
        
        # If PDB exists, create the parameter wrappers with correct shapes
        if pdb is not None:
            n_atoms = len(pdb)
            
            # Create MixedTensors with initial values from PDB (will be overwritten by load_state_dict)
            # Get refinable masks from state_dict if available
            xyz_mask = state_dict.get('xyz.refinable_mask')
            b_mask = state_dict.get('b.refinable_mask')
            u_mask = state_dict.get('u.refinable_mask')
            
            instance.xyz = MixedTensor(
                torch.tensor(pdb[['x', 'y', 'z']].values, dtype=saved_dtype),
                refinable_mask=xyz_mask, name='xyz'
            )
            instance.b = PositiveMixedTensor(
                torch.tensor(pdb['tempfactor'].values, dtype=saved_dtype),
                refinable_mask=b_mask, name='b_factor'
            )
            instance.u = MixedTensor(
                torch.tensor(pdb[['u11', 'u22', 'u33', 'u12', 'u13', 'u23']].values, dtype=saved_dtype),
                refinable_mask=u_mask, name='aniso_U'
            )
            
            # Create OccupancyTensor
            initial_occ = torch.tensor(pdb['occupancy'].values, dtype=saved_dtype)
            sharing_groups, altloc_groups, refinable_mask = instance._create_occupancy_groups(pdb, initial_occ)
            
            # Override mask if present in state_dict
            saved_occ_mask = state_dict.get('occupancy.refinable_mask')
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
                name='occupancy'
            )
            
            # Register buffers that are needed
            if 'cell' in state_dict:
                instance.register_buffer('cell', torch.zeros_like(state_dict['cell'], device=device))
            if 'aniso_flag' not in instance._buffers or instance.aniso_flag is None:
                instance.register_buffer('aniso_flag', torch.tensor(pdb['anisou_flag'].values, dtype=torch.bool))
            
            # Register mask buffers
            instance.register_buffer("xyz_mask", torch.ones(n_atoms, dtype=torch.bool, device=device))
            instance.register_buffer("b_mask", torch.ones(n_atoms, dtype=torch.bool, device=device))
            instance.register_buffer("u_mask", torch.ones(n_atoms, dtype=torch.bool, device=device))
            instance.register_buffer("occupancy_mask", torch.ones(n_atoms, dtype=torch.bool, device=device))
            
            # Register other buffers based on state_dict
            buffer_names = ['inv_fractional_matrix', 'fractional_matrix', 'recB', 'vdw_radii']
            for name in buffer_names:
                if name in state_dict and state_dict[name] is not None:
                    instance.register_buffer(name, torch.zeros_like(state_dict[name], device=device))
        
        # Now use PyTorch's default load_state_dict
        state_dict = {k: v for k, v in state_dict.items() if k.shape[0] > 0}
        instance.load_state_dict(state_dict, strict=False)
        
        if verbose > 0:
            n_atoms = len(instance.pdb) if instance.pdb is not None else 0
            print(f"Created Model from state_dict: {n_atoms} atoms")
        
        return instance