
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
    def __init__(self,dtype_float=torch.float32,verbose=1,device=torch.device('cpu'), strip_H: bool =True):
        super().__init__()
        self.altloc_pairs = []
        self.verbose = verbose
        self.initialized = False
        self.dtype_float = dtype_float
        self.device = device
        self.strip_H = strip_H
    
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

    def load_pdb(self,file):
        '''
        Load atomic model from PDB file.
        '''
        reader = legacy_format_readers.PDB(verbose=self.verbose).read(file)   
        return self.load(reader)
    
    def load_cif(self, file):
        """
        Load atomic model from mmCIF file.
        
        Args:
            file: Path to CIF/mmCIF file
            
        Returns:
            self (for method chaining)
        """
        
        if self.verbose > 0:
            print(f"Loading CIF file: {file}")
        
        # Read CIF file
        cif_reader = cif_readers.ModelCIFReader(file)

        return self.load(cif_reader)
    
    def _create_occupancy_groups(self, pdb_df, initial_occ):
        """
        Create sharing groups and altloc groups for occupancy.
        
        Logic:
        1. First identify alternative conformations (multiple altlocs per residue)
        2. For altloc groups: ALL atoms in each conformation share one collapsed index
        3. For non-altloc residues: group by similar occupancy (within 0.01 tolerance)
        4. Only refine occupancies that differ from 1.0
        
        Args:
            pdb_df: PDB DataFrame
            initial_occ: Tensor of initial occupancy values
        
        Returns:
            tuple: (sharing_groups_tensor, altloc_groups, refinable_mask)
                sharing_groups_tensor: Tensor of shape (n_atoms,) where each value is the
                                      collapsed index for that atom
                altloc_groups: List of tuples of atom index lists for alternative conformations
                refinable_mask: Boolean tensor indicating which atoms should be refined
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
        Get van der Waals radii for all atoms in the model based on their elements.
        Caches the result in self.vdw_radii for future calls.
        
        Returns:
        --------
        self.vdw_radii : torch.Tensor (n_atoms,)
            Van der Waals radii for each atom
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
        Create a deep copy of the Model with all its parameters, buffers, and submodules.
        
        This method uses PyTorch's state_dict mechanism to efficiently copy all:
        - Registered buffers (cell, fractional matrices, aniso_flag, etc.)
        - Module parameters (xyz, b, u, occupancy via their .copy() methods)
        - PDB DataFrame and metadata
        - Spacegroup and symmetry information
        
        Returns:
            Model: A new Model instance with copied data
            
        Example:
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
        
        Args:
            selection_string: Phenix-style selection string (see parse_phenix_selection docs)
            target: Parameter to update ('xyz', 'b', 'u', or 'occupancy')
            mode: How to combine with current mask:
                  - 'set': Replace mask with selection (default)
                  - 'add': Add selection to current mask
                  - 'remove': Remove selection from current mask
            freeze: If True (default), selected atoms will be frozen (mask=False).
                   If False, selected atoms will be unfrozen (mask=True).
        
        Examples:
            >>> # Freeze chain A coordinates
            >>> model.update_mask_from_selection("chain A", "xyz", mode='set', freeze=True)
            >>> model.apply_mask_to_parameter("xyz")
            >>> 
            >>> # Freeze residues 10-20
            >>> model.update_mask_from_selection("resseq 10:20", "xyz", freeze=True)
            >>> model.apply_mask_to_parameter("xyz")
            >>> 
            >>> # Unfreeze backbone atoms
            >>> model.update_mask_from_selection("name CA or name C or name N", "xyz", freeze=False)
            >>> model.apply_mask_to_parameter("xyz")
            >>> 
            >>> # Freeze B-factors for waters, add to existing frozen atoms
            >>> model.update_mask_from_selection("resname HOH", "b", mode='add', freeze=True)
            >>> model.apply_mask_to_parameter("b")
        
        Raises:
            ValueError: If target is not recognized or selection syntax is invalid
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
        
        This method takes the current state of the mask buffer (xyz_mask, b_mask, etc.)
        and applies it to the corresponding parameter tensor's refinable mask.
        
        Args:
            target: Parameter to update ('xyz', 'b', 'u', or 'occupancy')
        
        Examples:
            >>> # After updating xyz_mask with selections, apply it
            >>> model.update_mask_from_selection("chain A", "xyz", freeze=True)
            >>> model.apply_mask_to_parameter("xyz")
            >>> 
            >>> # Apply all masks after multiple updates
            >>> model.update_mask_from_selection("resname HOH", "b", freeze=True)
            >>> model.update_mask_from_selection("resname HOH", "occupancy", freeze=True)
            >>> model.apply_mask_to_parameter("b")
            >>> model.apply_mask_to_parameter("occupancy")
        
        Raises:
            ValueError: If target is not recognized
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
        
        This is a convenience method that combines update_mask_from_selection() and
        apply_mask_to_parameter() into a single call.
        
        Args:
            selection_string: Phenix-style selection string
            targets: Parameter(s) to freeze. Can be:
                    - 'all': Freeze xyz, b, u, and occupancy (default)
                    - str: Single parameter ('xyz', 'b', 'u', 'occupancy')
                    - list: List of parameters ['xyz', 'b']
        
        Examples:
            >>> # Freeze all parameters for chain A
            >>> model.freeze_selection("chain A", targets='all')
            >>> 
            >>> # Freeze only coordinates for residues 10-20
            >>> model.freeze_selection("resseq 10:20", targets='xyz')
            >>> 
            >>> # Freeze coordinates and B-factors for waters
            >>> model.freeze_selection("resname HOH", targets=['xyz', 'b'])
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
        
        This is a convenience method that combines update_mask_from_selection() and
        apply_mask_to_parameter() into a single call.
        
        Args:
            selection_string: Phenix-style selection string
            targets: Parameter(s) to unfreeze. Can be:
                    - 'all': Unfreeze xyz, b, u, and occupancy (default)
                    - str: Single parameter ('xyz', 'b', 'u', 'occupancy')
                    - list: List of parameters ['xyz', 'b']
        
        Examples:
            >>> # Unfreeze all parameters for chain A
            >>> model.unfreeze_selection("chain A", targets='all')
            >>> 
            >>> # Unfreeze only coordinates for backbone atoms
            >>> model.unfreeze_selection("name CA or name C or name N", targets='xyz')
            >>> 
            >>> # Unfreeze B-factors for non-water residues
            >>> model.unfreeze_selection("not resname HOH", targets='b')
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
        tuple contains tensors of atom indices for each alternative conformation
        of a residue.
        
        Example:
            For a residue with conformations A and B:
            - Conformation A has atoms at indices [100, 101, 102, ...]
            - Conformation B has atoms at indices [110, 111, 112, ...]
            Result: [(tensor([100, 101, 102, ...]), tensor([110, 111, 112, ...])), ...]
            
            For a residue with conformations A, B, C:
            [(tensor([200, 201, ...]), tensor([210, 211, ...]), tensor([220, 221, ...])), ...]
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
        
        This method perturbs the atomic coordinates by adding Gaussian noise
        with a specified standard deviation. The noise is applied to all atoms
        in the model.
        
        Args:
            stddev: Standard deviation of the Gaussian noise to be added (in Å).
        """
        xyz = self.xyz().detach()
        new_xyz = xyz + torch.normal(mean=0.0, std=stddev, size=xyz.shape)
        self.xyz = MixedTensor(new_xyz, refinable_mask=self.xyz.refinable_mask, name='xyz')
   
    def shake_b_factors(self, stddev: float):
        """
        Apply random Gaussian noise to B-factors (temperature factors).
        
        This method perturbs the B-factors by adding Gaussian noise
        with a specified standard deviation. The noise is applied to all atoms
        in the model.
        
        Args:
            stddev: Standard deviation of the Gaussian noise to be added (in 1/Å**2).
        """
        b_factors = self.b().detach()
        new_b = b_factors + torch.normal(mean=0.0, std=stddev, size=b_factors.shape)
        self.b = PositiveMixedTensor(new_b, refinable_mask=self.b.refinable_mask, name='b_factor')

    def adp_loss(self):
        """
        Compute the ADP (B-factor) regularization loss.
        
        This loss encourages B-factors to have similar values across the structure,
        helping to prevent overfitting during refinement.
        
        Returns:
            torch.Tensor: Scalar tensor representing the ADP loss.
        """
        b_current = self.b()
        b_mean = torch.mean(b_current)
        loss = torch.mean((b_current - b_mean) ** 2)
        return loss
    
    def adp_nll_loss(self, target_log_std: float = 0.2):
        """
        Compute the negative log-likelihood (NLL) of ADPs assuming a Gaussian distribution in log-space.
        
        This regularization penalizes B-factors that deviate from a target distribution with
        a FIXED standard deviation (hyperparameter), avoiding circular dependency on the 
        current distribution's statistics.
        
        The NLL for a Gaussian distribution in log-space is:
            NLL = 0.5 * mean[(log_b - μ)² / σ² + log(2πσ²)]
        
        Where:
            - μ (mu) = mean of log-space B-factors (computed from current data)
            - σ (sigma) = FIXED target standard deviation (hyperparameter, not computed)
        
        Args:
            target_log_std: Target standard deviation in log-space (default: 0.2)
                           - 0.1 = very tight (B-factors within ~10% of mean)
                           - 0.2 = moderate spread (B-factors within ~20% of mean) [RECOMMENDED]
                           - 0.3 = looser spread (B-factors within ~30% of mean)
        
        Returns:
            torch.Tensor: Scalar tensor representing the NLL. Lower values indicate
                         the distribution is closer to the target Gaussian with fixed σ.
        
        Example:
            >>> # During refinement
            >>> structure_factor_loss = compute_structure_factor_loss()
            >>> geometry_loss = compute_geometry_loss()
            >>> nll_reg = model.adp_nll_loss(target_log_std=0.2)
            >>> 
            >>> # Combined loss with NLL penalty
            >>> total_loss = structure_factor_loss + 0.1 * geometry_loss + 0.01 * nll_reg
            >>> total_loss.backward()
        
        Note:
            - Uses FIXED σ (no circular dependency on current distribution)
            - Penalizes deviations from mean with strength controlled by target_log_std
            - Smaller target_log_std = stronger regularization (tighter distribution)
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
        Compute per-atom negative log-likelihood (NLL) for B-factors in log-space.
        
        This returns the NLL contribution for each individual atom, useful for:
        - Identifying atoms with unusual B-factors (outliers)
        - Applying atom-specific regularization weights
        - Diagnostic analysis of B-factor distribution
        
        The per-atom NLL is:
            NLL_i = 0.5 * [(log_b_i - μ)² / σ² + log(2πσ²)]
        
        Args:
            target_log_std: Fixed target standard deviation in log-space (default: 0.2)
        
        Returns:
            torch.Tensor: Tensor of shape (n_atoms,) with per-atom NLL values.
                         Higher values indicate atoms farther from the mean.
        
        Example:
            >>> # Get per-atom NLL
            >>> atom_nll = model.adp_nll_loss_per_atom(target_log_std=0.2)
            >>> 
            >>> # Identify outlier atoms (high NLL)
            >>> threshold = atom_nll.mean() + 2 * atom_nll.std()
            >>> outliers = atom_nll > threshold
            >>> print(f"Found {outliers.sum()} outlier atoms")
            >>> print(model.pdb[outliers.cpu().numpy()])
            >>> 
            >>> # Use in loss with per-atom weighting
            >>> weighted_nll = torch.mean(weights * atom_nll)
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
        
        This measures how different the current log B-factor distribution is from
        a target Gaussian distribution with:
        - Mean: Current mean of log B-factors (detached, adapts to data)
        - Std: Fixed target standard deviation (regularization strength)
        
        KL divergence formula for two Gaussians:
            KL(q || p) = log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²) / (2σ_p²) - 0.5
        
        Where:
            q = actual distribution: N(μ_data, σ_data)
            p = target distribution: N(μ_data, σ_target)
        
        Since both distributions share the same mean (μ_data), the formula simplifies to:
            KL(q || p) = log(σ_target/σ_data) + σ_data² / (2σ_target²) - 0.5
        
        Args:
            target_log_std: Target standard deviation in log-space (default: 0.2)
                           Controls how tightly B-factors should cluster
        
        Returns:
            torch.Tensor: Scalar KL divergence value (always ≥ 0)
                         0 means distributions match perfectly
                         Higher values mean more deviation from target
        
        Example:
            >>> # Use in loss function
            >>> loss = xray_loss + w_restraints * restraints_loss + w_adp * model.adp_kl_divergence_loss(0.2)
            >>> 
            >>> # Monitor during refinement
            >>> kl_div = model.adp_kl_divergence_loss(0.2)
            >>> print(f"ADP KL divergence: {kl_div.item():.4f}")
        
        Notes:
            - Lower target_log_std = stronger regularization (tighter distribution)
            - Mean is detached so it adapts to the natural scale of the data
            - This is conceptually similar to NLL but explicitly measures distributional difference
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
        Returns a dictionary containing the complete state of the Model.
        
        This includes:
        - All registered buffers (via parent class state_dict)
        - Model parameters (xyz, b, u, occupancy)
        - PDB DataFrame
        - Metadata (spacegroup, device, dtype, etc.)
        
        Args:
            destination: Optional dict to populate
            prefix: Prefix for parameter names
            keep_vars: Whether to keep variables in computational graph
            
        Returns:
            dict: Complete state dictionary
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

    def load_state_dict(self, state_dict, strict=True):
        """
        Loads the Model state from a dictionary.
        
        Args:
            state_dict: Dictionary containing model state
            strict: Whether to strictly enforce that keys match
        """
        # Extract model-specific state
        self.pdb = state_dict.pop('pdb', None)
        self.spacegroup = state_dict.pop('spacegroup', None)
        self.initialized = state_dict.pop('initialized', False)
        self.dtype_float = state_dict.pop('dtype_float', torch.float32)
        self.device = state_dict.pop('device', torch.device('cpu'))
        self.strip_H = state_dict.pop('strip_H', True)
        self.altloc_pairs = state_dict.pop('altloc_pairs', [])
        
        # Reconstruct spacegroup_gemmi and spacegroup_function if spacegroup exists
        if self.spacegroup is not None:
            import gemmi
            self.spacegroup_gemmi = gemmi.SpaceGroup(self.spacegroup.replace('  ', ' '))
            self.spacegroup_function = sym.Symmetry(self.spacegroup)
        
        # If PDB is present, ensure submodules and buffers are instantiated
        if self.pdb is not None:
            # Instantiate MixedTensors if they don't exist
            # We use the PDB data to initialize them with correct shapes
            # We also try to retrieve the refinable_mask from state_dict to ensure parameter shapes match
            
            if not hasattr(self, 'xyz') or self.xyz is None:
                mask = state_dict.get('xyz.refinable_mask')
                self.xyz = MixedTensor(torch.tensor(self.pdb[['x', 'y', 'z']].values, dtype=self.dtype_float), refinable_mask=mask, name='xyz')
            
            if not hasattr(self, 'b') or self.b is None:
                mask = state_dict.get('b.refinable_mask')
                self.b = PositiveMixedTensor(torch.tensor(self.pdb['tempfactor'].values, dtype=self.dtype_float), refinable_mask=mask, name='b_factor')
                
            if not hasattr(self, 'u') or self.u is None:
                mask = state_dict.get('u.refinable_mask')
                self.u = MixedTensor(torch.tensor(self.pdb[['u11', 'u22', 'u33', 'u12', 'u13', 'u23']].values, dtype=self.dtype_float), refinable_mask=mask, name='aniso_U')
                
            if not hasattr(self, 'occupancy') or self.occupancy is None:
                initial_occ = torch.tensor(self.pdb['occupancy'].values, dtype=self.dtype_float)
                sharing_groups, altloc_groups, refinable_mask = self._create_occupancy_groups(self.pdb, initial_occ)
                
                # Override mask if present in state_dict
                # Note: saved mask is COLLAPSED, but constructor expects FULL mask
                saved_mask = state_dict.get('occupancy.refinable_mask')
                if saved_mask is not None:
                    # Expand the collapsed mask using sharing_groups
                    # sharing_groups maps atom_idx -> collapsed_idx
                    # So full_mask[atom_idx] = collapsed_mask[sharing_groups[atom_idx]]
                    if saved_mask.device != sharing_groups.device:
                        saved_mask = saved_mask.to(sharing_groups.device)
                    refinable_mask = saved_mask[sharing_groups]
                    
                self.occupancy = OccupancyTensor(
                    initial_values=initial_occ,
                    sharing_groups=sharing_groups,
                    altloc_groups=altloc_groups,
                    refinable_mask=refinable_mask,
                    dtype=self.dtype_float,
                    device=self.device,
                    name='occupancy'
                )
            
            # Register buffers that might be missing if __init__ didn't do it
            # We register them with dummy values (or values from PDB), 
            # load_state_dict will overwrite them with saved values
            
            if not hasattr(self, 'aniso_flag'):
                self.register_buffer('aniso_flag', torch.tensor(self.pdb['anisou_flag'].values, dtype=torch.bool))
            
            # For cell-dependent buffers, we need cell. 
            # If cell is in state_dict, we can't easily access it yet.
            # But we can register buffers with None or dummy tensors if we know the shape?
            # Actually, register_buffer(name, tensor) requires a tensor.
            # If we don't register them, load_state_dict might fail for these keys.
            # However, Model.load() registers them.
            # If we are loading into an uninitialized model, we should try to register them.
            
            # We can try to extract cell from state_dict if possible, or just use dummy
            # But state_dict keys might have prefixes.
            # Since we are in load_state_dict, the keys passed to us match our parameters.
            
            # Let's try to register them if they are missing.
            # We need 'cell' to compute them properly, but load_state_dict will overwrite.
            # So we just need to register them so they exist.
            
            if not hasattr(self, 'cell'):
                # Try to find cell in state_dict
                cell_key = 'cell'
                if cell_key in state_dict:
                    self.register_buffer('cell', state_dict[cell_key].clone())
                else:
                    # Fallback or maybe it's not in state_dict?
                    pass
            
            if hasattr(self, 'cell') and self.cell is not None:
                if not hasattr(self, 'inv_fractional_matrix'):
                    self.register_buffer('inv_fractional_matrix', torch.tensor(mnp.get_inv_fractional_matrix(self.cell), dtype=self.dtype_float))
                if not hasattr(self, 'fractional_matrix'):
                    self.register_buffer('fractional_matrix', torch.tensor(mnp.get_fractional_matrix(self.cell), dtype=self.dtype_float))
                if not hasattr(self, 'recB'):
                    self.register_buffer('recB', math_torch.reciprocal_basis_matrix(self.cell).to(dtype=self.dtype_float).to(self.device))
            
            # Register mask buffers
            if not hasattr(self, 'xyz_mask'):
                self.register_buffer("xyz_mask", torch.ones(len(self.pdb), dtype=torch.bool, device=self.device))
            if not hasattr(self, 'b_mask'):
                self.register_buffer("b_mask", torch.ones(len(self.pdb), dtype=torch.bool, device=self.device))
            if not hasattr(self, 'u_mask'):
                self.register_buffer("u_mask", torch.ones(len(self.pdb), dtype=torch.bool, device=self.device))
            if not hasattr(self, 'occupancy_mask'):
                self.register_buffer("occupancy_mask", torch.ones(len(self.pdb), dtype=torch.bool, device=self.device))
                
            # Register vdw_radii
            if not hasattr(self, 'vdw_radii'):
                self.get_vdw_radii()

        # Load parent class state_dict (buffers and parameters)
        return super().load_state_dict(state_dict, strict=strict)

    def save_state(self, path: str):
        """
        Save the complete state of the model to a file.
        
        Args:
            path (str): Path to save the state dictionary to.
        """
        torch.save(self.state_dict(), path)
        if self.verbose > 0:
            print(f"Saved model state to {path}")

    def load_state(self, path: str, strict: bool = True):
        """
        Load the complete state of the model from a file.
        
        Args:
            path (str): Path to load the state dictionary from.
            strict (bool): Whether to strictly enforce that keys match.
        """
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state_dict, strict=strict)
        if self.verbose > 0:
            print(f"Loaded model state from {path}")