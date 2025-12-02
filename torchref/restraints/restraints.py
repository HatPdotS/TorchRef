"""
Restraints Class for Crystallographic Model Refinement

This module provides a comprehensive restraints handler for crystallographic models.
It parses CIF (Crystallographic Information File) restraints dictionaries and builds
efficient tensor-based representations for bond lengths, angles, and torsion angles
across an entire molecular structure.

The restraints are stored in a format optimized for PyTorch operations:
- Bond lengths: (N, 2) tensor of atom indices
- Angles: (N, 3) tensor of atom indices  
- Torsions: (N, 4) tensor of atom indices


Each restraint type also stores the expected values and sigma (uncertainty) values.
"""

from networkx import sigma
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from torchref.restraints.restraints_helper import read_cif,find_cif_file_in_library,read_link_definitions
from torchref.model.model import Model
from torch.special import i0  # modified Bessel function of the first kind
from torch.nn import Module
from torchref.utils.debug_utils import DebugMixin
from torchref.utils.utils import ModuleReference


class Restraints(DebugMixin, Module):
    """
    Restraints handler for crystallographic model refinement.

    This class parses CIF restraints dictionaries and builds efficient tensor
    representations for the entire molecular structure. It stores restraints
    for bond lengths, angles, torsion angles, and planes with their expected values
    and uncertainties (sigma values).

    Supports two initialization patterns:

    1. Empty initialization (for state_dict loading)::

        >>> restraints = Restraints()  # Creates empty shell
        >>> restraints.load_state_dict(torch.load('restraints.pt'))

    2. Full initialization with model::

        >>> restraints = Restraints(model, cif_path='restraints.cif')

    The restraints are organized in a hierarchical dictionary structure:
    ``restraints[restraint_type][origin][property]``

    Where:

    - restraint_type: 'bond', 'angle', 'torsion', 'plane'
    - origin:

      - For bonds/angles/torsions: 'intra', 'peptide', 'disulfide', 'phi', 'psi', 'omega'
      - For planes: '3_atoms', '4_atoms', ..., '10_atoms' (grouped by atom count)

    - property: 'indices', 'references', 'sigmas', 'periods' (for torsions only)

    Parameters
    ----------
    model : Model, optional
        Model instance containing the atomic structure. If None, creates empty shell.
    cif_path : str or list of str, optional
        Path to the CIF restraints dictionary file(s).
    verbose : int, default 1
        Verbosity level (0=silent, 1=normal, 2=detailed).

    Attributes
    ----------
    model : ModuleReference
        Reference to the Model instance.
    cif_path : str or list
        Path to the CIF restraints dictionary file.
    cif_dict : dict
        Parsed CIF dictionary with restraints for each residue type.
    restraints : dict
        Hierarchical dictionary containing all restraints.
    unique_residues : list
        List of unique residue names in the model.

    Examples
    --------
    >>> from torchref.model import Model
    >>> from torchref.restraints import Restraints
    >>>
    >>> # Load model
    >>> model = Model()
    >>> model.load_pdb_from_file('structure.pdb')
    >>>
    >>> # Create restraints
    >>> restraints = Restraints(model)
    >>>
    >>> # Access via hierarchical structure (new way)
    >>> bond_indices = restraints.restraints['bond']['intra']['indices']
    >>> angle_refs = restraints.restraints['angle']['peptide']['references']
    >>> torsion_periods = restraints.restraints['torsion']['intra']['periods']
    >>> plane_4atom_indices = restraints.restraints['plane']['4_atoms']['indices']
    >>>
    >>> # Or use backward-compatible properties (old way)
    >>> bond_indices = restraints.bond_indices  # intra-residue bonds
    >>> bond_indices_inter = restraints.bond_indices_inter  # peptide bonds
    """
    
    def __init__(self, model: Model = None, cif_path=None, verbose: int = 1):
        """
        Initialize the Restraints handler.

        If model is provided, fully initializes the restraints.
        If not provided (empty init), creates a shell ready for load_state_dict().

        Parameters
        ----------
        model : Model, optional
            Model instance containing the atomic structure.
        cif_path : str or list of str, optional
            Path to the CIF restraints dictionary file(s).
        verbose : int, default 1
            Verbosity level (0=silent, 1=normal, 2=detailed).
        """
        super().__init__()  
        self.cif_path = cif_path
        self.verbose = verbose
        
        # Empty initialization
        if model is None:
            self.model = None
            self.cif_dict = {}
            self.unique_residues = []
            self.restraints = {}
            return
        
        # Full initialization with model
        self.model = ModuleReference(model)
        self.unique_residues = model.pdb.resname.unique()
        # Check unique residue have more than one atom
        self.unique_residues = [residue for residue in self.unique_residues if self.model.pdb.loc[self.model.pdb['resname'] == residue,'name'].nunique() > 1]
        # Parse the CIF file
        if cif_path:
            if isinstance(cif_path, str):
                try: 
                    self.cif_dict = read_cif(cif_path)
                except ValueError as e:
                    # Re-raise ValueError (validation errors) - these are user errors
                    print("Error reading CIF file:", e)
                    raise
                except Exception as e:
                    print("Error reading CIF file:", e)
                    self.cif_dict = {}
            elif isinstance(cif_path, list):
                self.cif_dict = {}
                for cif_file in cif_path:
                    try:
                        cif_dict_part = read_cif(cif_file)
                        self.cif_dict.update(cif_dict_part)
                    except ValueError as e:
                        # Re-raise ValueError (validation errors) - these are user errors
                        print("Error reading CIF file:", e)
                        raise
                    except Exception as e:
                        print("Error reading CIF file:", e)
            else:
                raise ValueError("cif_path must be a string or a list of strings")
        else:
            self.cif_dict = {}
        
        self.missing_residues = [res for res in self.unique_residues if res not in self.cif_dict]
        additional_files_to_load = [find_cif_file_in_library(res) for res in self.missing_residues]

        for cif_file in additional_files_to_load:
            if cif_file is not None:
                if self.verbose > 1: print(cif_file)
                try:
                    additional_cif_dict = read_cif(cif_file)
                    self.cif_dict.update(additional_cif_dict)
                except Exception as e:
                    print("Error reading CIF file:", e)
                    print('This residue will have no restraints applied.')
        
        self.missing_residues = [res for res in self.unique_residues if res not in self.cif_dict]
        
        if len(self.missing_residues) > 1: 
            if verbose > 0: print(f"Warning: The following residues are missing from the CIF dictionary and will have no restraints applied: {self.missing_residues}")

        # Load link definitions for inter-residue restraints (peptide bonds, etc.)
        if verbose > 1: print("Loading link definitions from monomer library...")
        self.link_dict, self.link_list = read_link_definitions()
        if verbose > 1: print(f"Loaded {len(self.link_dict)} link types")

        # Initialize hierarchical restraints dictionary
        # Structure: restraints[restraint_type][origin][property]
        # restraint_type: 'bond', 'angle', 'torsion', 'plane'
        # origin: 'intra', 'peptide', 'disulfide', 'phi', 'psi', 'omega', etc.
        # property: 'indices', 'references', 'sigmas', 'periods' (for torsions)
        # For planes: origin is atom count like '3_atoms', '4_atoms', etc.
        self.restraints = {
            'bond': {},
            'angle': {},
            'torsion': {},
            'plane': {}
        }
        
        # Build restraints for the entire structure
        self.build_restraints()
        if self.verbose > 0:
            self.summary()

    def build_restraints(self):
        """
        Build all restraints for the entire structure.

        This method iterates through all residues in the model and builds
        restraints for bond lengths, angles, torsions, and planes. The results are
        stored as tensors on the same device as the model coordinates.
        """
        try:
            # Build intra-residue restraints
            self._build_bond_restraints()
            self._build_angle_restraints()
            self._build_torsion_restraints()
            self._build_plane_restraints()
            self._build_chiral_restraints()
            
            # Build inter-residue restraints (peptide bonds, disulfide bonds, etc.)
            self._build_peptide_bond_restraints()
            self._build_disulfide_bond_restraints()
            
            # Build VDW (non-bonded) restraints
            # Note: This should be called AFTER all bonded restraints are built
            # so that the exclusion set can be properly constructed
            self._build_vdw_restraints(
                cutoff=5.0,               # Check atoms within 5 Å
                sigma=0.05,                # Strictness of enforcement
                inter_residue_only=False, # Include both inter- and intra-residue contacts
                use_spatial_hash=True     # Use O(N) spatial hashing algorithm
            )
            
            # Move tensors to the same device as model
            device = self.model.xyz().device
            self._move_to_device(device)
        except Exception as e:
            self.debug_on_error(e, context="Restraints.build_restraints")
            raise



    def expand_altloc(self, residue):
        """
        Expand residue with alternative conformations into separate conformations.

        Yields one DataFrame per altloc (with common atoms included in each).
        Normalizes altloc values: treats both '' and ' ' as no altloc.

        Parameters
        ----------
        residue : pandas.DataFrame
            DataFrame containing residue atoms with altloc column.

        Yields
        ------
        pandas.DataFrame
            DataFrame for each alternative conformation with common atoms included.
        """
        # Normalize altloc values: treat both '' and ' ' as no altloc
        residue = residue.copy()
        residue.loc[residue['altloc'].isin(['', ' ']), 'altloc'] = ' '
        
        alt_conf = residue['altloc'].unique()
        if ' ' in alt_conf:
            residue_no_alt = residue.loc[residue['altloc'] == ' ']
            for alt in alt_conf:
                if alt == ' ':
                    continue
                residue_alt = residue.loc[residue['altloc'] == alt]
                residue_combined = pd.concat([residue_no_alt, residue_alt], ignore_index=True)
                yield residue_combined
        else:
            # No common atoms, yield each altloc separately
            for alt_loc in alt_conf:
                residue_alt = residue.loc[residue['altloc'] == alt_loc]
                yield residue_alt
            

    def _build_bond_restraints(self):
        """
        Build bond length restraints for the entire structure.

        Iterates through all chains and residues, matching atoms with the
        restraints defined in the CIF dictionary. Only includes bonds where
        both atoms are present in the residue.
        """

        def __build_bonds(residue):
            # Filter to only include bonds where both atoms are present
            atom_names = residue['name'].values
            usable_bonds = cif_bonds[
                cif_bonds['atom1'].isin(atom_names) & 
                cif_bonds['atom2'].isin(atom_names)
            ]
            
            if len(usable_bonds) == 0:
                return
            
            # Create a mapping from atom names to indices
            residue_indexed = residue.set_index('name')
            
            # Check for duplicate atom names in the index
            if residue_indexed.index.has_duplicates:
                resname = residue['resname'].iloc[0] if len(residue) > 0 else 'UNKNOWN'
                chain_id = residue['chainid'].iloc[0] if len(residue) > 0 else 'UNKNOWN'
                resseq = residue['resseq'].iloc[0] if len(residue) > 0 else 'UNKNOWN'
                duplicates = residue_indexed.index[residue_indexed.index.duplicated()].unique()
                if self.verbose > 2:
                    print(f"WARNING: Residue {resname} {chain_id} {resseq} has duplicate atom names: {list(duplicates)}")
                    print(f"  Residue:\n{residue[['name', 'altloc', 'index']]}")
                # Skip this residue
                return
            
            # Get atom indices
            idx1 = residue_indexed.loc[usable_bonds['atom1'], 'index'].values
            idx2 = residue_indexed.loc[usable_bonds['atom2'], 'index'].values
            
            # Get reference distances and sigmas (standardized column names)
            references = usable_bonds['value'].values.astype(float)
            sigmas = usable_bonds['sigma'].values.astype(float)
            
            # Replace zero sigmas with a small value to avoid division by zero
            sigmas[sigmas == 0] = 1e-4
            
            # Append to lists
            bond_idx1_list.append(idx1)
            bond_idx2_list.append(idx2)
            bond_ref_list.append(references)
            bond_sigma_list.append(sigmas)

        bond_idx1_list = []
        bond_idx2_list = []
        bond_ref_list = []
        bond_sigma_list = []
        
        pdb = self.model.pdb
        # Iterate through all chains
        for chain_id in pdb['chainid'].unique():
            chain = pdb[pdb['chainid'] == chain_id]
            
            # Iterate through all residues in the chain
            for resseq in chain['resseq'].unique():
                residue = pdb[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
                
                # Get residue name
                resname = residue['resname'].values[0]
                
                # Check if restraints exist for this residue type
                if resname not in self.cif_dict:
                    continue
                
                cif_residue = self.cif_dict[resname]
                
                # Use standardized key 'bonds'
                if 'bonds' not in cif_residue:
                    continue
                
                cif_bonds = cif_residue['bonds']
                # Check if there are alternative conformations (normalize '' and ' ' as no altloc)
                has_altloc = any(~residue['altloc'].isin(['', ' ']))
                if has_altloc:
                    for residue_alt in self.expand_altloc(residue):
                        __build_bonds(residue_alt)
                else:
                    __build_bonds(residue)

                
        
        # Concatenate all bond restraints
        if len(bond_idx1_list) > 0:
            if self.verbose > 2:
                print(f"DEBUG: Building bond restraints from {len(bond_idx1_list)} lists")
                for i, (arr1, arr2, ref, sig) in enumerate(zip(bond_idx1_list, bond_idx2_list, bond_ref_list, bond_sigma_list)):
                    if len(arr1) != len(arr2):
                        print(f"  WARNING: Mismatch at list {i}: idx1={len(arr1)}, idx2={len(arr2)}, ref={len(ref)}, sigma={len(sig)}")
            
            bond_idx1 = np.concatenate(bond_idx1_list, dtype=int)
            bond_idx2 = np.concatenate(bond_idx2_list, dtype=int)
            
            if self.verbose > 2:
                print(f"DEBUG: After concatenation: idx1={len(bond_idx1)}, idx2={len(bond_idx2)}")
            
            # Store in hierarchical dictionary
            self.restraints['bond']['intra'] = {
                'indices': torch.tensor(
                    np.stack([bond_idx1, bond_idx2], axis=1), 
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.concatenate(bond_ref_list), 
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.concatenate(bond_sigma_list), 
                    dtype=torch.float32
                )
            }
            
            assert (self.restraints['bond']['intra']['indices'].shape[0] == 
                    self.restraints['bond']['intra']['references'].shape[0] == 
                    self.restraints['bond']['intra']['sigmas'].shape[0]), \
                    "Inconsistent bond restraint shapes"
        else:
            print("Warning: No bond restraints were built")
    
    def _build_angle_restraints(self):
        def __build_angle(residue):
            atom_names = residue['name'].values
            usable_angles = cif_angles.loc[
                cif_angles['atom1'].isin(atom_names) & 
                cif_angles['atom2'].isin(atom_names) &
                cif_angles['atom3'].isin(atom_names)
            ]
            if len(usable_angles) == 0:
                return
            
            # Create a mapping from atom names to indices
            residue_indexed = residue.set_index('name')
            
            # Check for duplicate atom names in the index
            if residue_indexed.index.has_duplicates:
                if self.verbose > 2:
                    resname = residue['resname'].iloc[0] if len(residue) > 0 else 'UNKNOWN'
                    chain_id = residue['chainid'].iloc[0] if len(residue) > 0 else 'UNKNOWN'
                    resseq = residue['resseq'].iloc[0] if len(residue) > 0 else 'UNKNOWN'
                    print(f"WARNING: Skipping angle restraints for residue {resname} {chain_id} {resseq} (duplicate atom names)")
                return
            
            # Get atom indices
            idx1 = residue_indexed.loc[usable_angles['atom1'], 'index'].values
            idx2 = residue_indexed.loc[usable_angles['atom2'], 'index'].values
            idx3 = residue_indexed.loc[usable_angles['atom3'], 'index'].values
                
            # Get reference values and sigmas (standardized column names)
            references = usable_angles['value'].values.astype(float)
            sigmas = usable_angles['sigma'].values.astype(float)
            
            # Replace zero sigmas with a small value to avoid division by zero
            sigmas[sigmas == 0] = 1e-4
            
            # Append to lists
            angle_idx1_list.append(idx1)
            angle_idx2_list.append(idx2)
            angle_idx3_list.append(idx3)
            angle_ref_list.append(references)
            angle_sigma_list.append(sigmas)
        """
        Build angle restraints for the entire structure.

        Iterates through all chains and residues, matching atom triplets with
        the restraints defined in the CIF dictionary. Only includes angles where
        all three atoms are present in the residue.
        """
        angle_idx1_list = []
        angle_idx2_list = []
        angle_idx3_list = []
        angle_ref_list = []
        angle_sigma_list = []
        
        pdb = self.model.pdb
        
        # Iterate through all chains
        for chain_id in pdb['chainid'].unique():
            chain = pdb[pdb['chainid'] == chain_id]
            
            # Iterate through all residues in the chain
            for resseq in chain['resseq'].unique():
                residue = pdb[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
                            
                # Get residue name
                resname = residue['resname'].values[0]
                
                # Check if restraints exist for this residue type
                if resname not in self.cif_dict:
                    continue
                
                cif_residue = self.cif_dict[resname]
                
                # Use standardized key 'angles'
                if 'angles' not in cif_residue:
                    continue
                
                cif_angles = cif_residue['angles']

                # Handle alternate conformations (normalize '' and ' ' as no altloc)
                has_altloc = any(~residue['altloc'].isin(['', ' ']))
                if has_altloc:
                    for residue_alt in self.expand_altloc(residue):
                        __build_angle(residue_alt)
                else:
                    __build_angle(residue)

        # Concatenate all angle restraints
        if len(angle_idx1_list) > 0:
            angle_idx1 = np.concatenate(angle_idx1_list, dtype=int)
            angle_idx2 = np.concatenate(angle_idx2_list, dtype=int)
            angle_idx3 = np.concatenate(angle_idx3_list, dtype=int)
            
            # Store in hierarchical dictionary
            self.restraints['angle']['intra'] = {
                'indices': torch.tensor(
                    np.stack([angle_idx1, angle_idx2, angle_idx3], axis=1), 
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.concatenate(angle_ref_list), 
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.concatenate(angle_sigma_list), 
                    dtype=torch.float32
                )
            }
            
            assert (self.restraints['angle']['intra']['indices'].shape[0] == 
                    self.restraints['angle']['intra']['references'].shape[0] == 
                    self.restraints['angle']['intra']['sigmas'].shape[0]), \
                    "Inconsistent angle restraint shapes"
        else:
            print("Warning: No angle restraints were built")
    
    def _build_torsion_restraints(self):
        def __build_torsion(residue):
            atom_names = residue['name'].values
            usable_torsions = cif_torsions[
                cif_torsions['atom1'].isin(atom_names) & 
                cif_torsions['atom2'].isin(atom_names) &
                cif_torsions['atom3'].isin(atom_names) &
                cif_torsions['atom4'].isin(atom_names)
            ]
            
            if len(usable_torsions) == 0:
                return
            
            # Create a mapping from atom names to indices
            residue_indexed = residue.set_index('name')
            
            # Check for duplicate atom names in the index
            if residue_indexed.index.has_duplicates:
                if self.verbose > 2:
                    resname = residue['resname'].iloc[0] if len(residue) > 0 else 'UNKNOWN'
                    chain_id = residue['chainid'].iloc[0] if len(residue) > 0 else 'UNKNOWN'
                    resseq = residue['resseq'].iloc[0] if len(residue) > 0 else 'UNKNOWN'
                    print(f"WARNING: Skipping torsion restraints for residue {resname} {chain_id} {resseq} (duplicate atom names)")
                return
            
            # Get atom indices
            idx1 = residue_indexed.loc[usable_torsions['atom1'], 'index'].values
            idx2 = residue_indexed.loc[usable_torsions['atom2'], 'index'].values
            idx3 = residue_indexed.loc[usable_torsions['atom3'], 'index'].values
            idx4 = residue_indexed.loc[usable_torsions['atom4'], 'index'].values
            
            # Get reference torsion angles, sigmas, and periods (standardized column names)
            references = usable_torsions['value'].values.astype(float)
            sigmas = usable_torsions['sigma'].values.astype(float)
            
            # Get period if available, default to 1 if not present
            if 'periodicity' in usable_torsions.columns:
                periods = usable_torsions['periodicity'].values.astype(int)
            else:
                periods = np.ones(len(usable_torsions), dtype=int)
            
            # Safety check: ensure no zero sigmas remain
            if np.any(sigmas == 0):
                if self.verbose > 4:
                    print(f"Warning: Found {(sigmas == 0).sum()} torsions with sigma=0 after filtering, setting to 0.1°")
                flagged = sigmas == 0
                idx1 = idx1[~flagged]
                idx2 = idx2[~flagged]
                idx3 = idx3[~flagged]
                idx4 = idx4[~flagged]
                references = references[~flagged]
                periods = periods[~flagged]
                sigmas = sigmas[~flagged]
                # sigmas[sigmas == 0] = 0.01
            
            # Skip if no valid torsions remain
            if len(idx1) == 0:
                return
            
            # Append to lists
            torsion_idx1_list.append(idx1)
            torsion_idx2_list.append(idx2)
            torsion_idx3_list.append(idx3)
            torsion_idx4_list.append(idx4)
            torsion_ref_list.append(references)
            torsion_sigma_list.append(sigmas)
            torsion_period_list.append(periods)
        """
        Build torsion angle restraints for the entire structure.

        Iterates through all chains and residues, matching atom quartets with
        the restraints defined in the CIF dictionary. Only includes torsions where
        all four atoms are present in the residue.
        """
        torsion_idx1_list = []
        torsion_idx2_list = []
        torsion_idx3_list = []
        torsion_idx4_list = []
        torsion_ref_list = []
        torsion_sigma_list = []
        torsion_period_list = []
        
        pdb = self.model.pdb
        
        # Iterate through all chains
        for chain_id in pdb['chainid'].unique():
            chain = pdb[pdb['chainid'] == chain_id]
            
            # Iterate through all residues in the chain
            for resseq in chain['resseq'].unique():
                residue = pdb[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
                
                # Get residue name
                resname = residue['resname'].values[0]
                
                # Check if restraints exist for this residue type
                if resname not in self.cif_dict:
                    continue
                
                cif_residue = self.cif_dict[resname]
                
                # Use standardized key 'torsions'
                if 'torsions' not in cif_residue:
                    continue
                
                cif_torsions = cif_residue['torsions']
                # Handle alternate conformations (normalize '' and ' ' as no altloc)
                has_altloc = any(~residue['altloc'].isin(['', ' ']))
                if has_altloc:
                    for residue_alt in self.expand_altloc(residue):
                        __build_torsion(residue_alt)
                else:
                    __build_torsion(residue)
                # Filter to only include torsions where all atoms are present

        
        # Concatenate all torsion restraints
        if len(torsion_idx1_list) > 0:
            torsion_idx1 = np.concatenate(torsion_idx1_list, dtype=int)
            torsion_idx2 = np.concatenate(torsion_idx2_list, dtype=int)
            torsion_idx3 = np.concatenate(torsion_idx3_list, dtype=int)
            torsion_idx4 = np.concatenate(torsion_idx4_list, dtype=int)
            
            # Store in hierarchical dictionary
            self.restraints['torsion']['intra'] = {
                'indices': torch.tensor(
                    np.stack([torsion_idx1, torsion_idx2, torsion_idx3, torsion_idx4], axis=1), 
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.concatenate(torsion_ref_list), 
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.concatenate(torsion_sigma_list), 
                    dtype=torch.float32
                ),
                'periods': torch.tensor(
                    np.concatenate(torsion_period_list),
                    dtype=torch.long
                )
            }
            
            assert (self.restraints['torsion']['intra']['indices'].shape[0] == 
                    self.restraints['torsion']['intra']['references'].shape[0] == 
                    self.restraints['torsion']['intra']['sigmas'].shape[0] == 
                    self.restraints['torsion']['intra']['periods'].shape[0]), \
                    "Inconsistent torsion restraint shapes"
        else:
            print("Warning: No torsion restraints were built")
    
    def _build_plane_restraints(self):
        """
        Build plane restraints for the entire structure.

        Planes have variable atom counts (minimum 3 atoms required, typically 3-10 atoms).
        This method groups planes by their atom count and stores them in separate dictionaries:
        restraints['plane']['3_atoms'], restraints['plane']['4_atoms'], etc.

        Planes with fewer than 3 atoms are skipped as they are mathematically invalid.

        Each plane is stored with:

        - indices: (N, max_atoms) tensor of atom indices (padded with -1 for unused slots)
        - sigmas: (N, max_atoms) tensor of sigma values
        - atom_counts: (N,) tensor indicating how many atoms each plane actually has
        """
        # Dictionary to collect planes by atom count
        # planes_by_count[atom_count] = {'atom_indices': [...], 'sigmas': [...]}
        planes_by_count = {}
        
        pdb = self.model.pdb
        
        # Iterate through all chains
        for chain_id in pdb['chainid'].unique():
            chain = pdb[pdb['chainid'] == chain_id]
            
            # Iterate through all residues in the chain
            for resseq in chain['resseq'].unique():
                residue = pdb[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
                
                # Get residue name
                resname = residue['resname'].values[0]
                
                # Check if restraints exist for this residue type
                if resname not in self.cif_dict:
                    continue
                
                cif_residue = self.cif_dict[resname]
                
                # Use standardized key 'planes'
                if 'planes' not in cif_residue:
                    continue
                
                cif_planes = cif_residue['planes']
                
                # Handle alternate conformations (normalize '' and ' ' as no altloc)
                has_altloc = any(~residue['altloc'].isin(['', ' ']))
                residue_variants = []
                if has_altloc:
                    residue_variants = list(self.expand_altloc(residue))
                else:
                    residue_variants = [residue]
                
                for residue_alt in residue_variants:
                    atom_names = residue_alt['name'].values
                    
                    # Filter to only include plane atoms that are present (standardized column name 'atom')
                    usable_planes = cif_planes[cif_planes['atom'].isin(atom_names)]
                    
                    if len(usable_planes) == 0:
                        continue
                    
                    # Create a mapping from atom names to indices
                    residue_indexed = residue_alt.set_index('name')
                    
                    # Check for duplicate atom names in the index
                    if residue_indexed.index.has_duplicates:
                        if self.verbose > 2:
                            resname = residue_alt['resname'].iloc[0] if len(residue_alt) > 0 else 'UNKNOWN'
                            chain_id = residue_alt['chainid'].iloc[0] if len(residue_alt) > 0 else 'UNKNOWN'
                            resseq = residue_alt['resseq'].iloc[0] if len(residue_alt) > 0 else 'UNKNOWN'
                            print(f"WARNING: Skipping plane restraints for residue {resname} {chain_id} {resseq} (duplicate atom names)")
                        continue
                    
                    # Group by plane_id
                    for plane_id in usable_planes['plane_id'].unique():
                        plane_atoms = usable_planes[usable_planes['plane_id'] == plane_id]
                        
                        # Get atom indices for this plane (standardized column name 'atom')
                        atom_ids = plane_atoms['atom'].values
                        atom_indices = residue_indexed.loc[atom_ids, 'index'].values
                        
                        # Get sigma values from CIF data (standardized column name 'sigma')
                        # These are clipped to minimum 0.1 Å in the CIF reader
                        sigmas = plane_atoms['sigma'].values.astype(float)
                        
                        # Get the atom count for this plane
                        atom_count = len(atom_indices)
                        
                        # Skip planes with fewer than 3 atoms (invalid)
                        if atom_count < 3:
                            continue
                        
                        # Initialize the list for this atom count if needed
                        if atom_count not in planes_by_count:
                            planes_by_count[atom_count] = {
                                'atom_indices': [],
                                'sigmas': []
                            }
                        
                        # Append to the appropriate list
                        planes_by_count[atom_count]['atom_indices'].append(atom_indices)
                        planes_by_count[atom_count]['sigmas'].append(sigmas)
        
        # Convert to tensors and store in hierarchical structure
        if len(planes_by_count) == 0:
            if self.verbose > 1:
                print("No plane restraints were built")
            return
        
        for atom_count, plane_data in planes_by_count.items():
            atom_indices_list = plane_data['atom_indices']
            sigmas_list = plane_data['sigmas']
            
            if len(atom_indices_list) > 0:
                # Stack into arrays
                indices_array = np.stack(atom_indices_list, axis=0)
                sigmas_array = np.stack(sigmas_list, axis=0)
                
                # Store in hierarchical dictionary
                key = f'{atom_count}_atoms'
                self.restraints['plane'][key] = {
                    'indices': torch.tensor(indices_array, dtype=torch.long),
                    'sigmas': torch.tensor(sigmas_array, dtype=torch.float32)
                }
                
                if self.verbose > 1:
                    print(f"Built {len(atom_indices_list)} plane restraints with {atom_count} atoms")
    
    def _build_chiral_restraints(self, ideal_volume: float = 2.5, sigma: float = 0.2):
        """
        Build chiral volume restraints for the entire structure.

        Chiral centers are always exactly 4 atoms: 1 center + 3 neighbors forming
        a tetrahedron. The chiral volume is computed as:

            V = v1 · (v2 × v3)

        where vi = position of neighbor i - position of center.

        The sign of the volume determines the handedness (R vs S configuration).
        For L-amino acids, the Cα chiral volume should be positive (with standard
        atom ordering: N, C, CB).

        Parameters
        ----------
        ideal_volume : float, default 2.5
            Target absolute chiral volume in Å³.
        sigma : float, default 0.2
            Standard deviation for restraint in Å³.

        Notes
        -----
        Stores the following in restraints['chiral']:

        - 'indices': (N, 4) tensor [center, atom1, atom2, atom3]
        - 'ideal_volumes': (N,) tensor of signed ideal volumes
        - 'sigmas': (N,) tensor of sigma values
        """
        chiral_indices = []  # List of [center, atom1, atom2, atom3]
        chiral_volumes = []  # Signed ideal volumes
        chiral_sigmas = []
        
        pdb = self.model.pdb
        
        # Iterate through all chains
        for chain_id in pdb['chainid'].unique():
            chain = pdb[pdb['chainid'] == chain_id]
            
            # Iterate through all residues in the chain
            for resseq in chain['resseq'].unique():
                residue = pdb[(pdb['resseq'] == resseq) & (pdb['chainid'] == chain_id)]
                
                # Get residue name
                resname = residue['resname'].values[0]
                
                # Check if restraints exist for this residue type
                if resname not in self.cif_dict:
                    continue
                
                cif_residue = self.cif_dict[resname]
                
                # Use standardized key 'chirals'
                if 'chirals' not in cif_residue:
                    continue
                
                cif_chirals = cif_residue['chirals']
                if cif_chirals.empty:
                    continue
                
                # Handle alternate conformations
                has_altloc = any(~residue['altloc'].isin(['', ' ']))
                residue_variants = list(self.expand_altloc(residue)) if has_altloc else [residue]
                
                for residue_alt in residue_variants:
                    atom_names = residue_alt['name'].values
                    
                    # Create a mapping from atom names to indices
                    residue_indexed = residue_alt.set_index('name')
                    
                    # Check for duplicate atom names
                    if residue_indexed.index.has_duplicates:
                        if self.verbose > 2:
                            print(f"WARNING: Skipping chiral restraints for residue "
                                  f"{resname} {chain_id} {resseq} (duplicate atom names)")
                        continue
                    
                    # Process each chiral center
                    for _, chiral_row in cif_chirals.iterrows():
                        center = chiral_row['atom_centre']
                        atom1 = chiral_row['atom1']
                        atom2 = chiral_row['atom2']
                        atom3 = chiral_row['atom3']
                        volume_sign = chiral_row['volume_sign']
                        
                        # Check all atoms are present
                        if not all(a in atom_names for a in [center, atom1, atom2, atom3]):
                            continue
                        
                        # Get atom indices
                        try:
                            idx_center = residue_indexed.loc[center, 'index']
                            idx1 = residue_indexed.loc[atom1, 'index']
                            idx2 = residue_indexed.loc[atom2, 'index']
                            idx3 = residue_indexed.loc[atom3, 'index']
                        except KeyError:
                            continue
                        
                        # Determine signed ideal volume based on chirality
                        if volume_sign == 'positive':
                            signed_volume = ideal_volume
                        elif volume_sign == 'negative':
                            signed_volume = -ideal_volume
                        elif volume_sign in ['both', 'either']:
                            # Achiral or racemic - skip restraint or use 0
                            # For 'both', we don't restrain the sign
                            signed_volume = 0.0  # Will use absolute value comparison
                        else:
                            # Unknown sign, skip
                            if self.verbose > 2:
                                print(f"WARNING: Unknown chiral sign '{volume_sign}' "
                                      f"for {resname} {chain_id} {resseq}")
                            continue
                        
                        chiral_indices.append([idx_center, idx1, idx2, idx3])
                        chiral_volumes.append(signed_volume)
                        chiral_sigmas.append(sigma)
        
        # Convert to tensors and store
        if len(chiral_indices) > 0:
            self.restraints['chiral'] = {
                'indices': torch.tensor(chiral_indices, dtype=torch.long),
                'ideal_volumes': torch.tensor(chiral_volumes, dtype=torch.float32),
                'sigmas': torch.tensor(chiral_sigmas, dtype=torch.float32)
            }
            
            if self.verbose > 0:
                n_positive = sum(1 for v in chiral_volumes if v > 0)
                n_negative = sum(1 for v in chiral_volumes if v < 0)
                n_both = sum(1 for v in chiral_volumes if v == 0)
                print(f"  Built {len(chiral_indices)} chiral restraints "
                      f"(+: {n_positive}, -: {n_negative}, both: {n_both})")
        else:
            self.restraints['chiral'] = {
                'indices': torch.tensor([], dtype=torch.long).reshape(0, 4),
                'ideal_volumes': torch.tensor([], dtype=torch.float32),
                'sigmas': torch.tensor([], dtype=torch.float32)
            }
            if self.verbose > 0:
                print("  No chiral restraints were built")
    
    def _build_peptide_bond_restraints(self):
        """
        Build peptide bond restraints between consecutive residues.

        This method iterates through all chains and identifies consecutive residues
        (resseq and resseq+1). For each consecutive pair, it adds TRANS peptide
        bond, angle, and torsion restraints.

        The restraints are stored separately from intra-residue restraints in:

        - bond_indices_inter, angle_indices_inter, torsion_indices_inter
        - bond_references_inter, angle_references_inter, torsion_references_inter
        - bond_sigmas_inter, angle_sigmas_inter, torsion_sigmas_inter
        """
        if 'TRANS' not in self.link_dict:
            if self.verbose > 0:
                print("Warning: TRANS link not found in link dictionary, skipping peptide bonds")
            return
        
        trans_link = self.link_dict['TRANS']
        
        # Get bond, angle, torsion, and plane definitions
        trans_bonds = trans_link.get('bonds', None)
        trans_angles = trans_link.get('angles', None)
        trans_torsions = trans_link.get('torsions', None)
        trans_planes = trans_link.get('planes', None)
        
        if trans_bonds is None:
            if self.verbose > 0:
                print("Warning: No bond restraints in TRANS link definition")
            return
        
        # Get the C-N bond parameters
        c_n_bond = trans_bonds[
            (trans_bonds['atom_1_comp_id'] == '1') & 
            (trans_bonds['atom1'] == 'C') &
            (trans_bonds['atom_2_comp_id'] == '2') & 
            (trans_bonds['atom2'] == 'N')
        ]
        
        if len(c_n_bond) == 0:
            if self.verbose > 0:
                print("Warning: C-N bond not found in TRANS link definition")
            return
        
        # Get bond parameters (standardized column names)
        bond_length = float(c_n_bond['value'].values[0])
        bond_sigma = float(c_n_bond['sigma'].values[0])
        
        if self.verbose > 1:
            print(f"Peptide bond parameters: C-N = {bond_length:.3f} Å ± {bond_sigma:.4f} Å")
        
        # Lists to accumulate peptide bond restraints
        bond_idx1_list = []
        bond_idx2_list = []
        bond_ref_list = []
        bond_sigma_list = []
        
        # Lists for angle restraints
        angle_idx1_list = []
        angle_idx2_list = []
        angle_idx3_list = []
        angle_ref_list = []
        angle_sigma_list = []
        
        # Lists for backbone torsion angles (phi, psi, omega tracked separately)
        phi_idx1_list = []
        phi_idx2_list = []
        phi_idx3_list = []
        phi_idx4_list = []
        phi_period_list = []
        
        psi_idx1_list = []
        psi_idx2_list = []
        psi_idx3_list = []
        psi_idx4_list = []
        psi_period_list = []
        
        omega_idx1_list = []
        omega_idx2_list = []
        omega_idx3_list = []
        omega_idx4_list = []
        omega_ref_list = []
        omega_sigma_list = []
        omega_period_list = []
        omega_is_proline_list = []  # Track if following residue is proline
        
        # Lists for plane restraints (organized by plane_id)
        # TRANS has plan-1 and plan-2, each with 4 atoms
        plane_indices_by_id = {}  # Dict of plane_id -> list of atom index lists
        plane_sigma_by_id = {}    # Dict of plane_id -> sigma value
        
        # Parse plane definitions if available
        if trans_planes is not None:
            # Group planes by plane_id
            for plane_id in trans_planes['plane_id'].unique():
                plane_atoms = trans_planes[trans_planes['plane_id'] == plane_id]
                plane_indices_by_id[plane_id] = []
                # All atoms in same plane should have same sigma (standardized column name)
                plane_sigma_by_id[plane_id] = float(plane_atoms['sigma'].values[0])
                
                if self.verbose > 1:
                    atom_list = ', '.join([f"{row['atom_comp_id']}:{row['atom']}" 
                                          for _, row in plane_atoms.iterrows()])
                    print(f"Plane {plane_id}: {atom_list} (σ={plane_sigma_by_id[plane_id]:.3f} Å)")
        
        pdb = self.model.pdb
        
        # Only consider protein residues (ATOM records, not HETATM)
        protein_atoms = pdb[pdb['ATOM'] == 'ATOM']
        
        # Iterate through all chains
        for chain_id in protein_atoms['chainid'].unique():
            chain = protein_atoms[protein_atoms['chainid'] == chain_id]
            
            # Get sorted list of residue sequence numbers
            resseq_list = sorted(chain['resseq'].unique())
            
            # Iterate through consecutive residue pairs
            for i in range(len(resseq_list) - 1):
                resseq_i = resseq_list[i]
                resseq_next = resseq_list[i + 1]
                
                # Skip if not consecutive (chain break)
                if resseq_next != resseq_i + 1:
                    if self.verbose > 1:
                        print(f"Skipping chain break: {chain_id}:{resseq_i} → {chain_id}:{resseq_next}")
                    continue
                
                # Get residues
                residue_i = chain[chain['resseq'] == resseq_i]
                residue_next = chain[chain['resseq'] == resseq_next]
                
                # Get residue names
                resname_next = residue_next['resname'].values[0]
                is_proline = (resname_next == 'PRO')
                
                # Create atom lookup dictionaries for both residues
                # Handle alternate conformations by preferring ' ' then 'A'
                def get_atom_index(residue, atom_name):
                    atoms = residue[residue['name'] == atom_name]
                    if len(atoms) == 0:
                        return None
                    if ' ' in atoms['altloc'].values:
                        return atoms[atoms['altloc'] == ' '].iloc[0]['index']
                    elif 'A' in atoms['altloc'].values:
                        return atoms[atoms['altloc'] == 'A'].iloc[0]['index']
                    else:
                        return atoms.iloc[0]['index']
                
                # Get atom indices for bond (C-N)
                c_idx = get_atom_index(residue_i, 'C')
                n_idx = get_atom_index(residue_next, 'N')
                
                if c_idx is None or n_idx is None:
                    continue
                
                # Add peptide bond restraint
                bond_idx1_list.append(c_idx)
                bond_idx2_list.append(n_idx)
                bond_ref_list.append(bond_length)
                bond_sigma_list.append(bond_sigma)
                
                # Add angle restraints if available
                if trans_angles is not None:
                    for _, angle_row in trans_angles.iterrows():
                        comp1 = angle_row['atom_1_comp_id']
                        comp2 = angle_row['atom_2_comp_id']
                        comp3 = angle_row['atom_3_comp_id']
                        atom1_name = angle_row['atom1']
                        atom2_name = angle_row['atom2']
                        atom3_name = angle_row['atom3']
                        
                        # Get atom indices based on which residue they belong to
                        res1 = residue_i if comp1 == '1' else residue_next
                        res2 = residue_i if comp2 == '1' else residue_next
                        res3 = residue_i if comp3 == '1' else residue_next
                        
                        idx1 = get_atom_index(res1, atom1_name)
                        idx2 = get_atom_index(res2, atom2_name)
                        idx3 = get_atom_index(res3, atom3_name)
                        
                        if idx1 is not None and idx2 is not None and idx3 is not None:
                            angle_idx1_list.append(idx1)
                            angle_idx2_list.append(idx2)
                            angle_idx3_list.append(idx3)
                            angle_ref_list.append(float(angle_row['value']))
                            angle_sigma_list.append(float(angle_row['sigma']))
                
                # Add torsion restraints if available - separate phi, psi, and omega
                if trans_torsions is not None:
                    for _, torsion_row in trans_torsions.iterrows():
                        comp1 = torsion_row['atom_1_comp_id']
                        comp2 = torsion_row['atom_2_comp_id']
                        comp3 = torsion_row['atom_3_comp_id']
                        comp4 = torsion_row['atom_4_comp_id']
                        atom1_name = torsion_row['atom1']
                        atom2_name = torsion_row['atom2']
                        atom3_name = torsion_row['atom3']
                        atom4_name = torsion_row['atom4']
                        torsion_id = torsion_row['id']
                        
                        # Get atom indices based on which residue they belong to
                        res1 = residue_i if comp1 == '1' else residue_next
                        res2 = residue_i if comp2 == '1' else residue_next
                        res3 = residue_i if comp3 == '1' else residue_next
                        res4 = residue_i if comp4 == '1' else residue_next
                        
                        idx1 = get_atom_index(res1, atom1_name)
                        idx2 = get_atom_index(res2, atom2_name)
                        idx3 = get_atom_index(res3, atom3_name)
                        idx4 = get_atom_index(res4, atom4_name)
                        
                        if idx1 is not None and idx2 is not None and idx3 is not None and idx4 is not None:
                            # Get period if available, default to 0
                            period = int(torsion_row['period']) if 'period' in torsion_row and pd.notna(torsion_row['period']) else 0
                            
                            # Separate phi, psi, and omega angles
                            if torsion_id == 'psi':
                                # psi: N(i) - CA(i) - C(i) - N(i+1)
                                psi_idx1_list.append(idx1)
                                psi_idx2_list.append(idx2)
                                psi_idx3_list.append(idx3)
                                psi_idx4_list.append(idx4)
                                psi_period_list.append(period)
                            elif torsion_id == 'phi':
                                # phi: C(i-1) - N(i) - CA(i) - C(i)
                                # In our case: C(i) - N(i+1) - CA(i+1) - C(i+1)
                                phi_idx1_list.append(idx1)
                                phi_idx2_list.append(idx2)
                                phi_idx3_list.append(idx3)
                                phi_idx4_list.append(idx4)
                                phi_period_list.append(period)
                            elif torsion_id == 'omega':
                                # omega: CA(i) - C(i) - N(i+1) - CA(i+1)
                                omega_idx1_list.append(idx1)
                                omega_idx2_list.append(idx2)
                                omega_idx3_list.append(idx3)
                                omega_idx4_list.append(idx4)
                                omega_ref_list.append(float(torsion_row['value']))
                                omega_sigma_list.append(float(torsion_row['sigma']))
                                omega_period_list.append(period)
                                omega_is_proline_list.append(is_proline)
                
                # Add plane restraints if available
                if trans_planes is not None:
                    for plane_id in plane_indices_by_id.keys():
                        plane_atoms = trans_planes[trans_planes['plane_id'] == plane_id]
                        
                        # Collect atom indices for this plane
                        atom_indices = []
                        all_found = True
                        
                        for _, plane_atom_row in plane_atoms.iterrows():
                            comp_id = plane_atom_row['atom_comp_id']
                            atom_name = plane_atom_row['atom']
                            
                            # Get the correct residue based on comp_id
                            residue = residue_i if comp_id == '1' else residue_next
                            
                            atom_idx = get_atom_index(residue, atom_name)
                            
                            if atom_idx is None:
                                # Skip this plane if any atom is missing (e.g., H in proline)
                                all_found = False
                                break
                            
                            atom_indices.append(atom_idx)
                        
                        # Add plane if all atoms were found
                        if all_found:
                            plane_indices_by_id[plane_id].append(atom_indices)
        
        # Create tensors for inter-residue bond restraints (peptide bonds)
        if len(bond_idx1_list) > 0:
            self.restraints['bond']['peptide'] = {
                'indices': torch.tensor(
                    np.stack([np.array(bond_idx1_list), np.array(bond_idx2_list)], axis=1),
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.array(bond_ref_list),
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.array(bond_sigma_list),
                    dtype=torch.float32
                )
            }
            
            if self.verbose > 0:
                print(f"Built {len(bond_idx1_list)} peptide bond restraints")
        else:
            if self.verbose > 0:
                print("Warning: No peptide bond restraints were built")
        
        # Create tensors for inter-residue angle restraints (peptide angles)
        if len(angle_idx1_list) > 0:
            self.restraints['angle']['peptide'] = {
                'indices': torch.tensor(
                    np.stack([np.array(angle_idx1_list), np.array(angle_idx2_list), np.array(angle_idx3_list)], axis=1),
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.array(angle_ref_list),
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.array(angle_sigma_list),
                    dtype=torch.float32
                )
            }
            
            if self.verbose > 0:
                print(f"Built {len(angle_idx1_list)} peptide angle restraints")
        
        # Create tensors for backbone torsion angles (phi, psi, omega)
        if len(phi_idx1_list) > 0:
            self.restraints['torsion']['phi'] = {
                'indices': torch.tensor(
                    np.stack([np.array(phi_idx1_list), np.array(phi_idx2_list), 
                             np.array(phi_idx3_list), np.array(phi_idx4_list)], axis=1),
                    dtype=torch.long
                ),
                'periods': torch.tensor(
                    np.array(phi_period_list),
                    dtype=torch.long
                )
            }
            if self.verbose > 2:
                print(f"Built {len(phi_idx1_list)} phi angle indices (C-N-CA-C)")
        
        if len(psi_idx1_list) > 0:
            self.restraints['torsion']['psi'] = {
                'indices': torch.tensor(
                    np.stack([np.array(psi_idx1_list), np.array(psi_idx2_list), 
                             np.array(psi_idx3_list), np.array(psi_idx4_list)], axis=1),
                    dtype=torch.long
                ),
                'periods': torch.tensor(
                    np.array(psi_period_list),
                    dtype=torch.long
                )
            }
            if self.verbose > 2:
                print(f"Built {len(psi_idx1_list)} psi angle indices (N-CA-C-N)")
        
        if len(omega_idx1_list) > 2:
            self.restraints['torsion']['omega'] = {
                'indices': torch.tensor(
                    np.stack([np.array(omega_idx1_list), np.array(omega_idx2_list), 
                             np.array(omega_idx3_list), np.array(omega_idx4_list)], axis=1),
                    dtype=torch.long
                ),
                'references': torch.tensor(
                    np.array(omega_ref_list),
                    dtype=torch.float32
                ),
                'sigmas': torch.tensor(
                    np.array(omega_sigma_list),
                    dtype=torch.float32
                ),
                'periods': torch.tensor(
                    np.array(omega_period_list),
                    dtype=torch.long
                ),
                'is_proline': torch.tensor(
                    np.array(omega_is_proline_list),
                    dtype=torch.bool
                )
            }
            if self.verbose > 2:
                n_proline = sum(omega_is_proline_list)
                print(f"Built {len(omega_idx1_list)} omega angle restraints (CA-C-N-CA, ~180°)")
                print(f"  {n_proline} omega angles preceding proline residues")
        
        # Create tensors for peptide plane restraints
        # Combine all planes from different plane_ids (plan-1, plan-2, etc.)
        if trans_planes is not None and len(plane_indices_by_id) > 0:
            all_plane_indices = []
            all_plane_sigmas = []
            
            for plane_id, plane_list in plane_indices_by_id.items():
                if len(plane_list) > 0:
                    sigma = plane_sigma_by_id[plane_id]
                    for atom_indices in plane_list:
                        all_plane_indices.append(atom_indices)
                        # Create sigma array with same length as number of atoms
                        all_plane_sigmas.append([sigma] * len(atom_indices))
            
            if len(all_plane_indices) > 0:
                # Determine atom count (should be 4 for peptide planes)
                atom_count = len(all_plane_indices[0])
                key = f'{atom_count}_atoms'
                
                # Check if this key exists, if not create it, otherwise append
                plane_indices_array = np.array(all_plane_indices, dtype=np.int64)
                plane_sigmas_array = np.array(all_plane_sigmas, dtype=np.float32)
                
                if key in self.restraints['plane']:
                    # Append to existing planes
                    existing_indices = self.restraints['plane'][key]['indices'].cpu().numpy()
                    existing_sigmas = self.restraints['plane'][key]['sigmas'].cpu().numpy()
                    
                    plane_indices_array = np.vstack([existing_indices, plane_indices_array])
                    plane_sigmas_array = np.vstack([existing_sigmas, plane_sigmas_array])
                
                self.restraints['plane'][key] = {
                    'indices': torch.tensor(plane_indices_array, dtype=torch.long),
                    'sigmas': torch.tensor(plane_sigmas_array, dtype=torch.float32)
                }
                
                if self.verbose > 0:
                    print(f"Built {len(all_plane_indices)} peptide plane restraints ({atom_count} atoms each)")
                    for plane_id, plane_list in plane_indices_by_id.items():
                        if len(plane_list) > 0:
                            print(f"  {plane_id}: {len(plane_list)} planes")
    
    def _build_disulfide_bond_restraints(self):
        """
        Build disulfide bond restraints between cysteine residues.

        This method identifies disulfide bonds by:

        1. Finding all cysteine residues (or residues with SG atoms)
        2. Computing distances between all pairs of SG atoms
        3. Creating disulfide restraints for SG-SG pairs closer than 2.0 Å

        The restraints are appended to the existing inter-residue restraints.
        This includes bonds, angles (CB-SG-SG), and torsions (CB-SG-SG-CB).
        """
        if 'disulf' not in self.link_dict:
            if self.verbose > 0:
                print("Warning: disulf link not found in link dictionary, skipping disulfide bonds")
            return
        
        disulf_link = self.link_dict['disulf']
        
        # Get bond, angle, and torsion definitions
        disulf_bonds = disulf_link.get('bonds', None)
        disulf_angles = disulf_link.get('angles', None)
        disulf_torsions = disulf_link.get('torsions', None)
        
        if disulf_bonds is None:
            if self.verbose > 0:
                print("Warning: No bond restraints in disulf link definition")
            return
        
        # Get the SG-SG bond parameters
        sg_sg_bond = disulf_bonds[
            (disulf_bonds['atom1'] == 'SG') & 
            (disulf_bonds['atom2'] == 'SG')
        ]
        
        if len(sg_sg_bond) == 0:
            if self.verbose > 0:
                print("Warning: SG-SG bond not found in disulf link definition")
            return
        
        # Get bond parameters (standardized column names)
        bond_length = float(sg_sg_bond['value'].values[0])
        bond_sigma = float(sg_sg_bond['sigma'].values[0])
        
        if self.verbose > 1:
            print(f"Disulfide bond parameters: SG-SG = {bond_length:.3f} Å ± {bond_sigma:.4f} Å")
        
        # Find all SG atoms (from cysteine residues)
        pdb = self.model.pdb
        sg_atoms = pdb[(pdb['name'] == 'SG') & (pdb['ATOM'] == 'ATOM')]
        
        if len(sg_atoms) == 0:
            if self.verbose > 1:
                print("No SG atoms found, skipping disulfide bonds")
            return
        
        if self.verbose > 1:
            print(f"Found {len(sg_atoms)} SG atoms")
        
        # Get coordinates of all SG atoms
        xyz = self.model.xyz()
        sg_indices = sg_atoms['index'].values
        sg_coords = xyz[sg_indices]
        
        # Get residue identifiers (chain + resseq) for each SG atom
        sg_residues = (sg_atoms['chainid'].astype(str) + '_' + sg_atoms['resseq'].astype(str)).values
        
        # Compute pairwise distances between all SG atoms
        distances = torch.cdist(sg_coords, sg_coords)
        
        # Find pairs closer than 4.0 Å (but not the same atom)
        threshold = 4.0
        close_pairs = torch.where((distances < threshold) & (distances > 0.1))
        
        # Filter out pairs from the same residue and only keep i < j
        valid_pairs = []
        for i, j in zip(close_pairs[0].cpu().numpy(), close_pairs[1].cpu().numpy()):
            if i < j and sg_residues[i] != sg_residues[j]:
                valid_pairs.append((i, j))
        
        if len(valid_pairs) == 0:
            if self.verbose > 1:
                print("No disulfide bonds found (no SG atoms from different residues within 2.0 Å)")
            return
        
        # Lists for disulfide restraints
        bond_idx1_list = []
        bond_idx2_list = []
        bond_ref_list = []
        bond_sigma_list = []
        
        angle_idx1_list = []
        angle_idx2_list = []
        angle_idx3_list = []
        angle_ref_list = []
        angle_sigma_list = []
        
        torsion_idx1_list = []
        torsion_idx2_list = []
        torsion_idx3_list = []
        torsion_idx4_list = []
        torsion_ref_list = []
        torsion_sigma_list = []
        torsion_period_list = []
        
        # Process each disulfide bond
        for i_local, j_local in valid_pairs:
            sg1_idx = sg_indices[i_local]
            sg2_idx = sg_indices[j_local]
            
            # Get the residues containing these SG atoms
            residue1 = pdb[pdb['index'] == sg1_idx].iloc[0]
            residue2 = pdb[pdb['index'] == sg2_idx].iloc[0]
            
            chain1 = residue1['chainid']
            resseq1 = residue1['resseq']
            chain2 = residue2['chainid']
            resseq2 = residue2['resseq']
            
            # Get all atoms from both residues
            res1_atoms = pdb[(pdb['chainid'] == chain1) & (pdb['resseq'] == resseq1)]
            res2_atoms = pdb[(pdb['chainid'] == chain2) & (pdb['resseq'] == resseq2)]
            
            # Helper function to get atom index
            def get_atom_index(residue, atom_name):
                atoms = residue[residue['name'] == atom_name]
                if len(atoms) == 0:
                    return None
                if ' ' in atoms['altloc'].values:
                    return atoms[atoms['altloc'] == ' '].iloc[0]['index']
                elif 'A' in atoms['altloc'].values:
                    return atoms[atoms['altloc'] == 'A'].iloc[0]['index']
                else:
                    return atoms.iloc[0]['index']
            
            # Add bond restraint (SG-SG)
            bond_idx1_list.append(sg1_idx)
            bond_idx2_list.append(sg2_idx)
            bond_ref_list.append(bond_length)
            bond_sigma_list.append(bond_sigma)
            
            # Add angle restraints if available
            if disulf_angles is not None:
                for _, angle_row in disulf_angles.iterrows():
                    comp1 = angle_row['atom_1_comp_id']
                    comp2 = angle_row['atom_2_comp_id']
                    comp3 = angle_row['atom_3_comp_id']
                    atom1_name = angle_row['atom1']
                    atom2_name = angle_row['atom2']
                    atom3_name = angle_row['atom3']
                    
                    # Get atom indices based on which residue they belong to
                    res1 = res1_atoms if comp1 == '1' else res2_atoms
                    res2 = res1_atoms if comp2 == '1' else res2_atoms
                    res3 = res1_atoms if comp3 == '1' else res2_atoms
                    
                    idx1 = get_atom_index(res1, atom1_name)
                    idx2 = get_atom_index(res2, atom2_name)
                    idx3 = get_atom_index(res3, atom3_name)
                    
                    if idx1 is not None and idx2 is not None and idx3 is not None:
                        angle_idx1_list.append(idx1)
                        angle_idx2_list.append(idx2)
                        angle_idx3_list.append(idx3)
                        angle_ref_list.append(float(angle_row['value']))
                        angle_sigma_list.append(float(angle_row['sigma']))
            
            # Add torsion restraints if available (CB-SG-SG-CB)
            if disulf_torsions is not None:
                for _, torsion_row in disulf_torsions.iterrows():
                    comp1 = torsion_row['atom_1_comp_id']
                    comp2 = torsion_row['atom_2_comp_id']
                    comp3 = torsion_row['atom_3_comp_id']
                    comp4 = torsion_row['atom_4_comp_id']
                    atom1_name = torsion_row['atom1']
                    atom2_name = torsion_row['atom2']
                    atom3_name = torsion_row['atom3']
                    atom4_name = torsion_row['atom4']
                    
                    # Get atom indices based on which residue they belong to
                    res1_tor = res1_atoms if comp1 == '1' else res2_atoms
                    res2_tor = res1_atoms if comp2 == '1' else res2_atoms
                    res3_tor = res1_atoms if comp3 == '1' else res2_atoms
                    res4_tor = res1_atoms if comp4 == '1' else res2_atoms
                    
                    idx1 = get_atom_index(res1_tor, atom1_name)
                    idx2 = get_atom_index(res2_tor, atom2_name)
                    idx3 = get_atom_index(res3_tor, atom3_name)
                    idx4 = get_atom_index(res4_tor, atom4_name)
                    
                    if idx1 is not None and idx2 is not None and idx3 is not None and idx4 is not None:
                        torsion_idx1_list.append(idx1)
                        torsion_idx2_list.append(idx2)
                        torsion_idx3_list.append(idx3)
                        torsion_idx4_list.append(idx4)
                        torsion_ref_list.append(float(torsion_row['value']))
                        torsion_sigma_list.append(float(torsion_row['sigma']))
                        # Disulfide CB-SG-SG-CB torsion can be +90° or -90° (both valid)
                        # Use period=2 so both conformations are equivalent
                        torsion_period_list.append(2)
        
        # Append to existing inter-residue restraints
        # Bonds
        if len(bond_idx1_list) > 0:
            new_bond_indices = torch.tensor(
                np.stack([np.array(bond_idx1_list), np.array(bond_idx2_list)], axis=1),
                dtype=torch.long
            )
            new_bond_references = torch.tensor(
                np.array(bond_ref_list),
                dtype=torch.float32
            )
            new_bond_sigmas = torch.tensor(
                np.array(bond_sigma_list),
                dtype=torch.float32
            )
            
            # Append to peptide bonds (or create disulfide as separate origin)
            if 'disulfide' in self.restraints['bond']:
                # Append to existing disulfide bonds
                self.restraints['bond']['disulfide']['indices'] = torch.cat(
                    [self.restraints['bond']['disulfide']['indices'], new_bond_indices], dim=0)
                self.restraints['bond']['disulfide']['references'] = torch.cat(
                    [self.restraints['bond']['disulfide']['references'], new_bond_references], dim=0)
                self.restraints['bond']['disulfide']['sigmas'] = torch.cat(
                    [self.restraints['bond']['disulfide']['sigmas'], new_bond_sigmas], dim=0)
            else:
                # Create new disulfide bond restraints
                self.restraints['bond']['disulfide'] = {
                    'indices': new_bond_indices,
                    'references': new_bond_references,
                    'sigmas': new_bond_sigmas
                }
            
            if self.verbose > 0:
                print(f"Built {len(bond_idx1_list)} disulfide bond restraints")
                if self.verbose > 1:
                    # Show which residues are bonded
                    for idx1, idx2 in zip(bond_idx1_list, bond_idx2_list):
                        atom1 = pdb.iloc[idx1]
                        atom2 = pdb.iloc[idx2]
                        print(f"  Disulfide: {atom1['chainid']}:{atom1['resname']}{atom1['resseq']} -- "
                              f"{atom2['chainid']}:{atom2['resname']}{atom2['resseq']}")
        
        # Angles
        if len(angle_idx1_list) > 0:
            new_angle_indices = torch.tensor(
                np.stack([np.array(angle_idx1_list), np.array(angle_idx2_list), np.array(angle_idx3_list)], axis=1),
                dtype=torch.long
            )
            new_angle_references = torch.tensor(
                np.array(angle_ref_list),
                dtype=torch.float32
            )
            new_angle_sigmas = torch.tensor(
                np.array(angle_sigma_list),
                dtype=torch.float32
            )
            
            # Append to disulfide angles
            if 'disulfide' in self.restraints['angle']:
                self.restraints['angle']['disulfide']['indices'] = torch.cat(
                    [self.restraints['angle']['disulfide']['indices'], new_angle_indices], dim=0)
                self.restraints['angle']['disulfide']['references'] = torch.cat(
                    [self.restraints['angle']['disulfide']['references'], new_angle_references], dim=0)
                self.restraints['angle']['disulfide']['sigmas'] = torch.cat(
                    [self.restraints['angle']['disulfide']['sigmas'], new_angle_sigmas], dim=0)
            else:
                self.restraints['angle']['disulfide'] = {
                    'indices': new_angle_indices,
                    'references': new_angle_references,
                    'sigmas': new_angle_sigmas
                }
            
            if self.verbose > 0:
                print(f"Built {len(angle_idx1_list)} disulfide angle restraints")
        
        # Torsions
        if len(torsion_idx1_list) > 0:
            new_torsion_indices = torch.tensor(
                np.stack([np.array(torsion_idx1_list), np.array(torsion_idx2_list), 
                         np.array(torsion_idx3_list), np.array(torsion_idx4_list)], axis=1),
                dtype=torch.long
            )
            new_torsion_references = torch.tensor(
                np.array(torsion_ref_list),
                dtype=torch.float32
            )
            new_torsion_sigmas = torch.tensor(
                np.array(torsion_sigma_list),
                dtype=torch.float32
            )
            new_torsion_periods = torch.tensor(
                np.array(torsion_period_list),
                dtype=torch.long
            )
            
            # Append to disulfide torsions
            if 'disulfide' in self.restraints['torsion']:
                self.restraints['torsion']['disulfide']['indices'] = torch.cat(
                    [self.restraints['torsion']['disulfide']['indices'], new_torsion_indices], dim=0)
                self.restraints['torsion']['disulfide']['references'] = torch.cat(
                    [self.restraints['torsion']['disulfide']['references'], new_torsion_references], dim=0)
                self.restraints['torsion']['disulfide']['sigmas'] = torch.cat(
                    [self.restraints['torsion']['disulfide']['sigmas'], new_torsion_sigmas], dim=0)
                self.restraints['torsion']['disulfide']['periods'] = torch.cat(
                    [self.restraints['torsion']['disulfide']['periods'], new_torsion_periods], dim=0)
            else:
                self.restraints['torsion']['disulfide'] = {
                    'indices': new_torsion_indices,
                    'references': new_torsion_references,
                    'sigmas': new_torsion_sigmas,
                    'periods': new_torsion_periods
                }
            
            if self.verbose > 0:
                print(f"Built {len(torsion_idx1_list)} disulfide torsion restraints")
    
    def _build_exclusion_set(self):
        """
        Build set of atom pairs to exclude from VDW calculations.

        Excludes:

        - 1-2 interactions: directly bonded atoms (from all bond types)
        - 1-3 interactions: atoms separated by 2 bonds (from all angle types)
        - 1-4 interactions: atoms separated by 3 bonds (from all torsion types)

        These are excluded because they're already handled by bond, angle, and torsion restraints.

        Returns
        -------
        set
            Set of tuples (i, j) where i < j, representing excluded atom pairs.
        """
        exclusions = set()
        
        # 1-2: Direct bonds (all bond types: intra, peptide, disulfide)
        for origin in self.restraints.get('bond', {}).keys():
            indices = self.restraints['bond'][origin].get('indices')
            if indices is not None and len(indices) > 0:
                # Vectorized: convert to numpy for fast set operations
                idx_np = indices.cpu().numpy()
                for i1, i2 in idx_np:
                    exclusions.add((int(min(i1, i2)), int(max(i1, i2))))
        
        # 1-3: Angles (all angle types)
        for origin in self.restraints.get('angle', {}).keys():
            indices = self.restraints['angle'][origin].get('indices')
            if indices is not None and len(indices) > 0:
                idx_np = indices.cpu().numpy()
                for i1, i2, i3 in idx_np:
                    # i1 and i3 are separated by 2 bonds (through i2)
                    exclusions.add((int(min(i1, i3)), int(max(i1, i3))))
        
        # 1-4: Torsions - exclude backbone phi, psi, omega and intra/disulfide torsions
        for origin in self.restraints.get('torsion', {}).keys():
            indices = self.restraints['torsion'][origin].get('indices')
            if indices is not None and len(indices) > 0:
                idx_np = indices.cpu().numpy()
                for i1, i2, i3, i4 in idx_np:
                    # i1 and i4 are separated by 3 bonds
                    exclusions.add((int(min(i1, i4)), int(max(i1, i4))))
        
        if self.verbose > 1:
            print(f"Built exclusion set with {len(exclusions)} bonded pairs (1-2, 1-3, 1-4)")
        
        return exclusions
    
    def _find_nearby_pairs_spatial_hash(self, xyz, cutoff=6.0):
        """
        Find all atom pairs within cutoff distance using spatial hashing.

        This is the key optimization for large systems. Complexity is O(N) instead of O(N²).

        Algorithm:

        1. Divide 3D space into cubic cells of size ~cutoff
        2. Assign each atom to a cell based on its coordinates
        3. For each atom, only check atoms in same cell and 26 neighboring cells
        4. This reduces the search space dramatically for sparse systems

        Parameters
        ----------
        xyz : torch.Tensor
            Atom coordinates tensor of shape (N, 3).
        cutoff : float, default 6.0
            Maximum distance to consider in Angstroms.

        Returns
        -------
        torch.Tensor
            Tensor of shape (M, 2) containing indices of nearby pairs (i, j) where i < j.
        """
        device = xyz.device
        n_atoms = xyz.shape[0]
        
        if n_atoms == 0:
            return torch.tensor([], dtype=torch.long, device=device).reshape(0, 2)
        
        # Determine cell grid
        cell_size = cutoff
        min_coords = xyz.min(dim=0)[0]
        max_coords = xyz.max(dim=0)[0]
        
        # Assign each atom to a cell - vectorized
        # cell_indices shape: (N, 3) with integer cell coordinates
        cell_indices = ((xyz - min_coords) / cell_size).long()
        
        # Create cell keys as single integers for efficient hashing
        # Use a large prime multiplier to avoid collisions
        cell_keys = (cell_indices[:, 0] * 100000 + 
                     cell_indices[:, 1] * 1000 + 
                     cell_indices[:, 2])
        
        # Group atoms by cell using torch operations
        unique_cells, inverse_indices = torch.unique(cell_keys, return_inverse=True)
        
        # Build cell dictionary: cell_key -> list of atom indices
        cell_dict = {}
        for atom_idx in range(n_atoms):
            cell_key = cell_keys[atom_idx].item()
            if cell_key not in cell_dict:
                cell_dict[cell_key] = []
            cell_dict[cell_key].append(atom_idx)
        
        # Pre-compute all 27 neighbor offsets (including self)
        neighbor_offsets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_offsets.append([dx * 100000, dy * 1000, dz])
        
        pairs_list = []
        cutoff_squared = cutoff ** 2
        
        # Iterate over all cells that contain atoms
        for cell_key, atoms_in_cell in cell_dict.items():
            # Get cell indices from key
            cx = cell_key // 100000
            cy = (cell_key % 100000) // 1000
            cz = cell_key % 1000
            
            # Check atoms in this cell and neighboring cells
            for offset in neighbor_offsets:
                neighbor_key = cell_key + offset[0] + offset[1] + offset[2]
                
                if neighbor_key not in cell_dict:
                    continue
                
                atoms_in_neighbor = cell_dict[neighbor_key]
                
                # Vectorized distance computation for pairs between cells
                for i in atoms_in_cell:
                    for j in atoms_in_neighbor:
                        # Only store each pair once (i < j)
                        if i >= j:
                            continue
                        
                        # Check distance (squared for efficiency)
                        dist_sq = ((xyz[i] - xyz[j]) ** 2).sum()
                        if dist_sq < cutoff_squared:
                            pairs_list.append([i, j])
        
        if len(pairs_list) == 0:
            return torch.tensor([], dtype=torch.long, device=device).reshape(0, 2)
        
        return torch.tensor(pairs_list, dtype=torch.long, device=device)
    
    def _build_vdw_restraints(
        self,
        cutoff=5.0,
        sigma=0.2,
        inter_residue_only=True,
        use_spatial_hash=True
    ):
        """
        Build van der Waals (non-bonded contact) restraints.

        This prevents atoms from getting too close together (clashing) by penalizing
        distances shorter than the sum of van der Waals radii.

        Algorithm:

        1. Build exclusion set from bonded atoms (1-2, 1-3, 1-4 interactions)
        2. Find nearby atom pairs within cutoff distance using spatial hashing
        3. Filter out excluded pairs and optionally intra-residue pairs
        4. Store minimum distances (sum of VDW radii) and sigmas

        Parameters
        ----------
        cutoff : float, default 5.0
            Maximum distance to check for contacts in Å. Larger values are more
            complete but slower.
        sigma : float, default 0.2
            Standard deviation for Gaussian repulsion in Å. Smaller values mean
            stricter enforcement.
        inter_residue_only : bool, default True
            If True, only check contacts between different residues. Intra-residue
            geometry is already constrained by bonds/angles/torsions.
        use_spatial_hash : bool, default True
            If True, use spatial hashing O(N). If False, use simple all-pairs
            check O(N²), which is slow for large systems.
        """
        if self.verbose > 0:
            print("\nBuilding VDW (non-bonded) restraints...")
        
        # Step 1: Build exclusion set
        exclusions = self._build_exclusion_set()
        
        # Step 2: Get VDW radii for all atoms
        vdw_radii = self.model.get_vdw_radii()  # (N,)
        
        # Step 3: Find nearby pairs
        xyz = self.model.xyz()
        
        if use_spatial_hash:
            nearby_pairs = self._find_nearby_pairs_spatial_hash(xyz, cutoff)
        else:
            # Fallback: simple all-pairs check (slow!)
            if self.verbose > 0:
                print("  Warning: Using O(N²) all-pairs check. This may be slow for large systems.")
            n_atoms = xyz.shape[0]
            pairs_list = []
            cutoff_sq = cutoff ** 2
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    dist_sq = ((xyz[i] - xyz[j]) ** 2).sum()
                    if dist_sq < cutoff_sq:
                        pairs_list.append([i, j])
            nearby_pairs = torch.tensor(pairs_list, dtype=torch.long, device=xyz.device) if pairs_list else torch.tensor([], dtype=torch.long, device=xyz.device).reshape(0, 2)
        
        if self.verbose > 1:
            print(f"  Found {len(nearby_pairs)} nearby pairs within {cutoff:.1f} Å")
        
        # Step 4: Filter pairs
        indices_list = []
        min_dist_list = []
        sigma_list = []
        
        pdb = self.model.pdb
        
        # Vectorized filtering - prepare residue identifiers
        if inter_residue_only and len(nearby_pairs) > 0:
            # Get residue IDs for all atoms once
            chainid_array = pdb['chainid'].values
            resseq_array = pdb['resseq'].values
        
        for pair_idx in range(len(nearby_pairs)):
            i1 = nearby_pairs[pair_idx, 0].item()
            i2 = nearby_pairs[pair_idx, 1].item()
            
            # Skip if excluded (bonded)
            if (i1, i2) in exclusions:
                continue
            
            # Skip if same residue (optional)
            if inter_residue_only:
                # Compare residue identifiers
                if (chainid_array[i1] == chainid_array[i2] and 
                    resseq_array[i1] == resseq_array[i2]):
                    continue
            
            # Compute minimum allowed distance (sum of VDW radii)
            min_distance = vdw_radii[i1].item() + vdw_radii[i2].item()
            
            indices_list.append([i1, i2])
            min_dist_list.append(min_distance)
            sigma_list.append(sigma)
        
        # Step 5: Store restraints
        device = xyz.device
        if len(indices_list) > 0:
            self.restraints['vdw'] = {
                'indices': torch.tensor(indices_list, dtype=torch.long, device=device),
                'min_distances': torch.tensor(min_dist_list, dtype=torch.float32, device=device),
                'sigmas': torch.tensor(sigma_list, dtype=torch.float32, device=device)
            }
            
            if self.verbose > 0:
                scope = "inter-residue" if inter_residue_only else "all"
                print(f"  Built {len(indices_list)} VDW restraints ({scope} contacts)")
                if self.verbose > 1:
                    print(f"  Minimum distance range: {min(min_dist_list):.2f} - {max(min_dist_list):.2f} Å")
                    print(f"  Using sigma = {sigma:.3f} Å")
                    print(f"  Cutoff distance: {cutoff:.1f} Å")
        else:
            self.restraints['vdw'] = {
                'indices': torch.tensor([], dtype=torch.long, device=device).reshape(0, 2),
                'min_distances': torch.tensor([], dtype=torch.float32, device=device),
                'sigmas': torch.tensor([], dtype=torch.float32, device=device)
            }
            if self.verbose > 0:
                print("  Warning: No VDW restraints were built")
    
    def _move_to_device(self, device):
        """
        Move all restraint tensors to the specified device.

        Parameters
        ----------
        device : torch.device
            PyTorch device (cpu, cuda, etc.).
        """
        # Iterate through hierarchical restraints structure
        for restraint_type in ['bond', 'angle', 'torsion', 'plane']:
            if restraint_type in self.restraints:
                for origin, properties in self.restraints[restraint_type].items():
                    if isinstance(properties, dict):
                        for prop_name, tensor in properties.items():
                            if tensor is not None and isinstance(tensor, torch.Tensor):
                                self.restraints[restraint_type][origin][prop_name] = tensor.to(device)
        
        # Handle VDW restraints (not hierarchical like others)
        if 'vdw' in self.restraints:
            vdw_data = self.restraints['vdw']
            for prop_name, tensor in vdw_data.items():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    self.restraints['vdw'][prop_name] = tensor.to(device)
    
    def cuda(self, device: Optional[int] = None):
        """
        Move all restraint tensors to CUDA device.

        Parameters
        ----------
        device : int, optional
            CUDA device index. If None, uses current device.

        Returns
        -------
        Restraints
            Self for method chaining.
        """
        cuda_device = torch.device('cuda' if device is None else f'cuda:{device}')
        self._move_to_device(cuda_device)
        return self
    
    def cpu(self):
        """
        Move all restraint tensors to CPU.

        Returns
        -------
        Restraints
            Self for method chaining.
        """
        self._move_to_device(torch.device('cpu'))
        return self
    
    def vdw_violations(self, threshold: float = 0.0):
        """
        Compute VDW (steric clash) violations.

        A violation occurs when the actual distance is less than the minimum
        allowed distance (sum of VDW radii).

        Parameters
        ----------
        threshold : float, default 0.0
            Additional tolerance in Angstroms. Distances < (min_distance - threshold)
            are counted as violations.

        Returns
        -------
        violations : torch.Tensor
            Tensor of violation amounts (min_dist - actual_dist) for violated pairs.
        n_violations : int
            Number of violations.
        """
        if 'vdw' not in self.restraints:
            return torch.tensor([]), 0
        
        vdw_data = self.restraints['vdw']
        indices = vdw_data.get('indices')
        
        if indices is None or len(indices) == 0:
            return torch.tensor([]), 0
        
        min_distances = vdw_data['min_distances']
        
        # Get current atomic positions
        xyz = self.model.xyz()
        pos1 = xyz[indices[:, 0]]
        pos2 = xyz[indices[:, 1]]
        
        # Compute actual distances
        actual_distances = torch.norm(pos2 - pos1, dim=-1)
        
        # Violations: where actual distance is less than minimum allowed
        violation_mask = actual_distances < (min_distances - threshold)
        violation_amounts = min_distances - actual_distances
        
        violations = violation_amounts[violation_mask]
        n_violations = violation_mask.sum().item()
        
        return violations, n_violations

    def __repr__(self):
        """
        Return string representation of the Restraints object.

        Returns
        -------
        str
            Detailed string representation showing restraint counts.
        """
        # Helper function to get count from hierarchical structure
        def get_count(rtype, origin):
            indices = self.restraints.get(rtype, {}).get(origin, {}).get('indices')
            return 0 if indices is None else indices.shape[0]
        
        n_bonds = get_count('bond', 'intra')
        n_angles = get_count('angle', 'intra')
        n_torsions = get_count('torsion', 'intra')
        
        n_bonds_peptide = get_count('bond', 'peptide')
        n_bonds_disulfide = get_count('bond', 'disulfide')
        n_angles_peptide = get_count('angle', 'peptide')
        n_angles_disulfide = get_count('angle', 'disulfide')
        n_torsions_disulfide = get_count('torsion', 'disulfide')
        
        n_phi = get_count('torsion', 'phi')
        n_psi = get_count('torsion', 'psi')
        n_omega = get_count('torsion', 'omega')
        
        # Count planes by atom count
        n_planes = 0
        plane_info = []
        for key in self.restraints.get('plane', {}).keys():
            count = get_count('plane', key)
            if count > 0:
                n_planes += count
                plane_info.append(f"{key}={count}")
        
        plane_str = ", ".join(plane_info) if plane_info else "none"
        
        # VDW restraints count
        n_vdw = 0
        if 'vdw' in self.restraints:
            vdw_indices = self.restraints['vdw'].get('indices')
            if vdw_indices is not None:
                n_vdw = vdw_indices.shape[0]
        
        # Get device from first available tensor
        device = 'not initialized'
        for rtype in ['bond', 'angle', 'torsion', 'plane', 'vdw']:
            for origin in self.restraints.get(rtype, {}).keys():
                indices = self.restraints[rtype][origin].get('indices') if isinstance(self.restraints[rtype], dict) else self.restraints[rtype].get('indices')
                if indices is not None:
                    device = str(indices.device)
                    break
            if device != 'not initialized':
                break
        
        return (f"Restraints(\n"
                f"  cif_path='{self.cif_path}',\n"
                f"  intra-residue: bonds={n_bonds}, angles={n_angles}, torsions={n_torsions},\n"
                f"  planes: {plane_str} (total={n_planes}),\n"
                f"  peptide: bonds={n_bonds_peptide}, angles={n_angles_peptide},\n"
                f"  disulfide: bonds={n_bonds_disulfide}, angles={n_angles_disulfide}, torsions={n_torsions_disulfide},\n"
                f"  backbone: phi={n_phi}, psi={n_psi}, omega={n_omega},\n"
                f"  vdw: {n_vdw} non-bonded contacts,\n"
                f"  device={device}\n"
                f")")
    
    def summary(self):
        """
        Print a detailed summary of all restraints.

        Displays counts and statistics for all restraint types including
        bonds, angles, torsions, planes, and VDW contacts.
        """
        print("=" * 80)
        print("Restraints Summary")
        print("=" * 80)
        print(f"CIF file: {self.cif_path}")
        print(f"Residue types in dictionary: {len(self.cif_dict)}")
        print()
        
        # Helper function to print restraint details
        def print_restraint_section(rtype, origin, title):
            data = self.restraints.get(rtype, {}).get(origin, {})
            indices = data.get('indices')
            
            if indices is None:
                print(f"{title}: None")
                return
            
            print(f"{title}: {indices.shape[0]}")
            print(f"  Shape: {indices.shape}")
            
            refs = data.get('references')
            sigmas = data.get('sigmas')
            periods = data.get('periods')
            
            if refs is not None:
                unit = "Å" if rtype == 'bond' else "°"
                print(f"  References: min={refs.min():.3f}{unit}, "
                      f"max={refs.max():.3f}{unit}, mean={refs.mean():.3f}{unit}")
            
            if sigmas is not None:
                unit = "Å" if rtype == 'bond' else "°"
                print(f"  Sigmas: min={sigmas.min():.4f}{unit}, "
                      f"max={sigmas.max():.4f}{unit}, mean={sigmas.mean():.4f}{unit}")
            
            if periods is not None:
                unique_periods = periods.unique().tolist()
                print(f"  Periods: {unique_periods}")
        
        # Intra-residue restraints
        print("INTRA-RESIDUE RESTRAINTS:")
        print("-" * 80)
        print_restraint_section('bond', 'intra', "Bonds")
        print()
        print_restraint_section('angle', 'intra', "Angles")
        print()
        print_restraint_section('torsion', 'intra', "Torsions")
        print()
        
        # Peptide restraints
        print("PEPTIDE LINK RESTRAINTS:")
        print("-" * 80)
        print_restraint_section('bond', 'peptide', "Bonds")
        print()
        print_restraint_section('angle', 'peptide', "Angles")
        print()
        
        # Disulfide restraints
        if 'disulfide' in self.restraints.get('bond', {}):
            print("DISULFIDE LINK RESTRAINTS:")
            print("-" * 80)
            print_restraint_section('bond', 'disulfide', "Bonds")
            print()
            print_restraint_section('angle', 'disulfide', "Angles")
            print()
            print_restraint_section('torsion', 'disulfide', "Torsions")
            print()
        
        # Backbone torsion angles
        print("BACKBONE TORSION ANGLES:")
        print("-" * 80)
        
        phi_data = self.restraints.get('torsion', {}).get('phi', {})
        if phi_data.get('indices') is not None:
            print(f"Phi: {phi_data['indices'].shape[0]}")
            print(f"  Definition: C(i) - N(i+1) - CA(i+1) - C(i+1)")
            if phi_data.get('periods') is not None:
                print(f"  Period: {phi_data['periods'][0].item()}")
        else:
            print("Phi: None")
        print()
        
        psi_data = self.restraints.get('torsion', {}).get('psi', {})
        if psi_data.get('indices') is not None:
            print(f"Psi: {psi_data['indices'].shape[0]}")
            print(f"  Definition: N(i) - CA(i) - C(i) - N(i+1)")
            if psi_data.get('periods') is not None:
                print(f"  Period: {psi_data['periods'][0].item()}")
        else:
            print("Psi: None")
        print()
        
        omega_data = self.restraints.get('torsion', {}).get('omega', {})
        if omega_data.get('indices') is not None:
            print(f"Omega: {omega_data['indices'].shape[0]}")
            print(f"  Definition: CA(i) - C(i) - N(i+1) - CA(i+1)")
            if omega_data.get('references') is not None:
                refs = omega_data['references']
                print(f"  References: min={refs.min():.2f}°, max={refs.max():.2f}°, mean={refs.mean():.2f}°")
            if omega_data.get('sigmas') is not None:
                sigs = omega_data['sigmas']
                print(f"  Sigmas: min={sigs.min():.4f}°, max={sigs.max():.4f}°, mean={sigs.mean():.4f}°")
            if omega_data.get('periods') is not None:
                print(f"  Period: {omega_data['periods'][0].item()}")
        else:
            print("Omega: None")
        
        # Plane restraints
        if len(self.restraints.get('plane', {})) > 0:
            print()
            print("PLANE RESTRAINTS:")
            print("-" * 80)
            for atom_count_key in sorted(self.restraints['plane'].keys()):
                plane_data = self.restraints['plane'][atom_count_key]
                indices = plane_data.get('indices')
                if indices is not None:
                    print(f"{atom_count_key}: {indices.shape[0]} planes")
                    print(f"  Shape: {indices.shape}")
                    sigmas = plane_data.get('sigmas')
                    if sigmas is not None:
                        print(f"  Sigmas: min={sigmas.min():.4f}Å, "
                              f"max={sigmas.max():.4f}Å, mean={sigmas.mean():.4f}Å")
                    print()
        
        # VDW restraints
        if 'vdw' in self.restraints:
            print()
            print("VDW (NON-BONDED) RESTRAINTS:")
            print("-" * 80)
            vdw_data = self.restraints['vdw']
            indices = vdw_data.get('indices')
            
            if indices is not None and len(indices) > 0:
                print(f"Pairs: {len(indices)}")
                min_dists = vdw_data['min_distances']
                sigmas = vdw_data['sigmas']
                print(f"  Min distances: {min_dists.min():.3f} - {min_dists.max():.3f} Å "
                      f"(mean={min_dists.mean():.3f} Å)")
                print(f"  Sigma: {sigmas[0]:.3f} Å")
                
                # Show current violations
                violations, n_violations = self.vdw_violations()
                if n_violations > 0:
                    print(f"  Current violations: {n_violations} ({100*n_violations/len(indices):.1f}%)")
                    print(f"  Violation range: {violations.min():.3f} - {violations.max():.3f} Å "
                          f"(mean={violations.mean():.3f} Å)")
                else:
                    print(f"  Current violations: None")
            else:
                print("No VDW restraints built")
        
        print("=" * 80)
    
    def _get_all_indices(self, restraint_type, keys_to_merge = None):
        """
        Gather all indices of a given restraint type across all origins.

        Parameters
        ----------
        restraint_type : str
            Type of restraint ('bond', 'angle', or 'torsion').
        keys_to_merge : list of str, optional
            Specific origins to include. If None, includes all origins.

        Returns
        -------
        torch.Tensor or None
            Concatenated tensor of all indices, or None if none exist.
        """
        indices_list = []
        for origin, data in self.restraints.get(restraint_type, {}).items():
            indices = data.get('indices')
            if indices is not None:
                if keys_to_merge is None:
                    indices_list.append(indices)
                elif origin in keys_to_merge:
                    indices_list.append(indices)
        
        if not indices_list:
            return None
        
        return torch.cat(indices_list, dim=0)
    
    def _get_all_property(self, restraint_type, property_name, keys_to_merge = None):
        """
        Gather all values of a given property across all origins.

        Parameters
        ----------
        restraint_type : str
            Type of restraint ('bond', 'angle', or 'torsion').
        property_name : str
            Property to gather ('references', 'sigmas', or 'periods').
        keys_to_merge : list of str, optional
            Specific origins to include. If None, includes all origins.

        Returns
        -------
        torch.Tensor or None
            Concatenated tensor of all property values, or None if none exist.
        """
        values_list = []
        for origin, data in self.restraints.get(restraint_type, {}).items():
            values = data.get(property_name)
            if values is not None:
                if keys_to_merge is None:
                    values_list.append(values)
                elif origin in keys_to_merge:
                    values_list.append(values)
        
        if not values_list:
            return None
        
        return torch.cat(values_list, dim=0)

    def bond_lengths(self, idx):
        """
        Compute current bond lengths from atomic coordinates.

        Parameters
        ----------
        idx : torch.Tensor
            Bond indices tensor of shape (N, 2).

        Returns
        -------
        torch.Tensor
            Tensor of bond lengths of shape (N,).
        """
        if idx is None:
            return torch.tensor([], device=self.model.xyz().device)
        xyz = self.model.xyz()
        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        return torch.linalg.norm(pos2 - pos1, dim=-1)
    
    def copy(self):
        """
        Create a deep copy of the Restraints object.

        Returns
        -------
        Restraints
            A deep copy of this Restraints instance.
        """
        import copy
        return copy.deepcopy(self)
    
    def bond_deviations(self):
        """
        Compute bond length deviations and sigmas.

        Returns
        -------
        deviations : torch.Tensor
            Calculated minus expected bond lengths in Angstroms.
        sigmas : torch.Tensor
            Standard deviations from CIF library in Angstroms.
        """
        if 'all' not in self.restraints['bond']:
            self.cat_dict()

        idx = self.restraints['bond']['all']['indices']
        references = self.restraints['bond']['all']['references']
        sigmas = self.restraints['bond']['all']['sigmas']

        # Get current bond lengths
        bond_lengths = self.bond_lengths(idx)
        deviations = bond_lengths - references
        
        return deviations, sigmas
    
    def nll_bonds(self):
        """
        Compute negative log-likelihood for bond length restraints.

        For Gaussian distribution: NLL = -log(P(x|μ,σ))
        NLL = 0.5 * ((x - μ) / σ)^2 + log(σ) + 0.5 * log(2π)

        This is the true NLL where exp(-NLL) = probability density.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_bonds,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll
        deviations, sigmas = self.bond_deviations()
        return gaussian_nll(deviations, sigmas)

    def angles(self, idx):
        """
        Compute current angle values for all angle restraints.

        Parameters
        ----------
        idx : torch.Tensor
            Angle indices tensor of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_angles,) with current angle values in degrees.
        """
        xyz = self.model.xyz()
        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        pos3 = xyz[idx[:, 2], :]
        
        # Compute vectors
        v1 = pos1 - pos2  # Vector from atom2 to atom1
        v2 = pos3 - pos2  # Vector from atom2 to atom3
        
        # Compute angle using dot product
        # cos(θ) = (v1 · v2) / (|v1| * |v2|)
        dot_product = torch.sum(v1 * v2, dim=-1)
        norm1 = torch.linalg.norm(v1, dim=-1)
        norm2 = torch.linalg.norm(v2, dim=-1)
        
        # Clamp to avoid numerical issues with arccos
        cos_angle = torch.clamp(dot_product / (norm1 * norm2), -1.0, 1.0)
        
        # Return angle in degrees
        angles_rad = torch.acos(cos_angle)
        angles_deg = torch.rad2deg(angles_rad)
        
        return angles_deg
    
    def angle_deviations(self):
        """
        Compute angle deviations and sigmas.

        Returns
        -------
        deviations : torch.Tensor
            Calculated minus expected angles in radians.
        sigmas : torch.Tensor
            Standard deviations in radians.
        """
        if 'all' not in self.restraints['angle']:
            self.cat_dict()
        
        idx = self.restraints['angle']['all']['indices']
        references_rad = self.restraints['angle']['all']['references'] * (torch.pi / 180.0)
        sigmas_rad = self.restraints['angle']['all']['sigmas'] * (torch.pi / 180.0)

        calculated_rad = self.angles(idx) * (torch.pi / 180.0)
        deviations = calculated_rad - references_rad
        
        return deviations, sigmas_rad
    
    def nll_angles(self):
        """
        Compute negative log-likelihood for angle restraints.

        For Gaussian distribution: NLL = -log(P(x|μ,σ))
        NLL = 0.5 * ((x - μ) / σ)^2 + log(σ) + 0.5 * log(2π)

        This is the true NLL where exp(-NLL) = probability density.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_angles,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll
        deviations, sigmas = self.angle_deviations()
        return gaussian_nll(deviations, sigmas)
    
    def cat_dict(self):
        """
        Concatenate all restraint dictionaries into 'all' keys.

        Creates restraints['bond']['all'], restraints['angle']['all'],
        and restraints['torsion']['all'] by concatenating all origins.
        """
        self.restraints['bond']['all'] = {
            'indices': self._get_all_indices('bond'),
            'references': self._get_all_property('bond', 'references'),
            'sigmas': self._get_all_property('bond', 'sigmas')
        }
        self.restraints['angle']['all'] = {
            'indices': self._get_all_indices('angle'),
            'references': self._get_all_property('angle', 'references'),
            'sigmas': self._get_all_property('angle', 'sigmas')
        }
        self.restraints['torsion']['all'] = {
            'indices': self._get_all_indices('torsion',['intra','disulfide']),
            'references': self._get_all_property('torsion', 'references',['intra','disulfide']),
            'sigmas': self._get_all_property('torsion', 'sigmas',['intra','disulfide']),
            'periods': self._get_all_property('torsion', 'periods',['intra','disulfide'])
        }

    def torsions(self, idx):
        """
        Compute current torsion angle values for all torsion restraints.

        Parameters
        ----------
        idx : torch.Tensor
            Torsion indices tensor of shape (N, 4).

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_torsions,) with current torsion values in degrees.
        """
        xyz = self.model.xyz()

        pos1 = xyz[idx[:, 0], :]
        pos2 = xyz[idx[:, 1], :]
        pos3 = xyz[idx[:, 2], :]
        pos4 = xyz[idx[:, 3], :]
        
        # Compute torsion angles using vector math
        b1 = pos2 - pos1
        b2 = pos3 - pos2
        b3 = pos4 - pos3
        
        # Normalize b2 for projection
        b2_norm = torch.linalg.norm(b2, dim=-1, keepdim=True)
        b2_unit = b2 / b2_norm
        
        # Compute normals to planes
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        
        # Normalize normals
        n1_unit = n1 / torch.linalg.norm(n1, dim=-1, keepdim=True)
        n2_unit = n2 / torch.linalg.norm(n2, dim=-1, keepdim=True)
        
        # Compute angle between normals
        m1 = torch.cross(n1_unit, b2_unit, dim=-1)
        
        x = torch.sum(n1_unit * n2_unit, dim=-1)
        y = torch.sum(m1 * n2_unit, dim=-1)
        
        torsions_rad = torch.atan2(y, x)
        torsions_deg = torch.rad2deg(torsions_rad)
        return torsions_deg
    
    def _wrap_torsion_periodicity(self, diff_rad, periods):
        """
        Find minimum angular deviation considering n-fold rotational symmetry.

        For period=n, angles differing by 360°/n are equivalent. This function
        finds the equivalent angle with the smallest absolute deviation.

        Parameters
        ----------
        diff_rad : torch.Tensor
            Tensor of angular deviations in radians (any shape).
        periods : torch.Tensor
            Tensor of periodicity values (same shape as diff_rad).
            Period=0 or 1 means no symmetry (simple wrapping).
            Period=n means n-fold rotational symmetry.

        Returns
        -------
        torch.Tensor
            Tensor of minimum wrapped deviations in radians (same shape as input).
            Values are wrapped to [-π, π] and account for rotational symmetry.

        Examples
        --------
        For period=6 (e.g., benzene), angles of 10°, 70°, 130°, 190°, 250°, 310°
        are all equivalent. The function returns the one closest to 0°.
        """
        # Clamp periods to minimum of 1 to avoid division by zero
        periods_safe = torch.clamp(periods, min=1)
        max_period = periods_safe.max().item()
        
        if max_period > 1:
            # Vectorized approach: generate all equivalent angles
            device = diff_rad.device
            original_shape = diff_rad.shape
            
            # Flatten input for processing
            diff_rad_flat = diff_rad.flatten()
            periods_flat = periods_safe.flatten()
            n_angles = len(diff_rad_flat)
            
            # Create offset matrix: k * (2π / period) for k in [0, 1, ..., period-1]
            # Shape: (n_angles, max_period)
            k_range = torch.arange(max_period, device=device).unsqueeze(0)  # (1, max_period)
            periods_expanded = periods_flat.unsqueeze(1).float()  # (n_angles, 1)
            
            # Offsets for each angle: k * 2π/period
            offsets = k_range * (2.0 * torch.pi / periods_expanded)  # (n_angles, max_period)
            
            # Apply offsets to differences: (n_angles, max_period)
            diff_rad_expanded = diff_rad_flat.unsqueeze(1)  # (n_angles, 1)
            equiv_diffs = diff_rad_expanded - offsets  # (n_angles, max_period)
            
            # Wrap all equivalent angles to [-pi, pi]
            equiv_diffs_wrapped = torch.atan2(torch.sin(equiv_diffs), torch.cos(equiv_diffs))
            
            # Mask out invalid offsets (where k >= period for each angle)
            valid_mask = k_range < periods_expanded  # (n_angles, max_period)
            
            # Set invalid positions to large value so they won't be selected
            equiv_diffs_wrapped_masked = torch.where(
                valid_mask,
                torch.abs(equiv_diffs_wrapped),
                torch.tensor(float('inf'), device=device)
            )
            
            # Find minimum absolute difference for each angle
            min_indices = torch.argmin(equiv_diffs_wrapped_masked, dim=1)  # (n_angles,)
            
            # Gather the best wrapped difference for each angle
            diff_wrapped_best = equiv_diffs_wrapped[torch.arange(n_angles, device=device), min_indices]
            
            # Reshape back to original shape
            return diff_wrapped_best.reshape(original_shape)
        else:
            # All periods are 0 or 1, simple wrapping
            return torch.atan2(torch.sin(diff_rad), torch.cos(diff_rad))
    
    def torsion_deviations(self, wrapped=True):
        """
        Compute deviations between calculated and expected torsion angles.

        Parameters
        ----------
        wrapped : bool, default True
            If True, wrap deviations accounting for periodicity.
            If False, return raw deviations (calculated - expected).

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_torsions,) with deviations in degrees.
            For wrapped=True, deviations are in range appropriate for the period.

        Notes
        -----
        Expected values from CIF library are discrete (typically -60°, 0°, 60°, 90°, 180°)
        while calculated values from structure are continuous. This is correct!
        Use wrapped=True for meaningful comparison and visualization.
        """
        if not 'all' in self.restraints['torsion']:
            self.cat_dict()
            
        idx = self.restraints['torsion']['all']['indices']
        expected = self.restraints['torsion']['all']['references']
        periods = self.restraints['torsion']['all']['periods']
        calculated = self.torsions(idx)
        
        if not wrapped:
            # Simple difference
            return calculated - expected
        else:
            # Use the helper function for periodicity handling
            diff_rad = (calculated - expected) * torch.pi / 180.0
            diff_wrapped_rad = self._wrap_torsion_periodicity(diff_rad, periods)
            
            # Convert back to degrees
            return torch.rad2deg(diff_wrapped_rad)
    
    def torsion_deviations_with_sigmas(self):
        """
        Compute torsion deviations (wrapped for periodicity) and sigmas.

        Returns
        -------
        deviations_rad : torch.Tensor
            Wrapped deviations in radians.
        sigmas_deg : torch.Tensor
            Standard deviations in degrees (for von Mises NLL).
        """
        if 'all' not in self.restraints['torsion']:
            self.cat_dict()
            
        idx = self.restraints['torsion']['all']['indices']
        expected = self.restraints['torsion']['all']['references']
        sigmas_deg = self.restraints['torsion']['all']['sigmas']
        periods = self.restraints['torsion']['all']['periods']
        
        calculated = self.torsions(idx)
        
        # Wrap for periodicity
        diff_rad = (calculated - expected) * (torch.pi / 180.0)
        deviations_rad = self._wrap_torsion_periodicity(diff_rad, periods)
        
        return deviations_rad, sigmas_deg
    
    def nll_torsions(self):
        """
        Compute negative log-likelihood for torsion angle restraints.

        For von Mises distribution: NLL = -log(P(θ|μ,κ))
        NLL = -κ*cos(θ-μ) + log(I₀(κ)) + log(2π)

        where κ = 1/σ² is the concentration parameter and I₀ is the modified
        Bessel function of the first kind.

        Notes
        -----
        Period indicates n-fold rotational symmetry (e.g., period=6 for benzene).
        We handle this by finding the minimum angular distance considering periodicity.
        For period=n, angles differing by 360°/n are equivalent.

        This is the true NLL where exp(-NLL) = probability density.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_torsions,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import von_mises_nll
        deviations_rad, sigmas_deg = self.torsion_deviations_with_sigmas()
        return von_mises_nll(deviations_rad, sigmas_deg)

    def nll_planes(self):
        """
        Compute negative log-likelihood for plane restraints.

        For each plane, computes the RMSD of atom deviations from the best-fit plane.
        Uses Gaussian NLL: NLL = 0.5 * (deviation / σ)² + log(σ) + 0.5 * log(2π)

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_planes,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll
        
        xyz = self.model.xyz()
        device = xyz.device
        
        all_nlls = []
        
        if 'plane' in self.restraints:
            for key, plane_data in self.restraints['plane'].items():
                indices = plane_data.get('indices')
                sigmas = plane_data.get('sigmas')
                
                if indices is None or len(indices) == 0:
                    continue
                
                # indices shape: (n_planes, n_atoms_per_plane)
                # sigmas shape: (n_planes, n_atoms_per_plane)
                n_planes, n_atoms = indices.shape
                
                for i in range(n_planes):
                    plane_indices = indices[i]
                    plane_sigmas = sigmas[i]
                    
                    # Get positions of atoms in this plane
                    positions = xyz[plane_indices]  # (n_atoms, 3)
                    
                    # Compute centroid
                    centroid = positions.mean(dim=0)
                    centered = positions - centroid
                    
                    # SVD to find best-fit plane normal
                    # The plane normal is the singular vector with smallest singular value
                    U, S, Vh = torch.linalg.svd(centered)
                    normal = Vh[-1]  # Normal to best-fit plane
                    
                    # Compute deviations from plane (distance to plane)
                    deviations = torch.abs(centered @ normal)
                    
                    # Compute NLL for each atom
                    nll = gaussian_nll(deviations, plane_sigmas)
                    all_nlls.append(nll)
        
        if all_nlls:
            return torch.cat(all_nlls)
        return torch.tensor([0.0], device=device)

    def nll_vdw(self):
        """
        Compute negative log-likelihood for VDW (non-bonded) restraints.

        Uses a soft-repulsive potential based on distance violations.
        NLL = 0.5 * (max(0, min_dist - actual_dist) / σ)² + log(σ) + 0.5 * log(2π)

        Only violations (distances shorter than minimum) contribute to the loss.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_pairs,) with negative log-likelihood values.
        """
        from torchref.refinement.targets import gaussian_nll
        
        xyz = self.model.xyz()
        device = xyz.device
        
        if 'vdw' not in self.restraints:
            return torch.tensor([0.0], device=device)
        
        vdw_data = self.restraints['vdw']
        indices = vdw_data.get('indices')
        
        if indices is None or len(indices) == 0:
            return torch.tensor([0.0], device=device)
        
        min_distances = vdw_data['min_distances']
        sigmas = vdw_data['sigmas']
        
        # Get current positions
        pos1 = xyz[indices[:, 0]]
        pos2 = xyz[indices[:, 1]]
        
        # Compute actual distances
        actual_distances = torch.norm(pos2 - pos1, dim=-1)
        
        # Violations: where actual distance is less than minimum
        # Deviation = max(0, min_dist - actual_dist)
        deviations = torch.clamp(min_distances - actual_distances, min=0.0)
        
        # Compute NLL (only non-zero for violations)
        nll = gaussian_nll(deviations, sigmas)
        
        return nll

    def adp_b_differences(self):
        """
        Compute B-factor differences between bonded atoms.

        Returns
        -------
        torch.Tensor
            Tensor of B-factor differences (B_i - B_j) for all bonds.
        """
        b_factors = self.model.b()
        
        diffs_list = []
        if 'bond' in self.restraints:
            for origin, restraint_group in self.restraints['bond'].items():
                if origin == 'all':
                    continue
                indices = restraint_group.get('indices')
                if indices is not None and len(indices) > 0:
                    b1 = b_factors[indices[:, 0]]
                    b2 = b_factors[indices[:, 1]]
                    diffs_list.append(b1 - b2)
        
        if diffs_list:
            return torch.cat(diffs_list, dim=0)
        return torch.tensor([], device=b_factors.device)

    def adp_similarity_loss(self, sigma: float = 2.0):
        """
        Compute ADP similarity loss (SIMU in Phenix/SHELX).

        This restrains the B-factors of bonded atoms to be similar.
        Loss = Σ ((B_i - B_j) / sigma)^2

        Parameters
        ----------
        sigma : float, default 2.0
            Target standard deviation for B-factor differences in Å².

        Returns
        -------
        torch.Tensor
            Mean similarity loss.
        """
        from torchref.refinement.targets import adp_similarity_nll
        b_diffs = self.adp_b_differences()
        if len(b_diffs) == 0:
            return torch.tensor(0.0, device=self.model.xyz().device)
        return adp_similarity_nll(b_diffs, sigma).mean()
