#!/usr/bin/env python3
"""
Test script with synthetic disulfide bond
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import Model
from torchref.restraints import Restraints
import torch
import pandas as pd
import numpy as np

print("=" * 80)
print("TESTING DISULFIDE BOND DETECTION WITH SYNTHETIC DATA")
print("=" * 80)

# Load the model
print("\n1. Loading model...")
model = Model()
model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

print(f"   Original atoms: {len(model.pdb)}")

# Find two cysteine residues that we can artificially bring together
cys_residues = model.pdb[(model.pdb['resname'] == 'CYS') & (model.pdb['ATOM'] == 'ATOM')]
cys_resseqs = cys_residues.groupby(['chainid', 'resseq']).first()

if len(cys_resseqs) >= 2:
    # Pick two cysteines
    cys1_chain, cys1_resseq = cys_resseqs.index[0]
    cys2_chain, cys2_resseq = cys_resseqs.index[1]
    
    print(f"\n2. Artificially creating disulfide bond between:")
    print(f"   CYS1: {cys1_chain}:{cys1_resseq}")
    print(f"   CYS2: {cys2_chain}:{cys2_resseq}")
    
    # Get the SG atoms
    sg1_atoms = model.pdb[(model.pdb['chainid'] == cys1_chain) & 
                           (model.pdb['resseq'] == cys1_resseq) & 
                           (model.pdb['name'] == 'SG')]
    if ' ' in sg1_atoms['altloc'].values:
        sg1 = sg1_atoms[sg1_atoms['altloc'] == ' '].iloc[0]
    else:
        sg1 = sg1_atoms.iloc[0]
    
    sg2_atoms = model.pdb[(model.pdb['chainid'] == cys2_chain) & 
                           (model.pdb['resseq'] == cys2_resseq) & 
                           (model.pdb['name'] == 'SG')]
    if ' ' in sg2_atoms['altloc'].values:
        sg2 = sg2_atoms[sg2_atoms['altloc'] == ' '].iloc[0]
    else:
        sg2 = sg2_atoms.iloc[0]
    
    sg1_idx = sg1['index']
    sg2_idx = sg2['index']
    
    # Get original coordinates
    xyz = model.xyz().clone()
    orig_dist = torch.linalg.norm(xyz[sg2_idx] - xyz[sg1_idx]).item()
    print(f"   Original SG-SG distance: {orig_dist:.3f} Å")
    
    # Move SG2 to be 2.05 Å from SG1
    direction = (xyz[sg2_idx] - xyz[sg1_idx]) / orig_dist
    xyz[sg2_idx] = xyz[sg1_idx] + direction * 2.05
    
    # Update model coordinates
    model.pdb['x'] = xyz[:, 0].detach().cpu().numpy()
    model.pdb['y'] = xyz[:, 1].detach().cpu().numpy()
    model.pdb['z'] = xyz[:, 2].detach().cpu().numpy()
    
    new_dist = torch.linalg.norm(xyz[sg2_idx] - xyz[sg1_idx]).item()
    print(f"   Modified SG-SG distance: {new_dist:.3f} Å")
    
    # Create restraints
    print("\n3. Building restraints...")
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
    restraints = Restraints(model, cif_path, verbose=2)
    
    print("\n4. Restraint statistics:")
    print("-" * 80)
    
    # Inter-residue bonds
    if restraints.bond_indices_inter is not None:
        n_bonds_inter = restraints.bond_indices_inter.shape[0]
        print(f"Inter-residue bonds total: {n_bonds_inter}")
        
        # Separate peptide bonds from disulfide bonds
        peptide_mask = restraints.bond_references_inter < 1.5
        disulfide_mask = restraints.bond_references_inter > 1.5
        
        n_peptide = peptide_mask.sum().item()
        n_disulfide = disulfide_mask.sum().item()
        
        print(f"  - Peptide bonds (< 1.5 Å): {n_peptide}")
        print(f"  - Disulfide bonds (> 1.5 Å): {n_disulfide}")
        
        if n_disulfide > 0:
            print("\n5. Disulfide bonds found:")
            print("-" * 80)
            disulfide_indices = restraints.bond_indices_inter[disulfide_mask]
            disulfide_refs = restraints.bond_references_inter[disulfide_mask]
            disulfide_sigmas = restraints.bond_sigmas_inter[disulfide_mask]
            
            for i in range(len(disulfide_indices)):
                idx1, idx2 = disulfide_indices[i]
                idx1 = int(idx1.item())
                idx2 = int(idx2.item())
                atom1 = model.pdb.iloc[idx1]
                atom2 = model.pdb.iloc[idx2]
                ref_dist = float(disulfide_refs[i].item())
                sigma = float(disulfide_sigmas[i].item())
                
                # Calculate actual distance
                coord1 = xyz[idx1]
                coord2 = xyz[idx2]
                actual_dist = torch.linalg.norm(coord2 - coord1).item()
                
                print(f"  {atom1['chainid']}:{atom1['resname']}{atom1['resseq']}:{atom1['name']} -- "
                      f"{atom2['chainid']}:{atom2['resname']}{atom2['resseq']}:{atom2['name']}  "
                      f"(ref: {ref_dist:.3f} ± {sigma:.4f} Å, actual: {actual_dist:.3f} Å)")
                
                if abs(actual_dist - ref_dist) < 0.1:
                    print("    ✓ Distance matches reference!")
        else:
            print("\n5. No disulfide bonds detected (unexpected!)")
    
    print("\n✓ Test complete!")
else:
    print("Not enough cysteine residues for test")
