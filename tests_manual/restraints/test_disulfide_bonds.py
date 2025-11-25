#!/usr/bin/env python3
"""
Test script to verify disulfide bond restraints are correctly built
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import Model
from torchref.restraints import Restraints
import torch

print("=" * 80)
print("TESTING DISULFIDE BOND RESTRAINTS")
print("=" * 80)

# Load the model
print("\n1. Loading model...")
model = Model()
model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

print(f"   Total atoms: {len(model.pdb)}")

# Check for cysteine residues
cys_residues = model.pdb[model.pdb['resname'] == 'CYS']
print(f"   Cysteine residues: {cys_residues['resseq'].nunique()}")

# Check for SG atoms
sg_atoms = model.pdb[(model.pdb['name'] == 'SG') & (model.pdb['ATOM'] == 'ATOM')]
print(f"   SG atoms: {len(sg_atoms)}")

if len(sg_atoms) > 1:
    print("\n   Checking SG-SG distances...")
    xyz = model.xyz()
    sg_indices = sg_atoms['index'].values
    sg_coords = xyz[sg_indices]
    
    # Compute pairwise distances
    distances = torch.cdist(sg_coords, sg_coords)
    
    # Find pairs closer than 3.0 Å
    close_pairs = torch.where((distances < 3.0) & (distances > 0.1))
    mask = close_pairs[0] < close_pairs[1]
    idx1_local = close_pairs[0][mask]
    idx2_local = close_pairs[1][mask]
    
    print(f"   SG pairs within 3.0 Å: {len(idx1_local)}")
    for i, (i1, i2) in enumerate(zip(idx1_local, idx2_local)):
        sg1 = sg_atoms.iloc[i1.item()]
        sg2 = sg_atoms.iloc[i2.item()]
        dist = distances[i1, i2].item()
        print(f"     {sg1['chainid']}:{sg1['resname']}{sg1['resseq']} -- "
              f"{sg2['chainid']}:{sg2['resname']}{sg2['resseq']}  ({dist:.3f} Å)")

# Create restraints
print("\n2. Building restraints...")
cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
restraints = Restraints(model, cif_path, verbose=2)

print("\n3. Restraint statistics:")
print("-" * 80)

# Intra-residue bonds
if restraints.bond_indices is not None:
    n_bonds_intra = restraints.bond_indices.shape[0]
    print(f"Intra-residue bonds: {n_bonds_intra}")
else:
    n_bonds_intra = 0
    print(f"Intra-residue bonds: None")

# Inter-residue bonds (peptide bonds + disulfide bonds)
if restraints.bond_indices_inter is not None:
    n_bonds_inter = restraints.bond_indices_inter.shape[0]
    print(f"Inter-residue bonds total: {n_bonds_inter}")
    
    # Try to separate peptide bonds from disulfide bonds
    # Peptide bonds have distance ~1.337 Å, disulfide bonds have distance ~2.031 Å
    peptide_mask = restraints.bond_references_inter < 1.5
    disulfide_mask = restraints.bond_references_inter > 1.5
    
    n_peptide = peptide_mask.sum().item()
    n_disulfide = disulfide_mask.sum().item()
    
    print(f"  - Peptide bonds (< 1.5 Å): {n_peptide}")
    print(f"  - Disulfide bonds (> 1.5 Å): {n_disulfide}")
    
    if n_disulfide > 0:
        print("\n4. Disulfide bonds found:")
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
    else:
        print("\n4. No disulfide bonds found")
else:
    n_bonds_inter = 0
    print(f"Inter-residue bonds: None")

print("\n5. Full summary:")
print("=" * 80)
restraints.summary()
