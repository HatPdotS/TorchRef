#!/usr/bin/env python3
"""
Test script to verify peptide bond restraints are correctly built
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import Model
from torchref.restraints import Restraints

print("=" * 80)
print("TESTING PEPTIDE BOND RESTRAINTS")
print("=" * 80)

# Load the model
print("\n1. Loading model...")
model = Model()
model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

print(f"   Total atoms: {len(model.pdb)}")
print(f"   Chains: {model.pdb['chainid'].unique().tolist()}")

# Count protein residues per chain
protein_atoms = model.pdb[model.pdb['ATOM'] == 'ATOM']
print(f"   Protein atoms (ATOM): {len(protein_atoms)}")

for chain_id in protein_atoms['chainid'].unique():
    chain = protein_atoms[protein_atoms['chainid'] == chain_id]
    n_residues = chain['resseq'].nunique()
    print(f"     Chain {chain_id}: {n_residues} residues")

# Create restraints
print("\n2. Building restraints...")
cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
restraints = Restraints(model, cif_path, verbose=1)

print("\n3. Restraint statistics:")
print("-" * 80)

# Intra-residue bonds
if restraints.bond_indices is not None:
    n_bonds_intra = restraints.bond_indices.shape[0]
    print(f"Intra-residue bonds: {n_bonds_intra}")
else:
    n_bonds_intra = 0
    print(f"Intra-residue bonds: None")

# Inter-residue bonds (peptide bonds)
if restraints.bond_indices_inter is not None:
    n_bonds_inter = restraints.bond_indices_inter.shape[0]
    print(f"Inter-residue bonds (peptide bonds): {n_bonds_inter}")
    
    # Show a few examples
    print("\n4. Sample peptide bonds:")
    print("-" * 80)
    for i in range(min(10, n_bonds_inter)):
        idx1, idx2 = restraints.bond_indices_inter[i]
        idx1 = int(idx1.item())
        idx2 = int(idx2.item())
        atom1 = model.pdb.iloc[idx1]
        atom2 = model.pdb.iloc[idx2]
        ref_dist = float(restraints.bond_references_inter[i].item())
        sigma = float(restraints.bond_sigmas_inter[i].item())
        
        print(f"  {atom1['chainid']}:{atom1['resname']}{atom1['resseq']}:{atom1['name']} -- "
              f"{atom2['chainid']}:{atom2['resname']}{atom2['resseq']}:{atom2['name']}  "
              f"({ref_dist:.3f} ± {sigma:.4f} Å)")
else:
    n_bonds_inter = 0
    print(f"Inter-residue bonds (peptide bonds): None")

print("\n5. Expected vs actual:")
print("-" * 80)
# Count expected peptide bonds
expected_peptide_bonds = 0
for chain_id in protein_atoms['chainid'].unique():
    chain = protein_atoms[protein_atoms['chainid'] == chain_id]
    n_residues = chain['resseq'].nunique()
    # Each chain should have (n_residues - 1) peptide bonds
    expected = n_residues - 1
    expected_peptide_bonds += expected
    print(f"Chain {chain_id}: {n_residues} residues → {expected} expected peptide bonds")

print(f"\nTotal expected peptide bonds: {expected_peptide_bonds}")
print(f"Total actual peptide bonds:   {n_bonds_inter}")

if n_bonds_inter == expected_peptide_bonds:
    print("\n✓ SUCCESS: Peptide bond count matches expected!")
elif n_bonds_inter < expected_peptide_bonds:
    print(f"\n⚠ WARNING: Missing {expected_peptide_bonds - n_bonds_inter} peptide bonds")
else:
    print(f"\n⚠ WARNING: {n_bonds_inter - expected_peptide_bonds} extra peptide bonds")

print("\n6. Full summary:")
print("=" * 80)
restraints.summary()
