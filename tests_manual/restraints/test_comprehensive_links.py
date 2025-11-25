#!/usr/bin/env python3
"""
Comprehensive test for peptide bonds and disulfide bonds
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import Model
from torchref.restraints import Restraints

print("=" * 80)
print("COMPREHENSIVE TEST: PEPTIDE BONDS AND DISULFIDE BONDS")
print("=" * 80)

# Load the model
print("\n1. Loading model...")
model = Model()
model.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

print(f"   Total atoms: {len(model.pdb)}")
print(f"   Chains: {sorted(model.pdb['chainid'].unique())}")

# Check for cysteines
cys_residues = model.pdb[model.pdb['resname'] == 'CYS']
print(f"   Cysteine residues: {cys_residues['resseq'].nunique()}")

# Create restraints
print("\n2. Building restraints (with verbose output)...")
cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
restraints = Restraints(model, cif_path, verbose=1)

print("\n3. Summary of restraints:")
print("=" * 80)
restraints.summary()

print("\n4. Analysis:")
print("=" * 80)

# Count protein residues to verify peptide bond count
protein_atoms = model.pdb[model.pdb['ATOM'] == 'ATOM']
total_expected_peptide_bonds = 0
for chain_id in protein_atoms['chainid'].unique():
    chain = protein_atoms[protein_atoms['chainid'] == chain_id]
    n_residues = chain['resseq'].nunique()
    expected = n_residues - 1
    total_expected_peptide_bonds += expected
    print(f"Chain {chain_id}: {n_residues} residues → {expected} peptide bonds expected")

print(f"\nTotal expected peptide bonds: {total_expected_peptide_bonds}")

if restraints.bond_indices_inter is not None:
    n_inter = restraints.bond_indices_inter.shape[0]
    # Separate by distance
    peptide_mask = restraints.bond_references_inter < 1.5
    disulfide_mask = restraints.bond_references_inter > 1.5
    n_peptide = peptide_mask.sum().item()
    n_disulfide = disulfide_mask.sum().item()
    
    print(f"Total actual inter-residue bonds: {n_inter}")
    print(f"  - Peptide bonds: {n_peptide}")
    print(f"  - Disulfide bonds: {n_disulfide}")
    
    if n_peptide == total_expected_peptide_bonds:
        print("\n✓ SUCCESS: Peptide bond count matches!")
    else:
        print(f"\n✗ ERROR: Expected {total_expected_peptide_bonds} peptide bonds, got {n_peptide}")
    
    if n_disulfide == 0:
        print("✓ Correctly found no disulfide bonds in this structure")
else:
    print("✗ ERROR: No inter-residue bonds found!")

print("\n5. Testing complete!")
print("=" * 80)
