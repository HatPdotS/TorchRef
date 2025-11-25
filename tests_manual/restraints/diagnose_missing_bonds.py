#!/usr/bin/env python3
"""
Investigate why bond count is low
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import Model
from torchref.restraints import Restraints
import pandas as pd

# Load the model
model = Model()
test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb'
model.load_pdb_from_file(test_pdb)

pdb = model.pdb

print("=" * 70)
print("RESIDUE TYPE ANALYSIS")
print("=" * 70)

# Check residue types
residue_counts = pdb.groupby('resname').size().sort_values(ascending=False)
print(f"\nTotal residues in model: {len(pdb.groupby(['chainid', 'resseq']))}")
print(f"Unique residue types: {len(residue_counts)}")
print(f"\nTop 10 residue types by atom count:")
for resname, count in residue_counts.head(10).items():
    print(f"  {resname:6s}: {count:4d} atoms")

# Check ATOM vs HETATM
atom_types = pdb['ATOM'].value_counts()
print(f"\nATOM record types:")
for atom_type, count in atom_types.items():
    print(f"  {atom_type}: {count:,} atoms")

# Check how many residues are being processed
print("\n" + "=" * 70)
print("CHECKING RESTRAINT BUILDING")
print("=" * 70)

cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'

# Read the CIF to see what's in it
from torchref.restraints_helper import read_cif
cif_dict = read_cif(cif_path)

print(f"\nResidues in CIF file: {list(cif_dict.keys())}")

# Check what the restraints builder is doing
restraints = Restraints(model, cif_path)

# Count bonds per residue type
print("\n" + "=" * 70)
print("RESIDUE PROCESSING CHECK")
print("=" * 70)

# Get unique residue combinations
unique_residues = pdb.groupby(['chainid', 'resseq', 'resname']).size()
print(f"\nTotal residues in model: {len(unique_residues)}")

# Count by ATOM type
hetatm_residues = pdb[pdb['ATOM'] == 'HETATM'].groupby(['chainid', 'resseq', 'resname']).size()
atom_residues = pdb[pdb['ATOM'] == 'ATOM'].groupby(['chainid', 'resseq', 'resname']).size()

print(f"ATOM residues: {len(atom_residues)}")
print(f"HETATM residues: {len(hetatm_residues)}")

if len(hetatm_residues) > 0:
    print(f"\nHETATM residue types:")
    hetatm_types = pdb[pdb['ATOM'] == 'HETATM']['resname'].value_counts()
    for resname, count in hetatm_types.items():
        num_residues = len(pdb[(pdb['ATOM'] == 'HETATM') & (pdb['resname'] == resname)].groupby(['chainid', 'resseq']))
        print(f"  {resname:6s}: {num_residues:3d} residues, {count:4d} atoms")

print("\n" + "=" * 70)
print("ISSUE DIAGNOSIS")
print("=" * 70)
print("\nPossible reasons for low bond count:")
print("1. HETATM residues are being skipped (check restraints_helper.py)")
print("2. Residues not in CIF dictionary are being skipped")
print("3. Incomplete atom names in residues (missing atoms)")
print("4. Bug in bond restraint building code")
