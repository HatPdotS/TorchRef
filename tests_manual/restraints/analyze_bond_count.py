#!/usr/bin/env python3
"""
Analyze bond restraints count vs atom count
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import Model
from torchref.restraints import Restraints

# Load the model
model = Model()
test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb'
model.load_pdb_from_file(test_pdb)

cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'
restraints = Restraints(model, cif_path)

# Get statistics
num_atoms = len(model.pdb)
num_bonds = len(restraints.bond_indices)
num_angles = len(restraints.angle_indices) if hasattr(restraints, 'angle_indices') else 0
num_torsions = len(restraints.torsion_indices) if hasattr(restraints, 'torsion_indices') else 0

print("=" * 70)
print("RESTRAINTS ANALYSIS")
print("=" * 70)
print(f"\nAtom Count: {num_atoms:,}")
print(f"Bond Restraints: {num_bonds:,}")
print(f"Bonds/Atom Ratio: {num_bonds/num_atoms:.2f}")

if num_angles > 0:
    print(f"Angle Restraints: {num_angles:,}")
    print(f"Angles/Atom Ratio: {num_angles/num_atoms:.2f}")

if num_torsions > 0:
    print(f"Torsion Restraints: {num_torsions:,}")
    print(f"Torsions/Atom Ratio: {num_torsions/num_atoms:.2f}")

print("\n" + "=" * 70)
print("EXPECTED VALUES FOR PROTEINS:")
print("=" * 70)
print("Typical protein atom types:")
print("  - Heavy atoms (C, N, O, S): ~3-4 bonds each")
print("  - Hydrogen atoms: 1 bond each")
print("  - Hydrogen ratio: ~50% of total atoms")
print()
print("Expected bonds/atom for proteins:")
print("  - Without hydrogens: ~1.8-2.2 bonds/atom")
print("  - With hydrogens: ~0.9-1.1 bonds/atom")
print()

# Count hydrogens vs heavy atoms
pdb = model.pdb
num_hydrogens = len(pdb[pdb['element'] == 'H'])
num_heavy = num_atoms - num_hydrogens

print(f"Heavy atoms: {num_heavy:,}")
print(f"Hydrogen atoms: {num_hydrogens:,}")
print(f"H-atom percentage: {100*num_hydrogens/num_atoms:.1f}%")

print("\n" + "=" * 70)
print("ASSESSMENT:")
print("=" * 70)

bonds_per_atom = num_bonds / num_atoms
expected_min = 0.8
expected_max = 1.2

if num_hydrogens > 0:  # Model has hydrogens
    expected_range = "0.9-1.1 bonds/atom (with H)"
else:  # All-atom model without H
    expected_range = "1.8-2.2 bonds/atom (no H)"
    expected_min = 1.7
    expected_max = 2.3

if expected_min <= bonds_per_atom <= expected_max:
    print(f"✓ Bond count looks REASONABLE")
    print(f"  {bonds_per_atom:.2f} bonds/atom is within expected range of {expected_range}")
else:
    print(f"⚠ Bond count may be UNUSUAL")
    print(f"  {bonds_per_atom:.2f} bonds/atom vs expected {expected_range}")
    if bonds_per_atom < expected_min:
        print(f"  → Fewer bonds than expected (possible missing restraints)")
    else:
        print(f"  → More bonds than expected (possible duplicate restraints)")

print("=" * 70)
