#!/usr/bin/env python3
"""Debug why we only see plan-1 planes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.model import Model
from torchref.restraints import Restraints

# Load test structure
pdb_file = Path(__file__).parent.parent.parent / "test_data" / "dark.pdb"
model = Model()
model.load_pdb_from_file(str(pdb_file))

# Build restraints
print("Building restraints...")
restraints = Restraints(model, verbose=0)

# Check plane counts
plane_data = restraints.restraints['plane']['4_atoms']
plane_indices = plane_data['indices']
plane_sigmas = plane_data['sigmas']

print(f"\nTotal 4-atom planes: {len(plane_indices)}")

# Identify peptide planes (inter-residue)
pdb = model.pdb
peptide_count = 0
aromatic_count = 0

for i, indices in enumerate(plane_indices):
    # Get residue info
    atom_residues = []
    atom_names = []
    for idx in indices:
        atom = pdb.iloc[idx.item()]
        atom_residues.append((atom['chainid'], atom['resseq']))
        atom_names.append(atom['name'])
    
    # Check if inter-residue (peptide) or intra-residue (aromatic)
    unique_residues = set(atom_residues)
    if len(unique_residues) > 1:
        peptide_count += 1
        if peptide_count <= 5:  # Show first 5
            print(f"\nPeptide plane {peptide_count}:")
            for (chain, resseq), name in zip(atom_residues, atom_names):
                print(f"  {chain}:{resseq} {name}")
    else:
        aromatic_count += 1

print(f"\n📊 Summary:")
print(f"  Peptide planes: {peptide_count}")
print(f"  Aromatic planes: {aromatic_count}")
print(f"  Total: {len(plane_indices)}")

# Check if we have H atoms
print(f"\n🔍 Checking for backbone H atoms...")
h_atoms = pdb[(pdb['name'] == 'H') & (pdb['ATOM'] == 'ATOM')]
print(f"  Found {len(h_atoms)} backbone H atoms")

if len(h_atoms) == 0:
    print("\n⚠️  NO backbone H atoms found!")
    print("     This explains why plan-2 planes are missing.")
    print("     Plan-2 includes the H atom (C-N-H-CA).")
