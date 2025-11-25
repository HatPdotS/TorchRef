#!/usr/bin/env python3
"""Debug script to understand what's happening with altlocs."""

import torch
import sys
import tempfile
import os

sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import Model

def create_altloc_pdb():
    pdb_content = """CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  1.00 20.00           C
ATOM      3  C   ALA A   1      12.000  12.000  12.000  1.00 20.00           C
ATOM      4  O   ALA A   1      13.000  13.000  13.000  1.00 20.00           O
ATOM      5  N  AVAL A   2      15.000  15.000  15.000  0.70 25.00           N
ATOM      6  CA AVAL A   2      16.000  16.000  16.000  0.70 25.00           C
ATOM      7  C  AVAL A   2      17.000  17.000  17.000  0.70 25.00           C
ATOM      8  O  AVAL A   2      18.000  18.000  18.000  0.70 25.00           O
ATOM      9  CB AVAL A   2      19.000  19.000  19.000  0.70 25.00           C
ATOM     10  N  BVAL A   2      15.000  15.000  15.000  0.30 25.00           N
ATOM     11  CA BVAL A   2      16.000  16.000  16.000  0.30 25.00           C
ATOM     12  C  BVAL A   2      17.000  17.000  17.000  0.30 25.00           C
ATOM     13  O  BVAL A   2      18.000  18.000  18.000  0.30 25.00           O
ATOM     14  CB BVAL A   2      19.000  19.000  19.000  0.30 25.00           C
END
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        return f.name

pdb_file = create_altloc_pdb()

try:
    model = Model(verbose=1)
    model.load_pdb_from_file(pdb_file)
    
    print("\n" + "="*70)
    print("DEBUGGING ALTLOC HANDLING")
    print("="*70)
    
    print(f"\nAltloc info: {len(model.occupancy.altloc_info)} groups")
    
    for i, altloc_dict in enumerate(model.occupancy.altloc_info):
        print(f"\nAltloc group {i}:")
        print(f"  Atom groups: {altloc_dict['atom_groups']}")
        print(f"  Collapsed indices: {altloc_dict['collapsed_indices']}")
        print(f"  Independent indices: {altloc_dict['independent_indices']}")
        print(f"  Placeholder collapsed idx: {altloc_dict['placeholder_collapsed_idx']}")
        print(f"  Placeholder conf idx: {altloc_dict['placeholder_conf_idx']}")
    
    print(f"\nExpansion mask: {model.occupancy.expansion_mask}")
    print(f"Altloc placeholder mask: {model.occupancy.altloc_placeholder_mask}")
    print(f"Refinable mask: {model.occupancy.refinable_mask}")
    
    # Get initial occupancies
    occ = model.occupancy()
    print(f"\nInitial occupancies (full space):")
    for i in range(len(occ)):
        print(f"  Atom {i}: {occ[i].item():.4f}")
    
    print(f"\nAltloc sums:")
    print(f"  Residue 2: atoms 4-8 + atoms 9-13 = {occ[4].item():.4f} + {occ[9].item():.4f} = {(occ[4] + occ[9]).item():.4f}")
    
    # Get collapsed values
    collapsed = model.occupancy.fixed_values.clone()
    collapsed[model.occupancy.refinable_mask] = model.occupancy.refinable_params
    print(f"\nCollapsed values (logit space):")
    for i in range(len(collapsed)):
        print(f"  Collapsed {i}: {collapsed[i].item():.4f} (sigmoid={torch.sigmoid(collapsed[i]).item():.4f})")
    
finally:
    os.remove(pdb_file)
