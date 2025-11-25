#!/usr/bin/env python3
"""Test script to verify CIF reading with multiple files"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.restraints_helper import read_cif
from pathlib import Path

# Test with several common amino acids
test_files = [
    'external_monomer_library/m/MET.cif',
    'external_monomer_library/a/ALA.cif',
    'external_monomer_library/g/GLY.cif',
    'external_monomer_library/l/LEU.cif',
    'external_monomer_library/t/TRP.cif',
]

base_path = Path('/das/work/p17/p17490/Peter/Library/multicopy_refinement')
success_count = 0
fail_count = 0

for test_file in test_files:
    cif_path = base_path / test_file
    res_name = test_file.split('/')[-1].replace('.cif', '')
    
    try:
        result = read_cif(str(cif_path))
        if len(result) > 0:
            comp_id = list(result.keys())[0]
            num_sections = len(result[comp_id])
            print(f"✓ {res_name:4s}: {num_sections} sections")
            success_count += 1
        else:
            print(f"✗ {res_name:4s}: No data found")
            fail_count += 1
    except Exception as e:
        print(f"✗ {res_name:4s}: {type(e).__name__}: {e}")
        fail_count += 1

print(f"\n{'='*50}")
print(f"Results: {success_count} passed, {fail_count} failed")
