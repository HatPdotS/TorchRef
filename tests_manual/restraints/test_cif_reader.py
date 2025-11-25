#!/usr/bin/env python3
"""Test script to debug CIF reading issues"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.restraints_helper import read_cif

# Test with MET.cif
cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/external_monomer_library/m/MET.cif'

print(f"Testing CIF reading with: {cif_path}")
print("=" * 80)

try:
    result = read_cif(cif_path)
    print(f"✓ Successfully read CIF file")
    print(f"✓ Found {len(result)} components")
    for comp_id, data in result.items():
        print(f"\nComponent: {comp_id}")
        print(f"  Data sections: {list(data.keys())}")
        for section_name, df in data.items():
            print(f"    {section_name}: {len(df)} entries, columns: {list(df.columns)}")
except Exception as e:
    print(f"✗ Error reading CIF file:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
