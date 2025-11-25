#!/usr/bin/env python3
"""Test GTP comprehensive parsing"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.restraints_helper import read_cif

cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/external_monomer_library/g/GTP.cif'

result = read_cif(cif_path)

print(f"Found {len(result)} components")
for comp_id, data in result.items():
    print(f"\nComponent: {comp_id}")
    print(f"  Number of sections: {len(data)}")
    for section_name, df in data.items():
        print(f"    {section_name}: {len(df)} rows, {len(df.columns)} columns")
        if section_name == '_chem_comp_bond':
            print(f"      Columns: {list(df.columns)}")
            print(f"      First few rows:")
            print(df.head())
