#!/usr/bin/env python3
"""
Demonstration that MET.cif now parses correctly.

This script tests the specific file mentioned in the request:
/das/work/p17/p17490/Peter/Library/multicopy_refinement/external_monomer_library/m/MET.cif
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.restraints_helper import read_cif

print("Testing MET.cif parsing")
print("=" * 70)

cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/external_monomer_library/m/MET.cif'

try:
    result = read_cif(cif_path)
    
    if 'MET' in result:
        met_data = result['MET']
        print(f"✓ Successfully parsed MET.cif")
        print(f"✓ Found {len(met_data)} data sections\n")
        
        print("Data sections found:")
        for section_name, df in met_data.items():
            print(f"  {section_name:40s} {len(df):3d} entries")
        
        # Show bond data specifically
        if '_chem_comp_bond' in met_data:
            bonds = met_data['_chem_comp_bond']
            print(f"\n✓ Bond restraints found: {len(bonds)} bonds")
            print(f"  Columns: {', '.join(bonds.columns)}")
            print(f"\n  Sample bonds:")
            for i in range(min(5, len(bonds))):
                row = bonds.iloc[i]
                print(f"    {row['atom_id_1']:4s} - {row['atom_id_2']:4s}  "
                      f"{row['type']:7s}  {row['value_dist']} Å")
        
        # Show angle data specifically  
        if '_chem_comp_angle' in met_data:
            angles = met_data['_chem_comp_angle']
            print(f"\n✓ Angle restraints found: {len(angles)} angles")
            print(f"  Sample angles:")
            for i in range(min(5, len(angles))):
                row = angles.iloc[i]
                print(f"    {row['atom_id_1']:4s} - {row['atom_id_2']:4s} - {row['atom_id_3']:4s}  "
                      f"{row['value_angle']}°")
        
        print("\n" + "=" * 70)
        print("SUCCESS: MET.cif parses correctly with all restraints!")
        
    else:
        print("✗ ERROR: MET component not found in parsed data")
        
except Exception as e:
    print(f"✗ ERROR: Failed to parse MET.cif")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
