#!/usr/bin/env python3
"""Comprehensive test of CIF reading across multiple files"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.restraints_helper import read_cif
from pathlib import Path

# Test with a broader set of residues including problem cases
test_residues = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'GTP',  # The problematic one with quotes
]

base_path = Path('/das/work/p17/p17490/Peter/Library/multicopy_refinement/external_monomer_library')
success_count = 0
fail_count = 0
detailed_results = []

for resname in test_residues:
    first_char = resname[0].lower()
    cif_path = base_path / first_char / f"{resname}.cif"
    
    try:
        result = read_cif(str(cif_path))
        if len(result) > 0:
            comp_id = list(result.keys())[0]
            num_sections = len(result[comp_id])
            
            # Check for critical sections
            has_bond = '_chem_comp_bond' in result[comp_id]
            has_angle = '_chem_comp_angle' in result[comp_id]
            
            status = "✓"
            if not has_bond:
                status = "⚠"  # Warning - missing bonds
                
            detailed_results.append({
                'name': resname,
                'status': status,
                'sections': num_sections,
                'has_bond': has_bond,
                'has_angle': has_angle
            })
            success_count += 1
        else:
            detailed_results.append({
                'name': resname,
                'status': "✗",
                'error': "No data found"
            })
            fail_count += 1
    except Exception as e:
        detailed_results.append({
            'name': resname,
            'status': "✗",
            'error': f"{type(e).__name__}: {str(e)[:50]}"
        })
        fail_count += 1

# Print results
print("CIF Reading Test Results")
print("=" * 70)
for r in detailed_results:
    if r['status'] == "✓":
        print(f"{r['status']} {r['name']:4s}: {r['sections']:2d} sections, "
              f"bonds={'Y' if r['has_bond'] else 'N'}, angles={'Y' if r['has_angle'] else 'N'}")
    else:
        error = r.get('error', 'Unknown error')
        print(f"{r['status']} {r['name']:4s}: {error}")

print("=" * 70)
print(f"Summary: {success_count}/{len(test_residues)} passed ({fail_count} failed)")

if fail_count == 0:
    print("\n🎉 All CIF files read successfully!")
