#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Edge case testing for RestraintCIFReader
Tests 1000 random files from the monomer library
"""

import sys
from pathlib import Path
from collections import defaultdict
import random
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.cif_readers import RestraintCIFReader

def test_restraint_cif(cif_path):
    """Test a single restraint CIF file and return results"""
    result = {
        'file': str(cif_path),
        'success': False,
        'error_type': None,
        'error_message': None,
        'compound_id': None,
        'has_bonds': False,
        'has_angles': False,
        'has_torsions': False,
        'has_planes': False,
        'has_chirality': False,
        'num_bonds': 0,
        'num_angles': 0,
        'bond_has_dist_values': False,
        'angle_has_value': False,
        'data_sections': []
    }
    
    try:
        reader = RestraintCIFReader(cif_path)
        result['compound_id'] = reader.get_compound_id()
        comp_id = result['compound_id']
        
        # Check what data is available
        result['has_bonds'] = reader.has_bond_restraints()
        result['has_angles'] = reader.has_angle_restraints()
        result['has_torsions'] = reader.has_torsion_restraints()
        result['has_planes'] = reader.has_plane_restraints()
        result['has_chirality'] = reader.has_chirality_restraints()
        
        # Get available data sections
        result['data_sections'] = list(reader.cif.data.keys())
        
        # Try to extract bond restraints
        if result['has_bonds']:
            try:
                bonds = reader.get_bond_restraints(comp_id)
                result['num_bonds'] = len(bonds)
                
                # Check if bond restraints have distance values
                if 'value_dist' in bonds.columns:
                    result['bond_has_dist_values'] = True
                    
            except Exception as e:
                result['error_type'] = 'bond_extraction'
                result['error_message'] = str(e)
                raise
        
        # Try to extract angle restraints
        if result['has_angles']:
            try:
                # Get from all_restraints instead since there's no get_angle_restraints method
                restraints = reader.get_compound_restraints(comp_id)
                angles = restraints['angles']
                result['num_angles'] = len(angles)
                
                # Check if angle restraints have values
                if 'value_angle' in angles.columns or '_chem_comp_angle.value_angle' in angles.columns:
                    result['angle_has_value'] = True
                    
            except Exception as e:
                result['error_type'] = 'angle_extraction'
                result['error_message'] = str(e)
                raise
        
        # Try to extract torsion restraints
        if result['has_torsions']:
            try:
                restraints = reader.get_compound_restraints(comp_id)
                torsions = restraints['torsions']
            except Exception as e:
                result['error_type'] = 'torsion_extraction'
                result['error_message'] = str(e)
                raise
        
        # Try to get all restraints
        try:
            all_restraints = reader.get_all_restraints()
        except Exception as e:
            result['error_type'] = 'all_restraints_extraction'
            result['error_message'] = str(e)
            raise
        
        # If we got here, success!
        result['success'] = True
        
    except Exception as e:
        if result['error_type'] is None:
            result['error_type'] = type(e).__name__
            result['error_message'] = str(e)
    
    return result

def main():
    print("=" * 80)
    print("EDGE CASE TESTING: RestraintCIFReader")
    print("=" * 80)
    print()
    
    # Find all restraint files in monomer library
    library_dir = Path('/das/work/p17/p17490/Peter/Library/multicopy_refinement/external_monomer_library')
    
    print("Scanning monomer library...")
    all_cif_files = []
    
    # Get CIF files from each subdirectory
    for subdir in library_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            all_cif_files.extend(list(subdir.glob('*.cif')))
    
    print(f"Found {len(all_cif_files)} total restraint files in library")
    
    # Sample 1000 random files (or all if less than 1000)
    num_to_test = min(1000, len(all_cif_files))
    test_files = random.sample(all_cif_files, num_to_test)
    test_files.sort()  # Sort for reproducibility
    
    print(f"Testing {num_to_test} randomly selected files")
    print()
    
    # Test each file
    results = []
    error_types = defaultdict(list)
    
    for i, cif_file in enumerate(test_files, 1):
        if i % 100 == 0 or i == 1:
            print(f"[{i}/{num_to_test}] Testing: {cif_file.name}")
        
        result = test_restraint_cif(cif_file)
        results.append(result)
        
        if not result['success']:
            if i % 100 == 0 or i <= 10:  # Show first 10 and every 100th
                print(f"  ✗ FAILED: {result['error_type']}")
                print(f"    Error: {result['error_message'][:100]}")
            error_types[result['error_type']].append(result)
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"Total files tested: {len(results)}")
    print(f"Successful: {successful} ({100*successful/len(results):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(results):.1f}%)")
    print()
    
    # Statistics on successful files
    if successful > 0:
        print("STATISTICS FROM SUCCESSFUL FILES:")
        print("-" * 80)
        successful_results = [r for r in results if r['success']]
        
        has_bonds = sum(1 for r in successful_results if r['has_bonds'])
        has_angles = sum(1 for r in successful_results if r['has_angles'])
        has_torsions = sum(1 for r in successful_results if r['has_torsions'])
        has_planes = sum(1 for r in successful_results if r['has_planes'])
        has_chirality = sum(1 for r in successful_results if r['has_chirality'])
        
        bond_with_dist = sum(1 for r in successful_results if r['bond_has_dist_values'])
        angle_with_val = sum(1 for r in successful_results if r['angle_has_value'])
        
        print(f"Files with bond restraints: {has_bonds} ({100*has_bonds/successful:.1f}%)")
        print(f"Files with angle restraints: {has_angles} ({100*has_angles/successful:.1f}%)")
        print(f"Files with torsion restraints: {has_torsions} ({100*has_torsions/successful:.1f}%)")
        print(f"Files with plane restraints: {has_planes} ({100*has_planes/successful:.1f}%)")
        print(f"Files with chirality restraints: {has_chirality} ({100*has_chirality/successful:.1f}%)")
        print()
        print(f"Bond restraints with distance values: {bond_with_dist} ({100*bond_with_dist/has_bonds:.1f}% of files with bonds)")
        print(f"Angle restraints with angle values: {angle_with_val} ({100*angle_with_val/has_angles:.1f}% of files with angles)")
        print()
    
    if error_types:
        print("ERROR BREAKDOWN:")
        print("-" * 80)
        for error_type, error_results in sorted(error_types.items(), key=lambda x: -len(x[1])):
            print(f"\n{error_type}: {len(error_results)} files ({100*len(error_results)/len(results):.1f}%)")
            print(f"  Example files:")
            for res in error_results[:5]:
                print(f"    - {Path(res['file']).name}")
            if len(error_results) > 5:
                print(f"    ... and {len(error_results)-5} more")
            
            # Show unique error messages
            unique_messages = list(set(r['error_message'] for r in error_results if r['error_message']))
            if unique_messages:
                print(f"  Unique error messages ({len(unique_messages)}):")
                for msg in unique_messages[:3]:
                    print(f"    - {msg[:150]}")
                if len(unique_messages) > 3:
                    print(f"    ... and {len(unique_messages)-3} more")
    
    # Save detailed results
    output_file = Path(__file__).parent / 'restraint_reader_edgecase_results.txt'
    with open(output_file, 'w') as f:
        f.write("RESTRAINT CIF READER - EDGE CASE TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Tested {len(results)} files from monomer library\n")
        f.write(f"Successful: {successful}, Failed: {failed}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Write all results
        for result in results:
            f.write(f"File: {result['file']}\n")
            f.write(f"Success: {result['success']}\n")
            if not result['success']:
                f.write(f"Error Type: {result['error_type']}\n")
                f.write(f"Error Message: {result['error_message']}\n")
            else:
                f.write(f"Compound ID: {result['compound_id']}\n")
                f.write(f"Has Bonds: {result['has_bonds']} (n={result['num_bonds']})\n")
                f.write(f"Has Angles: {result['has_angles']} (n={result['num_angles']})\n")
                f.write(f"Has Torsions: {result['has_torsions']}\n")
                f.write(f"Has Planes: {result['has_planes']}\n")
                f.write(f"Bond has dist values: {result['bond_has_dist_values']}\n")
                f.write(f"Angle has values: {result['angle_has_value']}\n")
            f.write("\n" + "-" * 80 + "\n\n")
        
        # Write error summary
        if error_types:
            f.write("\n" + "=" * 80 + "\n")
            f.write("ERROR SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            for error_type, error_results in sorted(error_types.items(), key=lambda x: -len(x[1])):
                f.write(f"{error_type}: {len(error_results)} files\n")
                f.write(f"Files: {', '.join([Path(r['file']).name for r in error_results[:20]])}\n")
                if len(error_results) > 20:
                    f.write(f"... and {len(error_results)-20} more\n")
                f.write("\n")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
