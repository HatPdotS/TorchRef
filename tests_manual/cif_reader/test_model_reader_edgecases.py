#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Edge case testing for ModelCIFReader
Tests all model CIF files in scientific_testing/data
"""

import sys
from pathlib import Path
from collections import defaultdict
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.cif_readers import ModelCIFReader

def test_model_cif(cif_path):
    """Test a single model CIF file and return results"""
    result = {
        'file': str(cif_path),
        'success': False,
        'error_type': None,
        'error_message': None,
        'num_atoms': 0,
        'has_coordinates': False,
        'has_cell': False,
        'has_spacegroup': False,
        'has_occupancy': False,
        'has_bfactor': False,
        'has_aniso': False,
        'columns_found': []
    }
    
    try:
        reader = ModelCIFReader(cif_path)
        
        # Check what data is available
        result['has_coordinates'] = reader.has_coordinates()
        result['has_cell'] = reader.has_cell_parameters()
        result['has_spacegroup'] = reader.has_space_group()
        result['has_occupancy'] = reader.has_occupancy()
        result['has_bfactor'] = reader.has_bfactor()
        result['has_aniso'] = reader.has_anisotropic_data()
        
        # Get columns
        if 'atom_site' in reader.cif.data:
            result['columns_found'] = list(reader.cif.data['atom_site'].columns)
            result['num_atoms'] = len(reader.cif.data['atom_site'])
        
        # Try to extract data
        try:
            coords = reader.get_coordinates()
            if coords is not None:
                result['num_atoms'] = len(coords)
        except Exception as e:
            result['error_type'] = 'coordinate_extraction'
            result['error_message'] = str(e)
            raise
        
        # Try to get atom info
        try:
            atom_info = reader.get_atom_info()
        except Exception as e:
            result['error_type'] = 'atom_info_extraction'
            result['error_message'] = str(e)
            raise
        
        # Try to get cell
        if result['has_cell']:
            try:
                cell = reader.get_cell_parameters()
            except Exception as e:
                result['error_type'] = 'cell_extraction'
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
    print("EDGE CASE TESTING: ModelCIFReader")
    print("=" * 80)
    print()
    
    # Find all model CIF files
    data_dir = Path('/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data')
    model_files = list(data_dir.glob('*/*.cif'))
    
    # Filter out structure factor files
    model_files = [f for f in model_files if not f.name.endswith('-sf.cif')]
    
    print(f"Found {len(model_files)} model CIF files")
    print()
    
    # Test each file
    results = []
    error_types = defaultdict(list)
    
    for i, cif_file in enumerate(model_files, 1):
        print(f"[{i}/{len(model_files)}] Testing: {cif_file.name}")
        result = test_model_cif(cif_file)
        results.append(result)
        
        if result['success']:
            print(f"  ✓ SUCCESS")
            print(f"    - Atoms: {result['num_atoms']}")
            print(f"    - Has cell: {result['has_cell']}, Has spacegroup: {result['has_spacegroup']}")
            print(f"    - Has aniso: {result['has_aniso']}")
        else:
            print(f"  ✗ FAILED: {result['error_type']}")
            print(f"    Error: {result['error_message'][:100]}")
            error_types[result['error_type']].append(result)
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"Total files tested: {len(results)}")
    print(f"Successful: {successful} ({100*successful/len(results):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(results):.1f}%)")
    print()
    
    if error_types:
        print("ERROR BREAKDOWN:")
        print("-" * 80)
        for error_type, error_results in sorted(error_types.items()):
            print(f"\n{error_type}: {len(error_results)} files")
            print(f"  Files affected:")
            for res in error_results[:5]:
                print(f"    - {Path(res['file']).name}")
            if len(error_results) > 5:
                print(f"    ... and {len(error_results)-5} more")
            
            # Show unique error messages
            unique_messages = list(set(r['error_message'] for r in error_results if r['error_message']))
            if unique_messages:
                print(f"  Error messages:")
                for msg in unique_messages[:3]:
                    print(f"    - {msg[:150]}")
            
            # Show column patterns
            if error_results[0]['columns_found']:
                print(f"  Example columns from affected files:")
                print(f"    {error_results[0]['columns_found'][:10]}")
    
    # Save detailed results
    output_file = Path(__file__).parent / 'model_reader_edgecase_results.txt'
    with open(output_file, 'w') as f:
        f.write("MODEL CIF READER - EDGE CASE TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"File: {result['file']}\n")
            f.write(f"Success: {result['success']}\n")
            if not result['success']:
                f.write(f"Error Type: {result['error_type']}\n")
                f.write(f"Error Message: {result['error_message']}\n")
            f.write(f"Atoms: {result['num_atoms']}\n")
            f.write(f"Has Coordinates: {result['has_coordinates']}\n")
            f.write(f"Has Cell: {result['has_cell']}\n")
            f.write(f"Has Space Group: {result['has_spacegroup']}\n")
            f.write(f"Has Aniso: {result['has_aniso']}\n")
            if result['columns_found']:
                f.write(f"Columns: {', '.join(result['columns_found'][:20])}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
