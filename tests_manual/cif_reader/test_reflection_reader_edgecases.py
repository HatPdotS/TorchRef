#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Edge case testing for ReflectionCIFReader
Tests all structure factor CIF files in scientific_testing/data
"""

import sys
from pathlib import Path
from collections import defaultdict
import traceback

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.cif_readers import ReflectionCIFReader

def test_reflection_cif(cif_path):
    """Test a single reflection CIF file and return results"""
    result = {
        'file': str(cif_path),
        'success': False,
        'error_type': None,
        'error_message': None,
        'has_indices': False,
        'has_amplitudes': False,
        'has_intensities': False,
        'has_phases': False,
        'has_rfree': False,
        'num_reflections': 0,
        'columns_found': []
    }
    
    try:
        reader = ReflectionCIFReader(cif_path)
        
        # Check what data is available
        result['has_indices'] = reader.has_miller_indices()
        result['has_amplitudes'] = reader.has_amplitudes()
        result['has_intensities'] = reader.has_intensities()
        result['has_phases'] = reader.has_phases()
        result['has_rfree'] = reader.has_rfree_flags()
        
        # Try to get available columns
        if 'refln' in reader.cif_reader:
            result['columns_found'] = list(reader.cif_reader['refln'].columns)
            result['num_reflections'] = len(reader.cif_reader['refln'])
        
        # Try to extract data (this is where most errors occur)
        try:
            indices = reader.get_miller_indices()
            if indices is not None:
                result['num_reflections'] = len(indices)
        except Exception as e:
            result['error_type'] = 'miller_indices_extraction'
            result['error_message'] = str(e)
            raise
            
        # Try amplitude extraction if available
        if result['has_amplitudes']:
            try:
                f_data = reader.get_amplitudes()
            except Exception as e:
                result['error_type'] = 'amplitude_extraction'
                result['error_message'] = str(e)
                raise
        
        # Try intensity extraction if available
        if result['has_intensities']:
            try:
                i_data = reader.get_intensities()
            except Exception as e:
                result['error_type'] = 'intensity_extraction'
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
    print("EDGE CASE TESTING: ReflectionCIFReader")
    print("=" * 80)
    print()
    
    # Find all structure factor CIF files
    data_dir = Path('/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data')
    sf_files = list(data_dir.glob('*/*-sf.cif'))
    
    print(f"Found {len(sf_files)} structure factor CIF files")
    print()
    
    # Test each file
    results = []
    error_types = defaultdict(list)
    
    for i, cif_file in enumerate(sf_files, 1):
        print(f"[{i}/{len(sf_files)}] Testing: {cif_file.name}")
        result = test_reflection_cif(cif_file)
        results.append(result)
        
        if result['success']:
            print(f"  ✓ SUCCESS")
            print(f"    - Reflections: {result['num_reflections']}")
            print(f"    - Has F: {result['has_amplitudes']}, Has I: {result['has_intensities']}")
            print(f"    - Has phases: {result['has_phases']}, Has R-free: {result['has_rfree']}")
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
            for res in error_results[:5]:  # Show first 5
                print(f"    - {Path(res['file']).name}")
            if len(error_results) > 5:
                print(f"    ... and {len(error_results)-5} more")
            
            # Show unique error messages
            unique_messages = list(set(r['error_message'] for r in error_results if r['error_message']))
            if unique_messages:
                print(f"  Error messages:")
                for msg in unique_messages[:3]:
                    print(f"    - {msg[:150]}")
            
            # Show column patterns for files with this error
            if error_results[0]['columns_found']:
                print(f"  Example columns from affected files:")
                print(f"    {error_results[0]['columns_found'][:10]}")
    
    # Save detailed results
    output_file = Path(__file__).parent / 'reflection_reader_edgecase_results.txt'
    with open(output_file, 'w') as f:
        f.write("REFLECTION CIF READER - EDGE CASE TEST RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            f.write(f"File: {result['file']}\n")
            f.write(f"Success: {result['success']}\n")
            if not result['success']:
                f.write(f"Error Type: {result['error_type']}\n")
                f.write(f"Error Message: {result['error_message']}\n")
            f.write(f"Reflections: {result['num_reflections']}\n")
            f.write(f"Has Amplitudes: {result['has_amplitudes']}\n")
            f.write(f"Has Intensities: {result['has_intensities']}\n")
            f.write(f"Has Phases: {result['has_phases']}\n")
            f.write(f"Has R-free: {result['has_rfree']}\n")
            if result['columns_found']:
                f.write(f"Columns: {', '.join(result['columns_found'][:20])}\n")
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
