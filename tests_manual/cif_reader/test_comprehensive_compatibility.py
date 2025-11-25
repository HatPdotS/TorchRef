#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Comprehensive test demonstrating that CIF readers are compatible with legacy readers.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.cif_readers import ReflectionCIFReader, ModelCIFReader
from torchref.legacy_format_readers import MTZ, PDB

def test_reflection_reader():
    """Test that ReflectionCIFReader returns data compatible with MTZ reader."""
    print("=" * 80)
    print("TEST 1: ReflectionCIFReader compatibility")
    print("=" * 80)
    
    # Test with a structure factor CIF
    sf_cif = Path('scientific_testing/data/3E98/3E98-sf.cif')
    
    if not sf_cif.exists():
        print(f"❌ File not found: {sf_cif}")
        return False
    
    print(f"\n📁 Loading: {sf_cif}")
    
    # Test the callable interface (like MTZ)
    reader = ReflectionCIFReader(str(sf_cif), verbose=0)
    data_dict, cell, spacegroup = reader()
    
    print(f"\n✅ Successfully called reader as: reader()")
    print(f"   Returns: (dict, ndarray, str)")
    
    # Validate return types
    assert isinstance(data_dict, dict), "data_dict should be a dict"
    assert isinstance(cell, np.ndarray), "cell should be a numpy array"
    assert isinstance(spacegroup, str), "spacegroup should be a string"
    
    print(f"\n📊 Data dictionary keys: {list(data_dict.keys())}")
    print(f"   Cell: {cell}")
    print(f"   Spacegroup: {spacegroup}")
    
    # Validate required keys (MTZ-compatible)
    required_keys = ['h', 'k', 'l']
    for key in required_keys:
        assert key in data_dict, f"Missing required key: {key}"
        assert isinstance(data_dict[key], np.ndarray), f"{key} should be numpy array"
    
    # Check optional keys
    optional_keys = ['F', 'SIGF', 'I', 'SIGI', 'R-free-flags']
    present_keys = [k for k in optional_keys if k in data_dict]
    print(f"   Present optional keys: {present_keys}")
    
    # Validate column metadata (like MTZ)
    if 'F' in data_dict:
        assert 'F_col' in data_dict, "Should have F_col metadata"
        print(f"   ✓ F source: {data_dict['F_col']}")
    
    if 'I' in data_dict:
        assert 'I_col' in data_dict, "Should have I_col metadata"
        print(f"   ✓ I source: {data_dict['I_col']}")
    
    if 'R-free-flags' in data_dict:
        assert 'R-free-source' in data_dict, "Should have R-free-source metadata"
        print(f"   ✓ R-free source: {data_dict['R-free-source']}")
    
    # Validate data consistency
    n_reflections = len(data_dict['h'])
    print(f"\n   Total reflections: {n_reflections}")
    
    for key in ['h', 'k', 'l', 'F', 'SIGF', 'I', 'SIGI']:
        if key in data_dict:
            assert len(data_dict[key]) == n_reflections, f"{key} has wrong length"
    
    print("\n✅ ReflectionCIFReader is fully compatible with MTZ reader interface!")
    return True


def test_model_reader():
    """Test that ModelCIFReader returns data compatible with PDB reader."""
    print("\n" + "=" * 80)
    print("TEST 2: ModelCIFReader compatibility")
    print("=" * 80)
    
    # Test with a model CIF
    model_cif = Path('scientific_testing/data/3E98/3E98.cif')
    
    if not model_cif.exists():
        print(f"❌ File not found: {model_cif}")
        return False
    
    print(f"\n📁 Loading: {model_cif}")
    
    # Test the callable interface (like PDB)
    reader = ModelCIFReader(str(model_cif), verbose=0)
    dataframe, cell, spacegroup = reader()
    
    print(f"\n✅ Successfully called reader as: reader()")
    print(f"   Returns: (DataFrame, list, str)")
    
    # Validate return types
    assert isinstance(dataframe, pd.DataFrame), "dataframe should be a pandas DataFrame"
    assert isinstance(cell, list) or cell is None, "cell should be a list or None"
    assert isinstance(spacegroup, str), "spacegroup should be a string"
    
    print(f"\n📊 DataFrame shape: {dataframe.shape}")
    print(f"   Cell: {cell}")
    print(f"   Spacegroup: {spacegroup}")
    
    # Validate required columns (PDB-compatible)
    required_cols = ['ATOM', 'serial', 'name', 'altloc', 'resname', 'chainid', 
                     'resseq', 'icode', 'x', 'y', 'z', 'occupancy', 'tempfactor', 
                     'element', 'charge']
    
    for col in required_cols:
        assert col in dataframe.columns, f"Missing required column: {col}"
    
    print(f"   ✓ All required columns present")
    
    # Validate anisotropic data columns
    aniso_cols = ['anisou_flag', 'u11', 'u22', 'u33', 'u12', 'u13', 'u23']
    for col in aniso_cols:
        assert col in dataframe.columns, f"Missing aniso column: {col}"
    
    print(f"   ✓ Anisotropic data columns present")
    
    # Validate index column (legacy PDB format)
    assert 'index' in dataframe.columns, "Missing index column"
    assert (dataframe['index'] == np.arange(len(dataframe))).all(), "index should be sequential"
    print(f"   ✓ Index column present and sequential")
    
    # Validate DataFrame attributes (like legacy PDB)
    assert hasattr(dataframe, 'attrs'), "DataFrame should have attrs"
    assert 'cell' in dataframe.attrs, "attrs should have cell"
    assert 'spacegroup' in dataframe.attrs, "attrs should have spacegroup"
    print(f"   ✓ DataFrame attrs present: {list(dataframe.attrs.keys())}")
    
    # Validate column dtypes
    assert dataframe['x'].dtype == np.float64, "x should be float64"
    assert dataframe['y'].dtype == np.float64, "y should be float64"
    assert dataframe['z'].dtype == np.float64, "z should be float64"
    assert dataframe['serial'].dtype == np.int64, "serial should be int64"
    assert dataframe['resseq'].dtype == np.int64, "resseq should be int64"
    print(f"   ✓ Column dtypes correct")
    
    # Show sample data
    print(f"\n   First 3 atoms:")
    print(dataframe[['ATOM', 'name', 'resname', 'chainid', 'resseq', 'x', 'y', 'z']].head(3).to_string(index=False))
    
    print("\n✅ ModelCIFReader is fully compatible with PDB reader interface!")
    return True


def test_chained_interface():
    """Test the .read() method chaining interface."""
    print("\n" + "=" * 80)
    print("TEST 3: Method chaining (.read() interface)")
    print("=" * 80)
    
    sf_cif = Path('scientific_testing/data/3E98/3E98-sf.cif')
    model_cif = Path('scientific_testing/data/3E98/3E98.cif')
    
    # Test ReflectionCIFReader with .read()
    print("\n1. Testing: ReflectionCIFReader().read(filepath)")
    reader = ReflectionCIFReader(str(sf_cif)).read()
    data, cell, sg = reader()
    print(f"   ✓ Works! Got {len(data['h'])} reflections")
    
    # Test ModelCIFReader with .read()
    print("\n2. Testing: ModelCIFReader().read(filepath)")
    reader = ModelCIFReader(str(model_cif)).read()
    df, cell, sg = reader()
    print(f"   ✓ Works! Got {len(df)} atoms")
    
    print("\n✅ Method chaining works like legacy readers (MTZ().read(), PDB().read())")
    return True


if __name__ == '__main__':
    print("\n" + "🧪" * 40)
    print("COMPREHENSIVE LEGACY COMPATIBILITY TEST")
    print("🧪" * 40)
    
    results = []
    
    # Run all tests
    results.append(("ReflectionCIFReader", test_reflection_reader()))
    results.append(("ModelCIFReader", test_model_reader()))
    results.append(("Method Chaining", test_chained_interface()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {name}")
    
    all_passed = all(p for _, p in results)
    
    if all_passed:
        print("\n" + "🎉" * 40)
        print("ALL TESTS PASSED!")
        print("CIF readers are fully compatible with legacy MTZ/PDB readers!")
        print("🎉" * 40)
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
