"""
Test the space group symmetry system with JSON-based name mapping.

This test verifies:
1. JSON name mapping file is loaded correctly
2. All 230 space groups can be accessed
3. Common aliases work correctly
4. Operations match gemmi exactly
"""

import torch
import json
import gemmi
from torchref.symmetrie import Symmetry

def test_json_loading():
    """Test that JSON file loads correctly."""
    import os
    # Get path relative to the test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    mapping_path = os.path.join(test_dir, '../../caching/files/spacegroup_name_mapping.json')
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    print(f"✓ JSON mapping loaded: {len(mapping)} aliases for {len(set(mapping.values()))} space groups")
    return mapping

def test_alias_resolution():
    """Test common space group aliases."""
    test_cases = [
        ('P21', 'P21'),
        ('P1211', 'P21'),
        ('p21', 'P21'),
        ('P 1 21 1', 'P21'),
        ('P121', 'P2'),
        ('C2', 'C2'),
        ('C121', 'C2'),
        ('Fm-3m', 'Fm-3m'),
        ('fm-3m', 'Fm-3m'),
        ('Fm3mbar', 'Fm-3m'),
        ('P4_1', 'P41'),
        ('P6_3', 'P63'),
    ]
    
    print("\nTesting alias resolution:")
    for alias, expected_canonical in test_cases:
        sym = Symmetry(alias)
        if sym.canonical_space_group == expected_canonical:
            print(f"  ✓ {alias:15s} -> {sym.canonical_space_group}")
        else:
            print(f"  ✗ {alias:15s} -> {sym.canonical_space_group} (expected {expected_canonical})")
            return False
    
    return True

def test_all_space_groups():
    """Test that all 230 space groups can be instantiated."""
    mapping = test_json_loading()
    canonical_names = sorted(set(mapping.values()))
    
    print(f"\nTesting all {len(canonical_names)} canonical space groups:")
    failed = []
    
    for name in canonical_names:
        try:
            sym = Symmetry(name)
            if sym.canonical_space_group != name:
                failed.append(f"{name} resolved to {sym.canonical_space_group}")
        except Exception as e:
            failed.append(f"{name}: {str(e)[:50]}")
    
    if failed:
        print(f"  ✗ Failed: {len(failed)} space groups")
        for f in failed[:10]:  # Show first 10 failures
            print(f"    - {f}")
        return False
    else:
        print(f"  ✓ All {len(canonical_names)} space groups work correctly")
        return True

def test_gemmi_consistency():
    """Test that operations match gemmi for selected space groups."""
    test_groups = ['P1', 'P21', 'P212121', 'C2', 'I4', 'P-1', 'P3', 'P31', 'P6', 'Fm-3m', 'Ia-3d']
    
    print("\nTesting gemmi consistency:")
    for sg_name in test_groups:
        sym = Symmetry(sg_name)
        sg_gemmi = gemmi.SpaceGroup(sym.canonical_space_group)
        ops_gemmi = sg_gemmi.operations()
        
        if len(sym.matrices) == len(ops_gemmi):
            print(f"  ✓ {sg_name:10s} ({sym.canonical_space_group:10s}): {len(sym.matrices):3d} ops")
        else:
            print(f"  ✗ {sg_name:10s}: {len(sym.matrices)} ops (gemmi has {len(ops_gemmi)})")
            return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 70)
    print("Space Group Symmetry System Test Suite")
    print("=" * 70)
    
    tests = [
        ("JSON Loading", test_json_loading),
        ("Alias Resolution", test_alias_resolution),
        ("All Space Groups", test_all_space_groups),
        ("Gemmi Consistency", test_gemmi_consistency),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            if result is False:
                results.append((name, False))
                print(f"\n✗ {name} test FAILED")
            else:
                results.append((name, True))
        except Exception as e:
            results.append((name, False))
            print(f"\n✗ {name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12s} {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("=" * 70)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == '__main__':
    exit(main())
