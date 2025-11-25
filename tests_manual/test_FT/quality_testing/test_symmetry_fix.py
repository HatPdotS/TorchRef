#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

"""
Test script to verify the symmetry coordinate fix.
Creates a simple map with a peak at a known position and checks
if symmetry operations place it correctly.
"""

import torch
import numpy as np
from torchref.map_symmetry import MapSymmetry

def test_symmetry_coordinates():
    """Test that symmetry operations work correctly with the voxel-center convention."""
    
    # Simple cubic cell
    map_shape = (64, 64, 64)
    cell = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0])
    
    print("=" * 60)
    print("Testing MapSymmetry coordinate fix")
    print("=" * 60)
    
    # Test 1: P21 symmetry (2-fold screw along b)
    print("\n### Test 1: P21 Symmetry ###")
    print(f"Map shape: {map_shape}")
    
    sym = MapSymmetry('P21', map_shape, cell)
    print(f"Number of symmetry operations: {sym.n_ops}")
    
    # Create a map with a single sharp peak
    test_map = torch.zeros(map_shape, dtype=torch.float64)
    
    # Put peak at a specific position (not on symmetry elements)
    peak_i, peak_j, peak_k = 20, 30, 16
    test_map[peak_i, peak_j, peak_k] = 100.0
    
    # Add some width to make it easier to see
    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                i, j, k = peak_i + di, peak_j + dj, peak_k + dk
                if 0 <= i < map_shape[0] and 0 <= j < map_shape[1] and 0 <= k < map_shape[2]:
                    if (di, dj, dk) != (0, 0, 0):
                        test_map[i, j, k] = 50.0
    
    print(f"\nOriginal peak at grid index: [{peak_i}, {peak_j}, {peak_k}]")
    print(f"Original peak fractional: [{peak_i/map_shape[0]:.4f}, {peak_j/map_shape[1]:.4f}, {peak_k/map_shape[2]:.4f}]")
    print(f"Original map sum: {test_map.sum():.2f}")
    print(f"Original map max: {test_map.max():.2f}")
    
    # Apply P21 symmetry: -x, y+1/2, -z (this is the correct P21 operation)
    # Fractional position of peak (grid-edge convention: i/N)
    fx_orig = peak_i / map_shape[0]
    fy_orig = peak_j / map_shape[1]
    fz_orig = peak_k / map_shape[2]
    
    # Expected symmetry mate position (P21 operation: -x, y+1/2, -z)
    fx_sym = (1.0 - fx_orig) % 1.0  # -x wrapped
    fy_sym = (fy_orig + 0.5) % 1.0
    fz_sym = (1.0 - fz_orig) % 1.0  # -z wrapped
    
    # Expected grid indices
    expected_i = int(fx_sym * map_shape[0])
    expected_j = int(fy_sym * map_shape[1])
    expected_k = int(fz_sym * map_shape[2])
    
    print(f"\nExpected symmetry mate fractional: [{fx_sym:.4f}, {fy_sym:.4f}, {fz_sym:.4f}]")
    print(f"Expected symmetry mate grid index: [{expected_i}, {expected_j}, {expected_k}]")
    
    # Get just the symmetry mate (operation 1, not identity)
    sym_mate = sym.get_symmetry_mate(test_map, 1)
    
    print(f"\nSymmetry mate map sum: {sym_mate.sum():.2f}")
    print(f"Symmetry mate map max: {sym_mate.max():.2f}")
    
    # Find where the peak actually ended up
    max_val = sym_mate.max()
    max_idx = (sym_mate == max_val).nonzero(as_tuple=False)
    
    if len(max_idx) > 0:
        actual_i, actual_j, actual_k = max_idx[0].tolist()
        print(f"\nActual symmetry mate peak at grid index: [{actual_i}, {actual_j}, {actual_k}]")
        print(f"Actual symmetry mate peak fractional: [{actual_i/map_shape[0]:.4f}, {actual_j/map_shape[1]:.4f}, {actual_k/map_shape[2]:.4f}]")
        
        # Check if it's within 1 voxel of expected
        error_i = abs(actual_i - expected_i)
        error_j = abs(actual_j - expected_j)
        error_k = abs(actual_k - expected_k)
        
        print(f"\nPosition error (voxels): [{error_i}, {error_j}, {error_k}]")
        
        if max(error_i, error_j, error_k) <= 1:
            print("✓ PASS: Symmetry mate is within 1 voxel of expected position")
        else:
            print("✗ FAIL: Symmetry mate is more than 1 voxel off")
    else:
        print("✗ FAIL: No peak found in symmetry mate")
    
    # Test 2: Apply full symmetry and check total
    print("\n### Test 2: Full Symmetry Application ###")
    full_sym_map = sym(test_map)
    print(f"Full symmetric map sum: {full_sym_map.sum():.2f}")
    print(f"Ratio to original: {full_sym_map.sum() / test_map.sum():.2f}")
    print(f"Expected ratio: {sym.n_ops} (number of symmetry operations)")
    
    if abs(full_sym_map.sum() / test_map.sum() - sym.n_ops) < 0.1:
        print("✓ PASS: Total density is preserved correctly")
    else:
        print("✗ FAIL: Total density is not correct")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_symmetry_coordinates()
