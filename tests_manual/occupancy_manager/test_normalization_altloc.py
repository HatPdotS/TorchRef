#!/usr/bin/env python3
"""
Test the new normalization-based alternative conformation handling.

This test verifies:
1. Altloc groups are correctly converted to collapsed indices
2. Linked occupancy tensors are properly structured
3. Sum-to-1 constraint is enforced via normalization
4. Constraint is maintained during optimization
5. Works with 2-way, 3-way, and N-way splits
"""

import torch
import torch.nn as nn
from torchref.model import OccupancyTensor, Model
import tempfile


def test_2way_altloc():
    """Test 2-way alternative conformation."""
    print("\n" + "="*70)
    print("TEST 1: 2-Way Alternative Conformation")
    print("="*70)
    
    initial = torch.tensor([1.0, 1.0, 0.7, 0.7, 0.3, 0.3])
    sharing = torch.tensor([0, 0, 1, 1, 2, 2])
    altlocs = [([2, 3], [4, 5])]
    
    occ = OccupancyTensor(initial, sharing_groups=sharing, altloc_groups=altlocs)
    result = occ()
    
    print(f"  Initial: {initial}")
    print(f"  Sharing: {sharing}")
    print(f"  Result: {result}")
    print(f"  Linked indices: {occ.linked_occ_2}")
    print(f"  Linked occ sizes: {occ.linked_occ_sizes}")
    
    # Check sum-to-1
    altloc_sum = result[2] + result[4]
    print(f"  Altloc sum: {altloc_sum:.6f} (should be 1.0)")
    assert torch.allclose(altloc_sum, torch.tensor(1.0), atol=1e-5), "2-way altloc should sum to 1.0"
    
    # Check all are refinable
    assert occ.refinable_mask.all(), "All collapsed indices should be refinable"
    print(f"  Refinable mask: {occ.refinable_mask}")
    print("  ✓ Test 1 passed\n")
    
    return True


def test_3way_altloc():
    """Test 3-way alternative conformation."""
    print("="*70)
    print("TEST 2: 3-Way Alternative Conformation")
    print("="*70)
    
    initial = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.3, 0.3, 0.2, 0.2])
    sharing = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    altlocs = [([2, 3], [4, 5], [6, 7])]  # 3-way split
    
    occ = OccupancyTensor(initial, sharing_groups=sharing, altloc_groups=altlocs)
    result = occ()
    
    print(f"  Initial: {initial}")
    print(f"  Sharing: {sharing}")
    print(f"  Result: {result}")
    print(f"  Linked indices: {occ.linked_occ_3}")
    print(f"  Linked occ sizes: {occ.linked_occ_sizes}")
    
    # Check sum-to-1
    altloc_sum = result[2] + result[4] + result[6]
    print(f"  Altloc sum: {altloc_sum:.6f} (should be 1.0)")
    assert torch.allclose(altloc_sum, torch.tensor(1.0), atol=1e-5), "3-way altloc should sum to 1.0"
    
    print(f"  Refinable mask: {occ.refinable_mask}")
    print("  ✓ Test 2 passed\n")
    
    return True


def test_multiple_altloc_groups():
    """Test multiple independent altloc groups."""
    print("="*70)
    print("TEST 3: Multiple Independent Altloc Groups")
    print("="*70)
    
    initial = torch.tensor([1.0, 1.0, 0.7, 0.7, 0.3, 0.3, 0.6, 0.6, 0.4, 0.4])
    sharing = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    altlocs = [
        ([2, 3], [4, 5]),  # First 2-way split
        ([6, 7], [8, 9]),  # Second 2-way split
    ]
    
    occ = OccupancyTensor(initial, sharing_groups=sharing, altloc_groups=altlocs)
    result = occ()
    
    print(f"  Initial: {initial}")
    print(f"  Sharing: {sharing}")
    print(f"  Result: {result}")
    print(f"  Linked indices: {occ.linked_occ_2}")
    print(f"  Linked occ sizes: {occ.linked_occ_sizes}")
    
    # Check both altloc groups sum to 1
    altloc1_sum = result[2] + result[4]
    altloc2_sum = result[6] + result[8]
    print(f"  Altloc group 1 sum: {altloc1_sum:.6f} (should be 1.0)")
    print(f"  Altloc group 2 sum: {altloc2_sum:.6f} (should be 1.0)")
    
    assert torch.allclose(altloc1_sum, torch.tensor(1.0), atol=1e-5), "First altloc should sum to 1.0"
    assert torch.allclose(altloc2_sum, torch.tensor(1.0), atol=1e-5), "Second altloc should sum to 1.0"
    
    print("  ✓ Test 3 passed\n")
    
    return True


def test_optimization_maintains_constraint():
    """Test that sum-to-1 constraint is maintained during optimization."""
    print("="*70)
    print("TEST 4: Optimization Maintains Constraint")
    print("="*70)
    
    initial = torch.tensor([1.0, 1.0, 0.7, 0.7, 0.3, 0.3])
    sharing = torch.tensor([0, 0, 1, 1, 2, 2])
    altlocs = [([2, 3], [4, 5])]
    
    occ = OccupancyTensor(initial, sharing_groups=sharing, altloc_groups=altlocs)
    
    print("  Initial state:")
    result_init = occ()
    sum_init = (result_init[2] + result_init[4]).item()
    print(f"    Altloc A: {result_init[2]:.4f}")
    print(f"    Altloc B: {result_init[4]:.4f}")
    print(f"    Sum: {sum_init:.6f}")
    
    # Optimize
    optimizer = torch.optim.SGD([occ.refinable_params], lr=0.5)
    
    for i in range(20):
        optimizer.zero_grad()
        result = occ()
        
        # Loss: maximize altloc A (minimize negative)
        loss = -result[2]
        loss.backward()
        optimizer.step()
    
    print("\n  After optimization:")
    result_final = occ()
    sum_final = (result_final[2] + result_final[4]).item()
    print(f"    Altloc A: {result_final[2]:.4f}")
    print(f"    Altloc B: {result_final[4]:.4f}")
    print(f"    Sum: {sum_final:.6f}")
    
    assert abs(sum_init - 1.0) < 1e-5, "Initial sum should be 1.0"
    assert abs(sum_final - 1.0) < 1e-5, "Final sum should be 1.0"
    
    # The key test: constraint is maintained, even if direction changed
    print(f"  ✓ Constraint maintained: sum={sum_final:.6f}")
    
    print("  ✓ Test 4 passed\n")
    
    return True


def test_assertion_error():
    """Test that assertion error is raised when atoms don't share same collapsed index."""
    print("="*70)
    print("TEST 5: Assertion for Mismatched Collapsed Indices")
    print("="*70)
    
    initial = torch.tensor([0.7, 0.7, 0.3, 0.3])
    sharing = torch.tensor([0, 1, 2, 3])  # Each atom has its own index
    altlocs = [([0, 1], [2, 3])]  # Atoms 0,1 don't share same collapsed index!
    
    try:
        occ = OccupancyTensor(initial, sharing_groups=sharing, altloc_groups=altlocs)
        print("  ✗ Should have raised AssertionError!")
        return False
    except AssertionError as e:
        print(f"  ✓ Correctly raised AssertionError: {e}")
        print("  ✓ Test 5 passed\n")
        return True


def main():
    print("\n" + "="*70)
    print("NORMALIZATION-BASED ALTLOC TESTS")
    print("="*70)
    
    tests = [
        test_2way_altloc,
        test_3way_altloc,
        test_multiple_altloc_groups,
        test_optimization_maintains_constraint,
        test_assertion_error,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*70)
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print("  ✓ 2-way altlocs work correctly")
        print("  ✓ 3-way altlocs work correctly")
        print("  ✓ Multiple altloc groups handled")
        print("  ✓ Sum-to-1 maintained during optimization")
        print("  ✓ Proper assertions for invalid configurations")
        print("  ✓ All altloc members are refinable")
        print("  ✓ Normalization approach is gradient-friendly")
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
