#!/usr/bin/env python3
"""
Comprehensive test suite for OccupancyTensor with collapsed storage.

Tests all features requested:
1. Creation with sharing groups creates collapsed internal storage
2. Forward() returns correct full tensor shape
3. Refinable params are collapsed (only stores unique values)
4. Editing parameters only affects refinable params
5. Expansion mask correctly maps collapsed to full space
"""

import torch
import torch.optim as optim
import sys
import os

# Add parent directory to path
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import OccupancyTensor

def test_basic_creation_no_sharing():
    """Test 1: Basic creation without sharing groups."""
    print("\n" + "="*70)
    print("TEST 1: Basic Creation Without Sharing Groups")
    print("="*70)
    
    n_atoms = 10
    initial_occ = torch.rand(n_atoms) * 0.5 + 0.5  # Random values in [0.5, 1.0]
    
    occ = OccupancyTensor(initial_values=initial_occ)
    
    # Check shapes
    print(f"Full shape: {occ.shape}")
    print(f"Collapsed shape: {occ.collapsed_shape}")
    print(f"Expected full: {(n_atoms,)}")
    print(f"Expected collapsed: {(n_atoms,)} (no sharing)")
    
    assert occ.shape == (n_atoms,), f"Full shape mismatch: {occ.shape} != {(n_atoms,)}"
    assert occ.collapsed_shape == (n_atoms,), f"Collapsed shape mismatch"
    
    # Check forward returns correct values
    result = occ()
    print(f"Initial values: {initial_occ}")
    print(f"Forward values: {result}")
    print(f"Max difference: {(initial_occ - result).abs().max().item():.6f}")
    
    assert result.shape == (n_atoms,), "Forward shape mismatch"
    assert torch.allclose(initial_occ, result, atol=1e-3), "Values don't match"
    
    # Check internal storage size
    print(f"Refinable params size: {occ.refinable_params.numel()}")
    print(f"Fixed values size: {occ.fixed_values.numel()}")
    
    print("✓ PASSED")


def test_creation_with_sharing():
    """Test 2: Creation with sharing groups creates collapsed storage."""
    print("\n" + "="*70)
    print("TEST 2: Creation With Sharing Groups (Collapsed Storage)")
    print("="*70)
    
    # 10 atoms: 2 groups of 3, 1 group of 2, 2 independent
    # Group 0: atoms 0,1,2 -> occupancy 1.0
    # Group 1: atoms 3,4,5 -> occupancy 0.8
    # Group 2: atoms 6,7   -> occupancy 0.6
    # Independent: atoms 8, 9 -> occupancies 0.4, 0.2
    
    initial_occ = torch.tensor([1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.6, 0.4, 0.2])
    sharing_groups = [
        [0, 1, 2],   # Group 0
        [3, 4, 5],   # Group 1
        [6, 7]       # Group 2
        # atoms 8, 9 are independent
    ]
    
    occ = OccupancyTensor(initial_values=initial_occ, sharing_groups=sharing_groups)
    
    # Check shapes
    print(f"Full shape: {occ.shape}")
    print(f"Collapsed shape: {occ.collapsed_shape}")
    print(f"Expected full: (10,)")
    print(f"Expected collapsed: (5,)  [3 groups + 2 independent atoms]")
    
    assert occ.shape == (10,), f"Full shape mismatch"
    assert occ.collapsed_shape == (5,), f"Collapsed shape should be 5 (3 groups + 2 independent)"
    
    # Check internal storage is collapsed
    print(f"\nInternal storage:")
    print(f"  refinable_params.numel() = {occ.refinable_params.numel()}")
    print(f"  fixed_values.numel() = {occ.fixed_values.numel()}")
    print(f"  Expected: 5 (not 10)")
    
    assert occ.refinable_params.numel() + occ.fixed_values[occ.fixed_mask].numel() == 5
    assert occ.fixed_values.numel() == 5, "Fixed values should be collapsed"
    
    # Check forward returns correct full size
    result = occ()
    print(f"\nForward result shape: {result.shape}")
    print(f"Forward values: {result}")
    
    assert result.shape == (10,), "Forward should return full tensor"
    
    # Check sharing constraints are maintained
    print(f"\nChecking sharing constraints:")
    print(f"  Group 0 (atoms 0,1,2): {result[0:3]}")
    print(f"  Group 1 (atoms 3,4,5): {result[3:6]}")
    print(f"  Group 2 (atoms 6,7): {result[6:8]}")
    print(f"  Independent (atoms 8,9): {result[8:10]}")
    
    assert torch.allclose(result[0:3], result[0].expand(3), atol=1e-5), "Group 0 not uniform"
    assert torch.allclose(result[3:6], result[3].expand(3), atol=1e-5), "Group 1 not uniform"
    assert torch.allclose(result[6:8], result[6].expand(2), atol=1e-5), "Group 2 not uniform"
    
    # Check expansion mask
    print(f"\nExpansion mask: {occ.expansion_mask}")
    print(f"  Expected: [0,0,0, 1,1,1, 2,2, 3, 4]")
    expected_expansion = torch.tensor([0,0,0, 1,1,1, 2,2, 3, 4], dtype=torch.long)
    assert torch.equal(occ.expansion_mask, expected_expansion), "Expansion mask incorrect"
    
    print("✓ PASSED - Storage is properly collapsed!")


def test_refinable_mask_collapsed():
    """Test 3: Refinable mask is properly handled in collapsed space."""
    print("\n" + "="*70)
    print("TEST 3: Refinable Mask in Collapsed Space")
    print("="*70)
    
    # 6 atoms: 2 groups of 2, 2 independent
    # Group 0: atoms 0,1 -> fixed
    # Group 1: atoms 2,3 -> refinable
    # Independent: atom 4 -> fixed, atom 5 -> refinable
    
    initial_occ = torch.tensor([1.0, 1.0, 0.8, 0.8, 0.6, 0.4])
    sharing_groups = [[0,1], [2,3]]
    
    # Refinable mask in FULL space
    refinable_mask_full = torch.tensor([False, False, True, True, False, True])
    
    occ = OccupancyTensor(
        initial_values=initial_occ,
        sharing_groups=sharing_groups,
        refinable_mask=refinable_mask_full
    )
    
    # Check collapsed dimensions
    print(f"Full shape: {occ.shape}")
    print(f"Collapsed shape: {occ.collapsed_shape}")
    print(f"Expected collapsed: (4,) [2 groups + 2 independent]")
    
    assert occ.collapsed_shape == (4,), "Collapsed shape mismatch"
    
    # Check refinable counts in collapsed space
    print(f"\nRefinable params count: {occ.get_refinable_count()}")
    print(f"Fixed params count: {occ.get_fixed_count()}")
    print(f"Expected refinable: 2 (group 1 + atom 5)")
    print(f"Expected fixed: 2 (group 0 + atom 4)")
    
    assert occ.get_refinable_count() == 2, "Should have 2 refinable in collapsed space"
    assert occ.get_fixed_count() == 2, "Should have 2 fixed in collapsed space"
    
    # Check refinable params storage size
    print(f"\nRefinable params size: {occ.refinable_params.numel()}")
    print(f"Expected: 2 (not 4)")
    
    assert occ.refinable_params.numel() == 2, "Refinable params should be collapsed"
    
    print("✓ PASSED - Refinable mask properly collapsed!")


def test_editing_only_refinable():
    """Test 4: Editing parameters only affects refinable params, not fixed ones."""
    print("\n" + "="*70)
    print("TEST 4: Editing Only Affects Refinable Parameters")
    print("="*70)
    
    # 8 atoms: 2 groups of 3, 2 independent
    # Group 0: atoms 0,1,2 -> fixed at 1.0
    # Group 1: atoms 3,4,5 -> refinable, start at 0.8
    # Independent: atom 6 -> fixed at 0.6, atom 7 -> refinable at 0.4
    
    initial_occ = torch.tensor([1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.6, 0.4])
    sharing_groups = [[0,1,2], [3,4,5]]
    refinable_mask_full = torch.tensor([False, False, False, True, True, True, False, True])
    
    occ = OccupancyTensor(
        initial_values=initial_occ,
        sharing_groups=sharing_groups,
        refinable_mask=refinable_mask_full,
        requires_grad=True
    )
    
    print(f"Initial forward values: {occ()}")
    initial_group0 = occ()[0:3].clone()
    initial_atom6 = occ()[6].clone()
    
    # Try to optimize - push all values toward 0.5
    optimizer = optim.Adam([occ.refinable_params], lr=0.1)
    target = torch.ones(8) * 0.5
    
    for _ in range(100):
        optimizer.zero_grad()
        values = occ()
        loss = ((values - target) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    final_values = occ()
    print(f"Final forward values: {final_values}")
    
    # Check fixed values didn't change
    print(f"\nGroup 0 (fixed) - Initial: {initial_group0[0].item():.4f}, Final: {final_values[0].item():.4f}")
    print(f"Atom 6 (fixed) - Initial: {initial_atom6.item():.4f}, Final: {final_values[6].item():.4f}")
    
    assert torch.allclose(final_values[0:3], initial_group0[0].expand(3), atol=1e-3), "Fixed group changed!"
    assert torch.allclose(final_values[6], initial_atom6, atol=1e-3), "Fixed atom changed!"
    
    # Check refinable values did change
    print(f"\nGroup 1 (refinable) - Initial: 0.8, Final: {final_values[3].item():.4f}")
    print(f"Atom 7 (refinable) - Initial: 0.4, Final: {final_values[7].item():.4f}")
    
    assert abs(final_values[3].item() - 0.8) > 0.1, "Refinable group didn't change"
    assert abs(final_values[7].item() - 0.4) > 0.05, "Refinable atom didn't change"
    
    # Check sharing is still maintained
    assert torch.allclose(final_values[3:6], final_values[3].expand(3), atol=1e-5), "Group sharing broken"
    
    print("✓ PASSED - Only refinable params changed, fixed stayed fixed!")


def test_expansion_mask_correctness():
    """Test 5: Expansion mask correctly maps collapsed to full space."""
    print("\n" + "="*70)
    print("TEST 5: Expansion Mask Correctness")
    print("="*70)
    
    # Create a specific pattern to test mapping
    # 12 atoms: 3 groups, 3 independent
    # Group 0: atoms 0,1,2,3 (4 atoms)
    # Group 1: atoms 4,5 (2 atoms)
    # Group 2: atoms 6,7,8 (3 atoms)
    # Independent: atoms 9, 10, 11
    
    initial_occ = torch.tensor([
        1.0, 1.0, 1.0, 1.0,  # Group 0
        0.8, 0.8,             # Group 1
        0.6, 0.6, 0.6,       # Group 2
        0.4, 0.3, 0.2        # Independent
    ])
    sharing_groups = [[0,1,2,3], [4,5], [6,7,8]]
    
    occ = OccupancyTensor(initial_values=initial_occ, sharing_groups=sharing_groups)
    
    print(f"Full shape: {occ.shape}")
    print(f"Collapsed shape: {occ.collapsed_shape}")
    print(f"Expansion mask: {occ.expansion_mask}")
    
    # Expected: [0,0,0,0, 1,1, 2,2,2, 3, 4, 5]
    # Collapsed has 6 values: group0, group1, group2, atom9, atom10, atom11
    assert occ.collapsed_shape == (6,), f"Collapsed shape should be 6, got {occ.collapsed_shape}"
    
    expected_expansion = torch.tensor([0,0,0,0, 1,1, 2,2,2, 3, 4, 5], dtype=torch.long)
    print(f"Expected: {expected_expansion}")
    
    assert torch.equal(occ.expansion_mask, expected_expansion), "Expansion mask incorrect"
    
    # Verify expansion works correctly
    result = occ()
    print(f"\nResult values: {result}")
    
    # All atoms in group should have same value
    assert torch.allclose(result[0:4], result[0].expand(4), atol=1e-5), "Group 0 expansion failed"
    assert torch.allclose(result[4:6], result[4].expand(2), atol=1e-5), "Group 1 expansion failed"
    assert torch.allclose(result[6:9], result[6].expand(3), atol=1e-5), "Group 2 expansion failed"
    
    # Independent atoms should have different values
    print(f"Independent atoms: {result[9]:.4f}, {result[10]:.4f}, {result[11]:.4f}")
    
    print("✓ PASSED - Expansion mask is correct!")


def test_memory_efficiency():
    """Test 6: Verify memory efficiency - collapsed storage saves memory."""
    print("\n" + "="*70)
    print("TEST 6: Memory Efficiency of Collapsed Storage")
    print("="*70)
    
    # Create two scenarios:
    # 1. Many small groups - high compression
    # 2. No groups - no compression
    
    n_atoms = 100
    
    # Scenario 1: 20 groups of 5 atoms each
    initial_occ_grouped = torch.ones(n_atoms)
    sharing_groups_many = [list(range(i*5, (i+1)*5)) for i in range(20)]
    
    occ_grouped = OccupancyTensor(
        initial_values=initial_occ_grouped,
        sharing_groups=sharing_groups_many
    )
    
    # Scenario 2: No groups
    initial_occ_independent = torch.ones(n_atoms)
    occ_independent = OccupancyTensor(initial_values=initial_occ_independent)
    
    print(f"\nScenario 1: 20 groups of 5 atoms")
    print(f"  Full shape: {occ_grouped.shape}")
    print(f"  Collapsed shape: {occ_grouped.collapsed_shape}")
    print(f"  Storage size: {occ_grouped.refinable_params.numel()} params")
    print(f"  Compression ratio: {n_atoms / occ_grouped.collapsed_shape[0]:.1f}x")
    
    print(f"\nScenario 2: No groups (all independent)")
    print(f"  Full shape: {occ_independent.shape}")
    print(f"  Collapsed shape: {occ_independent.collapsed_shape}")
    print(f"  Storage size: {occ_independent.refinable_params.numel()} params")
    print(f"  Compression ratio: {n_atoms / occ_independent.collapsed_shape[0]:.1f}x")
    
    assert occ_grouped.collapsed_shape[0] == 20, "Should have 20 collapsed values"
    assert occ_independent.collapsed_shape[0] == 100, "Should have 100 collapsed values"
    
    print("\n✓ PASSED - Collapsed storage is memory efficient!")


def test_group_operations():
    """Test 7: set_group_occupancy and get_group_occupancy work with collapsed storage."""
    print("\n" + "="*70)
    print("TEST 7: Group Operations (Set/Get) with Collapsed Storage")
    print("="*70)
    
    initial_occ = torch.tensor([1.0, 1.0, 0.8, 0.8, 0.8, 0.5, 0.5])
    sharing_groups = [[0,1], [2,3,4], [5,6]]
    
    occ = OccupancyTensor(
        initial_values=initial_occ,
        sharing_groups=sharing_groups
    )
    
    print(f"Initial values: {occ()}")
    print(f"Collapsed storage size: {occ.collapsed_shape[0]}")
    
    # Set group 1 occupancy
    occ.set_group_occupancy(1, 0.6)
    values = occ()
    print(f"After setting group 1 to 0.6: {values}")
    print(f"  Group 1 atoms (2,3,4): {values[2:5]}")
    
    assert torch.allclose(values[2:5], torch.tensor([0.6, 0.6, 0.6]), atol=1e-3), "Set group failed"
    
    # Get group 1 occupancy
    group1_occ = occ.get_group_occupancy(1)
    print(f"Get group 1 occupancy: {group1_occ:.4f}")
    
    assert abs(group1_occ - 0.6) < 1e-3, "Get group failed"
    
    # Set group 0 occupancy
    occ.set_group_occupancy(0, 0.3)
    values = occ()
    print(f"After setting group 0 to 0.3: {values}")
    
    assert torch.allclose(values[0:2], torch.tensor([0.3, 0.3]), atol=1e-3), "Set group 0 failed"
    
    print("✓ PASSED - Group operations work with collapsed storage!")


def test_gradient_flow():
    """Test 8: Gradients flow correctly through collapsed storage."""
    print("\n" + "="*70)
    print("TEST 8: Gradient Flow Through Collapsed Storage")
    print("="*70)
    
    initial_occ = torch.tensor([0.5, 0.5, 0.5, 0.7, 0.7])
    sharing_groups = [[0,1,2], [3,4]]
    
    occ = OccupancyTensor(
        initial_values=initial_occ,
        sharing_groups=sharing_groups,
        requires_grad=True
    )
    
    print(f"Collapsed shape: {occ.collapsed_shape}")
    print(f"Refinable params: {occ.refinable_params}")
    print(f"Refinable params require grad: {occ.refinable_params.requires_grad}")
    
    # Compute loss
    values = occ()
    target = torch.tensor([0.8, 0.8, 0.8, 0.4, 0.4])
    loss = ((values - target) ** 2).sum()
    
    print(f"\nInitial values: {values}")
    print(f"Target values: {target}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backprop
    loss.backward()
    
    print(f"\nGradients on refinable params: {occ.refinable_params.grad}")
    
    assert occ.refinable_params.grad is not None, "No gradients!"
    assert occ.refinable_params.grad.shape == (2,), "Gradient shape mismatch"
    assert not torch.all(occ.refinable_params.grad == 0), "Gradients are zero!"
    
    print("✓ PASSED - Gradients flow through collapsed storage!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("OCCUPANCY TENSOR COLLAPSED STORAGE - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    try:
        test_basic_creation_no_sharing()
        test_creation_with_sharing()
        test_refinable_mask_collapsed()
        test_editing_only_refinable()
        test_expansion_mask_correctness()
        test_memory_efficiency()
        test_group_operations()
        test_gradient_flow()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓✓✓")
        print("="*70)
        print("\nCollapsed storage implementation is working correctly:")
        print("  ✓ Storage is properly collapsed (memory efficient)")
        print("  ✓ Forward returns correct full tensor shape")
        print("  ✓ Refinable params are collapsed")
        print("  ✓ Editing only affects refinable params")
        print("  ✓ Expansion mask correctly maps collapsed to full")
        print("  ✓ Gradients flow correctly")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n" + "="*70)
        print(f"✗ TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
