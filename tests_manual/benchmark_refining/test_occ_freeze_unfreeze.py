#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

"""
Test freeze/unfreeze functionality for OccupancyTensor with compressed storage.

This test verifies that freeze/unfreeze correctly handles the conversion between
uncompressed (full atom) masks and compressed (grouped) storage.
"""

import torch
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.parameter_wrappers import OccupancyTensor

def test_basic_freeze_unfreeze():
    """Test basic freeze/unfreeze without sharing groups"""
    print("\n" + "="*80)
    print("TEST 1: Basic Freeze/Unfreeze (No Sharing Groups)")
    print("="*80)
    
    # Create simple occupancy tensor (no sharing)
    n_atoms = 10
    initial_values = torch.ones(n_atoms) * 0.8
    
    occ = OccupancyTensor(initial_values)
    
    print(f"Initial state:")
    print(f"  Total atoms: {n_atoms}")
    print(f"  Collapsed groups: {occ._collapsed_shape}")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    
    # Freeze atoms 0-4 (in full atom space)
    freeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
    freeze_mask[0:5] = True
    occ.freeze(freeze_mask)
    
    print(f"\nAfter freezing atoms 0-4:")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Fixed groups: {occ.get_fixed_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    print(f"  Frozen atoms: {occ.get_frozen_atoms().sum().item()}")
    
    # Verify
    refinable_atoms = occ.get_refinable_atoms()
    assert refinable_atoms[0:5].sum() == 0, "Atoms 0-4 should be frozen"
    assert refinable_atoms[5:10].sum() == 5, "Atoms 5-9 should be refinable"
    print("  ✓ Freeze mask correctly applied")
    
    # Unfreeze atoms 2-7
    unfreeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
    unfreeze_mask[2:8] = True
    occ.unfreeze(unfreeze_mask)
    
    print(f"\nAfter unfreezing atoms 2-7:")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    
    refinable_atoms = occ.get_refinable_atoms()
    assert refinable_atoms[0:2].sum() == 0, "Atoms 0-1 should still be frozen"
    assert refinable_atoms[2:10].sum() == 8, "Atoms 2-9 should be refinable"
    print("  ✓ Unfreeze mask correctly applied")
    
    print("\n✓ TEST 1 PASSED")
    return True


def test_freeze_with_sharing_groups():
    """Test freeze/unfreeze with sharing groups"""
    print("\n" + "="*80)
    print("TEST 2: Freeze/Unfreeze WITH Sharing Groups")
    print("="*80)
    
    # Create occupancy with sharing groups
    # Atoms 0-2 share (group 0), atoms 3-5 share (group 1), atoms 6-9 share (group 2)
    n_atoms = 10
    initial_values = torch.ones(n_atoms) * 0.7
    sharing_groups = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    
    occ = OccupancyTensor(initial_values, sharing_groups=sharing_groups)
    
    print(f"Initial state:")
    print(f"  Total atoms: {n_atoms}")
    print(f"  Sharing groups: {sharing_groups.tolist()}")
    print(f"  Collapsed groups: {occ._collapsed_shape}")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    
    # Freeze atom 1 (should freeze entire group 0, i.e., atoms 0-2)
    freeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
    freeze_mask[1] = True
    occ.freeze(freeze_mask)
    
    print(f"\nAfter freezing atom 1 (should freeze group 0 = atoms 0-2):")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Fixed groups: {occ.get_fixed_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    
    refinable_atoms = occ.get_refinable_atoms()
    assert refinable_atoms[0:3].sum() == 0, "Atoms 0-2 (group 0) should be frozen"
    assert refinable_atoms[3:6].sum() == 3, "Atoms 3-5 (group 1) should be refinable"
    assert refinable_atoms[6:10].sum() == 4, "Atoms 6-9 (group 2) should be refinable"
    print("  ✓ Freezing one atom froze entire sharing group")
    
    # Freeze atoms 4-8 (should freeze groups 1 and 2)
    freeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
    freeze_mask[4:9] = True
    occ.freeze(freeze_mask)
    
    print(f"\nAfter freezing atoms 4-8 (should freeze groups 1 and 2):")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    
    refinable_atoms = occ.get_refinable_atoms()
    assert refinable_atoms.sum() == 0, "All atoms should be frozen"
    print("  ✓ All groups now frozen")
    
    # Unfreeze atom 5 (should unfreeze group 1, i.e., atoms 3-5)
    unfreeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
    unfreeze_mask[5] = True
    occ.unfreeze(unfreeze_mask)
    
    print(f"\nAfter unfreezing atom 5 (should unfreeze group 1 = atoms 3-5):")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    
    refinable_atoms = occ.get_refinable_atoms()
    assert refinable_atoms[0:3].sum() == 0, "Atoms 0-2 should still be frozen"
    assert refinable_atoms[3:6].sum() == 3, "Atoms 3-5 (group 1) should be refinable"
    assert refinable_atoms[6:10].sum() == 0, "Atoms 6-9 should still be frozen"
    print("  ✓ Unfreezing one atom unfroze entire sharing group")
    
    print("\n✓ TEST 2 PASSED")
    return True


def test_freeze_unfreeze_all():
    """Test freeze_all and unfreeze_all convenience methods"""
    print("\n" + "="*80)
    print("TEST 3: Freeze All / Unfreeze All")
    print("="*80)
    
    n_atoms = 20
    initial_values = torch.ones(n_atoms) * 0.9
    sharing_groups = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5, 6, 6])
    
    occ = OccupancyTensor(initial_values, sharing_groups=sharing_groups)
    
    print(f"Initial state:")
    print(f"  Total atoms: {n_atoms}")
    print(f"  Collapsed groups: {occ._collapsed_shape}")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    
    # Freeze all
    occ.freeze_all()
    
    print(f"\nAfter freeze_all():")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    
    assert occ.get_refinable_count() == 0, "No groups should be refinable"
    assert occ.get_refinable_atoms().sum() == 0, "No atoms should be refinable"
    print("  ✓ All parameters frozen")
    
    # Unfreeze all
    occ.unfreeze_all()
    
    print(f"\nAfter unfreeze_all():")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    
    assert occ.get_refinable_count() == occ._collapsed_shape, "All groups should be refinable"
    assert occ.get_refinable_atoms().sum() == n_atoms, "All atoms should be refinable"
    print("  ✓ All parameters unfrozen")
    
    print("\n✓ TEST 3 PASSED")
    return True


def test_values_preserved():
    """Test that occupancy values are preserved through freeze/unfreeze"""
    print("\n" + "="*80)
    print("TEST 4: Values Preserved Through Freeze/Unfreeze")
    print("="*80)
    
    n_atoms = 12
    initial_values = torch.tensor([0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.9, 0.9, 0.9])
    sharing_groups = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    
    occ = OccupancyTensor(initial_values, sharing_groups=sharing_groups)
    
    # Get initial occupancies
    initial_occs = occ.forward().clone()
    print(f"Initial occupancies: {initial_occs}")
    
    # Freeze group 1 (atoms 3-5)
    freeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
    freeze_mask[3:6] = True
    occ.freeze(freeze_mask)
    
    # Check values unchanged
    after_freeze = occ.forward()
    print(f"\nAfter freezing atoms 3-5:")
    print(f"Occupancies: {after_freeze}")
    
    assert torch.allclose(initial_occs, after_freeze, atol=1e-5), "Values should be preserved after freeze"
    print("  ✓ Values preserved after freeze")
    
    # Unfreeze and check again
    occ.unfreeze(freeze_mask)
    after_unfreeze = occ.forward()
    print(f"\nAfter unfreezing atoms 3-5:")
    print(f"Occupancies: {after_unfreeze}")
    
    assert torch.allclose(initial_occs, after_unfreeze, atol=1e-5), "Values should be preserved after unfreeze"
    print("  ✓ Values preserved after unfreeze")
    
    print("\n✓ TEST 4 PASSED")
    return True


def test_refinement_with_freeze():
    """Test that frozen parameters don't change during optimization"""
    print("\n" + "="*80)
    print("TEST 5: Frozen Parameters Don't Change During Optimization")
    print("="*80)
    
    n_atoms = 9
    initial_values = torch.ones(n_atoms) * 0.5
    sharing_groups = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    
    occ = OccupancyTensor(initial_values, sharing_groups=sharing_groups)
    
    # Freeze group 1 (atoms 3-5)
    freeze_mask = torch.zeros(n_atoms, dtype=torch.bool)
    freeze_mask[3:6] = True
    occ.freeze(freeze_mask)
    
    initial_occs = occ.forward().clone()
    print(f"Initial occupancies: {initial_occs}")
    print(f"Refinable groups: {occ.get_refinable_count()}")
    
    # Simulate optimization
    optimizer = torch.optim.Adam([occ.refinable_params], lr=0.1)
    
    for i in range(10):
        optimizer.zero_grad()
        occs = occ.forward()
        # Dummy loss: try to push refinable occupancies to 0.8
        target = torch.ones_like(occs) * 0.8
        loss = ((occs - target) ** 2).sum()
        loss.backward()
        optimizer.step()
    
    final_occs = occ.forward()
    print(f"\nFinal occupancies: {final_occs}")
    
    # Check that frozen group didn't change
    frozen_atoms = occ.get_frozen_atoms()
    assert torch.allclose(initial_occs[frozen_atoms], final_occs[frozen_atoms], atol=1e-5), \
        "Frozen atoms should not change during optimization"
    print("  ✓ Frozen atoms (3-5) unchanged")
    
    # Check that refinable groups did change
    refinable_atoms = occ.get_refinable_atoms()
    assert not torch.allclose(initial_occs[refinable_atoms], final_occs[refinable_atoms], atol=1e-2), \
        "Refinable atoms should change during optimization"
    print("  ✓ Refinable atoms (0-2, 6-8) changed")
    
    print("\n✓ TEST 5 PASSED")
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*80)
    print("TEST 6: Edge Cases and Error Handling")
    print("="*80)
    
    n_atoms = 10
    initial_values = torch.ones(n_atoms) * 0.7
    sharing_groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    
    occ = OccupancyTensor(initial_values, sharing_groups=sharing_groups)
    
    # Test wrong mask size
    try:
        wrong_mask = torch.zeros(5, dtype=torch.bool)  # Wrong size
        occ.freeze(wrong_mask)
        print("  ✗ Should have raised ValueError for wrong mask size")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError for wrong mask size: {str(e)[:50]}...")
    
    # Test empty mask (no atoms to freeze)
    empty_mask = torch.zeros(n_atoms, dtype=torch.bool)
    occ.freeze(empty_mask)
    print(f"  ✓ Empty freeze mask handled: {occ.get_refinable_count()} groups still refinable")
    
    # Test full mask (all atoms to freeze)
    full_mask = torch.ones(n_atoms, dtype=torch.bool)
    occ.freeze(full_mask)
    assert occ.get_refinable_count() == 0, "All groups should be frozen"
    print(f"  ✓ Full freeze mask handled: {occ.get_refinable_count()} refinable groups")
    
    # Test unfreeze after full freeze
    occ.unfreeze(full_mask)
    assert occ.get_refinable_count() == occ._collapsed_shape, "All groups should be refinable"
    print(f"  ✓ Unfreeze after full freeze: {occ.get_refinable_count()} refinable groups")
    
    print("\n✓ TEST 6 PASSED")
    return True


def test_update_refinable_mask():
    """Test update_refinable_mask() method"""
    print("\n" + "="*80)
    print("TEST 7: Update Refinable Mask")
    print("="*80)
    
    # Create occupancy with sharing groups
    n_atoms = 12
    sharing_groups = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])  # 4 groups of 3
    initial_values = torch.ones(n_atoms) * 0.8
    
    occ = OccupancyTensor(initial_values, sharing_groups=sharing_groups)
    
    print(f"Initial state:")
    print(f"  Total atoms: {n_atoms}")
    print(f"  Collapsed groups: {occ._collapsed_shape}")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    
    # Test 1: Update with full atom space mask
    print("\nTest 1: Update with full atom space mask")
    atom_mask = torch.zeros(n_atoms, dtype=torch.bool)
    atom_mask[0:6] = True  # First two groups
    occ.update_refinable_mask(atom_mask, in_compressed_space=False)
    
    print(f"  After updating (atoms 0-5 refinable):")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    
    assert occ.get_refinable_count() == 2, "Should have 2 refinable groups"
    assert occ.get_refinable_atoms().sum().item() == 6, "Should have 6 refinable atoms"
    refinable = occ.get_refinable_atoms()
    assert refinable[0:6].all(), "Atoms 0-5 should be refinable"
    assert not refinable[6:12].any(), "Atoms 6-11 should be frozen"
    print("  ✓ Full atom space mask correctly applied")
    
    # Test 2: Update with compressed space mask
    print("\nTest 2: Update with compressed space mask")
    group_mask = torch.zeros(4, dtype=torch.bool)  # 4 groups
    group_mask[2:4] = True  # Last two groups
    occ.update_refinable_mask(group_mask, in_compressed_space=True)
    
    print(f"  After updating (groups 2-3 refinable):")
    print(f"  Refinable groups: {occ.get_refinable_count()}")
    print(f"  Refinable atoms: {occ.get_refinable_atoms().sum().item()}")
    
    assert occ.get_refinable_count() == 2, "Should have 2 refinable groups"
    assert occ.get_refinable_atoms().sum().item() == 6, "Should have 6 refinable atoms"
    refinable = occ.get_refinable_atoms()
    assert not refinable[0:6].any(), "Atoms 0-5 should be frozen"
    assert refinable[6:12].all(), "Atoms 6-11 should be refinable"
    print("  ✓ Compressed space mask correctly applied")
    
    # Test 3: Values preserved after update
    print("\nTest 3: Values preserved during mask update")
    initial_occ = occ()
    occ.update_refinable_mask(torch.ones(n_atoms, dtype=torch.bool), in_compressed_space=False)
    final_occ = occ()
    
    assert torch.allclose(initial_occ, final_occ), "Values should be preserved"
    print("  ✓ Values preserved after mask update")
    
    # Test 4: Update with all False (freeze all)
    print("\nTest 4: Update to freeze all")
    occ.update_refinable_mask(torch.zeros(n_atoms, dtype=torch.bool), in_compressed_space=False)
    
    assert occ.get_refinable_count() == 0, "No parameters should be refinable"
    assert occ.get_refinable_atoms().sum().item() == 0, "No atoms should be refinable"
    print("  ✓ Can freeze all parameters")
    
    # Test 5: Update with all True (unfreeze all)
    print("\nTest 5: Update to unfreeze all")
    occ.update_refinable_mask(torch.ones(n_atoms, dtype=torch.bool), in_compressed_space=False)
    
    assert occ.get_refinable_count() == 4, "All groups should be refinable"
    assert occ.get_refinable_atoms().sum().item() == n_atoms, "All atoms should be refinable"
    print("  ✓ Can unfreeze all parameters")
    
    # Test 6: Alternating pattern in compressed space
    print("\nTest 6: Alternating pattern in compressed space")
    group_mask = torch.tensor([True, False, True, False])
    occ.update_refinable_mask(group_mask, in_compressed_space=True)
    
    assert occ.get_refinable_count() == 2, "Should have 2 refinable groups"
    refinable = occ.get_refinable_atoms()
    assert refinable[0:3].all() and not refinable[3:6].any(), "Group 0 refinable, group 1 frozen"
    assert refinable[6:9].all() and not refinable[9:12].any(), "Group 2 refinable, group 3 frozen"
    print("  ✓ Alternating pattern correctly applied")
    
    print("\n✓ TEST 7 PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# OCCUPANCY TENSOR FREEZE/UNFREEZE TESTS")
    print("# Testing compressed storage with uncompressed masks")
    print("#"*80)
    
    tests = [
        ("Basic Freeze/Unfreeze", test_basic_freeze_unfreeze),
        ("With Sharing Groups", test_freeze_with_sharing_groups),
        ("Freeze/Unfreeze All", test_freeze_unfreeze_all),
        ("Values Preserved", test_values_preserved),
        ("Refinement Integration", test_refinement_with_freeze),
        ("Edge Cases", test_edge_cases),
        ("Update Refinable Mask", test_update_refinable_mask),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")
        if error:
            print(f"         Error: {error[:60]}...")
    
    print("="*80)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("✗ SOME TESTS FAILED")
        sys.exit(1)
