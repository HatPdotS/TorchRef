"""
Test script for centric/acentric determination using Symmetry class.

This script tests the new is_centric_from_hkl and get_centric_acentric_masks
functions to ensure they properly identify centric reflections based on
space group symmetry operations.
"""

import torch
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.french_wilson import is_centric_from_hkl, get_centric_acentric_masks


def test_p1_all_acentric():
    """Test P1 - all reflections should be acentric"""
    print("\nTest 1: P1 space group (all acentric)")
    print("-" * 50)
    
    hkl = torch.tensor([
        [1, 2, 3],
        [2, 0, 0],
        [0, 3, 0],
        [0, 0, 5],
        [1, 1, 1]
    ])
    
    is_centric = is_centric_from_hkl(hkl, "P1")
    centric_mask, acentric_mask = get_centric_acentric_masks(hkl, "P1")
    
    print(f"HKL indices:\n{hkl}")
    print(f"Centric mask: {is_centric}")
    print(f"Number centric: {is_centric.sum().item()}")
    print(f"Number acentric: {acentric_mask.sum().item()}")
    
    # All should be acentric in P1
    assert is_centric.sum() == 0, "P1 should have no centric reflections"
    assert acentric_mask.sum() == len(hkl), "All reflections should be acentric in P1"
    print("✓ P1 test passed!")


def test_p21_axial_centric():
    """Test P21 - reflections on certain axes should be centric"""
    print("\nTest 2: P21 space group")
    print("-" * 50)
    
    hkl = torch.tensor([
        [1, 0, 0],  # h00 - should be centric
        [0, 1, 0],  # 0k0 - depends on P21 along which axis
        [0, 0, 1],  # 00l - should be centric
        [1, 2, 3],  # General - acentric
        [2, 0, 4],  # h0l - mixed
    ])
    
    is_centric = is_centric_from_hkl(hkl, "P21")
    centric_mask, acentric_mask = get_centric_acentric_masks(hkl, "P21")
    
    print(f"HKL indices:\n{hkl}")
    print(f"Centric mask: {is_centric}")
    print(f"Number centric: {is_centric.sum().item()}")
    print(f"Number acentric: {acentric_mask.sum().item()}")
    
    for i, (h, k, l) in enumerate(hkl):
        status = "centric" if is_centric[i] else "acentric"
        print(f"  ({h}, {k}, {l}): {status}")
    
    print("✓ P21 test completed!")


def test_p212121_orthorhombic():
    """Test P212121 - orthorhombic space group"""
    print("\nTest 3: P212121 space group (orthorhombic)")
    print("-" * 50)
    
    hkl = torch.tensor([
        [1, 0, 0],  # h00 - should be centric
        [0, 1, 0],  # 0k0 - should be centric  
        [0, 0, 1],  # 00l - should be centric
        [2, 0, 0],  # h00 - should be centric
        [1, 2, 3],  # General - acentric
        [2, 3, 0],  # hk0 - might be centric
    ])
    
    is_centric = is_centric_from_hkl(hkl, "P212121")
    
    print(f"HKL indices:\n{hkl}")
    print(f"Centric mask: {is_centric}")
    print(f"Number centric: {is_centric.sum().item()}")
    
    for i, (h, k, l) in enumerate(hkl):
        status = "centric" if is_centric[i] else "acentric"
        print(f"  ({h}, {k}, {l}): {status}")
    
    # At least the axial reflections should be centric
    axial_reflections = torch.tensor([True, True, True, True, False, False])
    print("✓ P212121 test completed!")


def test_p_1_centrosymmetric():
    """Test P-1 - centrosymmetric, all reflections should be centric"""
    print("\nTest 4: P-1 space group (centrosymmetric)")
    print("-" * 50)
    
    hkl = torch.tensor([
        [1, 2, 3],
        [2, 0, 0],
        [0, 3, 0],
        [0, 0, 5],
        [1, 1, 1],
        [3, 4, 5]
    ])
    
    is_centric = is_centric_from_hkl(hkl, "P-1")
    
    print(f"HKL indices:\n{hkl}")
    print(f"Centric mask: {is_centric}")
    print(f"Number centric: {is_centric.sum().item()}")
    print(f"Number acentric: {(~is_centric).sum().item()}")
    
    # All should be centric in P-1 (centrosymmetric)
    assert is_centric.sum() == len(hkl), "P-1 should have all centric reflections"
    print("✓ P-1 test passed!")


def test_batch_processing():
    """Test that the function handles batched HKL arrays correctly"""
    print("\nTest 5: Batch processing")
    print("-" * 50)
    
    # Test with different shapes
    hkl_1d = torch.tensor([[1, 2, 3], [2, 0, 0]])
    hkl_2d = torch.tensor([[[1, 2, 3], [2, 0, 0]], [[0, 3, 0], [1, 1, 1]]])
    
    is_centric_1d = is_centric_from_hkl(hkl_1d, "P1")
    is_centric_2d = is_centric_from_hkl(hkl_2d, "P1")
    
    print(f"1D shape: {hkl_1d.shape} -> centric mask shape: {is_centric_1d.shape}")
    print(f"2D shape: {hkl_2d.shape} -> centric mask shape: {is_centric_2d.shape}")
    
    assert is_centric_1d.shape == hkl_1d.shape[:-1]
    assert is_centric_2d.shape == hkl_2d.shape[:-1]
    print("✓ Batch processing test passed!")


def test_multiple_spacegroups():
    """Test various common space groups"""
    print("\nTest 6: Multiple space groups")
    print("-" * 50)
    
    hkl = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [2, 3, 4]
    ])
    
    space_groups = ["P1", "P21", "P212121", "P-1", "C2", "P222"]
    
    for sg in space_groups:
        try:
            is_centric = is_centric_from_hkl(hkl, sg)
            n_centric = is_centric.sum().item()
            print(f"{sg:12s}: {n_centric} centric out of {len(hkl)}")
        except Exception as e:
            print(f"{sg:12s}: Error - {e}")
    
    print("✓ Multiple space groups test completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Centric/Acentric Determination with Symmetry Class")
    print("=" * 60)
    
    test_p1_all_acentric()
    test_p21_axial_centric()
    test_p212121_orthorhombic()
    test_p_1_centrosymmetric()
    test_batch_processing()
    test_multiple_spacegroups()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
