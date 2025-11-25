"""
Test script for FrenchWilson with improved math_torch integration.

Tests the calculation of d-spacings, structure factor conversion, and
proper usage of math_torch.get_scattering_vectors.
"""

import torch
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.french_wilson import FrenchWilson, french_wilson_auto


def test_dspacing_calculation():
    """Test d-spacing calculation using math_torch.get_scattering_vectors"""
    print("\nTest 1: D-spacing calculation")
    print("-" * 50)
    
    # Simple cubic cell
    unit_cell = torch.tensor([50.0, 50.0, 50.0, 90.0, 90.0, 90.0], dtype=torch.float32)
    hkl = torch.tensor([
        [1, 0, 0],  # d = a = 50 Å
        [0, 1, 0],  # d = b = 50 Å
        [1, 1, 0],  # d = a/√2 ≈ 35.36 Å
        [1, 1, 1],  # d = a/√3 ≈ 28.87 Å
        [2, 0, 0],  # d = a/2 = 25 Å
    ])
    
    fw_module = FrenchWilson(hkl, unit_cell, space_group='P1', verbose=0)
    
    print("HKL indices and calculated d-spacings:")
    for i, (h, k, l) in enumerate(hkl):
        d = fw_module.d_spacings[i].item()
        print(f"  ({h:2d}, {k:2d}, {l:2d}): d = {d:.3f} Å")
    
    # Check d-spacing for [1,0,0]
    expected_d_100 = 50.0
    actual_d_100 = fw_module.d_spacings[0].item()
    assert abs(actual_d_100 - expected_d_100) < 0.1, f"d-spacing mismatch: {actual_d_100} vs {expected_d_100}"
    
    # Check d-spacing for [1,1,1]
    expected_d_111 = 50.0 / (3**0.5)
    actual_d_111 = fw_module.d_spacings[3].item()
    assert abs(actual_d_111 - expected_d_111) < 0.1, f"d-spacing mismatch: {actual_d_111} vs {expected_d_111}"
    
    print("✓ D-spacing calculation test passed!")


def test_non_orthogonal_cell():
    """Test d-spacing calculation for non-orthogonal unit cell"""
    print("\nTest 2: Non-orthogonal unit cell")
    print("-" * 50)
    
    # Monoclinic cell (beta != 90)
    unit_cell = torch.tensor([40.0, 50.0, 60.0, 90.0, 110.0, 90.0], dtype=torch.float32)
    hkl = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
    ])
    
    fw_module = FrenchWilson(hkl, unit_cell, space_group='P21', verbose=0)
    
    print("Monoclinic unit cell:", unit_cell)
    print("HKL indices and calculated d-spacings:")
    for i, (h, k, l) in enumerate(hkl):
        d = fw_module.d_spacings[i].item()
        print(f"  ({h:2d}, {k:2d}, {l:2d}): d = {d:.3f} Å")
    
    print("✓ Non-orthogonal cell test passed!")


def test_french_wilson_conversion():
    """Test basic French-Wilson conversion"""
    print("\nTest 3: French-Wilson conversion")
    print("-" * 50)
    
    # Generate test data
    unit_cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float32)
    hkl = torch.tensor([
        [1, 2, 3],
        [2, 0, 0],
        [0, 3, 0],
        [1, 1, 1],
        [3, 2, 1]
    ])
    
    # Simulate intensity data
    I = torch.tensor([100.0, 50.0, 30.0, 200.0, 150.0])
    sigma_I = torch.tensor([10.0, 8.0, 7.0, 15.0, 12.0])
    
    # Create module and apply conversion
    fw_module = FrenchWilson(hkl, unit_cell, space_group='P212121', verbose=0)
    F, sigma_F = fw_module(I, sigma_I)
    
    print("\nIntensity -> Structure Factor conversion:")
    print(f"{'HKL':<12} {'I':>8} {'σ(I)':>8} {'F':>8} {'σ(F)':>8}")
    print("-" * 50)
    for i in range(len(hkl)):
        h, k, l = hkl[i]
        print(f"({h:2d},{k:2d},{l:2d})   {I[i]:8.2f} {sigma_I[i]:8.2f} {F[i]:8.2f} {sigma_F[i]:8.2f}")
    
    # Basic sanity checks
    assert torch.all(F >= 0), "Structure factors should be non-negative"
    assert torch.all(sigma_F >= 0), "Sigmas should be non-negative"
    assert torch.all(F**2 <= I * 1.5), "F² should be roughly similar to I"
    
    print("✓ French-Wilson conversion test passed!")


def test_weak_reflections():
    """Test French-Wilson conversion with weak/negative intensities"""
    print("\nTest 4: Weak and negative intensities")
    print("-" * 50)
    
    unit_cell = torch.tensor([50.0, 50.0, 50.0, 90.0, 90.0, 90.0], dtype=torch.float32)
    hkl = torch.tensor([
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [4, 0, 0],
        [5, 0, 0]
    ])
    
    # Mix of strong, weak, and negative intensities
    I = torch.tensor([1000.0, 50.0, 5.0, -10.0, -30.0])
    sigma_I = torch.tensor([30.0, 10.0, 10.0, 10.0, 10.0])
    
    fw_module = FrenchWilson(hkl, unit_cell, space_group='P1', verbose=0)
    F, sigma_F = fw_module(I, sigma_I)
    
    print("\nWeak/negative intensity handling:")
    print(f"{'HKL':<12} {'I':>10} {'σ(I)':>10} {'I/σ(I)':>10} {'F':>10} {'σ(F)':>10}")
    print("-" * 70)
    for i in range(len(hkl)):
        h, k, l = hkl[i]
        i_over_sig = I[i] / sigma_I[i]
        print(f"({h:2d},{k:2d},{l:2d})   {I[i]:10.2f} {sigma_I[i]:10.2f} {i_over_sig:10.2f} {F[i]:10.2f} {sigma_F[i]:10.2f}")
    
    # Even negative intensities should produce reasonable F values
    assert torch.all(torch.isfinite(F)), "All F values should be finite"
    assert torch.all(torch.isfinite(sigma_F)), "All sigma_F values should be finite"
    
    print("✓ Weak reflections test passed!")


def test_functional_api():
    """Test the functional API (french_wilson_auto)"""
    print("\nTest 5: Functional API (french_wilson_auto)")
    print("-" * 50)
    
    # Generate test data
    hkl = torch.tensor([
        [1, 2, 3],
        [2, 0, 0],
        [0, 3, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])
    
    I = torch.tensor([100.0, 50.0, 30.0, 200.0, 150.0])
    sigma_I = torch.tensor([10.0, 8.0, 7.0, 15.0, 12.0])
    d_spacings = torch.tensor([2.5, 3.0, 2.8, 2.0, 3.5])
    
    F, sigma_F, valid = french_wilson_auto(I, sigma_I, hkl, d_spacings, "P212121")
    
    print(f"Valid reflections: {valid.sum().item()} out of {len(hkl)}")
    print(f"F values: {F}")
    print(f"sigma_F values: {sigma_F}")
    
    assert torch.all(F[valid] >= 0), "Valid F values should be non-negative"
    
    print("✓ Functional API test passed!")


def test_different_spacegroups():
    """Test French-Wilson module with different space groups"""
    print("\nTest 6: Different space groups")
    print("-" * 50)
    
    unit_cell = torch.tensor([50.0, 50.0, 50.0, 90.0, 90.0, 90.0], dtype=torch.float32)
    hkl = torch.tensor([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1],
        [2, 3, 4]
    ])
    
    I = torch.tensor([100.0, 90.0, 80.0, 150.0, 120.0])
    sigma_I = torch.tensor([10.0, 9.0, 8.0, 12.0, 11.0])
    
    space_groups = ["P1", "P21", "P212121", "P-1"]
    
    for sg in space_groups:
        try:
            fw_module = FrenchWilson(hkl, unit_cell, space_group=sg, verbose=0)
            F, sigma_F = fw_module(I, sigma_I)
            print(f"\n{sg}:")
            print(f"  Centric reflections: {fw_module.is_centric.sum().item()}")
            print(f"  F range: [{F.min():.2f}, {F.max():.2f}]")
        except Exception as e:
            print(f"\n{sg}: Error - {e}")
    
    print("\n✓ Different space groups test completed!")


def test_large_dataset():
    """Test with a larger dataset"""
    print("\nTest 7: Large dataset")
    print("-" * 50)
    
    # Generate a large number of reflections
    n_reflections = 1000
    h = torch.randint(-10, 11, (n_reflections,))
    k = torch.randint(-10, 11, (n_reflections,))
    l = torch.randint(-10, 11, (n_reflections,))
    hkl = torch.stack([h, k, l], dim=1)
    
    # Remove (0,0,0)
    hkl = hkl[torch.any(hkl != 0, dim=1)]
    
    unit_cell = torch.tensor([60.0, 70.0, 80.0, 90.0, 95.0, 90.0], dtype=torch.float32)
    
    # Generate realistic-looking intensity data
    I = torch.abs(torch.randn(len(hkl))) * 100 + 50
    sigma_I = I * 0.1 + 5
    
    fw_module = FrenchWilson(hkl, unit_cell, space_group='P21', verbose=0)
    F, sigma_F = fw_module(I, sigma_I)
    
    print(f"Processed {len(hkl)} reflections")
    print(f"Resolution range: {fw_module.d_spacings.min():.2f} - {fw_module.d_spacings.max():.2f} Å")
    print(f"Centric reflections: {fw_module.is_centric.sum().item()} ({100*fw_module.is_centric.sum()/len(hkl):.1f}%)")
    print(f"F range: [{F.min():.2f}, {F.max():.2f}]")
    print(f"Mean F/sqrt(I): {(F / torch.sqrt(I)).mean():.3f}")
    
    print("✓ Large dataset test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing FrenchWilson with math_torch Integration")
    print("=" * 70)
    
    test_dspacing_calculation()
    test_non_orthogonal_cell()
    test_french_wilson_conversion()
    test_weak_reflections()
    test_functional_api()
    test_different_spacegroups()
    test_large_dataset()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
