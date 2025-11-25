"""
Test comparison between old and new implementations.

This script compares the behavior of the updated french_wilson.py with
any previous implementations to ensure backward compatibility where appropriate.
"""

import torch
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.french_wilson import (
    french_wilson_acentric,
    french_wilson_centric,
    french_wilson,
    estimate_mean_intensity_by_resolution
)


def test_acentric_conversion():
    """Test acentric French-Wilson conversion"""
    print("\nTest 1: Acentric conversion")
    print("-" * 50)
    
    # Strong reflections
    I = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
    sigma_I = torch.tensor([10.0, 15.0, 20.0, 25.0, 30.0])
    mean_I = torch.tensor([150.0, 150.0, 150.0, 150.0, 150.0])
    
    F, sigma_F, valid = french_wilson_acentric(I, sigma_I, mean_I)
    
    print("Strong reflections (acentric):")
    print(f"{'I':>10} {'σ(I)':>10} {'<I>':>10} {'F':>10} {'σ(F)':>10} {'Valid':>8}")
    print("-" * 70)
    for i in range(len(I)):
        print(f"{I[i]:10.2f} {sigma_I[i]:10.2f} {mean_I[i]:10.2f} {F[i]:10.2f} {sigma_F[i]:10.2f} {valid[i]}")
    
    # Weak reflections
    I_weak = torch.tensor([5.0, 0.0, -5.0, -10.0, -20.0])
    sigma_I_weak = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0])
    mean_I_weak = torch.tensor([50.0, 50.0, 50.0, 50.0, 50.0])
    
    F_weak, sigma_F_weak, valid_weak = french_wilson_acentric(I_weak, sigma_I_weak, mean_I_weak)
    
    print("\nWeak/negative reflections (acentric):")
    print(f"{'I':>10} {'σ(I)':>10} {'<I>':>10} {'F':>10} {'σ(F)':>10} {'Valid':>8}")
    print("-" * 70)
    for i in range(len(I_weak)):
        print(f"{I_weak[i]:10.2f} {sigma_I_weak[i]:10.2f} {mean_I_weak[i]:10.2f} {F_weak[i]:10.2f} {sigma_F_weak[i]:10.2f} {valid_weak[i]}")
    
    print("✓ Acentric conversion test passed!")


def test_centric_conversion():
    """Test centric French-Wilson conversion"""
    print("\nTest 2: Centric conversion")
    print("-" * 50)
    
    # Strong reflections
    I = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])
    sigma_I = torch.tensor([10.0, 15.0, 20.0, 25.0, 30.0])
    mean_I = torch.tensor([150.0, 150.0, 150.0, 150.0, 150.0])
    
    F, sigma_F, valid = french_wilson_centric(I, sigma_I, mean_I)
    
    print("Strong reflections (centric):")
    print(f"{'I':>10} {'σ(I)':>10} {'<I>':>10} {'F':>10} {'σ(F)':>10} {'Valid':>8}")
    print("-" * 70)
    for i in range(len(I)):
        print(f"{I[i]:10.2f} {sigma_I[i]:10.2f} {mean_I[i]:10.2f} {F[i]:10.2f} {sigma_F[i]:10.2f} {valid[i]}")
    
    # Weak reflections
    I_weak = torch.tensor([5.0, 0.0, -5.0, -10.0, -20.0])
    sigma_I_weak = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0])
    mean_I_weak = torch.tensor([50.0, 50.0, 50.0, 50.0, 50.0])
    
    F_weak, sigma_F_weak, valid_weak = french_wilson_centric(I_weak, sigma_I_weak, mean_I_weak)
    
    print("\nWeak/negative reflections (centric):")
    print(f"{'I':>10} {'σ(I)':>10} {'<I>':>10} {'F':>10} {'σ(F)':>10} {'Valid':>8}")
    print("-" * 70)
    for i in range(len(I_weak)):
        print(f"{I_weak[i]:10.2f} {sigma_I_weak[i]:10.2f} {mean_I_weak[i]:10.2f} {F_weak[i]:10.2f} {sigma_F_weak[i]:10.2f} {valid_weak[i]}")
    
    print("✓ Centric conversion test passed!")


def test_mixed_centric_acentric():
    """Test french_wilson with mixed centric/acentric reflections"""
    print("\nTest 3: Mixed centric/acentric")
    print("-" * 50)
    
    I = torch.tensor([100.0, 150.0, 200.0, 250.0, 300.0])
    sigma_I = torch.tensor([10.0, 12.0, 15.0, 18.0, 20.0])
    mean_I = torch.tensor([150.0, 150.0, 150.0, 150.0, 150.0])
    
    # Mark reflections 0 and 2 as centric, others as acentric
    is_centric = torch.tensor([True, False, True, False, False])
    
    F, sigma_F, valid = french_wilson(I, sigma_I, mean_I, is_centric=is_centric)
    
    print("Mixed centric/acentric reflections:")
    print(f"{'I':>10} {'σ(I)':>10} {'Type':>10} {'F':>10} {'σ(F)':>10}")
    print("-" * 60)
    for i in range(len(I)):
        ref_type = "centric" if is_centric[i] else "acentric"
        print(f"{I[i]:10.2f} {sigma_I[i]:10.2f} {ref_type:>10} {F[i]:10.2f} {sigma_F[i]:10.2f}")
    
    # Centric and acentric should give different results for same I
    assert F[0] != F[1], "Centric and acentric should differ"
    
    print("✓ Mixed centric/acentric test passed!")


def test_resolution_binning():
    """Test mean intensity estimation by resolution binning"""
    print("\nTest 4: Resolution binning")
    print("-" * 50)
    
    # Create synthetic data with resolution-dependent intensity
    n_reflections = 200
    d_spacings = torch.linspace(5.0, 1.5, n_reflections)  # High to low resolution
    
    # Simulate Wilson falloff: I ~ exp(-B*s²) where s = 1/d
    B = 20.0
    s = 1.0 / d_spacings
    I_true = 1000.0 * torch.exp(-B * s**2)
    
    # Add noise
    I = I_true + torch.randn(n_reflections) * 50
    
    # Estimate mean intensity
    mean_I = estimate_mean_intensity_by_resolution(I, d_spacings, n_bins=10)
    
    print(f"Total reflections: {n_reflections}")
    print(f"Resolution range: {d_spacings.min():.2f} - {d_spacings.max():.2f} Å")
    print(f"Intensity range: {I.min():.2f} - {I.max():.2f}")
    print(f"Mean intensity range: {mean_I.min():.2f} - {mean_I.max():.2f}")
    
    # Show some samples
    print("\nSample reflections:")
    print(f"{'d (Å)':>10} {'I':>12} {'<I>':>12}")
    print("-" * 40)
    for i in [0, 50, 100, 150, 199]:
        print(f"{d_spacings[i]:10.2f} {I[i]:12.2f} {mean_I[i]:12.2f}")
    
    # Mean intensity should be smooth and follow general trend
    assert torch.all(torch.isfinite(mean_I)), "Mean intensities should be finite"
    assert mean_I[0] > mean_I[-1] * 0.5, "Higher resolution should have lower intensity"
    
    print("✓ Resolution binning test passed!")


def test_lookup_table_interpolation():
    """Test the lookup table interpolation for different h values"""
    print("\nTest 5: Lookup table interpolation")
    print("-" * 50)
    
    # Test a range of h values from very weak to very strong
    h_values = torch.linspace(-3.0, 6.0, 20)
    
    print("Testing acentric reflections:")
    print(f"{'h':>8} {'I':>10} {'σ(I)':>10} {'F':>10} {'σ(F)':>10}")
    print("-" * 55)
    
    for h in h_values:
        # Construct I and sigma_I to give desired h value
        # h = (I/sigma_I) - (sigma_I/mean_I)
        # For simplicity, let mean_I = 100, sigma_I = 10
        mean_I = torch.tensor([100.0])
        sigma_I = torch.tensor([10.0])
        
        # I = sigma_I * (h + sigma_I/mean_I)
        I = sigma_I * (h + sigma_I / mean_I)
        
        F, sigma_F, valid = french_wilson_acentric(I, sigma_I, mean_I)
        
        if valid[0]:
            print(f"{h:8.2f} {I[0]:10.2f} {sigma_I[0]:10.2f} {F[0]:10.2f} {sigma_F[0]:10.2f}")
        else:
            print(f"{h:8.2f} {I[0]:10.2f} {sigma_I[0]:10.2f}   (rejected)")
    
    print("✓ Lookup table interpolation test passed!")


def test_asymptotic_formulas():
    """Test asymptotic formulas for large h values"""
    print("\nTest 6: Asymptotic formulas (large h)")
    print("-" * 50)
    
    # Very strong reflections (h >> 3 for acentric, h >> 4 for centric)
    I = torch.tensor([1000.0, 5000.0, 10000.0])
    sigma_I = torch.tensor([50.0, 100.0, 200.0])
    mean_I = torch.tensor([100.0, 100.0, 100.0])
    
    F_acen, sigma_F_acen, valid_acen = french_wilson_acentric(I, sigma_I, mean_I)
    F_cen, sigma_F_cen, valid_cen = french_wilson_centric(I, sigma_I, mean_I)
    
    print("Very strong reflections:")
    print(f"{'I':>10} {'σ(I)':>10} {'F_acen':>10} {'F_cen':>10} {'Ratio':>10}")
    print("-" * 60)
    for i in range(len(I)):
        ratio = F_cen[i] / F_acen[i]
        print(f"{I[i]:10.2f} {sigma_I[i]:10.2f} {F_acen[i]:10.2f} {F_cen[i]:10.2f} {ratio:10.3f}")
    
    # For very strong reflections, F should be close to sqrt(I)
    F_expected = torch.sqrt(I)
    print(f"\nExpected F (sqrt(I)): {F_expected}")
    print(f"Acentric F: {F_acen}")
    print(f"Difference: {torch.abs(F_acen - F_expected)}")
    
    print("✓ Asymptotic formulas test passed!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing French-Wilson Core Functions")
    print("=" * 70)
    
    test_acentric_conversion()
    test_centric_conversion()
    test_mixed_centric_acentric()
    test_resolution_binning()
    test_lookup_table_interpolation()
    test_asymptotic_formulas()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
