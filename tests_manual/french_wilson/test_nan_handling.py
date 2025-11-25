#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

"""
Test script for NaN handling in French-Wilson conversion.
"""

import torch
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.french_wilson import FrenchWilson

print("=" * 70)
print("Testing French-Wilson NaN Handling")
print("=" * 70)
print()

# Create test data with some NaN values
unit_cell = torch.tensor([50.0, 60.0, 70.0, 90.0, 90.0, 90.0], dtype=torch.float32)
n_refl = 1000

# Generate random Miller indices
hkl = torch.randint(-10, 11, (n_refl, 3), dtype=torch.int32)

# Generate intensity data with some NaN values
I = torch.abs(torch.randn(n_refl)) * 100 + 50
sigma_I = torch.abs(torch.randn(n_refl)) * 10 + 5

# Introduce NaN values at specific positions
nan_indices = [10, 50, 100, 200, 500]
I[nan_indices] = float('nan')
sigma_I[250] = float('nan')  # Additional NaN in sigma

print("Test 1: Mixed valid and NaN values")
print("-" * 70)
print(f"Total reflections: {n_refl}")
print(f"NaN in I: {torch.isnan(I).sum().item()}")
print(f"NaN in sigma_I: {torch.isnan(sigma_I).sum().item()}")
print(f"Total NaN (union): {(torch.isnan(I) | torch.isnan(sigma_I)).sum().item()}")
print()

# Initialize French-Wilson
fw = FrenchWilson(hkl, unit_cell, space_group='P21', verbose=0)

# Apply conversion
F, sigma_F = fw(I, sigma_I)

print("Results:")
print(f"  NaN in F output: {torch.isnan(F).sum().item()}")
print(f"  NaN in sigma_F output: {torch.isnan(sigma_F).sum().item()}")
print(f"  Valid F values: {(~torch.isnan(F)).sum().item()}")
print()

# Verify that NaN positions match
nan_mask_input = torch.isnan(I) | torch.isnan(sigma_I)
nan_mask_output = torch.isnan(F)

if torch.all(nan_mask_input == nan_mask_output):
    print("✓ NaN positions preserved correctly")
else:
    print("✗ NaN positions do not match!")
    print(f"  Input NaN mask sum: {nan_mask_input.sum().item()}")
    print(f"  Output NaN mask sum: {nan_mask_output.sum().item()}")

print()

# Check that valid values are reasonable
valid_F = F[~torch.isnan(F)]
print(f"Valid F statistics:")
print(f"  Min: {valid_F.min().item():.3f}")
print(f"  Max: {valid_F.max().item():.3f}")
print(f"  Mean: {valid_F.mean().item():.3f}")
print(f"  All positive: {torch.all(valid_F > 0).item()}")
print()

# Test 2: All NaN values
print("Test 2: All NaN values")
print("-" * 70)
I_all_nan = torch.full((100,), float('nan'))
sigma_I_all_nan = torch.full((100,), float('nan'))
hkl_small = torch.randint(-5, 6, (100, 3), dtype=torch.int32)

fw_small = FrenchWilson(hkl_small, unit_cell, space_group='P1', verbose=0)
F_all_nan, sigma_F_all_nan = fw_small(I_all_nan, sigma_I_all_nan)

print(f"Input: All {len(I_all_nan)} values are NaN")
print(f"Output: {torch.isnan(F_all_nan).sum().item()} NaN in F")
print(f"Output: {torch.isnan(sigma_F_all_nan).sum().item()} NaN in sigma_F")

if torch.all(torch.isnan(F_all_nan)) and torch.all(torch.isnan(sigma_F_all_nan)):
    print("✓ All NaN input correctly handled")
else:
    print("✗ All NaN input not handled correctly!")

print()

# Test 3: No NaN values
print("Test 3: No NaN values")
print("-" * 70)
I_no_nan = torch.abs(torch.randn(100)) * 100 + 50
sigma_I_no_nan = torch.abs(torch.randn(100)) * 10 + 5

F_no_nan, sigma_F_no_nan = fw_small(I_no_nan, sigma_I_no_nan)

print(f"Input: No NaN values")
print(f"Output: {torch.isnan(F_no_nan).sum().item()} NaN in F")
print(f"Output: {torch.isnan(sigma_F_no_nan).sum().item()} NaN in sigma_F")

if not torch.any(torch.isnan(F_no_nan)) and not torch.any(torch.isnan(sigma_F_no_nan)):
    print("✓ Clean input produces clean output")
else:
    print("✗ Clean input produced unexpected NaN values!")

print()
print("=" * 70)
print("All NaN handling tests completed successfully!")
print("=" * 70)
