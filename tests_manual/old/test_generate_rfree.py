#!/usr/bin/env python
"""
Test script for automatic R-free flag generation.
"""

import torch
import reciprocalspaceship as rs
from torchref.Data import ReflectionData


print("="*80)
print("Test 1: Load dataset with R-free flags (should flip)")
print("="*80)

data_with_flags = ReflectionData()
data_with_flags.load_from_mtz('test_data/tubulin/dark.mtz')

print("\n" + "="*80)
print("Test 2: Create dataset WITHOUT R-free flags and test auto-generation")
print("="*80)

# Load MTZ and remove R-free flags
print("Preparing test dataset without R-free flags...")
ds = rs.read_mtz('test_data/tubulin/dark.mtz')

# Remove R-free column
if 'R-free-flags' in ds.columns:
    ds = ds.drop(columns=['R-free-flags'])
    print(f"✓ Removed R-free-flags column")

# Save temporary MTZ without R-free flags
temp_file = '_temp_no_rfree.mtz'
ds.write_mtz(temp_file)
print(f"✓ Saved temporary MTZ: {temp_file}")

# Load with ReflectionData (should auto-generate flags)
print("\nLoading dataset without R-free flags...")
data_no_flags = ReflectionData()
data_no_flags.load_from_mtz(temp_file)

print("\n" + "="*80)
print("Test 3: Verify generated flags")
print("="*80)

if data_no_flags.rfree_flags is not None:
    print(f"✓ R-free flags were generated")
    print(f"  Source: {data_no_flags.rfree_source}")
    
    work_mask, test_mask = data_no_flags.get_rfree_masks()
    n_work = work_mask.sum().item()
    n_test = test_mask.sum().item()
    test_pct = 100.0 * n_test / len(data_no_flags)
    
    print(f"  Work set: {n_work} reflections")
    print(f"  Test set: {n_test} reflections")
    print(f"  Test %: {test_pct:.2f}%")
    
    # Check distribution across resolution bins
    print("\n  Checking distribution across resolution shells:")
    
    # Create 5 resolution shells
    res_sorted, _ = torch.sort(data_no_flags.resolution)
    n_per_shell = len(res_sorted) // 5
    
    for i in range(5):
        start_idx = i * n_per_shell
        end_idx = (i + 1) * n_per_shell if i < 4 else len(res_sorted)
        
        shell_min = res_sorted[start_idx].item()
        shell_max = res_sorted[end_idx - 1].item()
        
        # Get reflections in this shell
        shell_mask = (data_no_flags.resolution >= shell_min) & (data_no_flags.resolution <= shell_max)
        shell_test_mask = shell_mask & test_mask
        
        n_shell = shell_mask.sum().item()
        n_shell_test = shell_test_mask.sum().item()
        shell_test_pct = 100.0 * n_shell_test / n_shell if n_shell > 0 else 0
        
        print(f"    Shell {i+1} ({shell_min:.2f}-{shell_max:.2f} Å): "
              f"{n_shell_test}/{n_shell} test ({shell_test_pct:.1f}%)")
    
else:
    print("✗ ERROR: R-free flags were not generated")

print("\n" + "="*80)
print("Test 4: Manual regeneration with custom parameters")
print("="*80)

# Test regeneration with different parameters
print("\nTesting regeneration with 10% free and seed=42...")
data_no_flags.regenerate_rfree_flags(free_fraction=0.10, seed=42, force=True)

work_mask, test_mask = data_no_flags.get_rfree_masks()
test_pct = 100.0 * test_mask.sum() / len(data_no_flags)
print(f"✓ Regenerated with {test_pct:.2f}% test reflections")

print("\n" + "="*80)
print("Test 5: Test reproducibility with seed")
print("="*80)

# Generate flags with seed
data_seed1 = ReflectionData()
data_seed1.load_from_mtz(temp_file)
# Already generated automatically, regenerate with seed
data_seed1.regenerate_rfree_flags(free_fraction=0.05, seed=123, force=True)
flags1 = data_seed1.rfree_flags.clone()

# Generate again with same seed
data_seed2 = ReflectionData()
data_seed2.load_from_mtz(temp_file)
data_seed2.regenerate_rfree_flags(free_fraction=0.05, seed=123, force=True)
flags2 = data_seed2.rfree_flags.clone()

if torch.equal(flags1, flags2):
    print("✓ Flags are reproducible with same seed")
else:
    print("✗ ERROR: Flags differ with same seed")

# Test different seed gives different flags
data_seed3 = ReflectionData()
data_seed3.load_from_mtz(temp_file)
data_seed3.regenerate_rfree_flags(free_fraction=0.05, seed=999, force=True)
flags3 = data_seed3.rfree_flags.clone()

if not torch.equal(flags1, flags3):
    print("✓ Different seed produces different flags")
else:
    print("✗ ERROR: Different seeds produced identical flags")

# Cleanup
import os
if os.path.exists(temp_file):
    os.remove(temp_file)
    print(f"\n✓ Cleaned up temporary file: {temp_file}")

print("\n" + "="*80)
print("All R-free flag generation tests completed!")
print("="*80)
