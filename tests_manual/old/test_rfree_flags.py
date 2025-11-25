#!/usr/bin/env python
"""
Test script for R-free flag handling in ReflectionData class.
"""

import torch
from torchref.Data import ReflectionData


print("="*80)
print("Testing R-free Flag Handling")
print("="*80)

# Load data
data = ReflectionData()
data.load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz')

print("\n" + "="*80)
print("Test 1: Check R-free flags availability")
print("="*80)
if data.rfree_flags is not None:
    print(f"✓ R-free flags loaded from: {data.rfree_source}")
    print(f"  Flags shape: {data.rfree_flags.shape}")
    print(f"  Flags dtype: {data.rfree_flags.dtype}")
    unique_flags = torch.unique(data.rfree_flags).tolist()
    print(f"  Unique flag values: {unique_flags}")
else:
    print("✗ No R-free flags found")

print("\n" + "="*80)
print("Test 2: Get work/test masks")
print("="*80)
work_mask, test_mask = data.get_rfree_masks()
if work_mask is not None:
    print(f"✓ Work mask: {work_mask.sum()} reflections (flag != 0)")
    print(f"✓ Test mask: {test_mask.sum()} reflections (flag == 0)")
    print(f"  Total: {work_mask.sum() + test_mask.sum()} reflections")
    print(f"  Test set percentage: {100.0 * test_mask.sum() / len(data):.2f}%")
else:
    print("✗ No masks available (no R-free flags)")

print("\n" + "="*80)
print("Test 3: Get work set structure factors")
print("="*80)
F_work, sigma_work = data.get_work_set()
if F_work is not None:
    print(f"✓ Work set F: {F_work.shape}")
    if sigma_work is not None:
        print(f"✓ Work set σ: {sigma_work.shape}")
    print(f"  Work set F range: [{F_work.min():.2f}, {F_work.max():.2f}]")
else:
    print("✗ No work set data available")

print("\n" + "="*80)
print("Test 4: Get test set structure factors")
print("="*80)
try:
    F_test, sigma_test = data.get_test_set()
    if F_test is not None:
        print(f"✓ Test set F: {F_test.shape}")
        if sigma_test is not None:
            print(f"✓ Test set σ: {sigma_test.shape}")
        print(f"  Test set F range: [{F_test.min():.2f}, {F_test.max():.2f}]")
    else:
        print("✗ No test set data available")
except ValueError as e:
    print(f"✗ Error: {e}")

print("\n" + "="*80)
print("Test 5: Resolution filtering preserves R-free flags")
print("="*80)
filtered = data.filter_by_resolution(d_min=1.5, d_max=10.0)
if filtered.rfree_flags is not None:
    print(f"✓ Filtered data has R-free flags")
    print(f"  Original reflections: {len(data)}")
    print(f"  Filtered reflections: {len(filtered)}")
    
    work_mask_orig, test_mask_orig = data.get_rfree_masks()
    work_mask_filt, test_mask_filt = filtered.get_rfree_masks()
    
    orig_test_pct = 100.0 * test_mask_orig.sum() / len(data)
    filt_test_pct = 100.0 * test_mask_filt.sum() / len(filtered)
    
    print(f"  Original test %: {orig_test_pct:.2f}%")
    print(f"  Filtered test %: {filt_test_pct:.2f}%")
    print(f"  R-free source preserved: {filtered.rfree_source == data.rfree_source}")
else:
    print("✗ R-free flags not preserved in filtering")

print("\n" + "="*80)
print("All R-free flag tests completed!")
print("="*80)
