#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
"""
Debug: Check bin-wise scale factors
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
import numpy as np
from torchref.base_refinement import Refinement

# Load
instance = Refinement(
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz',
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb',
    cif='/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif',
    verbose=0
)

instance.scaler.setup_anisotropy_correction()

print("\n" + "="*80)
print("BIN-WISE SCALE FACTORS DEBUG")
print("="*80)

# Check log_scale parameters
log_scale = instance.scaler.log_scale.detach().cpu().numpy()
scale = np.exp(log_scale)

print(f"\nNumber of bins: {len(scale)}")
print(f"\nBin-wise scale factors:")
print(f"{'Bin':>5} {'log_scale':>12} {'scale':>12}")
print("-" * 32)
for i in range(len(scale)):
    print(f"{i:5d} {log_scale[i]:12.6f} {scale[i]:12.6f}")

print(f"\nScale factor statistics:")
print(f"  Mean:   {scale.mean():.6f}")
print(f"  Median: {np.median(scale):.6f}")
print(f"  Min:    {scale.min():.6f}")
print(f"  Max:    {scale.max():.6f}")

if scale.mean() < 0.1:
    print(f"\n⚠️  SCALE FACTORS ARE TOO SMALL!")
    print(f"   Mean scale = {scale.mean():.6f}")
    print(f"   This is reducing F_calc by {1/scale.mean():.1f}x")
    print(f"\n  LIKELY CAUSE: log_scale parameters are too negative")
    print(f"   log_scale should be near 0, not near {log_scale.mean():.2f}")

# Check bin assignments
print(f"\nBin assignment statistics:")
bins = instance.scaler.bins.cpu().numpy()
print(f"  Unique bins: {np.unique(bins)}")
print(f"  Bin counts:")
for i in range(len(scale)):
    count = np.sum(bins == i)
    print(f"    Bin {i}: {count} reflections")

print("="*80)
