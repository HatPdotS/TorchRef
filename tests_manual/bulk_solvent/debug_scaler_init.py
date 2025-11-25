#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
"""
Debug: Check scaler initialization
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
import numpy as np
from torchref.base_refinement import Refinement

# Load with verbose=2 to see scaler details
instance = Refinement(
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz',
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb',
    cif='/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif',
    verbose=0
)

print("\n" + "="*80)
print("SCALER INITIALIZATION DEBUG")
print("="*80)

# Manually check what F_calc was used to initialize scaler
print(f"\nChecking F_calc used for scaler initialization:")
print(f"  instance.scaler.fcalc (stored):")
print(f"    Mean |F_calc|: {torch.abs(instance.scaler.fcalc).mean().item():.2f}")
print(f"    Max |F_calc|:  {torch.abs(instance.scaler.fcalc).max().item():.2f}")

# Get current F_calc
f_calc_now = instance.get_fcalc()
print(f"\n  instance.get_fcalc() (current):")
print(f"    Mean |F_calc|: {torch.abs(f_calc_now).mean().item():.2f}")
print(f"    Max |F_calc|:  {torch.abs(f_calc_now).max().item():.2f}")

# Get F_obs
f_obs = instance.reflection_data.F
print(f"\n  F_obs:")
print(f"    Mean: {f_obs.mean().item():.2f}")
print(f"    Max:  {f_obs.max().item():.2f}")

# Check if they're the same
if not torch.allclose(instance.scaler.fcalc, f_calc_now, rtol=0.01):
    print(f"\n  ⚠️  F_CALC HAS CHANGED since scaler initialization!")
    print(f"     The scaler was initialized with different F_calc values")
    print(f"     This can happen if:")
    print(f"       1. Bulk solvent was added after scaler initialization")
    print(f"       2. Model parameters changed")
    print(f"       3. Scaler.fcalc wasn't updated")

# Manually calculate what log_scale should be
with torch.no_grad():
    log_ratios = torch.log(f_obs + 1e-6) - torch.log(torch.abs(instance.scaler.fcalc) + 1e-6)
    print(f"\n  log(F_obs / |F_calc_stored|) statistics:")
    print(f"    Mean:   {log_ratios.mean().item():.4f}")
    print(f"    Median: {torch.median(log_ratios).item():.4f}")
    print(f"    Std:    {log_ratios.std().item():.4f}")
    
    # Compare with actual log_scale
    actual_log_scale = instance.scaler.log_scale.detach()
    print(f"\n  Actual log_scale parameters:")
    print(f"    Mean:   {actual_log_scale.mean().item():.4f}")
    print(f"    Median: {torch.median(actual_log_scale).item():.4f}")
    print(f"    Std:    {actual_log_scale.std().item():.4f}")
    
    if abs(actual_log_scale.mean().item() - log_ratios.mean().item()) > 1.0:
        print(f"\n  ⚠️  LOG_SCALE IS VERY DIFFERENT FROM EXPECTED!")
        print(f"     Expected mean: {log_ratios.mean().item():.4f}")
        print(f"     Actual mean:   {actual_log_scale.mean().item():.4f}")
        print(f"     Difference:    {actual_log_scale.mean().item() - log_ratios.mean().item():.4f}")

print("="*80)
