#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
"""
Debug: Check F_obs, F_calc, and F_solvent magnitudes
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
from torchref.base_refinement import Refinement

# Load
instance = Refinement(
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz',
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb',
    cif='/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif',
    verbose=0
)

instance.scaler.setup_anisotropy_correction()
instance.setup_solvent()

print("\n" + "="*80)
print("STRUCTURE FACTOR MAGNITUDE DEBUG")
print("="*80)

with torch.no_grad():
    # Get F_obs
    f_obs = instance.reflection_data.F
    print(f"\nF_obs (observed):")
    print(f"  Mean: {f_obs.mean().item():.2f}")
    print(f"  Min:  {f_obs.min().item():.2f}")
    print(f"  Max:  {f_obs.max().item():.2f}")
    
    # Get F_calc BEFORE scaling
    f_calc_unscaled = instance.get_fcalc()  # This calls model() which may include solvent
    print(f"\nF_calc (calculated, possibly with solvent):")
    print(f"  Mean: {torch.abs(f_calc_unscaled).mean().item():.2f}")
    print(f"  Min:  {torch.abs(f_calc_unscaled).min().item():.2f}")
    print(f"  Max:  {torch.abs(f_calc_unscaled).max().item():.2f}")
    
    # Get F_calc AFTER scaling  
    f_calc_scaled = instance.scaler(f_calc_unscaled)
    print(f"\nF_calc (after scaler):")
    print(f"  Mean: {torch.abs(f_calc_scaled).mean().item():.2f}")
    print(f"  Min:  {torch.abs(f_calc_scaled).min().item():.2f}")
    print(f"  Max:  {torch.abs(f_calc_scaled).max().item():.2f}")
    
    # Check if they match F_obs scale
    print(f"\nScale check:")
    print(f"  F_obs / F_calc_scaled ratio: {(f_obs.mean() / torch.abs(f_calc_scaled).mean()).item():.3f}")
    
    if torch.abs(f_calc_scaled).mean() < 10:
        print(f"  ⚠️  F_calc_scaled is TOO SMALL!")
        print(f"     Expected: ~200-300, Got: {torch.abs(f_calc_scaled).mean().item():.2f}")
        print(f"     The scaler is crushing F_calc to near-zero!")

print("="*80)
