#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
"""
Diagnostic: Check resolution-dependent bulk solvent performance.

KEY INSIGHT FROM PHENIX LOG:
- Low resolution (9.5-3.0 Å): kmask = 0.22-0.285  
- High resolution (3.0-1.7 Å): kmask = 0.000

We currently apply constant k_sol across ALL resolutions.
This script checks if zeroing k_sol at high resolution would help.
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
import numpy as np
from torchref.base_refinement import Refinement
from torchref.math_functions.math_torch import get_scattering_vectors, rfactor, get_rfactors

# Load data  
instance = Refinement(
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz',
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb',
    cif='/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif',
    verbose=0
)

# Setup
instance.scaler.setup_anisotropy_correction()
instance.setup_solvent()

# Get resolution 
scattering_vectors = get_scattering_vectors(instance.hkl, instance.model.cell)
s = torch.norm(scattering_vectors, dim=1) / 2.0
d_spacing = 1.0 / (2.0 * s)

print("\n" + "="*80)
print("RESOLUTION-DEPENDENT BULK SOLVENT ANALYSIS")
print("="*80)

# Get data
f_obs = instance.reflection_data.F
rfree_flags = instance.reflection_data.rfree_flags
work_mask = (rfree_flags == 0)

# Resolution bins (similar to Phenix)
resolution_bins = [
    (9.5, 5.0, "Low"),
    (5.0, 3.0, "Medium-Low"),
    (3.0, 2.5, "Medium-High"),
    (2.5, 2.0, "High"),
    (2.0, 1.7, "Very High"),
]

print(f"\n{'Bin':>12} {'Resolution':>15} {'Nwork':>8} {'R_no_sol':>10} {'R_const_k':>10} {'R_zero_k':>10} {'Winner':>10}")
print("-" * 80)

# Get F_calc and F_solvent
with torch.no_grad():
    fcalc_protein = instance.get_fcalc()
    fcalc_scaled = instance.scaler(fcalc_protein)
    f_solvent = instance.solvent(instance.hkl, update_fsol=True, F_protein=fcalc_scaled)

for d_min, d_max, label in resolution_bins:
    bin_mask = (d_spacing >= d_min) & (d_spacing < d_max) & work_mask
    n_work = bin_mask.sum().item()
    
    if n_work == 0:
        continue
    
    # Get data for this bin
    f_obs_bin = f_obs[bin_mask]
    fcalc_bin = fcalc_scaled[bin_mask]
    f_sol_bin = f_solvent[bin_mask]
    
    # R-factor WITHOUT bulk solvent
    r_no_sol = rfactor(f_obs_bin, fcalc_bin)
    
    # R-factor WITH constant k_sol (current implementation)
    fcalc_with_sol = fcalc_bin + f_sol_bin
    r_const_k = rfactor(f_obs_bin, fcalc_with_sol)
    
    # R-factor WITH k_sol=0 (Phenix behavior at high resolution)
    r_zero_k = r_no_sol
    
    # Which is better?
    if r_const_k < r_no_sol:
        winner = "const_k"
        improvement = r_no_sol - r_const_k
    else:
        winner = "zero_k"
        improvement = 0.0
    
    print(f"{label:>12} {d_max:6.2f}-{d_min:5.2f} Å {n_work:8d} {r_no_sol:10.4f} {r_const_k:10.4f} {r_zero_k:10.4f} {winner:>10}")

print("-" * 80)

# Overall statistics
rwork_no_sol, rtest_no_sol = get_rfactors(f_obs, fcalc_scaled, rfree_flags)
rwork_with_sol, rtest_with_sol = get_rfactors(f_obs, fcalc_scaled + f_solvent, rfree_flags)

print(f"\nOVERALL STATISTICS:")
print(f"  Without bulk solvent: R_work = {rwork_no_sol:.4f}, R_test = {rtest_no_sol:.4f}")
print(f"  With constant k_sol:  R_work = {rwork_with_sol:.4f}, R_test = {rtest_with_sol:.4f}")
print(f"  Improvement:         ΔR_work = {rwork_no_sol - rwork_with_sol:.4f}, ΔR_test = {rtest_no_sol - rtest_with_sol:.4f}")

# Now test with resolution-dependent k_sol (Phenix style)
print(f"\n" + "="*80)
print("TESTING RESOLUTION-DEPENDENT K_SOL (PHENIX STYLE)")
print("="*80)

# Create resolution-dependent scaling like Phenix
# Phenix uses k_mask=0 above 3.0 Å resolution
high_res_cutoff = 3.0
mask_high_res = d_spacing < high_res_cutoff

# Zero out bulk solvent at high resolution
f_solvent_res_dep = f_solvent.clone()
f_solvent_res_dep[mask_high_res] = 0.0

rwork_res_dep, rtest_res_dep = get_rfactors(f_obs, fcalc_scaled + f_solvent_res_dep, rfree_flags)

print(f"\nWith resolution-dependent k_sol (zero above {high_res_cutoff} Å):")
print(f"  R_work = {rwork_res_dep:.4f}, R_test = {rtest_res_dep:.4f}")
print(f"  vs constant k_sol: ΔR_work = {rwork_with_sol - rwork_res_dep:.4f}, ΔR_test = {rtest_with_sol - rtest_res_dep:.4f}")

print(f"\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if rwork_res_dep < rwork_with_sol:
    print("✓ Resolution-dependent k_sol (Phenix style) is BETTER!")
    print(f"  It improves R_work by {rwork_with_sol - rwork_res_dep:.4f}")
    print(f"  This explains why Phenix achieves R_work~0.17 vs our ~0.20")
    print("\n  ACTION NEEDED: Implement resolution-dependent bulk solvent correction")
else:
    print("✗ Resolution-dependent k_sol does NOT help")
    print("  The problem lies elsewhere...")
print("="*80)
