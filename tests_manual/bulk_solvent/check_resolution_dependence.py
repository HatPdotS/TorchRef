#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
"""
Diagnostic: Check if resolution-dependent k_sol would improve results.

Phenix applies k_mask=0 at high resolution (>3.0 Å) but k_mask~0.25 at low resolution.
We currently apply constant k_sol across all resolutions.
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
import numpy as np
from torchref.base_refinement import Refinement
from torchref.math_functions.math_torch import get_scattering_vectors

# Load data
instance = Refinement(
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz',
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb',
    cif='/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif',
)

# Setup anisotropic correction
instance.scaler.setup_anisotropy_correction()

# Get working and test sets
rfree = instance.reflection_data.rfree
f_obs = instance.reflection_data.F
work_mask = ~rfree

# Get F_calc with NO bulk solvent
fcalc_no_solvent = instance.model.get_iso(instance.hkl, update_parametrization=False).detach()

# Scale
fcalc_no_solvent_scaled = instance.scaler(fcalc_no_solvent, instance.hkl, optimize_scale=False)

# Get resolution (d-spacing)
scattering_vectors = get_scattering_vectors(instance.hkl, instance.model.cell)
s = torch.norm(scattering_vectors, dim=1) / 2.0  # sin(θ)/λ
d_spacing = 1.0 / (2.0 * s)

print("\n" + "="*70)
print("RESOLUTION-DEPENDENT BULK SOLVENT ANALYSIS")
print("="*70)

# Define resolution bins like Phenix
resolution_bins = [
    (9.5, 5.0),
    (5.0, 3.0),
    (3.0, 2.5),
    (2.5, 2.0),
    (2.0, 1.7),
]

print(f"\nInitial R-factors (no bulk solvent):")
r_work, r_test = instance.get_rfactors(fcalc_no_solvent_scaled)
print(f"  R_work = {r_work:.4f}")
print(f"  R_test = {r_test:.4f}")

# Setup bulk solvent model
instance.setup_solvent_model()

# Get F_solvent
f_solvent = instance.solvent_model(instance.hkl, update_fsol=True, F_protein=fcalc_no_solvent_scaled)

print(f"\nBulk solvent parameters:")
k_sol = np.exp(instance.solvent_model.log_k_solvent.detach().cpu().numpy())
b_sol = np.exp(instance.solvent_model.log_b_solvent.detach().cpu().numpy())
print(f"  k_sol = {k_sol:.3f}")
print(f"  B_sol = {b_sol:.1f} Ų")

print(f"\n{'Resolution':>15} {'Nrefl':>8} {'R_no_sol':>10} {'R_const_k':>10} {'R_zero_k':>10} {'Best':>10}")
print("-" * 70)

work_mask = ~rfree

for d_min, d_max in resolution_bins:
    # Bin mask
    bin_mask = (d_spacing >= d_min) & (d_spacing < d_max) & work_mask
    n_refl = bin_mask.sum().item()
    
    if n_refl == 0:
        continue
    
    f_obs_bin = f_obs[bin_mask]
    fcalc_bin = fcalc_no_solvent_scaled[bin_mask]
    f_sol_bin = f_solvent[bin_mask]
    
    # R-factor with NO bulk solvent
    r_no_sol = (torch.abs(torch.abs(fcalc_bin) - f_obs_bin).sum() / f_obs_bin.sum()).item()
    
    # R-factor with CONSTANT k_sol (current implementation)
    fcalc_with_sol = fcalc_bin + f_sol_bin
    r_const_k = (torch.abs(torch.abs(fcalc_with_sol) - f_obs_bin).sum() / f_obs_bin.sum()).item()
    
    # R-factor with ZERO k_sol (Phenix behavior at high resolution)
    r_zero_k = r_no_sol  # Same as no solvent
    
    # Determine which is better
    best = "const_k" if r_const_k < r_no_sol else "zero_k"
    best_r = min(r_const_k, r_zero_k)
    
    print(f"{d_max:6.2f}-{d_min:5.2f} Å {n_refl:8d} {r_no_sol:10.4f} {r_const_k:10.4f} {r_zero_k:10.4f} {best_r:10.4f} ({best})")

print("-" * 70)

# Overall R-factors
print(f"\nOverall R-factors:")
print(f"  Without bulk solvent: R_work = {r_work:.4f}, R_test = {r_test:.4f}")

fcalc_with_sol_all = fcalc_no_solvent_scaled + f_solvent
r_work_sol, r_test_sol = instance.get_rfactors(fcalc_with_sol_all)
print(f"  With constant k_sol:  R_work = {r_work_sol:.4f}, R_test = {r_test_sol:.4f}")
print(f"  Improvement: ΔR_work = {r_work - r_work_sol:.4f}, ΔR_test = {r_test - r_test_sol:.4f}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("If high-resolution bins show better R with zero_k than const_k,")
print("then we need to implement resolution-dependent k_sol like Phenix.")
print("="*70)
