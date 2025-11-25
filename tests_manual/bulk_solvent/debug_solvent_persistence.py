#!/usr/bin/env python3
"""
Debug why "No solvent model found" appears repeatedly.
Check if solvent model persists after refine_solvent().
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
from torchref.base_refinement import Refinement

# Load test data
pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb'
mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz'

print("Loading model...")
instance = Refinement(mtz_path, pdb_path, verbose=False)

print("\n1. Before refine_solvent():")
print(f"   hasattr(instance, 'solvent') = {hasattr(instance, 'solvent')}")

print("\n2. Running refine_solvent()...")
instance.refine_solvent(iter=5, lr=0.001, optimize_phase=True)

print("\n3. After refine_solvent():")
print(f"   hasattr(instance, 'solvent') = {hasattr(instance, 'solvent')}")
if hasattr(instance, 'solvent'):
    print(f"   instance.solvent = {instance.solvent}")
    print(f"   k_sol = {torch.exp(instance.solvent.log_k_solvent).item():.4f}")
    print(f"   B_sol = {torch.exp(instance.solvent.log_b_solvent).item():.2f}")

print("\n4. Computing F_calc with use_solvent=True:")
with torch.no_grad():
    fcalc_with_solvent = instance.get_fcalc(use_solvent=True)
    print(f"   Mean |F_calc| = {torch.abs(fcalc_with_solvent).mean().item():.2f}")

print("\n5. Computing F_calc with use_solvent=False:")
with torch.no_grad():
    fcalc_without_solvent = instance.get_fcalc(use_solvent=False)
    print(f"   Mean |F_calc| = {torch.abs(fcalc_without_solvent).mean().item():.2f}")

print("\n6. Difference:")
with torch.no_grad():
    diff = torch.abs(fcalc_with_solvent - fcalc_without_solvent).mean().item()
    print(f"   Mean |F_with - F_without| = {diff:.2f}")
    print(f"   Relative difference = {100 * diff / torch.abs(fcalc_without_solvent).mean().item():.1f}%")

print("\n7. Computing R-factors:")
rwork, rtest = instance.get_rfactor()
print(f"   R_work = {rwork:.4f}")
print(f"   R_test = {rtest:.4f}")

print("\n8. Checking if solvent still exists:")
print(f"   hasattr(instance, 'solvent') = {hasattr(instance, 'solvent')}")
