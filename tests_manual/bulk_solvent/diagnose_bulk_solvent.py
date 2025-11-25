#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
"""
Diagnose bulk solvent optimization.

Check if:
1. Bulk solvent parameters are actually changing during optimization
2. F_solvent magnitude is reasonable
3. The correction is actually being applied to F_calc
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
import numpy as np
from torchref.base_refinement import Refinement

# Load data
instance = Refinement(
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz',
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb',
    cif='/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif',
    verbose=1
)

print("\n" + "="*80)
print("BULK SOLVENT DIAGNOSTIC")
print("="*80)

# Setup
instance.scaler.setup_anisotropy_correction()

# Get initial R-factors
rwork_init, rtest_init = instance.get_rfactor()
print(f"\nInitial (no bulk solvent):")
print(f"  R_work = {rwork_init:.4f}, R_test = {rtest_init:.4f}")

# Setup bulk solvent
instance.setup_solvent()

# Check initial parameters
k_sol_init = torch.exp(instance.solvent.log_k_solvent).item()
b_sol_init = torch.exp(instance.solvent.log_b_solvent).item()
print(f"\nInitial bulk solvent parameters:")
print(f"  k_sol = {k_sol_init:.4f}")
print(f"  B_sol = {b_sol_init:.2f} Ų")

# Get F_solvent magnitude
with torch.no_grad():
    f_protein = instance.get_fcalc()
    f_solvent = instance.solvent(instance.hkl, update_fsol=True, F_protein=f_protein)
    f_protein_scaled = instance.scaler(f_protein)
    
print(f"\nStructure factor magnitudes:")
print(f"  <|F_protein|> = {torch.abs(f_protein_scaled).mean().item():.2f}")
print(f"  <|F_solvent|> = {torch.abs(f_solvent).mean().item():.2f}")
print(f"  Ratio: |F_solvent|/|F_protein| = {(torch.abs(f_solvent).mean() / torch.abs(f_protein_scaled).mean()).item():.3f}")

# Check if F_solvent is zero (would indicate a problem)
if torch.abs(f_solvent).max().item() < 1e-6:
    print("  ⚠️  WARNING: F_solvent is essentially zero!")
else:
    print(f"  ✓ F_solvent has reasonable magnitude (max = {torch.abs(f_solvent).max().item():.2f})")

# Get R-factor WITH bulk solvent (before optimization)
rwork_with_sol, rtest_with_sol = instance.get_rfactor()
print(f"\nWith initial bulk solvent (before optimization):")
print(f"  R_work = {rwork_with_sol:.4f}, R_test = {rtest_with_sol:.4f}")
print(f"  ΔR_work = {rwork_init - rwork_with_sol:.4f}, ΔR_test = {rtest_init - rtest_with_sol:.4f}")

if abs(rwork_init - rwork_with_sol) < 0.001:
    print("  ⚠️  WARNING: Bulk solvent has almost NO effect before optimization!")
    print("     This suggests the bulk solvent model is not being applied correctly.")
else:
    print(f"  ✓ Bulk solvent changes R-factor by {abs(rwork_init - rwork_with_sol):.4f}")

# Now optimize
print(f"\n" + "-"*80)
print("Optimizing bulk solvent parameters (10 iterations, lr=0.1)...")
print("-"*80)

instance.refine_solvent(iter=10, lr=0.1)

# Check optimized parameters
k_sol_opt = torch.exp(instance.solvent.log_k_solvent).item()
b_sol_opt = torch.exp(instance.solvent.log_b_solvent).item()
print(f"\nOptimized bulk solvent parameters:")
print(f"  k_sol: {k_sol_init:.4f} → {k_sol_opt:.4f} (change: {k_sol_opt - k_sol_init:+.4f})")
print(f"  B_sol: {b_sol_init:.2f} → {b_sol_opt:.2f} Ų (change: {b_sol_opt - b_sol_init:+.2f})")

if abs(k_sol_opt - k_sol_init) < 0.01 and abs(b_sol_opt - b_sol_init) < 1.0:
    print("  ⚠️  WARNING: Parameters barely changed during optimization!")
    print("     Gradients may not be flowing correctly.")
else:
    print(f"  ✓ Parameters changed significantly during optimization")

# Get final R-factors
rwork_final, rtest_final = instance.get_rfactor()
print(f"\nAfter optimization:")
print(f"  R_work = {rwork_final:.4f}, R_test = {rtest_final:.4f}")
print(f"  Total improvement: ΔR_work = {rwork_init - rwork_final:.4f}, ΔR_test = {rtest_init - rtest_final:.4f}")

# Compare with Phenix
phenix_rwork_init = 0.2329
phenix_rwork_final = 0.2062
phenix_improvement = phenix_rwork_init - phenix_rwork_final

print(f"\n" + "="*80)
print("COMPARISON WITH PHENIX")
print("="*80)
print(f"Your improvement:   ΔR_work = {rwork_init - rwork_final:.4f} ({(rwork_init - rwork_final)/rwork_init*100:.1f}%)")
print(f"Phenix improvement: ΔR_work = {phenix_improvement:.4f} ({phenix_improvement/phenix_rwork_init*100:.1f}%)")

if (rwork_init - rwork_final) < phenix_improvement * 0.5:
    print("\n⚠️  YOUR IMPROVEMENT IS MUCH LESS THAN PHENIX!")
    print("   Likely causes:")
    print("   1. Bulk solvent correction not being applied during get_fcalc()")
    print("   2. Gradients not flowing to bulk solvent parameters")
    print("   3. F_solvent magnitude too small")
elif (rwork_init - rwork_final) > phenix_improvement:
    print("\n✓ YOUR IMPROVEMENT MATCHES OR EXCEEDS PHENIX!")
    print("  Bulk solvent optimization is working correctly.")
else:
    print("\n✓ YOUR IMPROVEMENT IS COMPARABLE TO PHENIX")
    print("  Bulk solvent optimization is working reasonably well.")

print("="*80)
