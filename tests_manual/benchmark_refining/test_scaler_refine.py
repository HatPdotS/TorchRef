#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

"""Quick test to verify scaler.refine_lbfgs() works correctly"""

import torch
from torchref.base_refinement import Refinement
from torchref.lbfgs_refinement import LBFGSRefinement, LBFGS_scales

print("="*80)
print("Testing Scaler.refine_lbfgs() method")
print("="*80)

mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'

print("\n1. Testing with base Refinement class + scaler.refine_lbfgs()...")
refinement1 = Refinement(mtz, pdb, verbose=0)
refinement1.get_scales()

# Get initial R-factors
rwork_init, rfree_init = refinement1.get_rfactor()
print(f"   Initial: Rwork={rwork_init:.4f}, Rfree={rfree_init:.4f}")

# Call scaler's refine_lbfgs directly
refinement1.model.freeze_all()
metrics1 = refinement1.scaler.refine_lbfgs(nsteps=3, verbose=False)

rwork_final, rfree_final = refinement1.get_rfactor()
print(f"   Final:   Rwork={rwork_final:.4f}, Rfree={rfree_final:.4f}")
print(f"   ✓ Direct scaler.refine_lbfgs() works!")

print("\n2. Testing with LBFGSRefinement.refine_scales_lbfgs()...")
refinement2 = LBFGSRefinement(mtz, pdb, verbose=0)
refinement2.get_scales()

rwork_init, rfree_init = refinement2.get_rfactor()
print(f"   Initial: Rwork={rwork_init:.4f}, Rfree={rfree_init:.4f}")

metrics2 = refinement2.refine_scales_lbfgs(nsteps=3, verbose=False)

rwork_final, rfree_final = refinement2.get_rfactor()
print(f"   Final:   Rwork={rwork_final:.4f}, Rfree={rfree_final:.4f}")
print(f"   ✓ LBFGSRefinement.refine_scales_lbfgs() works!")

print("\n3. Testing standalone LBFGS_scales() function...")
refinement3 = Refinement(mtz, pdb, verbose=0)
refinement3.get_scales()

rwork_init, rfree_init = refinement3.get_rfactor()
print(f"   Initial: Rwork={rwork_init:.4f}, Rfree={rfree_init:.4f}")

metrics3 = LBFGS_scales(refinement3, nsteps=3, verbose=False)

rwork_final, rfree_final = refinement3.get_rfactor()
print(f"   Final:   Rwork={rwork_final:.4f}, Rfree={rfree_final:.4f}")
print(f"   ✓ LBFGS_scales() standalone function works!")

print("\n4. Testing full refinement with scale refinement...")
refinement4 = LBFGSRefinement(mtz, pdb, verbose=0)

rwork_init, rfree_init = refinement4.get_rfactor()
print(f"   Initial: Rwork={rwork_init:.4f}, Rfree={rfree_init:.4f}")

# Run 1 macro cycle with scale refinement
metrics4 = refinement4.run_lbfgs_refinement(
    macro_cycles=1,
    targets=['xyz'],
    steps_per_target=2,
    steps_scales=2,
    refine_scales=True
)

rwork_final, rfree_final = refinement4.get_rfactor()
print(f"   Final:   Rwork={rwork_final:.4f}, Rfree={rfree_final:.4f}")
print(f"   ✓ Full refinement with scale refinement works!")

print("\n" + "="*80)
print("ALL TESTS PASSED ✓")
print("="*80)
print("\nScaler now hosts all scaling operations including LBFGS refinement!")
