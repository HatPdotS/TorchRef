#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/new_sol.log

from torchref.base_refinement import Refinement


mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/BR_LCLS_refine_8.mtz'
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/BR_LCLS_refine_8.pdb'

from time import time


start_time = time()
ref = Refinement(mtz, pdb, max_res=1.5)
print(f"Loaded data and model in {time() - start_time:.1f} seconds")

start_time = time()
ref.model(ref.reflection_data.get_hkl(), recalc=True)
print(f"Calculated initial model structure factors in {time() - start_time:.1f} seconds")
start_time = time()
ref.setup_solvent()
print(f"Set up solvent model in {time() - start_time:.1f} seconds")


ref.scaler.setup_anisotropy_correction()

start_time = time()
ref.refine_solvent(iter=50)

print(f"Refined solvent model for 50 iterations in {time() - start_time:.1f} seconds")
print(ref.get_rfactor())