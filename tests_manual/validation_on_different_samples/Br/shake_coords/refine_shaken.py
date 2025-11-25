#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
#SBATCH -c 16
#SBATCH -p gpu-day
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/shake_coords/refine_shaken_gpu.out
#SBATCH --gres=gpu:1

from torchref.base_refinement import Refinement
from time import time


mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/BR_LCLS_refine_8.mtz'
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/shake_coords/shaken_BR_LCLS_refine_8.pdb'
cif = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/LYR_open.cif'

refinement = Refinement(mtz, pdb, cif, max_res=1.5,verbose=1)

refinement.cuda()

refinement.get_scales()

print('Rfactor before refinement: ', refinement.get_rfactor())

print(50*'=')

print("Starting refinement on shaken structure...")

print(50*'=')

start_time = time()
refinement.run_refinement()

end_time = time()
refinement.model.write_pdb('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/validation_on_different_samples/Br/shake_coords/refined_shaken_BR_LCLS_refine_8.pdb')

print(50*'=')
print('Done in %.2f seconds' % (end_time - start_time))
print('Rfactor after refinement: ', refinement.get_rfactor())
print(50*'=')