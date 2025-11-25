#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/benchmark_refining/test_A10F.out

from glob import glob
from torchref.base_refinement import Refinement



mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F.mtz'
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A0F/1A0F_shaken.pdb'


refinement = Refinement(mtz, pdb, verbose=10,target_weights={'xray':1.0, 'restraints':1.0, 'adp':0.3})
refinement.model.verbose = 1
refinement.restraints.verbose = 1


def separate_refine(refinement):
    refinement.model.freeze_all()
    refinement.model.unfreeze('xyz')
    refinement.run_refinement(macro_cycles=1, n_steps=15, lr=[0.05,0.01])
    refinement.model.freeze('xyz')
    refinement.model.unfreeze('b')
    refinement.run_refinement(macro_cycles=1, n_steps=15, lr=[0.05,0.01])
    refinement.model.freeze('b')
    refinement.model.unfreeze('u')
    refinement.run_refinement(macro_cycles=1, n_steps=15, lr=[0.05,0.01])

for i in range(5):
    separate_refine(refinement)