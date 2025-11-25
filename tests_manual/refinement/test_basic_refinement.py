#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 32
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/test.log

from torchref.base_refinement import Refinement


pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb'
mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz'
cif = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'


instance = Refinement(mtz, pdb, cif=cif, verbose=1)


rwork, rtest = instance.get_rfactor()
print(f'Pre Rwork: {rwork}, Rtest: {rtest}')

instance.model.shake_coords(0.15)

rwork, rtest = instance.get_rfactor()

print(f'Initial Rwork: {rwork}, Rtest: {rtest}')
instance.run_refinement(n_cycles=20, lr=0.005)
instance.run_refinement(n_cycles=20, lr=0.003)
instance.run_refinement(n_cycles=20, lr=0.002)
instance.run_refinement(n_cycles=20, lr=0.001)




rwork, rtest = instance.get_rfactor()

print(f'Final Rwork: {rwork}, Rtest: {rtest}')