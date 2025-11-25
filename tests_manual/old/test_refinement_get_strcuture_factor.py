import torchref.Model as Model
import torchref.refinement as refinement
from torchref import Data 
import numpy as np
import reciprocalspaceship as rs
import cProfile
import pstats

M = Model.model()
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb')


hkl = Data.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_direct_summation.mtz')
F_calc_old = np.array(hkl.F)



ref = refinement.Refinement(hkl,model=M)

hkls = hkl[['h','k','l']].values

profiler = cProfile.Profile()
profiler.enable()

f = ref.get_structure_factor()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(50)


F_calc_new = np.abs(f.detach().numpy())

F = np.abs(f.detach().numpy())
phases = np.rad2deg(np.angle(f.detach().numpy()))

print(np.corrcoef(F_calc_old,F_calc_new)[0,1])

dataset = rs.DataSet({'H':hkls[:,0],'K':hkls[:,1],'L':hkls[:,2],'F':F,'PHI':phases}).set_index(['H','K','L'])

dataset.cell = M.cell

dataset.spacegroup = M.spacegroup
dataset.infer_mtz_dtypes(inplace=True)
dataset.write_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_direct_summation_new.mtz')
