from torchref import direct_summation as ds
from pdb_tools import load_pdb_as_pd
import reciprocalspaceship as rs
import numpy as np
import cProfile
import pstats


pdb = load_pdb_as_pd('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb')

data = rs.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.mtz').reset_index()

hkl = data[['H','K','L']].values.astype(np.int32)
cell = pdb.attrs['cell']
profiler = cProfile.Profile()
profiler.enable()

f = ds.direct_summation(pdb,hkl,cell)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(20)

F = np.abs(f)
phases = np.rad2deg(np.angle(f))

dataset = rs.DataSet({'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2],'F':F,'PHI':phases}).set_index(['H','K','L'])
dataset.cell = cell

dataset.spacegroup = pdb.attrs['spacegroup']
dataset.infer_mtz_dtypes(inplace=True)
dataset.write_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_direct_summation.mtz')

print('Corr:',np.corrcoef(F,data['F-model'].values)[0,1])