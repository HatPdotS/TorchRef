from torchref import Model
from time import time
from torchref import direct_summation as ds
import reciprocalspaceship as rs
import numpy as np
import matplotlib.pyplot as plt
import os

data = rs.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test.mtz').reset_index()

hkl = data[['H','K','L']].values.astype(np.int32)

m = Model.model()
m.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test.pdb')
m.setup_grids()

t = time()

scales = []
corrs = []

mappath = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/maps_scale_screen'
os.makedirs(mappath,exist_ok=True)
for scale in [0.15]:
    scales.append(scale)
    m.bscale = scale
    m.build()
    map = m.map
    cell = m.pdb.attrs['cell']
    f = m.get_f(hkl)
    F = np.abs(f)
    ds.write_numpy_to_ccp4(map,f'{mappath}/test.ccp4',cell)
    corr = np.corrcoef(F,data['F-model'].values)[0,1]
    corrs.append(corr)   

phases = np.rad2deg(np.angle(f))
print(phases)
F = np.abs(f)

dataset = rs.DataSet({'H':hkl[:,0],'K':hkl[:,1],'L':hkl[:,2],'F':F,'PHI':phases}).set_index(['H','K','L'])
dataset.cell = m.pdb.attrs['cell']
dataset.spacegroup = m.pdb.attrs['spacegroup']
dataset.infer_mtz_dtypes(inplace=True)  
dataset.write_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_2.mtz')

plt.plot(F,data['F-model'],'o')
plt.savefig('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Fcalc_fcalc.png')   

print(scales)
print(corrs)
