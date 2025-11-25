import pdb_tools 
import pandas as pd
import torchref.direct_summation as ds
from crystfel_tools.handling.fast_math import calculate_scattering_factor_cctbx,get_resolution
import numpy as np
from matplotlib import pyplot as plt    

pdb_name = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/temp.pdb'

df = pdb_tools.load_pdb_as_pd('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Alvra_BT_01-2025_refine_61.pdb')

df_test = df.iloc[:4]


df_test['anisou_flag'] = False
tempfactors = []
effectives = []
corrs = []

for tempfactor in np.logspace(-1,3,100):
    corr_temp = []
    ef = []

    df_test.loc[:,'tempfactor'] = tempfactor

    pdb_tools.write_file(df_test, pdb_name)


    F,hkl = calculate_scattering_factor_cctbx(pdb_name,d_min=2.0)
    F_cctbx = np.abs(F)

    cell = [14.97,18.85,18.89,89.37,84.94,67.84]
    spacegroup = 'P1'
    df = df_test.copy()
    for effective in np.logspace(-1,3,100):
        df.loc[:,'tempfactor'] = effective
        f = ds.direct_summation(df,hkl,cell)
        f_ds = np.abs(f)
        corr_temp.append(np.corrcoef(F_cctbx,f_ds)[0,1])
        corrs.append(np.corrcoef(F_cctbx,f_ds)[0,1])
        ef.append(effective)
    arg = np.argmax(np.array(corr_temp))
    ef_best = ef[arg]
    print(tempfactor,'Best effective:',ef_best)
    tempfactors.append(tempfactor)
    effectives.append(ef_best)

plt.plot(tempfactors,effectives)
plt.xscale('log')
plt.yscale('log')
plt.savefig('tempfactor_effective.png')