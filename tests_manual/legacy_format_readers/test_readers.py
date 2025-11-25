

mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/validation_on_different_samples/Br/BR_LCLS_refine_8.mtz'
cif = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/1A7V/1A7V-sf.cif'
from torchref.io.legacy_format_readers import MTZ,PDB
from torchref.Data import ReflectionData
import numpy as np


data = MTZ(verbose=2).read(mtz)

data_dictionary, spacegroup, cell = data()  # type: ignore

I = data_dictionary['I']
SigI = data_dictionary['SIGI']
print("Data keys:", data_dictionary)
print('Invalid Is:', np.isnan(I).sum())
print('Invalid SigIs:', np.isnan(SigI).sum())
print("Spacegroup:", spacegroup)
print("Cell parameters:", cell)

pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/validation_on_different_samples/Br/BR_LCLS_refine_8.pdb'

data_pdb = PDB(verbose=2).read(pdb)



df, cell, spacegroup = data_pdb()  # type: ignore


print(df)
print("Cell parameters:", cell)
print("Spacegroup:", spacegroup)


data = ReflectionData().load_mtz(mtz)

hkl, F, F_sigma, rfree = data()
print("HKL shape:", hkl.shape, "F shape:", F.shape, "F_sigma shape:", F_sigma.shape, "R-free flags shape:", rfree.shape)

data2 = ReflectionData().load_cif(cif)

hkl2, F2, F_sigma2, rfree2 = data2()
print("HKL shape:", hkl2.shape, "F shape:", F2.shape, "F_sigma shape:", F_sigma2.shape, "R-free flags shape:", rfree2.shape)    
