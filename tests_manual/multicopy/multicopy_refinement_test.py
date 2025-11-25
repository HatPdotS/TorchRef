#!/das/work/units/LBR-FEL/p17490/CONDA/cctbx_peter/bin/python -u 

from torchref import Data
from torchref import difference_refinement
import old.restraints_handler as restraints_handler
from torchref.Model import projected_residue
import os
import torchref.Model as Model

cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'

restraints = restraints_handler.restraints(cif_path)

outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/multicopy'

import pickle

with open('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/multicopy/one_copy_refined.pickle', 'rb') as f:
    M1 = pickle.load(f)

with open('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/multicopy/one_copy_refined.pickle', 'rb') as f:
    M2 = pickle.load(f)

two_comp_mtz = Data.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_multicopy.mtz')
one_comp_mtz = Data.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_onecopy.mtz')
print(one_comp_mtz)

ref = difference_refinement.Difference_refinement(Fobs_dark=one_comp_mtz,Fobs_light=two_comp_mtz,model_dark=M1,
                                                  model_light=M2,
                                                  restraints=restraints,weight_restraints=0.1,
                                                  alpha_start=0.1,refine_alpha=True)





print(ref.get_rfactor())
ref.refine(n_iter=1000,lr=0.02)
print(ref.get_rfactor())




print(ref.alpha)

ref.model_light.write_pdb(outdir + '/light.pdb')
ref.write_mtz(outdir+ '/light.mtz')
