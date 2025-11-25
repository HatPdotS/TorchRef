import torchref.Model as Model
import torchref.refinement as refinement
import old.restraints_handler as restraints_handler
import torchref.Data as Data
import pandas as pd
import torch
import os
import pickle


input_model = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_onecopy.pdb'

mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_onecopy.mtz'

M = Model.model()
M.load_pdb_from_file(input_model)

hkl = Data.read_mtz(mtz_path)


ref = refinement.Refinement(hkl,model=M,
                            structure_factors_to_refine=['Pd'],use_parametrization=True,weigth_xray=1,weight_restraints=0.1)


res1 = list(M.residues.values())[0]

SF_new = torch.abs(res1.get_structure_factor(ref.hkl,ref.scattering_vectors,ref.s))
print(SF_new.shape)
SF_old = torch.abs(ref.get_structure_factor_for_residue(res1))

print("New SF:", SF_new)
print("Old SF:", SF_old)
print(torch.corrcoef(torch.stack([SF_new.flatten(), SF_old.flatten()]))[0, 1])
