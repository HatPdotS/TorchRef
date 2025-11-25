#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/functional_test.out



from torchref.Data import ReflectionData
from torchref.model_ft import ModelFT
from torchref.scaler import Scaler
from matplotlib import pyplot as plt
import torch
from torchref.solvent import SolventModel

def rfactor(fobs, fcalc):
    fobs_amp = torch.abs(fobs)
    fcalc_amp = torch.abs(fcalc)
    r = torch.sum(torch.abs(fobs_amp - fcalc_amp)) / torch.sum(fobs_amp)
    return r.item()


M = ModelFT(verbose=2,max_res=1.7).load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/dark.pdb')

sol = SolventModel(M)

sol.get_solvent_mask()

sol.save_solvent_mask('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/solvent_mask.ccp4')