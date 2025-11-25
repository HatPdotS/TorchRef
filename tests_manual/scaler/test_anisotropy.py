#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 32
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/anisotropy_base.out


from torchref.model_ft import ModelFT
from torchref.scaler import Scaler
from torchref.Data import ReflectionData
import torch

data = ReflectionData(verbose=0).load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/dark.mtz')

M = ModelFT(verbose=0).load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/dark.pdb')

fcalc = M(data.get_hkl())

scaler = Scaler(fcalc, data, nbins=50,verbose=0)

scaler.setup_anisotropy_correction()

correction = scaler.anisotropy_correction()


print("Anisotropy correction factors:", correction.detach().cpu().numpy())

print('Anisotropy correction all equal 1:', torch.allclose(correction, torch.ones_like(correction)))