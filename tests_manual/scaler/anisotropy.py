#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 32
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/anisotropy_test.out

from torchref.scaler import Scaler
from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.math_functions.math_torch import rfactor
import torch

pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb'
mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz'
cif = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'


data = ReflectionData(verbose=1).load_from_mtz(mtz)
model = ModelFT(verbose=1).load_pdb_from_file(pdb)

fcalc = model.get_structure_factor(data.get_hkl())

Fcalc = torch.abs(fcalc).detach()

hkl, fobs, sigma, rfree = data()

scaler = Scaler(Fcalc, data, nbins=50)

rwork = rfactor(fobs[rfree], scaler(Fcalc)[rfree])
rtest = rfactor(fobs[~rfree], scaler(Fcalc)[~rfree])

print(f"Initial R-factor (working): {rwork:.4f}")
print(f"Initial R-factor (test): {rtest:.4f}")

scaler.fit_anisotropy(Fcalc)

print('Anisotropy U parameters:', scaler.U.detach().cpu().numpy())

rwork = rfactor(fobs[rfree], scaler(Fcalc)[rfree])
rtest = rfactor(fobs[~rfree], scaler(Fcalc)[~rfree])
print(f"Refined R-factor (working): {rwork:.4f}")
print(f"Refined R-factor (test): {rtest:.4f}")