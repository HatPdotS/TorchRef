#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/Data/new_french_wilson.out

from torchref.model_ft import ModelFT
from torchref.scaler import Scaler
from torchref.Data import ReflectionData
import torch
import matplotlib.pyplot as plt
import os
from torchref.french_wilson import FrenchWilson
import reciprocalspaceship as rs

data = ReflectionData(verbose=2).load_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz')


sigmas = data.F_sigma
print(sigmas.shape)

plt.hist(sigmas.cpu().numpy(), bins=100)
plt.xlabel('Sigma values from MTZ')
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/Data'
plt.savefig(f'{outdir}/hist_sigma_values_data.png', dpi=300)

mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz'


data = rs.read_mtz(mtz)


cell = torch.tensor([data.cell.a, data.cell.b, data.cell.c,
                     data.cell.alpha, data.cell.beta, data.cell.gamma], dtype=torch.float32)


spacegroup = 'P21'


hkl = torch.tensor(data.reset_index()[['H','K','L']].values.astype(int), dtype=torch.int32)

I = data['I-obs'].values.astype(float)
SigI = data['SIGI-obs'].values.astype(float)

I_torch = torch.tensor(I, dtype=torch.float32)
SigI_torch = torch.tensor(SigI, dtype=torch.float32)

FW = FrenchWilson(hkl, cell, spacegroup)
F, sigma_F = FW(I_torch, SigI_torch)

print(sigma_F.shape)


plt.hist(sigma_F.detach().cpu().numpy(), bins=100)
plt.xlabel('Sigma_F values from French-Wilson')
plt.savefig(f'{outdir}/hist_sigma_french_wilson.png', dpi=300)

