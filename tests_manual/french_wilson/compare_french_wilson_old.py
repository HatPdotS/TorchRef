#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/french_wilson/compare_french_wilson_old.out

from torchref.model_ft import ModelFT
from torchref.scaler import Scaler
from torchref.Data import ReflectionData
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from torchref.math_functions.math_torch import french_wilson_conversion
import torch
import reciprocalspaceship as rs

mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz'
from torchref.french_wilson import FrenchWilson

import torch
import torch.nn.functional as F

outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/french_wilson/old'
os.makedirs(outdir, exist_ok=True)

data = rs.read_mtz(mtz)

cell = torch.tensor([data.cell.a, data.cell.b, data.cell.c,
                     data.cell.alpha, data.cell.beta, data.cell.gamma], dtype=torch.float32)

spacegroup = 'P21'

data = data.dropna()

hkl = torch.tensor(data.reset_index()[['H','K','L']].values.astype(int), dtype=torch.int32)

I = data['I-obs'].values.astype(float)
SigI = data['SIGI-obs'].values.astype(float)

I_torch = torch.tensor(I, dtype=torch.float32)
SigI_torch = torch.tensor(SigI, dtype=torch.float32)



F, sigma_F = french_wilson_conversion(I_torch, SigI_torch)

F = F.detach().cpu().numpy()

f = data['F-obs-filtered'].values.astype(float)


plt.plot(F, f, 'o', alpha=0.5)
plt.xlabel('Fobs my french wilson (multicopy_refinement)')
plt.ylabel('Fobs (phenix)')
plt.savefig(f'{outdir}/compare_french_wilson.png', dpi=300)
plt.close()

ideal_scale = np.mean(np.log(f) - np.log(F))

F_scaled = F * np.exp(ideal_scale)

plt.plot(F_scaled, f, 'o', alpha=0.5)
plt.xlabel('Fobs my french wilson scaled (multicopy_refinement)')
plt.ylabel('Fobs (phenix)')
plt.savefig(f'{outdir}/compare_french_wilson_scaled.png', dpi=300)
print(f"Ideal scale factor (log space): {ideal_scale:.4f}")



rfactor = np.sum(np.abs(f - F_scaled)) / np.sum(np.abs(f))
print(f"R-factor after ideal scaling: {rfactor:.4f}")