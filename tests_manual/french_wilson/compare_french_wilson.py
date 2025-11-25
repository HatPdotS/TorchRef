#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/french_wilson/compare_french_wilson_new.out


import torch
import matplotlib.pyplot as plt

import numpy as np
import torch
import reciprocalspaceship as rs

mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz'
from torchref.french_wilson import FrenchWilson

import torch
import torch.nn.functional as F
from time import time
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/french_wilson'

timestart = time()
data = rs.read_mtz(mtz)
print(f"Loaded MTZ in {time() - timestart:.2f} seconds")

cell = torch.tensor([data.cell.a, data.cell.b, data.cell.c,
                     data.cell.alpha, data.cell.beta, data.cell.gamma], dtype=torch.float32)


spacegroup = 'P21'

data = data.dropna()

hkl = torch.tensor(data.reset_index()[['H','K','L']].values.astype(int), dtype=torch.int32)

I = data['I-obs'].values.astype(float)
SigI = data['SIGI-obs'].values.astype(float)

I_torch = torch.tensor(I, dtype=torch.float32)
SigI_torch = torch.tensor(SigI, dtype=torch.float32)

timestart = time()

FW = FrenchWilson(hkl, cell, spacegroup)

print(f"Initialized French-Wilson in {time() - timestart:.2f} seconds")

timestart = time()

F, sigma_F = FW(I_torch, SigI_torch)

print(f"Computed French-Wilson in {time() - timestart:.2f} seconds")

F = F.detach().cpu().numpy()

f = data['F-obs-filtered'].values.astype(float)
sigma_f = data['SIGF-obs-filtered'].values.astype(float)


plt.plot(F, f, 'o', alpha=0.5)
plt.xlabel('Fobs my french wilson (multicopy_refinement)')
plt.ylabel('Fobs (phenix)')
plt.savefig(f'{outdir}/compare_french_wilson.png', dpi=300)
plt.close()

plt.plot(sigma_F.detach().cpu().numpy(), sigma_f, 'o', alpha=0.5)
plt.xlabel('SigF my french wilson (multicopy_refinement)')
plt.ylabel('SigF (phenix)')
plt.savefig(f'{outdir}/compare_french_wilson_sigma.png', dpi=300)
plt.close()


ideal_scale = np.mean(np.log(f) - np.log(F))

F_scaled = F * np.exp(ideal_scale)

plt.plot(F_scaled, f, 'o', alpha=0.5)
plt.xlabel('Fobs my french wilson scaled (multicopy_refinement)')
plt.ylabel('Fobs (phenix)')
plt.savefig(f'{outdir}/compare_french_wilson_scaled.png', dpi=300)
plt.close()
print(f"Ideal scale factor (log space): {ideal_scale:.4f}")



rfactor = np.sum(np.abs(f - F_scaled)) / np.sum(np.abs(f))
print(f"R-factor after ideal scaling: {rfactor:.4f}")


print(sigma_F)
print(sigma_f[sigma_f <= 0])


log_Fsigma = torch.log(sigma_F).detach().cpu().numpy()
log_Fsigma = (log_Fsigma - np.mean(log_Fsigma) ) / np.std(log_Fsigma)
print(log_Fsigma.shape)
plt.hist(log_Fsigma, bins=100)
plt.xlabel('log(SigF) my french wilson (multicopy_refinement)')
plt.savefig(f'{outdir}/hist_log_sigma_french_wilson.png', dpi=300)    
plt.close()


