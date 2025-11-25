#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 32
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/functional_test.out



from torchref.Data import ReflectionData
from torchref.model_ft import ModelFT
from torchref.scaler import Scaler
from matplotlib import pyplot as plt
import torch
from time import time, sleep

def rfactor(fobs, fcalc):
    fobs_amp = torch.abs(fobs)
    fcalc_amp = torch.abs(fcalc)
    r = torch.sum(torch.abs(fobs_amp - fcalc_amp)) / torch.sum(fobs_amp)
    return r.item()

data = ReflectionData(verbose=2).load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/dark.mtz')

M = ModelFT(verbose=2).load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/dark.pdb')

fcalc = M.get_structure_factor(data.get_hkl())
_,_,_,rfree = data()

print('Done with prep')
sleep(2)

print('Calculating initial scaler...')

scaler = Scaler(fcalc, data, nbins=50)

scaled_fcalc = scaler(fcalc)


_, fobs, _, _ = data()



print('unscaled total rfactor:', rfactor(fobs, fcalc), 'scaled total rfactor:', rfactor(fobs, scaled_fcalc))

print('unscaled rfactor work:',rfactor(fobs[rfree], fcalc[rfree]),'scaled rfactor work:', rfactor(fobs[rfree], scaled_fcalc[rfree]))
print('unscaled rfactor test:',rfactor(fobs[~rfree], fcalc[~rfree]),'scaled rfactor test:', rfactor(fobs[~rfree], scaled_fcalc[~rfree]))

plt.figure(figsize=(10,5))  
plt.subplot(1, 2, 1)
plt.title('Before Scaling')
plt.scatter(torch.abs(fcalc).detach().cpu().numpy(), fobs.cpu().numpy(), alpha=0.5)
plt.xlabel('Calculated |F|')
plt.ylabel('Observed |F|')
plt.grid()

plt.subplot(1, 2, 2)
plt.title('After Scaling')
plt.scatter(torch.abs(scaled_fcalc).detach().cpu().numpy(), fobs.cpu().numpy(), alpha=0.5)
plt.xlabel('Calculated |F| (Scaled)')
plt.ylabel('Observed |F|')
plt.grid()

plt.tight_layout()
plt.savefig('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler/scaling_comparison.png')