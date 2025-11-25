#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/refinement/initial_corr_2JBV.log

from torchref.base_refinement import Refinement
from torchref.Data import ReflectionData
from torchref.model_ft import ModelFT
from torchref.scaler import Scaler

import torch


pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/4L7Z/4L7Z_shaken.pdb'
mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/scientific_testing/data/4L7Z/4L7Z-sf.cif'
# cif = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'

data = ReflectionData(verbose=2).load_cif(mtz)
hkl, fobs, sigfobs, rfree_flags = data()

print('nan values in fobs and sigfobs:',torch.sum(torch.isnan(fobs)).item(), torch.sum(torch.isnan(sigfobs)).item())

model = ModelFT().load_pdb(pdb)

fcalc = model(hkl)


correlation = torch.corrcoef(torch.stack([fobs, torch.abs(fcalc)]))[0, 1].item()

bins, nbins = data.get_bins(n_bins=20)

Scale = Scaler(model, data, 20, verbose=2)
Scale.initialize()
fcalc = Scale(model(hkl))

rfactor = torch.abs(fobs - torch.abs(fcalc)).sum() / torch.abs(fobs).sum()

print(f'Initial correlation: {correlation}')
print(f'Initial R-factor: {rfactor}')


Ref = Refinement(mtz, pdb, verbose=2)

rwork, rfree = Ref.get_rfactor()

print('When using Refinement class:')
print(f'Initial Rwork: {rwork}, Rfree: {rfree}')

Ref.scaler.initialize()

rwork, rfree = Ref.get_rfactor()

f_calc = Ref.scaler(Ref.model(hkl))

reader = Ref.reflection_data.reader




print('When using Refinement class:')
print(f'Initial Rwork: {rwork}, Rfree: {rfree}')

Ref.scaler.fit_all_scales()

rwork, rfree = Ref.get_rfactor()

print('After fitting scales:')
print(f'Rwork: {rwork}, Rfree: {rfree}')


Ref._print_recursive_debug_summaries()