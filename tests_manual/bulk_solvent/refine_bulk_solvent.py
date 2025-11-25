#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 32
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/refine_bulk_solvent_long.out
#SBATCH -p day
#SBATCH -t 1-00:00:00

import torch
from torchref.base_refinement import Refinement
from torchref.solvent_new import SolventModel
import numpy as np
from collections import defaultdict
import pandas as pd
pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb'
mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz'
cif = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif'


instance = Refinement(mtz, pdb, cif=cif, verbose=1)

instance.write_out_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/refine_bulk_solvent_initial.mtz')

instance.scaler.nbins = 1


rwork, rtest = instance.get_rfactor()
print(f'Pre Rwork: {rwork}, Rtest: {rtest}')

log_ksol = torch.tensor(-1.1)
log_bsol = torch.tensor(4.0)

k_sol = torch.exp(log_ksol)
b_sol = torch.exp(log_bsol)
print('-------' * 10)
print(f'Starting bulk solvent refinement with log_k_sol: {log_ksol}, log_b_sol: {log_bsol}')
try: del instance.solvent
except: pass
d = 1.3
transition = 1.2        
radius = 0.9

print(f'Using dilation radius: {d} Å, transition: {transition} voxels, and radius: {radius} Å')
instance.solvent = SolventModel(instance.model,k_solvent=k_sol,b_solvent=b_sol,radius=radius,
                                transition=transition,verbose=0)
instance.scaler.calc_initial_scale()
instance.scaler.setup_anisotropy_correction()
rwork, rtest = instance.get_rfactor(recompute_fcalc=True)
print(f'Initial log_k_sol: {log_ksol}, log_b_sol: {log_bsol} => Rwork: {rwork}, Rtest: {rtest}')
for i in range(3):
    instance.refine_solvent(iter=20, lr=0.1)
    instance.refine_solvent(iter=20, lr=0.05)
    instance.refine_solvent(iter=20, lr=0.01)
    instance.refine_solvent(iter=20, lr=0.005)
    instance.refine_solvent(iter=20, lr=0.001)

rwork, rtest = instance.get_rfactor(recompute_fcalc=True)
log_ksol_final = instance.solvent.log_k_solvent.item()
log_bsol_final = instance.solvent.log_b_solvent.item()
print(instance.scaler.U.detach().cpu().numpy())
print(f'After training with log_k_sol: {log_ksol}, log_b_sol: {log_bsol}, Final Rwork: {rwork}, Rtest: {rtest}')
print(f'Initial log_k_sol: {log_ksol}, log_b_sol: {log_bsol} => Final log_k_sol: {log_ksol_final}, log_b_sol: {log_bsol_final}')
print('-------' * 10)



instance.write_out_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/bulk_solvent/refine_bulk_solvent_final.mtz')