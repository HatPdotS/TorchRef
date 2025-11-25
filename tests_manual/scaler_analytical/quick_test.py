#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Quick test: Does adding bulk-solvent before binned scaling reach R~0.17?
"""

import torch
import sys

sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.solvent import SolventModel
from torchref.scaler import Scaler
from torchref.math_functions.math_torch import rfactor

# Load data
pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.pdb'
mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz'

model = ModelFT(max_res=1.5, verbose=0)
model.load_pdb_from_file(pdb_path)

data = ReflectionData(verbose=0)
data.load_from_mtz(mtz_path)
data = data.filter_by_resolution(d_min=1.5, d_max=50.0)

F_calc = model.get_structure_factor(data.hkl, recalc=False)
F_obs = data.F

solvent = SolventModel(model, verbose=0)
F_mask = solvent.get_rec_solvent(data.hkl)

print("Testing: F_calc only (no solvent)")
scaler_no_solvent = Scaler(F_calc, data, nbins=20, verbose=0)
F_no_solvent = scaler_no_solvent.forward(F_calc)
r_no_solvent = rfactor(F_obs, torch.abs(F_no_solvent))
print(f"  R = {r_no_solvent:.4f}")

print("\nTesting: F_calc + k_mask*F_mask, then binned scaling")
for k_mask in [0.1, 0.2, 0.3, 0.4]:
    F_with_solvent = F_calc + k_mask * F_mask
    scaler_with_solvent = Scaler(F_with_solvent, data, nbins=20, verbose=0)
    F_scaled = scaler_with_solvent.forward(F_with_solvent)
    r = rfactor(F_obs, torch.abs(F_scaled))
    print(f"  k_mask={k_mask:.1f}: R = {r:.4f}")
    del scaler_with_solvent  # Free memory

print(f"\nTarget: R ~ 0.17")
