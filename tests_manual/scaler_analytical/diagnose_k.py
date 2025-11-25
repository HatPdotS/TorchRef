#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Diagnostic script to understand K values in analytical scaler.
"""

import torch
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.solvent import SolventModel

# Load data
pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.pdb'
mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz'

model = ModelFT(max_res=1.5, verbose=0)
model.load_pdb_from_file(pdb_path)

data = ReflectionData(verbose=0)
data.load_from_mtz(mtz_path)
data = data.filter_by_resolution(d_min=1.5, d_max=50.0)

# Get F_calc
F_calc = model.get_structure_factor(data.hkl, recalc=False)
F_calc_amp = torch.abs(F_calc)

# Get F_obs  
F_obs = data.F

# Compare scales
print(f"\nF_calc statistics:")
print(f"  Mean: {F_calc_amp.mean():.4f}")
print(f"  Std:  {F_calc_amp.std():.4f}")
print(f"  Max:  {F_calc_amp.max():.4f}")

print(f"\nF_obs statistics:")
print(f"  Mean: {F_obs.mean():.4f}")
print(f"  Std:  {F_obs.std():.4f}")
print(f"  Max:  {F_obs.max():.4f}")

# Compute simple scale
simple_scale = F_obs.sum() / F_calc_amp.sum()
print(f"\nSimple amplitude scale (sum F_obs / sum |F_calc|): {simple_scale:.4f}")

# What would K be if it represents intensity ratio?
I_calc = F_calc_amp ** 2
I_obs = F_obs ** 2
K_intensity = I_obs.sum() / I_calc.sum()
print(f"Intensity scale K (sum I_obs / sum I_calc): {K_intensity:.4f}")
print(f"  sqrt(K) = {torch.sqrt(K_intensity):.4f}")
print(f"  1/sqrt(K) = {1.0/torch.sqrt(K_intensity):.4f}")

# Now let's check what the analytical method should give
# For a single bin with all data
A_2 = torch.sum(F_calc_amp ** 2)
Y_2 = torch.sum(I_obs)
K_analytical_simple = A_2 / Y_2
print(f"\nAnalytical K (no solvent, A_2/Y_2): {K_analytical_simple:.4f}")
print(f"  This is: sum(|F_calc|^2) / sum(F_obs^2)")
print(f"  sqrt(K) = {torch.sqrt(K_analytical_simple):.4f}")
print(f"  1/sqrt(K) = {1.0/torch.sqrt(K_analytical_simple):.4f}")
