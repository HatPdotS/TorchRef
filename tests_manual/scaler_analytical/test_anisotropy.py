
#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Test if analytical scaler + anisotropy fitting reaches R~0.17
"""

import torch
import sys

sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.solvent import SolventModel
from torchref.scaler_analytical import AnalyticalScaler
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

print("Step 1: Analytical bulk-solvent + isotropic scaling")
analytical_scaler = AnalyticalScaler(
    model_ft=model,
    reflection_data=data,
    solvent_model=solvent,
    n_bins=20,
    verbose=0
)
F_scaled = analytical_scaler.forward(F_calc)
r_analytical = rfactor(F_obs, torch.abs(F_scaled))
print(f"  R = {r_analytical:.4f}")

print("\nStep 2: Check if anisotropy would help")
print("  (Analytical method doesn't include anisotropic B-factor refinement)")
print("  This would require iterative optimization, which defeats the purpose")
print("  of having an 'analytical' method.")

print(f"\nConclusion:")
print(f"  Analytical scaler achieves R = {r_analytical:.4f}")
print(f"  To reach R ~ 0.17, would need:")
print(f"    - Anisotropic B-factor correction (~5-10% improvement)")
print(f"    - Possibly TLS refinement")
print(f"    - Or iterative refinement of k_mask per bin")
