#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
#SBATCH --job-name=compare_scalers
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler_analytical/compare_scalers.out
#SBATCH -c 16
#SBATCH --mem=64G

"""
Compare different scaling approaches to understand why analytical scaler isn't reaching R~0.17
"""

import torch
import sys

sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.solvent import SolventModel
from torchref.scaler_analytical import AnalyticalScaler
from torchref.scaler import Scaler
from torchref.math_functions.math_torch import rfactor

print("="*80)
print("COMPARING SCALING APPROACHES")
print("="*80)

# Load data
pdb_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.pdb'
mtz_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/tubulin/dark.mtz'

print("\nLoading model and data...")
model = ModelFT(max_res=1.5, verbose=0)
model.load_pdb_from_file(pdb_path)

data = ReflectionData(verbose=0)
data.load_from_mtz(mtz_path)
data = data.filter_by_resolution(d_min=1.5, d_max=50.0)

print(f"✓ Loaded {len(data.hkl)} reflections")

# Get structure factors
F_calc = model.get_structure_factor(data.hkl, recalc=False)
F_obs = data.F

# Create solvent model
solvent = SolventModel(model, verbose=0)
F_mask = solvent.get_rec_solvent(data.hkl)

print("\n" + "="*80)
print("SCENARIO 1: No scaling, no solvent")
print("="*80)
F_1 = F_calc
F_1_amp = torch.abs(F_1)
simple_scale_1 = F_obs.sum() / F_1_amp.sum()
F_1_scaled = F_1_amp * simple_scale_1
r_1 = rfactor(F_obs, F_1_scaled)
print(f"R-factor: {r_1:.4f}")

print("\n" + "="*80)
print("SCENARIO 2: Binned amplitude scaling only (like old Scaler)")
print("="*80)
old_scaler = Scaler(F_calc, data, nbins=20, verbose=0)
F_2 = old_scaler.forward(F_calc)
F_2_amp = torch.abs(F_2)
r_2 = rfactor(F_obs, F_2_amp)
print(f"R-factor: {r_2:.4f}")
print(f"Mean scale applied: {(F_2_amp.mean()/torch.abs(F_calc).mean()):.4f}")

print("\n" + "="*80)
print("SCENARIO 3: Simple uniform bulk-solvent + overall scale")
print("="*80)
print("Testing different uniform k_mask values:")
best_r = 1.0
best_k = 0.0
for k_mask in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
    F_3 = F_calc + k_mask * F_mask
    F_3_amp = torch.abs(F_3)
    simple_scale = F_obs.sum() / F_3_amp.sum()
    F_3_scaled = F_3_amp * simple_scale
    r_3 = rfactor(F_obs, F_3_scaled)
    marker = " ← best" if r_3 < best_r else ""
    print(f"  k_mask={k_mask:.2f}: R={r_3:.4f}{marker}")
    if r_3 < best_r:
        best_r = r_3
        best_k = k_mask

print(f"\nBest: k_mask={best_k:.2f}, R={best_r:.4f}")

print("\n" + "="*80)
print("SCENARIO 4: Uniform bulk-solvent + binned amplitude scaling")
print("="*80)
print("Using best k_mask from above, then applying binned scaling:")
F_with_solvent = F_calc + best_k * F_mask
old_scaler_4 = Scaler(F_with_solvent, data, nbins=20, verbose=0)
F_4 = old_scaler_4.forward(F_with_solvent)
F_4_amp = torch.abs(F_4)
r_4 = rfactor(F_obs, F_4_amp)
print(f"R-factor: {r_4:.4f}")

print("\n" + "="*80)
print("SCENARIO 5: Analytical scaler (current implementation)")
print("="*80)
analytical_scaler = AnalyticalScaler(
    model_ft=model,
    reflection_data=data,
    solvent_model=solvent,
    n_bins=20,
    verbose=0
)
F_5 = analytical_scaler.forward(F_calc)
F_5_amp = torch.abs(F_5)
r_5 = rfactor(F_obs, F_5_amp)
print(f"R-factor: {r_5:.4f}")

# Check which bins have significant k_mask
nonzero_bins = [(i+1, analytical_scaler.bin_info['kmask_values'][i]) 
                for i in range(len(analytical_scaler.bin_info['kmask_values'])) 
                if analytical_scaler.bin_info['kmask_values'][i] > 0.01]
print(f"Bins with k_mask > 0.01: {len(nonzero_bins)}/20")
if nonzero_bins:
    print("  ", nonzero_bins[:10])  # Show first 10

print("\n" + "="*80)
print("SCENARIO 6: Resolution-dependent bulk-solvent + binned scaling")
print("="*80)
print("Apply simple resolution-dependent k_mask, then binned scaling:")

# Get bins
from torchref.math_functions.math_torch import get_scattering_vectors
s = get_scattering_vectors(data.hkl, data.cell)
s_mag = torch.sqrt(torch.sum(s**2, dim=1))
bins, n_bins = data.get_bins(20)

# Create simple resolution-dependent k_mask (decreases with resolution)
# Higher resolution (high s) → lower k_mask
s_min = s_mag.min()
s_max = s_mag.max()
s_normalized = (s_mag - s_min) / (s_max - s_min)  # 0 to 1

# Try different functions
for profile_name, k_mask_func in [
    ("Constant 0.3", lambda s: 0.3),
    ("Linear decay", lambda s: 0.4 * (1 - s)),
    ("Quadratic decay", lambda s: 0.5 * (1 - s)**2),
    ("Low-res only", lambda s: 0.4 if s < 0.3 else 0.0),
]:
    k_mask_per_ref = torch.tensor([k_mask_func(s.item()) for s in s_normalized])
    F_6 = F_calc + k_mask_per_ref.unsqueeze(-1) * F_mask
    
    # Apply binned scaling
    old_scaler_6 = Scaler(F_6, data, nbins=20, verbose=0)
    F_6_scaled = old_scaler_6.forward(F_6)
    F_6_amp = torch.abs(F_6_scaled)
    r_6 = rfactor(F_obs, F_6_amp)
    print(f"  {profile_name:20s}: R={r_6:.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"1. No scaling, no solvent:                    R = {r_1:.4f}")
print(f"2. Binned scaling only:                       R = {r_2:.4f}")
print(f"3. Uniform bulk-solvent + simple scale:       R = {best_r:.4f} (k_mask={best_k:.2f})")
print(f"4. Uniform bulk-solvent + binned scaling:     R = {r_4:.4f}")
print(f"5. Analytical scaler (current):               R = {r_5:.4f}")
print("="*80)

if r_5 > 0.17:
    print(f"\n⚠ Target R~0.17 not achieved. Current: {r_5:.4f}")
    print(f"  Best alternative was scenario 4: R={r_4:.4f}")
