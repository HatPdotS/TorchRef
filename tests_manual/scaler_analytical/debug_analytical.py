#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
#SBATCH --job-name=debug_analytical
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/scaler_analytical/debug_analytical.out
#SBATCH -c 16

"""
Debug why analytical scaler performs poorly compared to old scaler.
"""

import torch
import numpy as np
import sys

sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.solvent import SolventModel
from torchref.scaler_analytical import AnalyticalScaler
from torchref.scaler import Scaler
from torchref.math_functions.math_torch import rfactor

print("="*80)
print("DEBUGGING ANALYTICAL SCALER")
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

# Get basic structure factors
F_calc = model.get_structure_factor(data.hkl, recalc=False)
F_obs = data.F

print(f"\n{'='*80}")
print("1. CHECKING F_CALC AND F_MASK SCALES")
print(f"{'='*80}")

solvent = SolventModel(model, verbose=0)
F_mask = solvent.get_rec_solvent(data.hkl)

F_calc_amp = torch.abs(F_calc)
F_mask_amp = torch.abs(F_mask)

print(f"\nF_calc amplitudes:")
print(f"  Mean: {F_calc_amp.mean():.2f}")
print(f"  Std:  {F_calc_amp.std():.2f}")
print(f"  Max:  {F_calc_amp.max():.2f}")

print(f"\nF_mask amplitudes:")
print(f"  Mean: {F_mask_amp.mean():.2f}")
print(f"  Std:  {F_mask_amp.std():.2f}")
print(f"  Max:  {F_mask_amp.max():.2f}")

print(f"\nF_obs amplitudes:")
print(f"  Mean: {F_obs.mean():.2f}")
print(f"  Std:  {F_obs.std():.2f}")
print(f"  Max:  {F_obs.max():.2f}")

print(f"\nRatio F_mask/F_calc: {(F_mask_amp.mean()/F_calc_amp.mean()):.4f}")

print(f"\n{'='*80}")
print("2. CHECKING ANALYTICAL SCALER INTERNALS")
print(f"{'='*80}")

# Create analytical scaler with detailed output
scaler = AnalyticalScaler(
    model_ft=model,
    reflection_data=data,
    solvent_model=solvent,
    n_bins=20,
    verbose=2
)

print(f"\n{'='*80}")
print("3. EXAMINING BIN-BY-BIN RESULTS")
print(f"{'='*80}")

# Get bin assignments
from torchref.math_functions.math_torch import get_scattering_vectors
s = get_scattering_vectors(data.hkl, data.cell)
s_mag = torch.sqrt(torch.sum(s**2, dim=1))
bins, n_bins = data.get_bins(20)

print(f"\nAnalyzing each bin in detail:")
print(f"\n{'Bin':>4} {'N_ref':>6} {'d(Å)':>6} {'k_mask':>10} {'K':>15} {'|F_calc|':>10} {'|F_mask|':>10} {'F_obs':>10}")
print("-"*95)

for bin_idx in range(20):
    mask = (bins == bin_idx)
    n_ref = mask.sum().item()
    
    if n_ref == 0:
        continue
    
    s_bin = s_mag[mask]
    d_mean = 1.0 / (2.0 * s_bin.mean().item())
    
    F_calc_bin_amp = F_calc_amp[mask].mean().item()
    F_mask_bin_amp = F_mask_amp[mask].mean().item()
    F_obs_bin = F_obs[mask].mean().item()
    
    k_mask = scaler.bin_info['kmask_values'][bin_idx]
    K = scaler.bin_info['K_values'][bin_idx]
    
    print(f"{bin_idx+1:4d} {n_ref:6d} {d_mean:6.2f} {k_mask:10.4f} {K:15.2e} {F_calc_bin_amp:10.2f} {F_mask_bin_amp:10.2f} {F_obs_bin:10.2f}")

print(f"\n{'='*80}")
print("4. CHECKING WHY K_MASK IS ZERO IN MOST BINS")
print(f"{'='*80}")

# Let's manually check one bin where k_mask=0
print("\nInvestigating bin 2 (k_mask=0)...")
mask = (bins == 1)  # bin index 1 (second bin)
F_calc_bin = F_calc[mask]
F_mask_bin = F_mask[mask]
F_obs_bin = F_obs[mask]

# Compute u, v, w
u_s = torch.abs(F_calc_bin) ** 2
v_s = torch.real(F_calc_bin * torch.conj(F_mask_bin))
w_s = torch.abs(F_mask_bin) ** 2
I_s = F_obs_bin ** 2

print(f"\nBin 2 statistics:")
print(f"  N reflections: {len(F_calc_bin)}")
print(f"  u_s (|F_calc|^2) - mean: {u_s.mean():.2e}, std: {u_s.std():.2e}")
print(f"  v_s (Re[F_calc*F_mask*]) - mean: {v_s.mean():.2e}, std: {v_s.std():.2e}")
print(f"  w_s (|F_mask|^2) - mean: {w_s.mean():.2e}, std: {w_s.std():.2e}")
print(f"  I_s (F_obs^2) - mean: {I_s.mean():.2e}, std: {I_s.std():.2e}")

# Compute sums
A_2 = torch.sum(u_s)
B_2 = torch.sum(2 * u_s * v_s)
C_2 = torch.sum(v_s ** 2 + u_s * w_s)
D_3 = torch.sum(v_s * w_s)
Y_2 = torch.sum(I_s)
Y_3 = torch.sum(I_s * v_s)
E_4 = torch.sum(w_s ** 2)
F_4 = torch.sum(2 * v_s * w_s)
G_4 = torch.sum(v_s ** 2)

print(f"\nCubic equation terms:")
print(f"  A_2: {A_2:.2e}")
print(f"  B_2: {B_2:.2e}")
print(f"  C_2: {C_2:.2e}")
print(f"  D_3: {D_3:.2e}")
print(f"  Y_2: {Y_2:.2e}")
print(f"  Y_3: {Y_3:.2e}")
print(f"  E_4: {E_4:.2e}")
print(f"  F_4: {F_4:.2e}")
print(f"  G_4: {G_4:.2e}")

# Compute cubic coefficients
denom = Y_2 * E_4 - Y_3 * F_4 / 2
print(f"\nDenominator: {denom:.2e}")

if torch.abs(denom) > 1e-10:
    a = (Y_2 * F_4 - 2 * Y_3 * E_4) / (2 * denom)
    b = (Y_2 * G_4 + Y_2 * C_2 - Y_3 * B_2 - Y_3 * D_3) / denom
    c = (Y_2 * D_3 - Y_3 * C_2) / denom
    
    print(f"\nCubic coefficients (k^3 + a*k^2 + b*k + c = 0):")
    print(f"  a: {a:.2e}")
    print(f"  b: {b:.2e}")
    print(f"  c: {c:.2e}")
    
    # Solve cubic
    coeffs = np.array([1.0, a.item(), b.item(), c.item()])
    roots = np.roots(coeffs)
    
    print(f"\nRoots of cubic equation:")
    for i, root in enumerate(roots):
        is_real = np.isreal(root)
        is_positive = np.real(root) > 0 if is_real else False
        print(f"  Root {i+1}: {root} (real: {is_real}, positive: {is_positive})")
    
    # Check positive real roots
    positive_real_roots = [np.real(r) for r in roots if np.isreal(r) and np.real(r) > 0]
    print(f"\nPositive real roots: {len(positive_real_roots)}")
    
    if len(positive_real_roots) > 0:
        print("\nEvaluating loss for each positive root:")
        for kmask_candidate in positive_real_roots:
            K_candidate = (kmask_candidate**2 * C_2 + kmask_candidate * B_2 + A_2) / Y_2
            model_intensity = (u_s + 2 * kmask_candidate * v_s + kmask_candidate**2 * w_s)
            residual = model_intensity - K_candidate * I_s
            loss = torch.sum(residual ** 2).item()
            print(f"  k_mask={kmask_candidate:.4f}, K={K_candidate:.2e}, loss={loss:.2e}")
        
        # Also check k_mask=0
        K_zero = A_2 / Y_2
        model_intensity_zero = u_s
        residual_zero = model_intensity_zero - K_zero * I_s
        loss_zero = torch.sum(residual_zero ** 2).item()
        print(f"  k_mask=0.0000, K={K_zero:.2e}, loss={loss_zero:.2e} (fallback)")

print(f"\n{'='*80}")
print("5. COMPARING SCALERS ON SAME DATA")
print(f"{'='*80}")

# Test analytical scaler
F_analytical = scaler.forward(F_calc)
F_analytical_amp = torch.abs(F_analytical)
r_analytical = rfactor(F_obs, F_analytical_amp)

print(f"\nAnalytical scaler:")
print(f"  R-factor: {r_analytical:.4f}")
print(f"  F_model mean: {F_analytical_amp.mean():.2f}")
print(f"  F_model/F_obs ratio: {(F_analytical_amp.mean()/F_obs.mean()):.4f}")

# Test old scaler
old_scaler = Scaler(F_calc, data, nbins=20, verbose=0)
F_old = old_scaler.forward(F_calc)
F_old_amp = torch.abs(F_old)
r_old = rfactor(F_obs, F_old_amp)

print(f"\nOld scaler:")
print(f"  R-factor: {r_old:.4f}")
print(f"  F_model mean: {F_old_amp.mean():.2f}")
print(f"  F_model/F_obs ratio: {(F_old_amp.mean()/F_obs.mean()):.4f}")

# Check what scales the old scaler computed
print(f"\nOld scaler bin scales:")
scales = torch.exp(old_scaler.log_scale).detach()
for bin_idx in range(min(20, len(scales))):
    mask = (bins == bin_idx)
    if mask.sum() > 0:
        s_bin = s_mag[mask]
        d_mean = 1.0 / (2.0 * s_bin.mean().item())
        print(f"  Bin {bin_idx+1}: d={d_mean:.2f}Å, scale={scales[bin_idx]:.4f}")

print(f"\n{'='*80}")
print("6. TESTING SIMPLE BULK-SOLVENT SCALES")
print(f"{'='*80}")

# Try simple k_mask values manually
test_kmasks = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
print(f"\nTesting different k_mask values (uniform across all bins):")
print(f"{'k_mask':>8} {'R-factor':>10} {'Improvement':>12}")
print("-"*32)

for test_k in test_kmasks:
    # Apply simple uniform bulk-solvent
    F_test = F_calc + test_k * F_mask
    F_test_amp = torch.abs(F_test)
    
    # Scale to match F_obs
    simple_scale = F_obs.sum() / F_test_amp.sum()
    F_test_scaled_amp = F_test_amp * simple_scale
    
    r_test = rfactor(F_obs, F_test_scaled_amp)
    improvement = r_analytical - r_test
    marker = " ←" if r_test < r_analytical else ""
    print(f"{test_k:8.2f} {r_test:10.4f} {improvement:12.4f}{marker}")

print("\n" + "="*80)
print("DEBUGGING COMPLETE")
print("="*80)
