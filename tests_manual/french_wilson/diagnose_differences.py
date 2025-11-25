#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u

"""
Diagnostic script to understand why our French-Wilson implementation
disagrees with Phenix for weak/negative reflections.
"""

import torch
import reciprocalspaceship as rs
import numpy as np
from torchref.french_wilson import FrenchWilson, estimate_mean_intensity_by_resolution
from torchref.math_functions import math_torch

# Load data
mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz'
data = rs.read_mtz(mtz)

cell = torch.tensor([data.cell.a, data.cell.b, data.cell.c,
                     data.cell.alpha, data.cell.beta, data.cell.gamma], dtype=torch.float32)

spacegroup = 'P21'
data = data.dropna()

hkl = torch.tensor(data.reset_index()[['H','K','L']].values.astype(int), dtype=torch.int32)
I = torch.tensor(data['I-obs'].values.astype(float), dtype=torch.float32)
SigI = torch.tensor(data['SIGI-obs'].values.astype(float), dtype=torch.float32)
F_phenix = data['F-obs-filtered'].values.astype(float)

# Get d-spacings
d_spacings = math_torch.get_d_spacing(hkl, cell)

# Estimate mean intensity
mean_I = estimate_mean_intensity_by_resolution(I, d_spacings, n_bins=60, min_per_bin=40)

# Calculate h parameter for acentric reflections
h = (I / SigI) - (SigI / mean_I)
i_over_sig = I / SigI

# Run our French-Wilson
FW = FrenchWilson(hkl, cell, spacegroup, verbose=0)
F_ours, sigma_F_ours = FW(I, SigI)
F_ours = F_ours.detach().cpu().numpy()

# Find problematic reflections
# Focus on weak/negative reflections where agreement is poor
weak_mask = (i_over_sig < 3.0).numpy()

# Calculate absolute and relative errors
abs_error = np.abs(F_ours - F_phenix)
rel_error = abs_error / np.clip(F_phenix, 1e-6, None)

# Sort by absolute error (worst first)
sorted_idx = np.argsort(abs_error)[::-1]

print("=" * 100)
print("WORST DISAGREEMENTS (sorted by absolute error)")
print("=" * 100)
print()
print(f"{'Rank':>5} {'H':>4} {'K':>4} {'L':>4} {'I':>10} {'SigI':>10} {'<I>':>10} {'h':>8} {'I/sig':>8} "
      f"{'F_ours':>10} {'F_phenix':>10} {'Abs_Err':>10} {'Rel_Err':>10}")
print("-" * 130)

for rank, idx in enumerate(sorted_idx[:30]):  # Top 30 worst
    h_val = h[idx].item()
    mean_I_val = mean_I[idx].item()
    h_idx, k_idx, l_idx = hkl[idx].tolist()
    I_val = I[idx].item()
    SigI_val = SigI[idx].item()
    I_sig_val = i_over_sig[idx].item()
    F_o = F_ours[idx]
    F_p = F_phenix[idx]
    abs_err = abs_error[idx]
    rel_err = rel_error[idx]
    
    print(f"{rank+1:>5} {h_idx:>4} {k_idx:>4} {l_idx:>4} {I_val:>10.2f} {SigI_val:>10.2f} {mean_I_val:>10.2f} "
          f"{h_val:>8.3f} {I_sig_val:>8.3f} {F_o:>10.3f} {F_p:>10.3f} {abs_err:>10.3f} {rel_err:>10.3f}")

print()
print("=" * 100)
print("STATISTICS BY I/sigma RANGE")
print("=" * 100)
print()

ranges = [
    ("Very negative", (None, -3.0)),
    ("Negative", (-3.0, 0.0)),
    ("Weak positive", (0.0, 1.0)),
    ("Medium weak", (1.0, 3.0)),
    ("Strong", (3.0, None))
]

for label, (low, high) in ranges:
    if low is None:
        mask = i_over_sig.numpy() < high
    elif high is None:
        mask = i_over_sig.numpy() >= low
    else:
        mask = (i_over_sig.numpy() >= low) & (i_over_sig.numpy() < high)
    
    if mask.sum() == 0:
        continue
    
    abs_err_range = abs_error[mask]
    rel_err_range = rel_error[mask]
    
    print(f"{label:20s}: n={mask.sum():6d}, "
          f"Mean abs err={abs_err_range.mean():8.4f}, "
          f"Median abs err={np.median(abs_err_range):8.4f}, "
          f"Mean rel err={rel_err_range.mean():8.4f}")

print()
print("=" * 100)
print("SAMPLE WEAK REFLECTIONS WITH DETAILED CALCULATION")
print("=" * 100)
print()

# Pick a few specific weak reflections to trace through the calculation
weak_indices = np.where(weak_mask)[0][:5]

for idx in weak_indices:
    h_val = h[idx].item()
    mean_I_val = mean_I[idx].item()
    h_idx, k_idx, l_idx = hkl[idx].tolist()
    I_val = I[idx].item()
    SigI_val = SigI[idx].item()
    I_sig_val = i_over_sig[idx].item()
    F_o = F_ours[idx]
    F_p = F_phenix[idx]
    
    print(f"Reflection ({h_idx}, {k_idx}, {l_idx}):")
    print(f"  I = {I_val:.3f}, σ_I = {SigI_val:.3f}, <I> = {mean_I_val:.3f}")
    print(f"  I/σ_I = {I_sig_val:.3f}")
    print(f"  h = (I/σ_I) - (σ_I/<I>) = {I_sig_val:.3f} - {SigI_val/mean_I_val:.3f} = {h_val:.3f}")
    
    # For weak reflections (h < 3), show lookup table calculation
    if h_val < 3.0:
        h_clamped = max(h_val, -4.0)
        point = 10.0 * (h_clamped + 4.0)
        pt_1 = int(point)
        pt_2 = pt_1 + 1
        delta = point - pt_1
        print(f"  Lookup: h_clamped={h_clamped:.3f}, point={point:.2f}, pt_1={pt_1}, delta={delta:.3f}")
        print(f"  zf × √σ_I = zf × √{SigI_val:.3f} = zf × {np.sqrt(SigI_val):.3f}")
        print(f"  Expected zf from F_ours: {F_o / np.sqrt(SigI_val):.3f}")
        print(f"  Expected zf from F_phenix: {F_p / np.sqrt(SigI_val):.3f}")
    
    print(f"  F_ours = {F_o:.3f}, F_phenix = {F_p:.3f}, Error = {abs(F_o - F_p):.3f}")
    print()
