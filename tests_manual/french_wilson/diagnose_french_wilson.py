#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python
"""
Diagnostic script to compare French-Wilson implementations.
Identifies where our implementation differs from Phenix.
"""

import torch
import numpy as np
import reciprocalspaceship as rs
import matplotlib.pyplot as plt
from torchref.french_wilson import FrenchWilsonModule
import sys

mtz = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/scaler/dark.mtz'
outdir = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests_manual/french_wilson'

# Load data
data = rs.read_mtz(mtz)
cell = [data.cell.a, data.cell.b, data.cell.c,
        data.cell.alpha, data.cell.beta, data.cell.gamma]
spacegroup = 'P21'
data = data.dropna()

# Get HKL, I, sigI
hkl = torch.tensor(data.reset_index()[['H','K','L']].values.astype(int), dtype=torch.int32)
I = torch.tensor(data['I-obs'].values.astype(float), dtype=torch.float32)
SigI = torch.tensor(data['SIGI-obs'].values.astype(float), dtype=torch.float32)

# Our French-Wilson
fw_module = FrenchWilsonModule(hkl, cell, spacegroup, verbose=0)
F_ours, sigma_F_ours = fw_module(I, SigI)

# Phenix French-Wilson
F_phenix = torch.tensor(data['F-obs-filtered'].values.astype(float), dtype=torch.float32)

# Convert to numpy
F_ours_np = F_ours.detach().cpu().numpy()
F_phenix_np = F_phenix.detach().cpu().numpy()
I_np = I.cpu().numpy()
SigI_np = SigI.cpu().numpy()

# Calculate I/SigI
I_over_SigI = I_np / SigI_np

print("=" * 80)
print("French-Wilson Comparison Diagnostics")
print("=" * 80)

# Overall statistics
print(f"\nTotal reflections: {len(I_np)}")
print(f"Negative intensities: {(I_np < 0).sum()} ({100*(I_np < 0).sum()/len(I_np):.1f}%)")
print(f"Weak reflections (I/σI < 0): {(I_over_SigI < 0).sum()}")
print(f"Very weak (I/σI < -3.7): {(I_over_SigI < -3.7).sum()}")

# Check where our F=0 but Phenix has values
zero_ours = (F_ours_np == 0)
nonzero_phenix = (F_phenix_np > 0)
problematic = zero_ours & nonzero_phenix

print(f"\nOur F=0 but Phenix F>0: {problematic.sum()} reflections")

if problematic.sum() > 0:
    print("\nSample of problematic reflections:")
    prob_indices = np.where(problematic)[0][:20]
    print(f"{'I':>12} {'σI':>12} {'I/σI':>12} {'F_ours':>12} {'F_phenix':>12}")
    print("-" * 70)
    for idx in prob_indices:
        print(f"{I_np[idx]:12.2f} {SigI_np[idx]:12.2f} {I_over_SigI[idx]:12.2f} "
              f"{F_ours_np[idx]:12.2f} {F_phenix_np[idx]:12.2f}")

# Analyze by I/SigI bins
print("\n" + "=" * 80)
print("Analysis by I/σI bins")
print("=" * 80)

bins = [(-np.inf, -4.0), (-4.0, -3.0), (-3.0, -2.0), (-2.0, -1.0), (-1.0, 0.0),
        (0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, np.inf)]

for bin_min, bin_max in bins:
    mask = (I_over_SigI >= bin_min) & (I_over_SigI < bin_max)
    if mask.sum() == 0:
        continue
    
    n = mask.sum()
    mean_F_ours = F_ours_np[mask].mean()
    mean_F_phenix = F_phenix_np[mask].mean()
    ratio = mean_F_ours / mean_F_phenix if mean_F_phenix > 0 else 0
    
    print(f"I/σI ∈ [{bin_min:6.1f}, {bin_max:6.1f}): N={n:5d}  "
          f"F_ours={mean_F_ours:8.2f}  F_phenix={mean_F_phenix:8.2f}  "
          f"Ratio={ratio:.3f}")

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Overall comparison
ax = axes[0, 0]
ax.scatter(F_phenix_np, F_ours_np, alpha=0.3, s=1)
ax.plot([0, F_phenix_np.max()], [0, F_phenix_np.max()], 'r--', label='y=x')
ax.set_xlabel('F (Phenix)')
ax.set_ylabel('F (Ours)')
ax.set_title('Overall Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Focus on weak reflections
ax = axes[0, 1]
weak_mask = (I_over_SigI < 3.0)
ax.scatter(F_phenix_np[weak_mask], F_ours_np[weak_mask], alpha=0.3, s=2)
ax.plot([0, F_phenix_np[weak_mask].max()], [0, F_phenix_np[weak_mask].max()], 'r--', label='y=x')
ax.set_xlabel('F (Phenix)')
ax.set_ylabel('F (Ours)')
ax.set_title('Weak Reflections (I/σI < 3)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Ratio vs I/SigI
ax = axes[1, 0]
valid_ratio = (F_ours_np > 0) & (F_phenix_np > 0)
ratio = F_ours_np[valid_ratio] / F_phenix_np[valid_ratio]
ax.scatter(I_over_SigI[valid_ratio], ratio, alpha=0.3, s=1)
ax.axhline(y=1.0, color='r', linestyle='--', label='Perfect agreement')
ax.set_xlabel('I/σI')
ax.set_ylabel('F_ours / F_phenix')
ax.set_title('Ratio vs Signal Strength')
ax.set_ylim(0, 2)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Histogram of differences
ax = axes[1, 1]
diff = F_ours_np - F_phenix_np
ax.hist(diff, bins=100, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='r', linestyle='--', label='Zero difference')
ax.set_xlabel('F_ours - F_phenix')
ax.set_ylabel('Count')
ax.set_title(f'Difference Distribution (mean={diff.mean():.2f})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{outdir}/french_wilson_diagnostics.png', dpi=300)
print(f"\nDiagnostic plots saved to {outdir}/french_wilson_diagnostics.png")

# Compute R-factor
valid = (F_phenix_np > 0)
rfactor = np.sum(np.abs(F_phenix_np[valid] - F_ours_np[valid])) / np.sum(F_phenix_np[valid])
print(f"\nR-factor (without scaling): {rfactor:.4f}")

# Find optimal scale
scale = np.sum(F_phenix_np[valid] * F_ours_np[valid]) / np.sum(F_ours_np[valid]**2)
print(f"Optimal scale factor: {scale:.4f}")
rfactor_scaled = np.sum(np.abs(F_phenix_np[valid] - scale*F_ours_np[valid])) / np.sum(F_phenix_np[valid])
print(f"R-factor (with optimal scaling): {rfactor_scaled:.4f}")

print("\n" + "=" * 80)
