from torchref.math_functions.math_torch import french_wilson_conversion
import reciprocalspaceship as rs
import torch
import numpy as np
import matplotlib.pyplot as plt

mtzin = rs.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz')
mtzin.dropna(inplace=True)

Fobs = torch.tensor(mtzin['F-obs-filtered'].values.astype(np.float32), dtype=torch.float32)
I_obs = torch.tensor(mtzin['I-obs'].values.astype(np.float32), dtype=torch.float32)
Sigma_Iobs = torch.tensor(mtzin['SIGI-obs'].values.astype(np.float32), dtype=torch.float32)
Fcalc = torch.tensor(mtzin['F-model'].values.astype(np.float32), dtype=torch.float32)

Fobs_fw, sigma_fw = french_wilson_conversion(I_obs, Sigma_Iobs)

# Don't clamp - let's see the actual values
Fobs_np = Fobs.detach().cpu().numpy()
Fobs_fw_np = Fobs_fw.detach().cpu().numpy()
Fcalc_np = Fcalc.detach().cpu().numpy()
I_obs_np = I_obs.detach().cpu().numpy()
Sigma_Iobs_np = Sigma_Iobs.detach().cpu().numpy()

# Calculate log ratios
# For Fcalc vs Fobs (reference)
mask_positive = (Fobs_np > 1e-6) & (Fcalc_np > 1e-6)
log_ratio_ref = np.log(Fcalc_np[mask_positive]) - np.log(Fobs_np[mask_positive])

# For Fcalc vs Fobs_FW (our implementation)
mask_positive_fw = (Fobs_fw_np > 1e-6) & (Fcalc_np > 1e-6)
log_ratio_fw = np.log(Fcalc_np[mask_positive_fw]) - np.log(Fobs_fw_np[mask_positive_fw])

# Create comprehensive plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Log ratio histogram - Reference
axes[0, 0].hist(log_ratio_ref, bins=100, density=True, alpha=0.6, color='blue', range=(-3, 3))
axes[0, 0].set_xlabel('log(Fcalc) - log(Fobs_reference)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title(f'Reference: log(Fcalc/Fobs)\nMean={np.mean(log_ratio_ref):.3f}, Std={np.std(log_ratio_ref):.3f}')
axes[0, 0].axvline(0, color='r', linestyle='--', alpha=0.5)

# Plot 2: Log ratio histogram - French-Wilson
axes[0, 1].hist(log_ratio_fw, bins=100, density=True, alpha=0.6, color='green', range=(-3, 3))
axes[0, 1].set_xlabel('log(Fcalc) - log(Fobs_FW)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title(f'French-Wilson: log(Fcalc/Fobs_FW)\nMean={np.mean(log_ratio_fw):.3f}, Std={np.std(log_ratio_fw):.3f}')
axes[0, 1].axvline(0, color='r', linestyle='--', alpha=0.5)

# Plot 3: Overlay comparison
axes[0, 2].hist(log_ratio_ref, bins=100, density=True, alpha=0.5, color='blue', range=(-3, 3), label='Reference')
axes[0, 2].hist(log_ratio_fw, bins=100, density=True, alpha=0.5, color='green', range=(-3, 3), label='French-Wilson')
axes[0, 2].set_xlabel('log(Fcalc/Fobs)')
axes[0, 2].set_ylabel('Density')
axes[0, 2].set_title('Comparison')
axes[0, 2].legend()
axes[0, 2].axvline(0, color='r', linestyle='--', alpha=0.5)

# Plot 4: Fobs_FW vs Fobs scatter
axes[1, 0].scatter(Fobs_np, Fobs_fw_np, alpha=0.1, s=1, c='blue')
axes[1, 0].plot([0, np.max(Fobs_np)], [0, np.max(Fobs_np)], 'r--', alpha=0.5, label='y=x')
axes[1, 0].set_xlabel('Fobs (reference)')
axes[1, 0].set_ylabel('Fobs (French-Wilson)')
axes[1, 0].set_title('F values comparison')
axes[1, 0].legend()

# Plot 5: Distribution by I/sigma category
I_over_sigma = I_obs_np / Sigma_Iobs_np
categories = [
    (I_over_sigma < -1, 'Very negative (I/σ < -1)'),
    ((I_over_sigma >= -1) & (I_over_sigma < 0), 'Slightly negative (-1 ≤ I/σ < 0)'),
    ((I_over_sigma >= 0) & (I_over_sigma < 1), 'Weak positive (0 ≤ I/σ < 1)'),
    ((I_over_sigma >= 1) & (I_over_sigma < 3), 'Medium (1 ≤ I/σ < 3)'),
    (I_over_sigma >= 3, 'Strong (I/σ ≥ 3)')
]

for mask, label in categories:
    if mask.sum() > 0:
        axes[1, 1].hist(Fobs_fw_np[mask], bins=50, alpha=0.5, label=f'{label} (n={mask.sum()})', range=(0, 200))

axes[1, 1].set_xlabel('Fobs (French-Wilson)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('F distribution by I/σ category')
axes[1, 1].legend(fontsize=8)
axes[1, 1].set_yscale('log')

# Plot 6: Statistics summary
axes[1, 2].axis('off')
stats_text = f"""
STATISTICS SUMMARY

Reference (Fobs):
  Mean: {np.mean(Fobs_np):.2f}
  Std:  {np.std(Fobs_np):.2f}
  Min:  {np.min(Fobs_np):.2f}
  Max:  {np.max(Fobs_np):.2f}

French-Wilson (Fobs_FW):
  Mean: {np.mean(Fobs_fw_np):.2f}
  Std:  {np.std(Fobs_fw_np):.2f}
  Min:  {np.min(Fobs_fw_np):.2f}
  Max:  {np.max(Fobs_fw_np):.2f}

Intensity statistics:
  Negative I: {(I_obs_np < 0).sum()} ({100*(I_obs_np < 0).sum()/len(I_obs_np):.1f}%)
  I/σ < -3: {(I_over_sigma < -3).sum()}
  I/σ < 0: {(I_over_sigma < 0).sum()}
  I/σ < 3: {(I_over_sigma < 3).sum()}

Log ratio (Fcalc vs Fobs):
  Reference:     μ={np.mean(log_ratio_ref):.3f}, σ={np.std(log_ratio_ref):.3f}
  French-Wilson: μ={np.mean(log_ratio_fw):.3f}, σ={np.std(log_ratio_fw):.3f}

Correlation:
  Fobs vs Fobs_FW: {np.corrcoef(Fobs_np, Fobs_fw_np)[0,1]:.4f}
"""
axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, family='monospace', verticalalignment='center')

plt.tight_layout()
plt.savefig('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/io/detailed_french_wilson_analysis.png', dpi=150)
print("Detailed analysis plot saved!")

print(stats_text)

# Additional diagnostic: show some examples
print("\n" + "="*60)
print("Sample of very negative intensities and their F values:")
print("="*60)
very_neg = I_over_sigma < -1
if very_neg.sum() > 0:
    indices = np.where(very_neg)[0][:10]
    for idx in indices:
        print(f"I={I_obs_np[idx]:8.2f}, σ={Sigma_Iobs_np[idx]:8.2f}, I/σ={I_over_sigma[idx]:6.2f}")
        print(f"  Fobs_ref={Fobs_np[idx]:8.2f}, Fobs_FW={Fobs_fw_np[idx]:8.2f}, Fcalc={Fcalc_np[idx]:8.2f}")
        print()
