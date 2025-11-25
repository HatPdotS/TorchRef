from torchref.math_functions.math_torch import french_wilson_conversion
import reciprocalspaceship as rs
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

mtzin = rs.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz')
mtzin.dropna(inplace=True)

Fobs_ref = mtzin['F-obs-filtered'].values
I_obs = mtzin['I-obs'].values
Sigma_Iobs = mtzin['SIGI-obs'].values
Fcalc = mtzin['F-model'].values

Fobs_fw, _ = french_wilson_conversion(torch.tensor(I_obs, dtype=torch.float32), torch.tensor(Sigma_Iobs, dtype=torch.float32))
Fobs_fw = Fobs_fw.detach().cpu().numpy()

# Log ratios
mask_ref = (Fobs_ref > 0) & (Fcalc > 0)
mask_fw = (Fobs_fw > 0) & (Fcalc > 0)

log_ratio_ref = np.log(Fcalc[mask_ref]) - np.log(Fobs_ref[mask_ref])
log_ratio_fw = np.log(Fcalc[mask_fw]) - np.log(Fobs_fw[mask_fw])

# Create comparison plot
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Reference histogram
axes[0, 0].hist(log_ratio_ref, bins=100, density=True, alpha=0.6, color='blue', range=(-3, 3))
axes[0, 0].set_xlabel('log(Fcalc/Fobs)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title(f'Reference MTZ Fobs\nμ={np.mean(log_ratio_ref):.3f}, σ={np.std(log_ratio_ref):.3f}\nskew={stats.skew(log_ratio_ref):.2f}, kurt={stats.kurtosis(log_ratio_ref):.2f}')
axes[0, 0].axvline(0, color='r', linestyle='--', alpha=0.5)
# Add Gaussian fit
x = np.linspace(-3, 3, 100)
axes[0, 0].plot(x, stats.norm.pdf(x, np.mean(log_ratio_ref), np.std(log_ratio_ref)), 'r-', lw=2, alpha=0.6, label='Gaussian fit')
axes[0, 0].legend()

# Plot 2: French-Wilson histogram  
axes[0, 1].hist(log_ratio_fw, bins=100, density=True, alpha=0.6, color='green', range=(-3, 5))
axes[0, 1].set_xlabel('log(Fcalc/Fobs_FW)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title(f'French-Wilson Fobs\nμ={np.mean(log_ratio_fw):.3f}, σ={np.std(log_ratio_fw):.3f}\nskew={stats.skew(log_ratio_fw):.2f}, kurt={stats.kurtosis(log_ratio_fw):.2f}')
axes[0, 1].axvline(0, color='r', linestyle='--', alpha=0.5)
# Add Gaussian fit
x = np.linspace(-3, 5, 100)
axes[0, 1].plot(x, stats.norm.pdf(x, np.mean(log_ratio_fw), np.std(log_ratio_fw)), 'r-', lw=2, alpha=0.6, label='Gaussian fit')
axes[0, 1].legend()

# Plot 3: Q-Q plot for reference
stats.probplot(log_ratio_ref, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Q-Q Plot: Reference')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Q-Q plot for French-Wilson
stats.probplot(log_ratio_fw, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: French-Wilson')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Centered comparison (remove mean)
log_ratio_ref_centered = log_ratio_ref - np.mean(log_ratio_ref)
log_ratio_fw_centered = log_ratio_fw - np.mean(log_ratio_fw)

axes[1, 1].hist(log_ratio_ref_centered, bins=100, density=True, alpha=0.5, color='blue', range=(-3, 3), label='Reference (centered)')
axes[1, 1].hist(log_ratio_fw_centered, bins=100, density=True, alpha=0.5, color='green', range=(-3, 3), label='FW (centered)')
axes[1, 1].set_xlabel('log(Fcalc/Fobs) - mean')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Centered Distributions (mean removed)')
axes[1, 1].legend()
axes[1, 1].axvline(0, color='r', linestyle='--', alpha=0.5)

# Plot 6: Tail analysis
# Count reflections in tails
ref_left_tail = np.sum(log_ratio_ref < np.mean(log_ratio_ref) - 2*np.std(log_ratio_ref))
ref_right_tail = np.sum(log_ratio_ref > np.mean(log_ratio_ref) + 2*np.std(log_ratio_ref))
fw_left_tail = np.sum(log_ratio_fw < np.mean(log_ratio_fw) - 2*np.std(log_ratio_fw))
fw_right_tail = np.sum(log_ratio_fw > np.mean(log_ratio_fw) + 2*np.std(log_ratio_fw))

axes[1, 2].bar(['Ref\nLeft tail', 'Ref\nRight tail', 'FW\nLeft tail', 'FW\nRight tail'],
               [ref_left_tail, ref_right_tail, fw_left_tail, fw_right_tail],
               color=['blue', 'blue', 'green', 'green'], alpha=0.6)
axes[1, 2].set_ylabel('Count (beyond ±2σ)')
axes[1, 2].set_title('Tail Analysis\n(Gaussian should have ~5% in tails)')
axes[1, 2].axhline(0.05 * len(log_ratio_ref), color='r', linestyle='--', alpha=0.5, label='Expected for Gaussian')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/io/gaussian_comparison.png', dpi=150)
print("Saved comparison plot!")

# Print summary
neg_mask = I_obs < 0
print('\nSUMMARY:')
print('=' * 60)
print(f'Reference: σ={np.std(log_ratio_ref):.4f}, skew={stats.skew(log_ratio_ref):.2f}, kurt={stats.kurtosis(log_ratio_ref):.2f}')
print(f'French-Wilson: σ={np.std(log_ratio_fw):.4f}, skew={stats.skew(log_ratio_fw):.2f}, kurt={stats.kurtosis(log_ratio_fw):.2f}')
print(f'\nNegative I reflections: {neg_mask.sum()}')
print(f'  F_FW mean: {np.mean(Fobs_fw[neg_mask]):.2f}')
print(f'  F_ref mean: {np.mean(Fobs_ref[neg_mask]):.2f}')
print(f'\nImprovement: French-Wilson now assigns reasonable F values to negative intensities')
print(f'(was ~0 before, now ~{np.mean(Fobs_fw[neg_mask]):.1f})')
