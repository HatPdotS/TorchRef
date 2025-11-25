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

Fobs_fw,sigma_fw = french_wilson_conversion(I_obs, Sigma_Iobs)
Fobs_fw = Fobs_fw.clamp(min=1e-6,max=1e6)
Fobs = Fobs.clamp(min=1e-6,max=1e6)
log_ratios = torch.log(Fobs_fw) - torch.log(Fobs)

log_ratios_np = log_ratios.detach().cpu().numpy()

# Create figure with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Histogram of log ratios
axes[0, 0].hist(log_ratios_np, bins=100, density=True, alpha=0.6, color='g')
axes[0, 0].set_xlabel('log(F_FW) - log(F_obs)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Distribution of log ratios')
axes[0, 0].axvline(0, color='r', linestyle='--', alpha=0.5)

# Plot 2: Scatter plot of F_FW vs F_obs
axes[0, 1].scatter(Fobs.detach().cpu().numpy(), Fobs_fw.detach().cpu().numpy(), 
                   alpha=0.1, s=1)
axes[0, 1].plot([0, Fobs.max().item()], [0, Fobs.max().item()], 'r--', alpha=0.5)
axes[0, 1].set_xlabel('F-obs-filtered')
axes[0, 1].set_ylabel('F French-Wilson')
axes[0, 1].set_title('F comparison')
axes[0, 1].set_aspect('equal')

# Plot 3: Distribution of I-obs
I_obs_np = I_obs.detach().cpu().numpy()
axes[1, 0].hist(I_obs_np, bins=100, alpha=0.6, color='b', range=(-100, 500))
axes[1, 0].set_xlabel('I-obs')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Distribution of observed intensities')
axes[1, 0].axvline(0, color='r', linestyle='--', alpha=0.5, label='I=0')
axes[1, 0].legend()

# Plot 4: Log ratios vs I/sigma
I_over_sigma = (I_obs / Sigma_Iobs).detach().cpu().numpy()
axes[1, 1].scatter(I_over_sigma, log_ratios_np, alpha=0.1, s=1)
axes[1, 1].set_xlabel('I / sigma(I)')
axes[1, 1].set_ylabel('log(F_FW) - log(F_obs)')
axes[1, 1].set_title('Log ratio vs signal-to-noise')
axes[1, 1].axhline(0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].axvline(0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].set_xlim(-5, 10)

plt.tight_layout()
plt.savefig('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/io/log_ratio_histogram_french_wilson.png', dpi=150)
print("Plot saved!")

large_log_ratios_mismatch = log_ratios_np < -4

print('correlation Fobs, French wilson Fobs',np.corrcoef(Fobs_fw.detach().cpu().numpy(),Fobs.detach().cpu().numpy())[0,1])
print(f'\nNumber of reflections with large negative log ratios (< -4): {large_log_ratios_mismatch.sum()}')
print(f'Total negative intensities: {(I_obs_np < 0).sum()}')
print(f'Reflections with I/sigma < -3: {(I_over_sigma < -3).sum()}')

if large_log_ratios_mismatch.sum() > 0:
    print('\nReflections with large log ratio mismatch:')
    print(mtzin[large_log_ratios_mismatch])

