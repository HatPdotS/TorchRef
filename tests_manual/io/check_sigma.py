from torchref.math_functions.math_torch import french_wilson_conversion
import reciprocalspaceship as rs
import torch
import numpy as np

mtzin = rs.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz')
mtzin.dropna(inplace=True)

I_obs = torch.tensor(mtzin['I-obs'].values.astype(np.float32), dtype=torch.float32)
Sigma_Iobs = torch.tensor(mtzin['SIGI-obs'].values.astype(np.float32), dtype=torch.float32)

# Check what Sigma is
positive_mask = I_obs > 0
mean_I = torch.mean(I_obs[positive_mask])
Sigma = torch.sqrt(mean_I)

print(f"Mean positive intensity: {mean_I.item():.2f}")
print(f"Sigma (sqrt of mean I): {Sigma.item():.2f}")
print(f"F_prior = Sigma * sqrt(pi/2): {(Sigma * np.sqrt(np.pi/2)).item():.2f}")
print(f"\nTypical sigma_I values: min={Sigma_Iobs.min().item():.2f}, max={Sigma_Iobs.max().item():.2f}, mean={Sigma_Iobs.mean().item():.2f}")
