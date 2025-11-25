from torchref.math_functions.math_torch import french_wilson_conversion
import reciprocalspaceship as rs
import torch
import numpy as np

mtzin = rs.read_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz')
mtzin.dropna(inplace=True)

Fobs = torch.tensor(mtzin['F-obs-filtered'].values.astype(np.float32), dtype=torch.float32)
I_obs = torch.tensor(mtzin['I-obs'].values.astype(np.float32), dtype=torch.float32)
Sigma_Iobs = torch.tensor(mtzin['SIGI-obs'].values.astype(np.float32), dtype=torch.float32)

Fobs_fw, sigma_fw = french_wilson_conversion(I_obs, Sigma_Iobs)

# Look at specific examples
print("Examples of negative intensity reflections:\n")
negative_mask = I_obs < 0
indices = torch.where(negative_mask)[0][:10]  # First 10 negative intensities

for idx in indices:
    i = idx.item()
    I = I_obs[i].item()
    sig_I = Sigma_Iobs[i].item()
    F_ref = Fobs[i].item()
    F_fw = Fobs_fw[i].item()
    sig_fw = sigma_fw[i].item()
    
    print(f"I={I:8.2f}, sigma_I={sig_I:8.2f}, I/sigma={I/sig_I:6.2f}")
    print(f"  F_reference={F_ref:8.2f}, F_FW={F_fw:8.2f}, ratio={F_fw/F_ref:6.3f}")
    print(f"  sqrt(|I|)={np.sqrt(abs(I)):8.2f}")
    print()

print("\nExamples of weak positive intensity reflections:\n")
weak_positive_mask = (I_obs > 0) & (I_obs < 3 * Sigma_Iobs)
indices = torch.where(weak_positive_mask)[0][:10]

for idx in indices:
    i = idx.item()
    I = I_obs[i].item()
    sig_I = Sigma_Iobs[i].item()
    F_ref = Fobs[i].item()
    F_fw = Fobs_fw[i].item()
    sig_fw = sigma_fw[i].item()
    
    print(f"I={I:8.2f}, sigma_I={sig_I:8.2f}, I/sigma={I/sig_I:6.2f}")
    print(f"  F_reference={F_ref:8.2f}, F_FW={F_fw:8.2f}, ratio={F_fw/F_ref:6.3f}")
    print(f"  sqrt(I)={np.sqrt(I):8.2f}")
    print()
