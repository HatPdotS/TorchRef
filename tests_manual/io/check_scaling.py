#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/io/check_scaling.log

import torch
from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.scaler import Scaler
import numpy as np

# Load model and data
Model = ModelFT().load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb')
Data = ReflectionData(verbose=2).load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz')

# Get Fcalc and Fobs
Fcalc_complex = Model(Data.get_hkl())
Fcalc = torch.abs(Fcalc_complex)
_, Fobs, _, _ = Data()

print("\n" + "="*60)
print("BEFORE SCALING")
print("="*60)

# Compute log-ratio before scaling
eps = 1e-10
log_ratio_unscaled = torch.log(Fobs + eps) - torch.log(Fcalc + eps)
log_ratio_np = log_ratio_unscaled.detach().cpu().numpy()

print(f"Fobs: mean={Fobs.mean().item():.2f}, std={Fobs.std().item():.2f}")
print(f"Fcalc: mean={Fcalc.mean().item():.2f}, std={Fcalc.std().item():.2f}")
print(f"Ratio Fobs/Fcalc: mean={torch.mean(Fobs/Fcalc).item():.4f}")
print(f"Log-ratio: mean={log_ratio_np.mean():.4f}, std={log_ratio_np.std():.4f}")

# Create scaler
print("\n" + "="*60)
print("CREATING SCALER")
print("="*60)
scaler = Scaler(Fcalc_complex, Data, verbose=2, nbins=50)

# Get scaled Fcalc
Fcalc_scaled = torch.abs(scaler(Fcalc_complex))

print("\n" + "="*60)
print("AFTER SCALING")
print("="*60)

log_ratio_scaled = torch.log(Fobs + eps) - torch.log(Fcalc_scaled + eps)
log_ratio_scaled_np = log_ratio_scaled.detach().cpu().numpy()

print(f"Fcalc_scaled: mean={Fcalc_scaled.mean().item():.2f}, std={Fcalc_scaled.std().item():.2f}")
print(f"Ratio Fobs/Fcalc_scaled: mean={torch.mean(Fobs/Fcalc_scaled).item():.4f}")
print(f"Log-ratio: mean={log_ratio_scaled_np.mean():.4f}, std={log_ratio_scaled_np.std():.4f}")

# Check if there's a bias in the scaling
print("\n" + "="*60)
print("SCALE FACTOR DIAGNOSTICS")
print("="*60)

# Get scale factors
scale_factors = scaler.scale.detach().cpu().numpy()
print(f"Scale factors per bin: min={scale_factors.min():.4f}, max={scale_factors.max():.4f}, mean={scale_factors.mean():.4f}")

# Expected correction to center the distribution
correction_factor = np.exp(-log_ratio_scaled_np.mean())
print(f"\nTo center log-ratio at 0, multiply all scales by: {correction_factor:.4f}")
print(f"Current log-ratio mean: {log_ratio_scaled_np.mean():.4f}")
print(f"After correction would be: {(log_ratio_scaled_np + np.log(correction_factor)).mean():.6f}")

# Check for systematic bias in negative intensity reflections
print("\n" + "="*60)
print("NEGATIVE INTENSITY ANALYSIS")
print("="*60)

# Need to check which reflections had negative intensities
# This information should be in the Data object
if hasattr(Data, 'I'):
    I = Data.I
    neg_mask = I < 0
    print(f"Number of negative intensities: {neg_mask.sum().item()} ({100*neg_mask.sum().item()/len(I):.2f}%)")
    
    # Compare log-ratio for negative vs positive intensity reflections
    log_ratio_neg = log_ratio_scaled[neg_mask]
    log_ratio_pos = log_ratio_scaled[~neg_mask]
    
    print(f"Log-ratio for negative I: mean={log_ratio_neg.mean().item():.4f}, std={log_ratio_neg.std().item():.4f}")
    print(f"Log-ratio for positive I: mean={log_ratio_pos.mean().item():.4f}, std={log_ratio_pos.std().item():.4f}")
    print(f"Difference in means: {(log_ratio_neg.mean() - log_ratio_pos.mean()).item():.4f}")
    
    # Check F values for negative vs positive I
    F_neg = Fobs[neg_mask]
    F_pos = Fobs[~neg_mask]
    print(f"\nFobs for negative I: mean={F_neg.mean().item():.2f}, median={F_neg.median().item():.2f}")
    print(f"Fobs for positive I: mean={F_pos.mean().item():.2f}, median={F_pos.median().item():.2f}")
