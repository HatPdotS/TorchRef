from torchref.math_functions.math_torch import french_wilson_conversion
import torch
import numpy as np

# Simulate data
I_obs = torch.tensor([-10.0, -5.0, -2.0, -1.0, 0.5, 1.0, 5.0, 10.0, 100.0, 1000.0], dtype=torch.float32)
sigma_I = torch.tensor([20.0, 15.0, 10.0, 8.0, 5.0, 5.0, 10.0, 15.0, 50.0, 150.0], dtype=torch.float32)

print("Testing French-Wilson with simulated data")
print("=" * 60)

# Current implementation
F, sigma_F = french_wilson_conversion(I_obs, sigma_I)

mean_I = torch.mean(torch.clamp(I_obs, min=0))
wilson_param = mean_I / 2.0

print(f"Mean I (positive only): {mean_I:.2f}")
print(f"Wilson param (mean_I/2): {wilson_param:.2f}")
print(f"sqrt(wilson_param): {torch.sqrt(wilson_param):.2f}")
print()

print("Results:")
print(f"{'I':>8} {'sigma_I':>8} {'I/sigma':>8} {'correction':>12} {'I+corr':>10} {'F':>8}")
print("-" * 70)

for i in range(len(I_obs)):
    I_val = I_obs[i].item()
    sig_val = sigma_I[i].item()
    ratio = I_val / sig_val
    
    # Calculate what the correction term is
    if I_val <= 3.0 * sig_val:  # weak/negative
        correction = sig_val**2 / (2.0 * wilson_param.item())
        I_corrected = max(0, I_val + correction)
    else:  # strong
        correction = 0
        I_corrected = I_val
    
    F_val = F[i].item()
    
    print(f"{I_val:8.2f} {sig_val:8.2f} {ratio:8.2f} {correction:12.2f} {I_corrected:10.2f} {F_val:8.2f}")

print()
print("Analysis of negative intensities:")
neg_mask = I_obs < 0
if neg_mask.any():
    I_neg = I_obs[neg_mask]
    sig_neg = sigma_I[neg_mask]
    F_neg = F[neg_mask]
    
    corrections = sig_neg**2 / (2.0 * wilson_param)
    
    print(f"  Number of negative: {neg_mask.sum()}")
    print(f"  Mean |I|: {torch.mean(torch.abs(I_neg)):.2f}")
    print(f"  Mean sigma: {torch.mean(sig_neg):.2f}")
    print(f"  Mean correction: {torch.mean(corrections):.2f}")
    print(f"  Mean F: {torch.mean(F_neg):.2f}")
    print()
    print("The problem: correction term is dominated by sigma^2, not by the intensity signal")
    print(f"  For I=-10, sigma=20: correction = 20^2/(2*{wilson_param:.0f}) = {400/(2*wilson_param.item()):.2f}")
    print(f"  This gives F = sqrt(max(0, -10 + {400/(2*wilson_param.item()):.2f})) = {torch.sqrt(torch.clamp(torch.tensor(-10.0 + 400/(2*wilson_param.item())), min=0)):.2f}")
