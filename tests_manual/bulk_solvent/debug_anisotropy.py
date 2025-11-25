#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u
"""
Debug: Check anisotropic U parameters
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
import numpy as np
from torchref.base_refinement import Refinement
from torchref.math_functions.math_torch import get_scattering_vectors

# Load
instance = Refinement(
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz',
    '/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb',
    cif='/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/elbow.AZO.dark_pdb.001.cif',
    verbose=0
)

instance.scaler.setup_anisotropy_correction()

print("\n" + "="*80)
print("ANISOTROPIC CORRECTION DEBUG")
print("="*80)

# Check U parameters
U_params = instance.scaler.U.detach().cpu().numpy()
print(f"\nU parameters (Ų):")
print(f"  U11 = {U_params[0]:+.6f}")
print(f"  U22 = {U_params[1]:+.6f}")
print(f"  U33 = {U_params[2]:+.6f}")
print(f"  U12 = {U_params[3]:+.6f}")
print(f"  U13 = {U_params[4]:+.6f}")
print(f"  U23 = {U_params[5]:+.6f}")

# Compute anisotropic correction for some reflections
with torch.no_grad():
    scattering_vectors = get_scattering_vectors(instance.hkl, instance.model.cell)
    s = scattering_vectors / 2.0  # This is the scattering vector s = h*
    
    # Compute s·U·s^T for each reflection
    # U is 3x3 symmetric matrix
    U_matrix = torch.zeros((3, 3), device=instance.scaler.U.device, dtype=instance.scaler.U.dtype)
    U_matrix[0, 0] = instance.scaler.U[0]  # U11
    U_matrix[1, 1] = instance.scaler.U[1]  # U22
    U_matrix[2, 2] = instance.scaler.U[2]  # U33
    U_matrix[0, 1] = U_matrix[1, 0] = instance.scaler.U[3]  # U12
    U_matrix[0, 2] = U_matrix[2, 0] = instance.scaler.U[4]  # U13
    U_matrix[1, 2] = U_matrix[2, 1] = instance.scaler.U[5]  # U23
    
    # s·U·s^T
    sU = torch.matmul(s, U_matrix)  # (N, 3)
    sUs = (sU * s).sum(dim=1)  # (N,)
    
    # Anisotropic correction: exp(-2π² * s·U·s^T)
    aniso_correction = torch.exp(-2 * np.pi**2 * sUs)
    
    print(f"\nAnisotropic correction statistics:")
    print(f"  s·U·s^T:")
    print(f"    Mean: {sUs.mean().item():.6f}")
    print(f"    Min:  {sUs.min().item():.6f}")
    print(f"    Max:  {sUs.max().item():.6f}")
    print(f"\n  exp(-2π² * s·U·s^T):")
    print(f"    Mean: {aniso_correction.mean().item():.6f}")
    print(f"    Min:  {aniso_correction.min().item():.6f}")
    print(f"    Max:  {aniso_correction.max().item():.6f}")
    
    if aniso_correction.mean() < 0.1:
        print(f"\n  ⚠️  ANISOTROPIC CORRECTION IS CRUSHING STRUCTURE FACTORS!")
        print(f"     Mean correction factor is {aniso_correction.mean().item():.4f}")
        print(f"     This is scaling F_calc down by {1/aniso_correction.mean().item():.1f}x")
        print(f"\n  LIKELY CAUSE: U parameters are TOO POSITIVE")
        print(f"     When U values are positive, exp(-2π²·s·U·s^T) < 1, reducing magnitudes")
        print(f"     For typical data, U should be small or slightly negative")

print("="*80)
