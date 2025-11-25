#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/io/test_solvent_forward.log

import torch
from torchref.model_ft import ModelFT
from torchref.Data import ReflectionData
from torchref.solvent import SolventModel
import numpy as np

print("="*60)
print("Testing SolventModel.forward() method")
print("="*60)

# Load model and data
print("\n1. Loading model and data...")
Model = ModelFT().load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb')
Data = ReflectionData(verbose=1).load_from_mtz('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.mtz')

# Create solvent model
print("\n2. Creating SolventModel...")
solvent = SolventModel(Model, radius=1.1, k_solvent=0.35, b_solvent=46.0, verbose=1)

# Get HKL from data
hkl = Data.get_hkl()
print(f"\n3. Computing solvent structure factors for {len(hkl)} reflections...")

# Compute solvent contribution
F_solvent = solvent.forward(hkl)

print(f"\nSolvent structure factors computed:")
print(f"  Shape: {F_solvent.shape}")
print(f"  Dtype: {F_solvent.dtype}")
print(f"  Device: {F_solvent.device}")
print(f"  |F_solvent| mean: {torch.abs(F_solvent).mean().item():.4f}")
print(f"  |F_solvent| std: {torch.abs(F_solvent).std().item():.4f}")
print(f"  |F_solvent| max: {torch.abs(F_solvent).max().item():.4f}")
print(f"  |F_solvent| min: {torch.abs(F_solvent).min().item():.4f}")

# Test gradient flow
print("\n" + "="*60)
print("4. Testing gradient flow through k_solvent and b_solvent")
print("="*60)

# Enable gradient tracking
solvent.k_solvent.requires_grad = True
solvent.b_solvent.requires_grad = True

# Compute solvent contribution
F_solvent = solvent.forward(hkl[:10000])  # Use subset for speed

# Create a dummy loss (e.g., sum of absolute values)
loss = torch.abs(F_solvent).sum()

print(f"\nLoss value: {loss.item():.4f}")
print(f"k_solvent = {solvent.k_solvent.item():.4f}, requires_grad = {solvent.k_solvent.requires_grad}")
print(f"b_solvent = {solvent.b_solvent.item():.4f}, requires_grad = {solvent.b_solvent.requires_grad}")

# Backpropagate
print("\nComputing gradients...")
loss.backward()

print(f"\nGradients:")
print(f"  dk_solvent/dloss = {solvent.k_solvent.grad.item():.6f}")
print(f"  db_solvent/dloss = {solvent.b_solvent.grad.item():.6f}")

if solvent.k_solvent.grad is not None and solvent.b_solvent.grad is not None:
    print("\n✓ Gradients computed successfully!")
    print("  The forward method is differentiable with respect to both parameters.")
else:
    print("\n✗ Gradient computation failed!")

# Test effect of parameters
print("\n" + "="*60)
print("5. Testing parameter effects")
print("="*60)

# Reset gradients
solvent.k_solvent.grad = None
solvent.b_solvent.grad = None

# Test with different k_solvent values
print("\nEffect of k_solvent on |F_solvent|:")
for k in [0.1, 0.35, 0.5, 1.0]:
    with torch.no_grad():
        solvent.k_solvent.data = torch.tensor(k, dtype=torch.float32)
    F_test = solvent.forward(hkl[:1000])
    print(f"  k_solvent = {k:.2f} → |F_solvent| mean = {torch.abs(F_test).mean().item():.4f}")

# Reset k_solvent
solvent.k_solvent.data = torch.tensor(0.35, dtype=torch.float32)

# Test with different b_solvent values
print("\nEffect of b_solvent on |F_solvent|:")
for b in [10.0, 30.0, 50.0, 100.0]:
    with torch.no_grad():
        solvent.b_solvent.data = torch.tensor(b, dtype=torch.float32)
    F_test = solvent.forward(hkl[:1000])
    print(f"  b_solvent = {b:.1f} → |F_solvent| mean = {torch.abs(F_test).mean().item():.4f}")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
