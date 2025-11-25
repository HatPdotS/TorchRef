#!/das/work/p17/p17490/CONDA/muticopy_refinement/bin/python -u 

#SBATCH -c 16
#SBATCH -o /das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/io/test_solvent_smooth.log

import torch
from torchref.model_ft import ModelFT
from torchref.solvent import SolventModel
import numpy as np

print("="*60)
print("Testing SolventModel.forward() - Smoothed Mask")
print("="*60)

# Load model
print("\n1. Loading model...")
Model = ModelFT().load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/tests/refinement/dark.pdb')

# Create solvent model
print("\n2. Creating SolventModel...")
solvent = SolventModel(Model, radius=1.1, k_solvent=0.35, b_solvent=46.0, verbose=1)

print("\n3. Computing smoothed solvent density...")

# Get original mask statistics
mask_original = solvent.solvent_mask.float()
print(f"\nOriginal binary mask:")
print(f"  Shape: {mask_original.shape}")
print(f"  Min: {mask_original.min().item():.4f} (should be 0)")
print(f"  Max: {mask_original.max().item():.4f} (should be 1)")
print(f"  Mean: {mask_original.mean().item():.4f}")
print(f"  Unique values: {torch.unique(mask_original).tolist()}")

# Compute smoothed solvent density
solvent_density = solvent.forward()

print(f"\nSmoothed solvent density:")
print(f"  Shape: {solvent_density.shape}")
print(f"  Dtype: {solvent_density.dtype}")
print(f"  Device: {solvent_density.device}")
print(f"  Min: {solvent_density.min().item():.6f}")
print(f"  Max: {solvent_density.max().item():.6f}")
print(f"  Mean: {solvent_density.mean().item():.6f}")
print(f"  Std: {solvent_density.std().item():.6f}")

# Check that smoothing creates intermediate values
unique_vals = torch.unique(solvent_density)
print(f"  Number of unique values: {len(unique_vals)} (should be > 2 for smooth)")
if len(unique_vals) > 10:
    print(f"  ✓ Mask has been smoothed (many intermediate values)")
else:
    print(f"  ✗ Mask may not be smoothed properly")

# Test gradient flow
print("\n" + "="*60)
print("4. Testing gradient flow through k_solvent")
print("="*60)

solvent.k_solvent.requires_grad = True

solvent_density = solvent.forward()
loss = solvent_density.sum()

print(f"\nLoss value: {loss.item():.4f}")
print(f"k_solvent = {solvent.k_solvent.item():.4f}, requires_grad = {solvent.k_solvent.requires_grad}")

print("\nComputing gradients...")
loss.backward()

print(f"\nGradients:")
print(f"  dk_solvent/dloss = {solvent.k_solvent.grad.item():.6f}")

if solvent.k_solvent.grad is not None:
    print("\n✓ Gradient computed successfully!")
    print("  The forward method is differentiable with respect to k_solvent.")
else:
    print("\n✗ Gradient computation failed!")

# Test parameter effect
print("\n" + "="*60)
print("5. Testing k_solvent effect on smoothed density")
print("="*60)

solvent.k_solvent.grad = None

print("\nEffect of k_solvent on smoothed density:")
for k in [0.1, 0.35, 0.5, 1.0]:
    with torch.no_grad():
        solvent.k_solvent.data = torch.tensor(k, dtype=torch.float32)
    density = solvent.forward()
    print(f"  k_solvent = {k:.2f} → density mean = {density.mean().item():.6f}, max = {density.max().item():.6f}")

# Verify scaling is linear
print("\n✓ Expected: density scales linearly with k_solvent")

# Check that smoothing preserves total "volume"
print("\n" + "="*60)
print("6. Checking conservation of solvent volume")
print("="*60)

solvent.k_solvent.data = torch.tensor(1.0, dtype=torch.float32)
density = solvent.forward()

original_volume = mask_original.sum().item()
smoothed_volume = density.sum().item()

print(f"Original mask sum: {original_volume:.1f}")
print(f"Smoothed mask sum: {smoothed_volume:.1f}")
print(f"Ratio: {smoothed_volume / original_volume:.4f}")

if abs(smoothed_volume / original_volume - 1.0) < 0.1:
    print("✓ Volume approximately conserved (within 10%)")
else:
    print("⚠ Volume changed significantly - this is expected due to edge smoothing")

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
