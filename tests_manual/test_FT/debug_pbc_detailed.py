"""
Detailed debugging of PBC coordinate conversion.
"""
import torch
import numpy as np
from torchref.model_ft import ModelFT

print("=" * 70)
print("DETAILED PBC DEBUGGING")
print("=" * 70)

# Load structure
M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')
M.max_res = 1.2
M.setup_grid()

# Test atom: N MET 1
row = M.pdb.iloc[0]
pos_cart = torch.tensor([row['x'], row['y'], row['z']], dtype=torch.float64)

print(f"\nTest atom: N MET 1")
print(f"Cartesian position: {pos_cart.numpy()}")

# Grid parameters
grid_shape = torch.tensor(M.real_space_grid.shape[:3], device=pos_cart.device)
grid_origin = M.real_space_grid[0, 0, 0]

print(f"\nGrid shape: {grid_shape.numpy()}")
print(f"Grid origin: {grid_origin.numpy()}")

# Conversion using inv_frac_matrix
print(f"\nInv frac matrix shape: {M.inv_frac_matrix.shape}")
print(f"Inv frac matrix:\n{M.inv_frac_matrix.numpy()}")

# Convert Cartesian -> Fractional
pos_cart_unsqueezed = pos_cart.unsqueeze(0)  # (1, 3)
xyz_frac = torch.matmul(M.inv_frac_matrix, pos_cart_unsqueezed.T).T  # (1, 3)

print(f"\nCartesian (as 2D): {pos_cart_unsqueezed.numpy()}")
print(f"Fractional: {xyz_frac.numpy()}")
print(f"Fractional (wrapped [0,1]): {(xyz_frac % 1.0).numpy()}")

# Convert to grid indices
center_idx = torch.round(xyz_frac * grid_shape.unsqueeze(0)).to(torch.int64)
print(f"\nGrid indices (from fractional): {center_idx.numpy()}")

# What coordinate is at this grid position?
idx = center_idx[0]
if all(0 <= idx[d] < grid_shape[d] for d in range(3)):
    coord_at_idx = M.real_space_grid[idx[0], idx[1], idx[2]]
    print(f"Coordinate at grid index: {coord_at_idx.numpy()}")
    dist = torch.norm(pos_cart - coord_at_idx)
    print(f"Distance: {dist.item():.4f} Å")
else:
    print(f"Grid index out of bounds!")

# Now wrap fractional coordinates and try again
xyz_frac_wrapped = xyz_frac % 1.0
center_idx_wrapped = torch.round(xyz_frac_wrapped * grid_shape.unsqueeze(0)).to(torch.int64)

print(f"\n--- With fractional wrapping ---")
print(f"Wrapped fractional: {xyz_frac_wrapped.numpy()}")
print(f"Grid indices (from wrapped fractional): {center_idx_wrapped.numpy()}")

idx_w = center_idx_wrapped[0]
if all(0 <= idx_w[d] < grid_shape[d] for d in range(3)):
    coord_at_idx_w = M.real_space_grid[idx_w[0], idx_w[1], idx_w[2]]
    print(f"Coordinate at grid index: {coord_at_idx_w.numpy()}")
    
    # Calculate the wrapped Cartesian position
    pos_cart_wrapped = torch.matmul(M.frac_matrix, xyz_frac_wrapped.T).T
    print(f"Wrapped Cartesian (from frac): {pos_cart_wrapped.numpy()}")
    
    dist_w = torch.norm(pos_cart_wrapped[0] - coord_at_idx_w)
    print(f"Distance (wrapped): {dist_w.item():.4f} Å")
    
    # Also check via PBC
    diff = pos_cart_wrapped[0] - coord_at_idx_w
    print(f"Raw difference: {diff.numpy()}")
    
    # Convert diff to fractional
    diff_frac = torch.matmul(M.inv_frac_matrix, diff)
    print(f"Difference in fractional: {diff_frac.numpy()}")
    
    # Round to nearest periodic image
    translation = torch.round(diff_frac)
    print(f"Translation (cells): {translation.numpy()}")
    
    # Apply translation
    corrected_frac = torch.matmul(M.frac_matrix, translation)
    print(f"Translation (Cartesian): {corrected_frac.numpy()}")
    
    diff_pbc = diff - corrected_frac
    print(f"PBC corrected difference: {diff_pbc.numpy()}")
    print(f"PBC distance: {torch.norm(diff_pbc).item():.4f} Å")
else:
    print(f"Grid index out of bounds!")

print("\n" + "=" * 70)
