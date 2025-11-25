"""
Investigate the real_space_grid structure to understand PBC issues.
"""
import torch
import numpy as np
from torchref.model_ft import ModelFT

M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')
M.max_res = 1.2
M.setup_grid()

print("=" * 70)
print("INVESTIGATING REAL_SPACE_GRID STRUCTURE")
print("=" * 70)

print(f"\nGrid shape: {M.real_space_grid.shape}")
print(f"Voxel size: {M.voxel_size}")
print(f"Unit cell: {M.cell}")

# Check corners of the grid
print("\nGrid corners:")
corners = [
    ([0, 0, 0], "origin"),
    ([-1, 0, 0], "x_max"),
    ([0, -1, 0], "y_max"),
    ([0, 0, -1], "z_max"),
    ([-1, -1, -1], "opposite corner"),
]

for idx, name in corners:
    coord = M.real_space_grid[idx[0], idx[1], idx[2]]
    print(f"  {name:20s} {idx}: {coord.numpy()}")

# Check if grid is orthogonal
print("\nGrid structure analysis:")
print(f"  X direction (voxel 1,0,0 - voxel 0,0,0): {(M.real_space_grid[1,0,0] - M.real_space_grid[0,0,0]).numpy()}")
print(f"  Y direction (voxel 0,1,0 - voxel 0,0,0): {(M.real_space_grid[0,1,0] - M.real_space_grid[0,0,0]).numpy()}")
print(f"  Z direction (voxel 0,0,1 - voxel 0,0,0): {(M.real_space_grid[0,0,1] - M.real_space_grid[0,0,0]).numpy()}")

# Check the span of the grid
print("\nGrid span:")
for dim, name in enumerate(['X', 'Y', 'Z']):
    coords = M.real_space_grid[..., dim]
    print(f"  {name}: min={coords.min().item():.4f}, max={coords.max().item():.4f}, span={coords.max().item()-coords.min().item():.4f}")

# Check frac/inv_frac matrices
print(f"\nFrac matrix (Fractional -> Cartesian):")
print(M.frac_matrix.numpy())
print(f"\nInv frac matrix (Cartesian -> Fractional):")
print(M.inv_frac_matrix.numpy())

# Test conversion
test_frac = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float64)
test_cart = M.frac_matrix @ test_frac
print(f"\nTest conversion:")
print(f"  Fractional [0.5, 0.5, 0.5]")
print(f"  Cartesian: {test_cart.numpy()}")

# Convert back
test_frac_back = M.inv_frac_matrix @ test_cart
print(f"  Back to fractional: {test_frac_back.numpy()}")

# Now test with an actual atom
row = M.pdb.iloc[0]
atom_cart = torch.tensor([row['x'], row['y'], row['z']], dtype=torch.float64)
atom_frac = M.inv_frac_matrix @ atom_cart

print(f"\nFirst atom (N MET 1):")
print(f"  Cartesian: {atom_cart.numpy()}")
print(f"  Fractional: {atom_frac.numpy()}")
print(f"  Fractional (wrapped to [0,1]): {(atom_frac - torch.floor(atom_frac)).numpy()}")

# What if we convert fractional back to cart?
atom_frac_wrapped = atom_frac - torch.floor(atom_frac)
atom_cart_wrapped = M.frac_matrix @ atom_frac_wrapped
print(f"  Wrapped back to Cartesian: {atom_cart_wrapped.numpy()}")

# Where should this atom be in the grid?
grid_origin = M.real_space_grid[0, 0, 0]
print(f"\nGrid origin: {grid_origin.numpy()}")

# Convert wrapped position to grid index
grid_idx = torch.round((atom_cart_wrapped - grid_origin) / M.voxel_size).to(torch.int64)
print(f"Grid index for wrapped atom: {grid_idx.numpy()}")

# Check what coord is at that grid position
if all(0 <= grid_idx[d] < M.map.shape[d] for d in range(3)):
    grid_coord = M.real_space_grid[grid_idx[0], grid_idx[1], grid_idx[2]]
    print(f"Coordinate at that grid position: {grid_coord.numpy()}")
    print(f"Distance: {torch.norm(atom_cart_wrapped - grid_coord).item():.4f} Å")
else:
    print("Grid index out of bounds!")

print("\n" + "=" * 70)
