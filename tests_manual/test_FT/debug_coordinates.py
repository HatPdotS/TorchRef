"""
Simple test to understand coordinate systems and the missing density issue.
"""
import torch
import numpy as np
from torchref.model_ft import ModelFT

print("=" * 70)
print("COORDINATE SYSTEM DEBUG")
print("=" * 70)

# Load structure
M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')
M.max_res = 1.2
M.setup_grid()

print(f"\nUnit cell: {M.cell}")
print(f"Map shape: {M.map.shape}")
print(f"Voxel size: {M.voxel_size}")

# Check the frac_matrix
print(f"\nFractional matrix shape: {M.frac_matrix.shape}")
print(f"Fractional matrix:\n{M.frac_matrix}")
print(f"\nInv fractional matrix shape: {M.inv_frac_matrix.shape}")
print(f"Inv fractional matrix:\n{M.inv_frac_matrix}")

# Test: what is the origin in fractional coords?
origin_cart = np.array([0.0, 0.0, 0.0])
origin_frac = M.frac_matrix.numpy() @ origin_cart
print(f"\nOrigin (0,0,0) in Cartesian:")
print(f"  Fractional: {origin_frac}")
print(f"  Should be: [0, 0, 0]")

# Test: convert grid corner back to fractional
grid_origin = M.real_space_grid[0, 0, 0]
print(f"\nGrid origin (first voxel corner):")
print(f"  Cartesian: {grid_origin}")
grid_origin_frac = M.frac_matrix.numpy() @ grid_origin.numpy()
print(f"  Fractional: {grid_origin_frac}")

# Check a few atoms
print("\n" + "=" * 70)
print("Checking atom coordinates")
print("=" * 70)

for i in [0, 100, 415]:  # Include index 415 (SER 54 CA)
    if i >= len(M.pdb):
        continue
    row = M.pdb.iloc[i]
    print(f"\nAtom {i}: {row['name']} {row['resname']} {row['resseq']}")
    
    pos_cart = np.array([row['x'], row['y'], row['z']])
    print(f"  Cartesian: {pos_cart}")
    
    # Convert to fractional
    pos_frac = M.frac_matrix.numpy() @ pos_cart
    print(f"  Fractional (frac_matrix @ pos): {pos_frac}")
    
    # Check if in [0, 1] range
    if np.all((pos_frac >= 0) & (pos_frac <= 1)):
        print(f"  ✓ In valid range [0, 1]")
    else:
        print(f"  ✗ OUT OF RANGE! Should be in [0, 1]")
    
    # Convert to grid indices
    grid_idx = (pos_frac * np.array(M.map.shape)).astype(int)
    print(f"  Grid indices: {grid_idx}")
    
    # Check bounds
    in_bounds = np.all((grid_idx >= 0) & (grid_idx < np.array(M.map.shape)))
    print(f"  In bounds: {in_bounds}")

# Check the real_space_grid
print("\n" + "=" * 70)
print("Checking real_space_grid")
print("=" * 70)

print(f"Grid shape: {M.real_space_grid.shape}")
print(f"Grid min: {M.real_space_grid.min(dim=0).values.min(dim=0).values.min(dim=0).values}")
print(f"Grid max: {M.real_space_grid.max(dim=0).values.max(dim=0).values.max(dim=0).values}")

# Convert grid extremes to fractional
grid_min_cart = M.real_space_grid[0, 0, 0].numpy()
grid_max_cart = M.real_space_grid[-1, -1, -1].numpy()

grid_min_frac = M.frac_matrix.numpy() @ grid_min_cart
grid_max_frac = M.frac_matrix.numpy() @ grid_max_cart

print(f"\nGrid min corner:")
print(f"  Cartesian: {grid_min_cart}")
print(f"  Fractional: {grid_min_frac}")

print(f"\nGrid max corner:")
print(f"  Cartesian: {grid_max_cart}")
print(f"  Fractional: {grid_max_frac}")

print("\n" + "=" * 70)
