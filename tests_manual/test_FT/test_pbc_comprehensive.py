"""
Comprehensive test of periodic boundary conditions and density placement.
"""
import torch
import numpy as np
from torchref.model_ft import ModelFT
from torchref.math_functions.math_torch import find_relevant_voxels

print("=" * 70)
print("COMPREHENSIVE PBC AND DENSITY PLACEMENT TEST")
print("=" * 70)

# Load structure
M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')
M.max_res = 1.2
M.setup_grid()

print(f"\nUnit cell: {M.cell}")
print(f"Map shape: {M.map.shape}")
print(f"Voxel size: {M.voxel_size}")

# Check grid coverage
grid_min = M.real_space_grid.min(dim=0).values.min(dim=0).values.min(dim=0).values
grid_max = M.real_space_grid.max(dim=0).values.max(dim=0).values.max(dim=0).values
print(f"\nGrid coverage (Cartesian):")
print(f"  Min: {grid_min}")
print(f"  Max: {grid_max}")

# Check atom coordinate distribution
pdb_xyz = M.pdb[['x', 'y', 'z']].values
print(f"\nAtom coordinate range (Cartesian):")
print(f"  Min: {pdb_xyz.min(axis=0)}")
print(f"  Max: {pdb_xyz.max(axis=0)}")

# Test the wrapping function
print("\n" + "=" * 70)
print("Testing find_relevant_voxels wrapping")
print("=" * 70)

# Test a few specific atoms
test_atoms = [
    ("N", "MET", 1, 0),      # First atom
    ("CA", "SER", 54, 415),  # Problem atom mentioned
    ("CA", "ALA", 100, None), # Another test
]

for atom_name, res_name, res_num, expected_idx in test_atoms:
    # Find the atom
    found_idx = None
    for i in range(len(M.pdb)):
        row = M.pdb.iloc[i]
        if (row['name'].strip() == atom_name and 
            row['resname'].strip() == res_name and 
            row['resseq'] == res_num):
            found_idx = i
            break
    
    if found_idx is None:
        print(f"\nAtom {atom_name} {res_name} {res_num}: NOT FOUND")
        continue
    
    row = M.pdb.iloc[found_idx]
    pos_cart = torch.tensor([row['x'], row['y'], row['z']], dtype=torch.float64)
    
    print(f"\n{atom_name} {res_name} {res_num} (index {found_idx}):")
    print(f"  Original Cartesian: {pos_cart.numpy()}")
    
    # Call find_relevant_voxels with radius=1 to see what happens
    surrounding_coords, voxel_indices = find_relevant_voxels(
        M.real_space_grid, 
        pos_cart, 
        radius=1,
        inv_frac_matrix=M.inv_frac_matrix
    )
    
    # The center voxel should be at index [1,1,1] in the 3x3x3 cube (13th element of 27)
    center_voxel_idx = voxel_indices[0, 13]  # Center of 3x3x3 cube
    center_coord = surrounding_coords[0, 13]
    
    print(f"  Center voxel index: {center_voxel_idx.numpy()}")
    print(f"  Center voxel coord: {center_coord.numpy()}")
    
    # Check distance from atom to center voxel
    dist = torch.norm(pos_cart - center_coord)
    print(f"  Distance to center voxel: {dist.item():.4f} Å")
    print(f"  Expected: < {M.voxel_size.norm().item():.4f} Å (voxel diagonal)")
    
    # Check if the center voxel is in bounds
    in_bounds = all(0 <= center_voxel_idx[d] < M.map.shape[d] for d in range(3))
    print(f"  Center voxel in bounds: {in_bounds}")
    
    # Now check if wrapping happened
    # Extract the wrapping calculation from find_relevant_voxels
    unit_cell_dims = torch.tensor([
        M.real_space_grid[-1, 0, 0, 0] - M.real_space_grid[0, 0, 0, 0] + M.voxel_size[0],
        M.real_space_grid[0, -1, 0, 1] - M.real_space_grid[0, 0, 0, 1] + M.voxel_size[1],
        M.real_space_grid[0, 0, -1, 2] - M.real_space_grid[0, 0, 0, 2] + M.voxel_size[2]
    ], dtype=torch.float64)
    
    grid_origin = M.real_space_grid[0, 0, 0]
    
    pos_shifted = pos_cart - grid_origin
    pos_wrapped = torch.remainder(pos_shifted, unit_cell_dims)
    pos_final = pos_wrapped + grid_origin
    
    print(f"  Grid origin: {grid_origin.numpy()}")
    print(f"  Unit cell dims: {unit_cell_dims.numpy()}")
    print(f"  After shift: {pos_shifted.numpy()}")
    print(f"  After wrap: {pos_wrapped.numpy()}")
    print(f"  Final position: {pos_final.numpy()}")
    print(f"  Moved by: {(pos_final - pos_cart).numpy()} Å")

# Now test the full density building
print("\n" + "=" * 70)
print("Building density map and checking specific atoms")
print("=" * 70)

M.build_density_map(apply_symmetry=False, radius=10)

# Check density at the test atom positions
for atom_name, res_name, res_num, expected_idx in test_atoms:
    # Find the atom
    found_idx = None
    for i in range(len(M.pdb)):
        row = M.pdb.iloc[i]
        if (row['name'].strip() == atom_name and 
            row['resname'].strip() == res_name and 
            row['resseq'] == res_num):
            found_idx = i
            break
    
    if found_idx is None:
        continue
    
    row = M.pdb.iloc[found_idx]
    pos_cart = torch.tensor([row['x'], row['y'], row['z']], dtype=torch.float64)
    
    # Get wrapped position
    grid_origin = M.real_space_grid[0, 0, 0]
    unit_cell_dims = torch.tensor([
        M.real_space_grid[-1, 0, 0, 0] - M.real_space_grid[0, 0, 0, 0] + M.voxel_size[0],
        M.real_space_grid[0, -1, 0, 1] - M.real_space_grid[0, 0, 0, 1] + M.voxel_size[1],
        M.real_space_grid[0, 0, -1, 2] - M.real_space_grid[0, 0, 0, 2] + M.voxel_size[2]
    ], dtype=torch.float64)
    
    pos_shifted = pos_cart - grid_origin
    pos_wrapped = torch.remainder(pos_shifted, unit_cell_dims)
    pos_final = pos_wrapped + grid_origin
    
    # Find nearest voxel
    center_idx = torch.round(pos_final / M.voxel_size).to(torch.int64)
    center_idx = center_idx % torch.tensor(M.map.shape, dtype=torch.int64)
    
    print(f"\n{atom_name} {res_name} {res_num}:")
    print(f"  Original position: {pos_cart.numpy()}")
    print(f"  Wrapped position: {pos_final.numpy()}")
    print(f"  Nearest voxel: {center_idx.numpy()}")
    
    # Check density in 3x3x3 region
    ix, iy, iz = center_idx
    density_sum = 0.0
    density_max = 0.0
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                i = (ix + di) % M.map.shape[0]
                j = (iy + dj) % M.map.shape[1]
                k = (iz + dk) % M.map.shape[2]
                val = M.map[i, j, k].item()
                density_sum += val
                density_max = max(density_max, val)
    
    print(f"  Density in 3x3x3: sum={density_sum:.4f}, max={density_max:.4f}")
    if density_max < 0.1:
        print(f"  ⚠️  WARNING: Very low density! Atom may not be placed correctly!")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
