"""
Debug missing electron density for specific atoms.
"""
import torch
import numpy as np
from torchref.model_ft import ModelFT

print("=" * 70)
print("DEBUG: Missing Electron Density")
print("=" * 70)

# Load structure
M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')
M.max_res = 1.2
M.setup_grid()

# Find the problematic atoms
print("\n" + "-" * 70)
print("Searching for problematic atoms:")
print("-" * 70)

# CA SER 54 (working)
# CA ALA 74 (problematic)

target_atoms = [
    ("CA", "SER", 54),
    ("CA", "ALA", 74),
]

atom_info = []
for atom_name, res_name, res_num in target_atoms:
    found = False
    for i in range(len(M.pdb)):
        row = M.pdb.iloc[i]
        if (row['name'].strip() == atom_name and 
            row['resname'].strip() == res_name and 
            row['resseq'] == res_num):
            print(f"\nFound: {atom_name} {res_name} {res_num}")
            print(f"  PDB index: {i}")
            print(f"  Position (Å): [{row['x']:.3f}, {row['y']:.3f}, {row['z']:.3f}]")
            print(f"  Chain: {row['chainid']}")
            print(f"  Occupancy: {row['occupancy']:.3f}")
            print(f"  B-factor: {row['tempfactor']:.3f}")
            print(f"  Element: {row['element']}")
            
            # Convert to fractional coordinates
            pos_cart = np.array([row['x'], row['y'], row['z']])
            pos_frac = np.linalg.solve(M.inv_frac_matrix.numpy(), pos_cart)
            print(f"  Fractional coords: {pos_frac}")
            
            # Convert to grid indices
            grid_idx = (pos_frac * np.array(M.map.shape)).astype(int)
            print(f"  Grid indices: {grid_idx}")
            print(f"  Map shape: {M.map.shape}")
            
            # Check if within bounds
            in_bounds = all(0 <= grid_idx[d] < M.map.shape[d] for d in range(3))
            print(f"  In bounds: {in_bounds}")
            
            atom_info.append((i, row, pos_frac, atom_name, res_name, res_num))
            found = True
            break
    
    if not found:
        print(f"\nWARNING: Could not find {atom_name} {res_name} {res_num}")

# Build density map without symmetry
print("\n" + "-" * 70)
print("Building density map...")
print("-" * 70)
M.build_density_map(apply_symmetry=False, radius=10)

print(f"Map sum: {M.map.sum():.2f}")
print(f"Map max: {M.map.max():.4f}")
print(f"Non-zero voxels: {(M.map > 0).sum()}")

# Check density around each atom
print("\n" + "-" * 70)
print("Checking density around atoms:")
print("-" * 70)

for idx, row, pos_frac, atom_name, res_name, res_num in atom_info:
    print(f"\n{atom_name} {res_name} {res_num} (PDB index {idx}):")
    
    # Convert to grid coordinates (continuous)
    grid_coords = pos_frac * np.array(M.map.shape)
    print(f"  Grid coordinates (continuous): {grid_coords}")
    
    # Get integer grid indices
    grid_idx = grid_coords.astype(int)
    print(f"  Grid indices (integer): {grid_idx}")
    
    # Check density at this voxel
    ix, iy, iz = grid_idx
    if 0 <= ix < M.map.shape[0] and 0 <= iy < M.map.shape[1] and 0 <= iz < M.map.shape[2]:
        density_at_atom = M.map[ix, iy, iz]
        print(f"  Density at atom position: {density_at_atom:.6f}")
    else:
        print(f"  Atom position outside map bounds!")
        continue
    
    # Check density in a 3x3x3 cube around the atom
    print(f"  Density in 3x3x3 cube around atom:")
    radius = 1
    cube_sum = 0.0
    cube_max = 0.0
    cube_count = 0
    
    for di in range(-radius, radius+1):
        for dj in range(-radius, radius+1):
            for dk in range(-radius, radius+1):
                i = ix + di
                j = iy + dj
                k = iz + dk
                if 0 <= i < M.map.shape[0] and 0 <= j < M.map.shape[1] and 0 <= k < M.map.shape[2]:
                    val = M.map[i, j, k]
                    cube_sum += val
                    cube_max = max(cube_max, val)
                    cube_count += 1
    
    print(f"    Sum: {cube_sum:.6f}")
    print(f"    Max: {cube_max:.6f}")
    print(f"    Mean: {cube_sum/cube_count:.6f}")
    print(f"    Voxels checked: {cube_count}")
    
    # Check a larger region (5x5x5)
    radius = 2
    region_sum = 0.0
    region_max = 0.0
    region_count = 0
    
    for di in range(-radius, radius+1):
        for dj in range(-radius, radius+1):
            for dk in range(-radius, radius+1):
                i = ix + di
                j = iy + dj
                k = iz + dk
                if 0 <= i < M.map.shape[0] and 0 <= j < M.map.shape[1] and 0 <= k < M.map.shape[2]:
                    val = M.map[i, j, k]
                    region_sum += val
                    region_max = max(region_max, val)
                    region_count += 1
    
    print(f"  Density in 5x5x5 region:")
    print(f"    Sum: {region_sum:.6f}")
    print(f"    Max: {region_max:.6f}")
    print(f"    Mean: {region_sum/region_count:.6f}")

# Now check what radius was used for density calculation
print("\n" + "-" * 70)
print("Checking FT cache for these atoms:")
print("-" * 70)

if hasattr(M, 'ft_cache') and M.ft_cache is not None:
    for idx, row, pos_frac, atom_name, res_name, res_num in atom_info:
        print(f"\n{atom_name} {res_name} {res_num} (PDB index {idx}):")
        
        # Check if this atom is in the cache
        atom_indices = M.ft_cache.get('atom_indices', None)
        if atom_indices is not None and idx < len(atom_indices):
            cache_idx = atom_indices[idx]
            print(f"  Cache index: {cache_idx}")
            
            if cache_idx >= 0:
                # Get the precomputed FT for this atom
                ft_real = M.ft_cache.get('ft_real', None)
                if ft_real is not None and cache_idx < len(ft_real):
                    ft_data = ft_real[cache_idx]
                    print(f"  FT data shape: {ft_data.shape}")
                    print(f"  FT data sum: {ft_data.sum():.6f}")
                    print(f"  FT data max: {ft_data.max():.6f}")
                    print(f"  FT data min: {ft_data.min():.6f}")
            else:
                print(f"  Not in cache (cache_idx = {cache_idx})")
        else:
            print(f"  Index out of cache range or no atom_indices")
else:
    print("  No FT cache found")

print("\n" + "=" * 70)
print("Diagnosis complete. Check if density is present for both atoms.")
print("=" * 70)
