"""
Debug symmetry operations by saving individual symmetry mates for visual inspection.
"""
import torch
import numpy as np
from torchref.model_ft import ModelFT
from torchref.map_symmetry import MapSymmetry

print("=" * 60)
print("SYMMETRY DEBUG TEST")
print("=" * 60)

# Load structure
M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')

# Setup grids
M.setup_grid(1.2)
print(f"\nStructure info:")
print(f"  Space group: {M.spacegroup}")
print(f"  Cell parameters: {M.cell}")
print(f"  Map shape: {M.map.shape}")
print(f"  Parametrization: {M.parametrization}")

# Build asymmetric unit density (no symmetry)
print("\n" + "=" * 60)
print("STEP 1: Build asymmetric unit map (no symmetry)")
print("=" * 60)
M.build_density_map(apply_symmetry=False)
print(f"Asymmetric unit map sum: {M.map.sum():.2f}")
print(f"Asymmetric unit map max: {M.map.max():.4f}")
print(f"Non-zero voxels: {(M.map > 0).sum()}")

# Save asymmetric unit
asym_file = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/asymmetric_unit.ccp4'
M.save_map(asym_file)
print(f"\n✓ Saved asymmetric unit map to: {asym_file}")

# Now manually apply symmetry operations one by one
print("\n" + "=" * 60)
print("STEP 2: Apply individual symmetry operations")
print("=" * 60)

# Get the symmetry object
map_sym = M.map_symmetry
print(f"\nSymmetry info:")
print(f"  Number of operations: {map_sym.n_ops}")
print(f"  Rotation matrices shape: {map_sym.symmetry.matrices.shape}")
print(f"  Translations shape: {map_sym.symmetry.translations.shape}")

# Print each symmetry operation
for i in range(map_sym.n_ops):
    print(f"\nOperation {i}:")
    print(f"  Rotation matrix:\n{map_sym.symmetry.matrices[i]}")
    print(f"  Translation: {map_sym.symmetry.translations[i]}")

# Get the asymmetric unit map
asym_map = M.map.clone()

# Apply each symmetry operation separately and save
import torch.nn.functional as F

for i in range(map_sym.n_ops):
    print(f"\n--- Processing symmetry operation {i} ---")
    
    # Prepare map for grid_sample
    map_5d = asym_map.unsqueeze(0).unsqueeze(0)  # (1, 1, nx, ny, nz)
    
    # Get sampling grid for this operation
    sampling_grid = map_sym.sampling_grids[i]
    sampling_grid_batch = sampling_grid.unsqueeze(0)  # (1, nx, ny, nz, 3)
    
    print(f"  Sampling grid shape: {sampling_grid_batch.shape}")
    print(f"  Sampling grid range: [{sampling_grid_batch.min():.3f}, {sampling_grid_batch.max():.3f}]")
    print(f"  (Should be in [-1, 1] for grid_sample)")
    
    # Interpolate
    transformed_map = F.grid_sample(
        map_5d, 
        sampling_grid_batch,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    # Remove batch and channel dimensions
    transformed_map = transformed_map.squeeze(0).squeeze(0)
    
    print(f"  Transformed map sum: {transformed_map.sum():.2f}")
    print(f"  Transformed map max: {transformed_map.max():.4f}")
    print(f"  Non-zero voxels: {(transformed_map > 0).sum()}")
    
    # Save this symmetry mate
    M.map = transformed_map
    mate_file = f'/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/symmetry_mate_{i}.ccp4'
    M.save_map(mate_file)
    print(f"  ✓ Saved to: {mate_file}")

# Now apply full symmetry (sum all mates)
print("\n" + "=" * 60)
print("STEP 3: Apply full symmetry (sum all mates)")
print("=" * 60)

M.map = asym_map.clone()
symmetric_map = map_sym(M.map, apply_symmetry=True)
print(f"Full symmetric map sum: {symmetric_map.sum():.2f}")
print(f"Full symmetric map max: {symmetric_map.max():.4f}")
print(f"Non-zero voxels: {(symmetric_map > 0).sum()}")

# Check if sum equals sum of individual mates
expected_sum = sum(M.map.sum() for _ in range(map_sym.n_ops))
print(f"\nExpected sum (asym × n_ops): {asym_map.sum() * map_sym.n_ops:.2f}")
print(f"Actual sum: {symmetric_map.sum():.2f}")
print(f"Ratio: {symmetric_map.sum() / (asym_map.sum() * map_sym.n_ops):.4f}")

# Save full symmetric map
M.map = symmetric_map
symm_file = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/symmetric_full.ccp4'
M.save_map(symm_file)
print(f"\n✓ Saved full symmetric map to: {symm_file}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY - Files created for visual inspection:")
print("=" * 60)
print(f"1. {asym_file}")
print(f"   - Original asymmetric unit")
for i in range(map_sym.n_ops):
    print(f"{i+2}. /das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/symmetry_mate_{i}.ccp4")
    print(f"   - Symmetry mate {i}")
print(f"{map_sym.n_ops+2}. {symm_file}")
print(f"   - Full symmetric map (sum of all mates)")

print("\n" + "=" * 60)
print("Load these in Coot/PyMOL/Chimera to visually verify symmetry!")
print("=" * 60)
