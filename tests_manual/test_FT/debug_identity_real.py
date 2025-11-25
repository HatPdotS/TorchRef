"""
Debug identity operation on the actual tubulin map.
"""
import torch
from torchref.model_ft import ModelFT
from torchref.map_symmetry import MapSymmetry

print("=" * 70)
print("DEBUG: Identity Operation on Real Map")
print("=" * 70)

# Load structure and setup
M = ModelFT()   
M.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_FT/dark.pdb')
M.max_res = 1.2
M.setup_grid()

# Build asymmetric unit map
M.build_density_map(apply_symmetry=False)
map_original = M.map.clone()

print(f"\nOriginal map info:")
print(f"  Shape: {map_original.shape}")
print(f"  Dtype: {map_original.dtype}")
print(f"  Device: {map_original.device}")
print(f"  Sum: {map_original.sum():.6f}")
print(f"  Max: {map_original.max():.6f}")
print(f"  Min: {map_original.min():.6f}")

# Create symmetry object
sym = MapSymmetry(M.spacegroup, map_original.shape, M.cell)

print(f"\nSymmetry object info:")
print(f"  Grid frac dtype: {sym.grid_frac.dtype}")
print(f"  Grid frac device: {sym.grid_frac.device}")
print(f"  Sampling grids dtype: {sym.sampling_grids.dtype}")
print(f"  Sampling grids device: {sym.sampling_grids.device}")

# Apply identity operation
print("\n" + "-" * 70)
print("Applying identity operation (mate 0)")
print("-" * 70)

mate_0 = sym.get_symmetry_mate(map_original, 0)

print(f"\nIdentity mate info:")
print(f"  Shape: {mate_0.shape}")
print(f"  Dtype: {mate_0.dtype}")
print(f"  Device: {mate_0.device}")
print(f"  Sum: {mate_0.sum():.6f}")
print(f"  Max: {mate_0.max():.6f}")
print(f"  Min: {mate_0.min():.6f}")

# Compare
print("\n" + "-" * 70)
print("Comparison")
print("-" * 70)

diff = (map_original - mate_0).abs()
print(f"Absolute difference:")
print(f"  Max: {diff.max():.10f}")
print(f"  Mean: {diff.mean():.10f}")
print(f"  Std: {diff.std():.10f}")
print(f"  Sum: {diff.sum():.6f}")

# Count how many voxels are different
threshold = 1e-6
different = diff > threshold
print(f"\nVoxels with |diff| > {threshold}:")
print(f"  Count: {different.sum()}")
print(f"  Percentage: {100 * different.sum().item() / diff.numel():.4f}%")

# Check with different tolerances
for tol in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
    close = torch.allclose(map_original, mate_0, rtol=tol, atol=tol)
    print(f"  allclose(tol={tol}): {close}")

# Check specific statistics
rel_diff = diff / (map_original.abs() + 1e-10)
print(f"\nRelative difference:")
print(f"  Max: {rel_diff.max():.10f}")
print(f"  Mean: {rel_diff.mean():.10f}")

# Find where the biggest differences are
max_diff_idx = diff.argmax()
i, j, k = torch.unravel_index(max_diff_idx, map_original.shape)
print(f"\nLocation of max difference: [{i}, {j}, {k}]")
print(f"  Original value: {map_original[i, j, k]:.10f}")
print(f"  Identity mate value: {mate_0[i, j, k]:.10f}")
print(f"  Difference: {diff[i, j, k]:.10f}")

# Check if it's an interpolation issue
print("\n" + "-" * 70)
print("Checking grid_sample behavior")
print("-" * 70)

# Check the sampling grid coordinates
sampling_grid_0 = sym.sampling_grids[0]
print(f"Sampling grid 0:")
print(f"  Shape: {sampling_grid_0.shape}")
print(f"  Dtype: {sampling_grid_0.dtype}")
print(f"  Min: {sampling_grid_0.min():.10f}")
print(f"  Max: {sampling_grid_0.max():.10f}")

# Check if any coordinates are exactly at -1 or 1 (boundaries)
at_minus_one = (sampling_grid_0 <= -1.0).sum()
at_plus_one = (sampling_grid_0 >= 1.0).sum()
print(f"  Coords <= -1.0: {at_minus_one}")
print(f"  Coords >= 1.0: {at_plus_one}")

# Sample a few coordinates
nx, ny, nz = map_original.shape
print(f"\nSample coordinates at corners and center:")
for i, j, k in [(0, 0, 0), (nx//2, ny//2, nz//2), (nx-1, ny-1, nz-1)]:
    coords = sampling_grid_0[i, j, k]
    print(f"  [{i:3d},{j:3d},{k:3d}]: {coords}")
