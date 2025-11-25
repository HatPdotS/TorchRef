"""
Debug why identity operation doesn't return identical map.
"""
import torch
import torch.nn.functional as F
import numpy as np
from torchref.map_symmetry import MapSymmetry

# Simple test case
print("=" * 70)
print("DEBUG: Identity Operation Test")
print("=" * 70)

# Create a simple test map
map_shape = (16, 16, 16)
cell = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0])

# Create P21 symmetry
sym = MapSymmetry('P21', map_shape, cell)

# Create a test map with a single peak
test_map = torch.zeros(map_shape, dtype=torch.float64)
test_map[8, 8, 8] = 100.0  # Peak at center

print(f"Original map:")
print(f"  Sum: {test_map.sum():.6f}")
print(f"  Max: {test_map.max():.6f}")
print(f"  Max position: {test_map.argmax().item()}")

# Apply identity operation (operation 0)
mate_0 = sym.get_symmetry_mate(test_map, 0)

print(f"\nIdentity mate (operation 0):")
print(f"  Sum: {mate_0.sum():.6f}")
print(f"  Max: {mate_0.max():.6f}")
print(f"  Max position: {mate_0.argmax().item()}")

# Check difference
diff = (test_map - mate_0).abs()
print(f"\nDifference:")
print(f"  Max diff: {diff.max():.10f}")
print(f"  Mean diff: {diff.mean():.10f}")
print(f"  Are they close? {torch.allclose(test_map, mate_0)}")

# Let's check the sampling grid for operation 0
sampling_grid_0 = sym.sampling_grids[0]
print(f"\nSampling grid for operation 0:")
print(f"  Shape: {sampling_grid_0.shape}")
print(f"  Min: {sampling_grid_0.min():.6f}")
print(f"  Max: {sampling_grid_0.max():.6f}")

# For identity, the sampling grid should map each voxel to itself
# Let's check a few points
nx, ny, nz = map_shape
print(f"\nChecking sampling coordinates:")
print(f"  Expected: identity mapping, each voxel samples itself")

# Create what the identity grid SHOULD be
# For grid_sample, coordinates are in [-1, 1] range
# For a grid of size N, voxel i should sample at: 2*(i+0.5)/N - 1

fx_expected = 2.0 * (torch.arange(nx, dtype=torch.float64) + 0.5) / nx - 1.0
fy_expected = 2.0 * (torch.arange(ny, dtype=torch.float64) + 0.5) / ny - 1.0
fz_expected = 2.0 * (torch.arange(nz, dtype=torch.float64) + 0.5) / nz - 1.0

grid_fz_exp, grid_fy_exp, grid_fx_exp = torch.meshgrid(fz_expected, fy_expected, fx_expected, indexing='ij')
expected_grid = torch.stack([grid_fz_exp, grid_fy_exp, grid_fx_exp], dim=-1)

print(f"\nExpected identity grid:")
print(f"  Shape: {expected_grid.shape}")
print(f"  Min: {expected_grid.min():.6f}")
print(f"  Max: {expected_grid.max():.6f}")

# Compare
grid_diff = (sampling_grid_0 - expected_grid).abs()
print(f"\nDifference from expected:")
print(f"  Max diff: {grid_diff.max():.10f}")
print(f"  Mean diff: {grid_diff.mean():.10f}")

# Check a few specific points
print(f"\nSample coordinate checks:")
for i, j, k in [(0, 0, 0), (8, 8, 8), (15, 15, 15)]:
    actual = sampling_grid_0[i, j, k]
    expected = expected_grid[i, j, k]
    print(f"  [{i},{j},{k}]: actual={actual}, expected={expected}")

# Now let's test grid_sample directly with identity coordinates
print("\n" + "=" * 70)
print("Testing grid_sample with identity coordinates")
print("=" * 70)

map_5d = test_map.unsqueeze(0).unsqueeze(0)
expected_grid_batch = expected_grid.unsqueeze(0)

result_identity = F.grid_sample(
    map_5d,
    expected_grid_batch,
    mode='bilinear',
    padding_mode='zeros',
    align_corners=False
).squeeze()

print(f"Result with expected identity grid:")
print(f"  Sum: {result_identity.sum():.6f}")
print(f"  Max: {result_identity.max():.6f}")
print(f"  Diff from original: {(test_map - result_identity).abs().max():.10f}")
print(f"  Are they close? {torch.allclose(test_map, result_identity)}")
