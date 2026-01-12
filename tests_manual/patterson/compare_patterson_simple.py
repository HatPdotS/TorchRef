"""
Simple Patterson comparison using raw FFT without torchref fill/expand machinery.
"""
import numpy as np
import torch

mtz_path = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/patterson/dark.mtz'

# Load data with CCTBX
from iotbx import mtz
from cctbx import miller
from cctbx.array_family import flex

mtz_obj = mtz.object(mtz_path)
miller_arrays = mtz_obj.as_miller_arrays()

# Use amplitude array
f_obs = miller_arrays[2]  # F-obs-filtered
print(f"Using: {f_obs.info().label_string()}")
print(f"  N reflections: {f_obs.size()}")
print(f"  Space group: {f_obs.space_group().info()}")

# Get data
cctbx_f = f_obs.data().as_numpy_array()
cctbx_hkl = np.array(f_obs.indices())

# CCTBX Patterson
patterson_cctbx_obj = f_obs.patterson_map(symmetry_flags=None)
patterson_cctbx = patterson_cctbx_obj.real_map_unpadded().as_numpy_array()
grid_size = patterson_cctbx.shape

print(f"\nCCTBX Patterson:")
print(f"  Shape: {grid_size}")
print(f"  Origin: {patterson_cctbx[0,0,0]:.4e}")
print(f"  Max: {patterson_cctbx.max():.4e}")

# Get unit cell from CCTBX
uc = f_obs.unit_cell()
cell_params = list(uc.parameters())
print(f"  Unit cell: {cell_params}")

# Now compute Patterson directly using numpy FFT
# Step 1: Expand to P1 using CCTBX's symmetry operations
print("\n--- Manual Patterson calculation ---")

# Use CCTBX to expand to P1
f_obs_p1 = f_obs.expand_to_p1()
print(f"After P1 expansion: {f_obs_p1.size()} reflections")

hkl_p1 = np.array(f_obs_p1.indices())
f_p1 = f_obs_p1.data().as_numpy_array()

print(f"  F² sum: {(f_p1**2).sum():.4e}")

# Step 2: Place F² on grid
Nx, Ny, Nz = grid_size
grid = np.zeros((Nx, Ny, Nz), dtype=np.complex128)

h, k, l = hkl_p1[:, 0], hkl_p1[:, 1], hkl_p1[:, 2]
F_sq = f_p1**2

# Map negative indices using modular arithmetic
hi = np.mod(h, Nx)
ki = np.mod(k, Ny)
li = np.mod(l, Nz)

# Add F² values to grid
for i in range(len(f_p1)):
    grid[hi[i], ki[i], li[i]] += F_sq[i]

print(f"  Grid F² sum: {grid.real.sum():.4e}")

# Step 3: FFT with no normalization (like CCTBX)
# numpy ifft normalizes by 1/N by default, so we multiply by N
patterson_manual = np.fft.ifftn(grid).real * (Nx * Ny * Nz)

print(f"\nManual Patterson (numpy):")
print(f"  Shape: {patterson_manual.shape}")
print(f"  Origin: {patterson_manual[0,0,0]:.4e}")
print(f"  Max: {patterson_manual.max():.4e}")

# Compare
corr = np.corrcoef(patterson_cctbx.flatten(), patterson_manual.flatten())[0, 1]
print(f"\nCorrelation (CCTBX vs manual): {corr:.6f}")

# Check origin ratio
origin_ratio = patterson_cctbx[0,0,0] / patterson_manual[0,0,0]
print(f"Origin ratio (CCTBX/manual): {origin_ratio:.4f}")

# Now also try with torchref FFT code
print("\n--- Torchref FFT comparison ---")
from torchref.math_functions.math_torch import place_on_grid

hkl_torch = torch.tensor(hkl_p1, dtype=torch.int32)
F_sq_torch = torch.tensor(F_sq, dtype=torch.float32)

grid_torch = place_on_grid(hkl_torch, F_sq_torch, grid_size, enforce_hermitian=False)
patterson_torch = torch.fft.ifftn(grid_torch, dim=(0, 1, 2), norm="forward").real.numpy()

print(f"Torchref FFT Patterson:")
print(f"  Origin: {patterson_torch[0,0,0]:.4e}")
print(f"  Max: {patterson_torch.max():.4e}")

corr_torch = np.corrcoef(patterson_cctbx.flatten(), patterson_torch.flatten())[0, 1]
print(f"\nCorrelation (CCTBX vs torchref FFT): {corr_torch:.6f}")

# Now test with torchref P1 expansion
print("\n--- Testing torchref P1 expansion ---")
from torchref.io import ReflectionData

data = ReflectionData(verbose=0).load_mtz(mtz_path)
data_p1 = data.expand_to_p1()

# Get the data after P1 expansion
hkl_torchref, F_torchref, _, _ = data_p1.data_fill_masked()
hkl_torchref = hkl_torchref.numpy()
F_torchref = F_torchref.numpy()

print(f"Torchref P1 expansion: {len(F_torchref)} reflections")
print(f"  F² sum: {(F_torchref**2).sum():.4e}")

# Check if same HKL indices
# Compare by sorting both
hkl_cctbx_set = set(tuple(x) for x in hkl_p1)
hkl_torchref_set = set(tuple(x) for x in hkl_torchref)

common = len(hkl_cctbx_set & hkl_torchref_set)
only_cctbx = len(hkl_cctbx_set - hkl_torchref_set)
only_torchref = len(hkl_torchref_set - hkl_cctbx_set)

print(f"  Common HKL indices: {common}")
print(f"  Only in CCTBX: {only_cctbx}")
print(f"  Only in torchref: {only_torchref}")
