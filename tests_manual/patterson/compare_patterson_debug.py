"""
Diagnostic script to compare Patterson maps between torchref and CCTBX.
"""
from torchref.io import ReflectionData
import numpy as np
import torch

mtz_path = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/patterson/dark.mtz'

# Load data with torchref
data = ReflectionData(verbose=0).load_mtz(mtz_path)

# Load data with CCTBX
from iotbx import mtz
from cctbx import miller
from cctbx.array_family import flex

mtz_obj = mtz.object(mtz_path)
miller_arrays = mtz_obj.as_miller_arrays()

# Print available miller arrays to understand what we're comparing
print("Available miller arrays:")
for i, arr in enumerate(miller_arrays):
    print(f"  [{i}] {arr.info().label_string()} - type: {arr.observation_type()}, size: {arr.size()}")

# Find amplitude array (F-obs) - use index 2 which has F-obs-filtered
# Index 0 is I-obs (intensities), which is different data!
f_obs = miller_arrays[2]  # F-obs-filtered - actual amplitudes
print(f"\nUsing array: {f_obs.info().label_string()}")
print(f"  Is anomalous: {f_obs.anomalous_flag()}")
print(f"  Space group: {f_obs.space_group().info()}")

# Get F values from CCTBX
cctbx_f = f_obs.data().as_numpy_array()
cctbx_hkl = np.array(f_obs.indices())

print(f"\nCCTBX F statistics:")
print(f"  N reflections: {len(cctbx_f)}")
print(f"  F range: {cctbx_f.min():.2f} - {cctbx_f.max():.2f}")
print(f"  F mean: {cctbx_f.mean():.2f}")

# Get F values from torchref (after fill + expand_to_p1)
print(f"\nTorchref original data:")
print(f"  N reflections: {len(data.F)}")
print(f"  NaN count: {torch.isnan(data.F).sum().item()}")
valid_F = data.F[~torch.isnan(data.F)]
print(f"  Valid F range: {valid_F.min().item():.2f} - {valid_F.max().item():.2f}")
print(f"  Valid F mean: {valid_F.mean().item():.2f}")

# Check the mask
mask = data.masks()
print(f"  Valid mask count: {mask.sum().item()}")
print(f"  Masked F range: {data.F[mask].min().item():.2f} - {data.F[mask].max().item():.2f}")

# Trace through the torchref Patterson calculation
print("\n--- Tracing torchref Patterson calculation ---")
from torchref.math_functions.math_torch import place_on_grid, find_grid_size

# Step 1: fill + expand_to_p1
data_filled = data.fill()
print(f"After fill(): {len(data_filled.F)} reflections")
data_p1 = data_filled.expand_to_p1()
print(f"After expand_to_p1(): {len(data_p1.F)} reflections")
print(f"  Spacegroup: {data_p1.spacegroup}")

# Step 2: data_fill_masked
hkl_p1, F_p1, _, _ = data_p1.data_fill_masked()
print(f"After data_fill_masked(): {len(F_p1)} reflections")
print(f"  F range: {F_p1.min().item():.2f} - {F_p1.max().item():.2f}")
print(f"  F mean: {F_p1.mean().item():.2f}")
print(f"  F² sum (should be ~ Patterson origin): {(F_p1**2).sum().item():.2e}")

# For comparison, compute what CCTBX should have
cctbx_f_sq_sum = (cctbx_f**2).sum()
print(f"  CCTBX F² sum: {cctbx_f_sq_sum:.2e}")

# Try computing Patterson without fill() - just expand_to_p1()
print("\n--- Computing Patterson without fill() ---")
data_p1_nofill = data.expand_to_p1()
print(f"After expand_to_p1() only: {len(data_p1_nofill.F)} reflections")
hkl_nofill, F_nofill, _, _ = data_p1_nofill.data_fill_masked()
print(f"After data_fill_masked: {len(F_nofill)} reflections")
print(f"  F² sum: {(F_nofill**2).sum().item():.2e}")

# Now compute Patterson maps
print("\n--- Computing Patterson maps ---")

# CCTBX Patterson
patterson_cctbx_obj = f_obs.patterson_map(symmetry_flags=None)
patterson_cctbx = patterson_cctbx_obj.real_map_unpadded().as_numpy_array()

print(f"\nCCTBX Patterson:")
print(f"  Shape: {patterson_cctbx.shape}")
print(f"  Origin value: {patterson_cctbx[0, 0, 0]:.2e}")
print(f"  Max value: {patterson_cctbx.max():.2e}")
print(f"  Min value: {patterson_cctbx.min():.2e}")

# Torchref Patterson with same grid size
patterson_torchref = data.calc_patterson(patterson_cctbx.shape).detach().cpu().numpy()

# Also compute Patterson directly from CCTBX F values using torchref machinery
# This isolates whether the issue is F values or the FFT code
print("\n--- Computing Patterson with CCTBX F values via torchref ---")
# Create a simple ReflectionData with CCTBX values
from torchref.io import ReflectionData as RD
data_cctbx_style = RD(verbose=0)
data_cctbx_style.hkl = torch.tensor(cctbx_hkl, dtype=torch.int32)
data_cctbx_style.F = torch.tensor(cctbx_f, dtype=torch.float32)
data_cctbx_style.F_sigma = torch.tensor(np.ones_like(cctbx_f), dtype=torch.float32)
data_cctbx_style.cell = data.cell.clone()
data_cctbx_style.spacegroup = data.spacegroup
data_cctbx_style._calculate_resolution()
# Note: this won't have proper masks set up, so we need to set up basic mask
data_cctbx_style.masks['sanity_F'] = torch.ones(len(cctbx_f), dtype=torch.bool)
data_cctbx_style.masks['flagged_initial'] = torch.ones(len(cctbx_f), dtype=torch.bool)
data_cctbx_style.masks['flagged_sigma'] = torch.ones(len(cctbx_f), dtype=torch.bool)

try:
    patterson_from_cctbx_F = data_cctbx_style.calc_patterson(patterson_cctbx.shape).detach().cpu().numpy()
    print(f"  Shape: {patterson_from_cctbx_F.shape}")
    print(f"  Origin value: {patterson_from_cctbx_F[0, 0, 0]:.2e}")
    corr_cctbx_style = np.corrcoef(patterson_cctbx.flatten(), patterson_from_cctbx_F.flatten())[0, 1]
    print(f"  Correlation with CCTBX Patterson: {corr_cctbx_style:.4f}")
except Exception as e:
    print(f"  Error: {e}")

print(f"\nTorchref Patterson:")
print(f"  Shape: {patterson_torchref.shape}")
print(f"  Origin value: {patterson_torchref[0, 0, 0]:.2e}")
print(f"  Max value: {patterson_torchref.max():.2e}")
print(f"  Min value: {patterson_torchref.min():.2e}")

# Compute correlation
corr = np.corrcoef(patterson_cctbx.flatten(), patterson_torchref.flatten())[0, 1]
print(f"\nCorrelation: {corr:.4f}")

# Check if maps are simply scaled differently
# Normalize both maps and check correlation
cctbx_norm = (patterson_cctbx - patterson_cctbx.mean()) / patterson_cctbx.std()
torchref_norm = (patterson_torchref - patterson_torchref.mean()) / patterson_torchref.std()
corr_norm = np.corrcoef(cctbx_norm.flatten(), torchref_norm.flatten())[0, 1]
print(f"Correlation (normalized): {corr_norm:.4f}")

# Check if one map is shifted relative to the other
print("\n--- Checking for origin shifts ---")
# Find indices of max value in each map
max_idx_cctbx = np.unravel_index(np.argmax(patterson_cctbx), patterson_cctbx.shape)
max_idx_torchref = np.unravel_index(np.argmax(patterson_torchref), patterson_torchref.shape)
print(f"CCTBX max at: {max_idx_cctbx}")
print(f"Torchref max at: {max_idx_torchref}")

# Check values at specific positions
print("\n--- Sample values at various positions ---")
positions = [(0, 0, 0), (10, 10, 10), (50, 50, 50), (67, 90, 75)]  # origin and some others
for pos in positions:
    if all(p < s for p, s in zip(pos, patterson_cctbx.shape)):
        c_val = patterson_cctbx[pos]
        t_val = patterson_torchref[pos]
        print(f"  {pos}: CCTBX={c_val:.2e}, torchref={t_val:.2e}, ratio={c_val/t_val if t_val != 0 else 'inf':.4f}")

# Try flipping/shifting to see if there's a coordinate transformation
print("\n--- Testing coordinate transformations ---")

# Test flip along each axis
for ax in [0, 1, 2]:
    flipped = np.flip(patterson_torchref, axis=ax)
    corr_flip = np.corrcoef(patterson_cctbx.flatten(), flipped.flatten())[0, 1]
    print(f"  Flip axis {ax}: corr = {corr_flip:.4f}")

# Test flip along all axes
flipped_all = np.flip(patterson_torchref)
corr_flip_all = np.corrcoef(patterson_cctbx.flatten(), flipped_all.flatten())[0, 1]
print(f"  Flip all axes: corr = {corr_flip_all:.4f}")

# Test fftshift
shifted = np.fft.fftshift(patterson_torchref)
corr_shifted = np.corrcoef(patterson_cctbx.flatten(), shifted.flatten())[0, 1]
print(f"  fftshift: corr = {corr_shifted:.4f}")

# Test roll by half grid
roll_half = np.roll(patterson_torchref,
                    [s//2 for s in patterson_torchref.shape],
                    axis=(0, 1, 2))
corr_roll = np.corrcoef(patterson_cctbx.flatten(), roll_half.flatten())[0, 1]
print(f"  Roll by half grid: corr = {corr_roll:.4f}")

# Test combined operations
flipped_shifted = np.fft.fftshift(np.flip(patterson_torchref))
corr_combined = np.corrcoef(patterson_cctbx.flatten(), flipped_shifted.flatten())[0, 1]
print(f"  Flip all + fftshift: corr = {corr_combined:.4f}")
