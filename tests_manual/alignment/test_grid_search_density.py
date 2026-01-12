"""
Test grid search at different angular step densities to find minimum required for convergence.
"""
import sys
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from torchref.alignment.align import PattersonAligner
from torchref.io import ReflectionData
from torchref.model.model_ft import ModelFT as Model
import torch
import numpy as np

pdb = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/alignment/dark.pdb'
mtz = '/das/work/p17/p17490/Peter/Library/torchref/tests_manual/alignment/dark.mtz'

print("Loading model and data...")
M = Model().load_pdb(pdb)
D = ReflectionData().load_mtz(mtz)

# Remove waters for cleaner test
M = M.select('not resname HOH')

def rotation_error_degrees(R_true, R_found):
    """Calculate angular error between two rotation matrices."""
    R_error = R_found @ R_true.T
    trace = torch.trace(R_error)
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle_rad = torch.acos(cos_angle)
    return angle_rad.item() * 180 / np.pi

def random_rotation_uniform():
    """Generate a random rotation matrix uniformly distributed on SO(3)."""
    from torchref.math_functions.math_torch import random_rotation_uniform as _random_rotation_uniform
    return _random_rotation_uniform(1, dtype=torch.float64)

# Test with fewer trials for faster feedback
n_trials = 3
angular_steps = [30.0, 20.0, 15.0, 10.0]

print("\n" + "="*70)
print("Grid Search Density Test")
print("="*70)
print(f"Testing {len(angular_steps)} angular step sizes with {n_trials} random rotations each")

# First, show grid sizes for each angular step
print("\nGrid sizes:")
aligner_temp = PattersonAligner(data=D, model=M, verbose=0)
for step in angular_steps:
    rotations = aligner_temp.generate_rotation_grid(step)
    print(f"  {step:5.1f}° step: {rotations.shape[0]:>6} rotations")

results = {step: [] for step in angular_steps}

for trial in range(n_trials):
    # Generate random rotation
    torch.manual_seed(42 + trial)
    R_true = random_rotation_uniform()

    print(f"\n--- Trial {trial + 1}/{n_trials} ---")

    # Apply rotation to model
    M_rotated = M.copy()
    xyz_original = M.xyz().to(torch.float64)
    M_rotated.xyz[:] = xyz_original @ R_true.T

    # Create aligner
    aligner = PattersonAligner(data=D, model=M, verbose=0)

    for angular_step in angular_steps:
        print(f"  Testing step={angular_step}°...", end=" ", flush=True)

        # Run grid search alignment
        aligned_model, result = aligner.align_grid_search(
            model=M_rotated,
            angular_step=angular_step,
            n_refine=5,
            n_vectors=500,
            seed=42
        )

        # Calculate rotation error
        R_found = result.rotation.to(torch.float64)
        error_deg = rotation_error_degrees(R_true, R_found)
        results[angular_step].append(error_deg)

        print(f"error = {error_deg:6.2f}°, score = {result.score:.4f}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Step (°)':<10} {'Mean Error (°)':<15} {'Max Error (°)':<15} {'Success (<10°)':<15}")
print("-"*55)
for angular_step in angular_steps:
    errors = results[angular_step]
    mean_err = np.mean(errors)
    max_err = np.max(errors)
    success_rate = sum(1 for e in errors if e < 10) / len(errors) * 100
    print(f"{angular_step:<10.1f} {mean_err:<15.2f} {max_err:<15.2f} {success_rate:<15.0f}%")

print("\nConclusion:")
best_step = None
for step in reversed(angular_steps):  # Start from coarsest
    success_rate = sum(1 for e in results[step] if e < 10) / len(results[step]) * 100
    if success_rate >= 80:
        best_step = step
        break

if best_step:
    print(f"  Recommended angular step: {best_step}° (achieves >=80% success)")
else:
    print("  No angular step achieved 80% success rate. Consider finer grid (5°).")
