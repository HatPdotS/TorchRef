#!/usr/bin/env python
"""
Test with manual rate constants that might cause issues.
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

print("\n" + "="*70)
print("TEST: MANUAL RATE CONSTANTS WITH LARGE TIMEPOINTS")
print("="*70 + "\n")

# Case 1: Fast rates with large timepoints
print("Case 1: Fast rates (k=100) with large timepoints")
print("-"*70)

t1 = np.logspace(-1, 6, 100)
print(f"Timepoints: {t1.min():.2e} to {t1.max():.2e} ps")

try:
    model1 = KineticModel(
        flow_chart="A->B,B->C",
        timepoints=t1,
        rate_constants={'A->B': 100.0, 'B->C': 50.0},  # Fast rates
        instrument_function='none',
        verbose=0
    )
    
    pops1 = model1().detach().numpy()
    print(f"Population min: {pops1.min():.6e}")
    print(f"Population max: {pops1.max():.6e}")
    print(f"Sum range: {pops1.sum(axis=1).min():.6f} to {pops1.sum(axis=1).max():.6f}")
    
    if pops1.max() > 1e10:
        print("✗ NUMERICAL INSTABILITY!")
    else:
        print("✓ OK")
except Exception as e:
    print(f"✗ ERROR: {e}")

print()

# Case 2: Very fast rates with very large timepoints
print("Case 2: Very fast rates (k=1000) with extreme timepoints")
print("-"*70)

t2 = np.logspace(0, 9, 100)  # 1 ps to 1 second (1e9 ps)
print(f"Timepoints: {t2.min():.2e} to {t2.max():.2e} ps")

try:
    model2 = KineticModel(
        flow_chart="A->B,B->C",
        timepoints=t2,
        rate_constants={'A->B': 1000.0, 'B->C': 500.0},
        instrument_function='none',
        verbose=0
    )
    
    pops2 = model2().detach().numpy()
    print(f"Population min: {pops2.min():.6e}")
    print(f"Population max: {pops2.max():.6e}")
    print(f"Sum range: {pops2.sum(axis=1).min():.6f} to {pops2.sum(axis=1).max():.6f}")
    
    if pops2.max() > 1e10:
        print("✗ NUMERICAL INSTABILITY!")
        # Find where it breaks
        for i in range(len(t2)):
            if pops2[i, :].max() > 1e10:
                print(f"  Breaks at t = {t2[i]:.6e} ps")
                print(f"  Populations: {pops2[i, :]}")
                break
    else:
        print("✓ OK")
except Exception as e:
    print(f"✗ ERROR: {e}")

print()

# Case 3: Check what happens without baseline
print("Case 3: Without baseline initialization")
print("-"*70)

# Temporarily test without baseline by checking old behavior
# Let's compute matrix exponential directly
K = torch.tensor([
    [-10.0, 0.0],
    [10.0, -5.0]
])

print("Rate matrix K:")
print(K.numpy())

test_times = [1e0, 1e3, 1e6, 1e9]
print("\nMatrix exponential for different times:")
for t_val in test_times:
    exp_Kt = torch.matrix_exp(K * t_val)
    print(f"  t = {t_val:.0e}: max = {exp_Kt.abs().max():.6e}, sum = {exp_Kt.sum():.6e}")
    if exp_Kt.abs().max() > 1e10:
        print(f"    ✗ OVERFLOW at t = {t_val:.0e}!")

print("\n" + "="*70)
