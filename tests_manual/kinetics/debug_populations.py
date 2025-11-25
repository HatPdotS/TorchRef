#!/usr/bin/env python
"""
Debug the population calculation issue.
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

print("\n" + "="*70)
print("DEBUG POPULATION CALCULATION")
print("="*70 + "\n")

t = np.linspace(-0.5, 15, 300)

model = KineticModel(
    flow_chart="A->B,B->C,C->D",
    timepoints=t,
    instrument_function='gaussian',
    instrument_width=0.2,
    verbose=0
)

# Get populations
populations = model().detach().numpy()

print(f"Population shape: {populations.shape}")
print(f"Population min: {populations.min():.6f}")
print(f"Population max: {populations.max():.6f}")
print(f"Population sum (per timepoint): min={populations.sum(axis=1).min():.6f}, max={populations.sum(axis=1).max():.6f}")
print()

# Check individual states
for i, state in enumerate(model.states):
    print(f"State {state}:")
    print(f"  min={populations[:, i].min():.6f}, max={populations[:, i].max():.6f}")
    
print("\nFirst few timepoints:")
for i in range(5):
    print(f"t={t[i]:.3f}: {populations[i, :]}")
