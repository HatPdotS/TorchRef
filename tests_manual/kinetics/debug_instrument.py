#!/usr/bin/env python
"""
Debug: check populations before and after instrument function.
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

print("\n" + "="*70)
print("DEBUG: POPULATIONS BEFORE/AFTER INSTRUMENT FUNCTION")
print("="*70 + "\n")

t = np.linspace(-0.5, 15, 300)

model = KineticModel(
    flow_chart="A->B,B->C,C->D",
    timepoints=t,
    instrument_function='gaussian',
    instrument_width=0.2,
    verbose=0
)

# Get rate matrix
rate_constants = torch.exp(model.log_rate_constants)
efficiencies = torch.sigmoid(model.logit_efficiencies)
rate_matrix = model._build_rate_matrix(rate_constants, efficiencies)

print("Rate matrix:")
print(rate_matrix.detach().numpy())
print()

# Solve kinetics (before instrument function)
pops_before = model._solve_kinetics(rate_matrix).detach().numpy()

print(f"Populations BEFORE instrument function:")
print(f"  Shape: {pops_before.shape}")
print(f"  Min: {pops_before.min():.6f}")
print(f"  Max: {pops_before.max():.6f}")
print(f"  Sum (per timepoint): min={pops_before.sum(axis=1).min():.6f}, max={pops_before.sum(axis=1).max():.6f}")
print()

# Apply instrument function
pops_after = model._apply_instrument_function(torch.tensor(pops_before)).detach().numpy()

print(f"Populations AFTER instrument function:")
print(f"  Shape: {pops_after.shape}")
print(f"  Min: {pops_after.min():.6f}")
print(f"  Max: {pops_after.max():.6f}")
print(f"  Sum (per timepoint): min={pops_after.sum(axis=1).min():.6f}, max={pops_after.sum(axis=1).max():.6f}")
print()

# Add baselines (should be zeros)
baselines = model._baseline_occupancies.detach().numpy()
print(f"Baselines: {baselines}")

pops_final = pops_after + baselines
print(f"\nFinal populations (after baseline):")
print(f"  Min: {pops_final.min():.6f}")
print(f"  Max: {pops_final.max():.6f}")
print(f"  Sum (per timepoint): min={pops_final.sum(axis=1).min():.6f}, max={pops_final.sum(axis=1).max():.6f}")
