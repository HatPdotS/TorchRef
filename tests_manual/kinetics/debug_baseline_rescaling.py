#!/usr/bin/env python
"""
Check baseline values during execution.
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

timepoints = torch.logspace(-9,-2,100)  # Fewer points for debugging

Br_rate_constants = {
    'A->K': 1e10,
    'K->L': 5e5,
    'L->M': 2.5e4,
    'M->N': 200,
    'N->O': 200,
    'O->A': 200
}

flow_chart = 'A->K,K->L,L->M,M->N,N->O,O->A'

kinetic_model = KineticModel(
    flow_chart=flow_chart,
    timepoints=timepoints,
    instrument_width=1e-8,
    rate_constants=Br_rate_constants,
    light_activated=True,
    verbose=0
)

print("States:", kinetic_model.states)
print("\nBaseline occupancies (_baseline_occupancies):")
print(kinetic_model._baseline_occupancies)
print(f"Sum of baselines: {kinetic_model._baseline_occupancies.sum():.6f}")

# Check if it's a proper buffer
print(f"\nIs registered buffer: {hasattr(kinetic_model, '_baseline_occupancies')}")
print(f"Type: {type(kinetic_model._baseline_occupancies)}")

# Manually compute what should happen
print("\n" + "="*70)
print("MANUAL FORWARD CALCULATION")
print("="*70)

# Get populations
rate_constants = torch.exp(kinetic_model.log_rate_constants)
efficiencies = torch.sigmoid(kinetic_model.logit_efficiencies)
rate_matrix = kinetic_model._build_rate_matrix(rate_constants, efficiencies)
pops_dynamic = kinetic_model._solve_kinetics(rate_matrix)
pops_after_irf = kinetic_model._apply_instrument_function(pops_dynamic)

print(f"\nDynamic populations (after IRF) at t={timepoints[-1]:.2e}:")
print(f"  Sum: {pops_after_irf[-1, :].sum():.6f}")
print(f"  Values: {pops_after_irf[-1, :].numpy()}")

# Now apply baseline rescaling
total_baseline = kinetic_model._baseline_occupancies.sum()
print(f"\nTotal baseline: {total_baseline:.6f}")

reactive_fraction = 1.0 - total_baseline
print(f"Reactive fraction (1 - baseline): {reactive_fraction:.6f}")

pops_rescaled = pops_after_irf[-1, :] * reactive_fraction
print(f"\nAfter rescaling:")
print(f"  Sum: {pops_rescaled.sum():.6f}")
print(f"  Values: {pops_rescaled.numpy()}")

pops_final = pops_rescaled + kinetic_model._baseline_occupancies
print(f"\nAfter adding baseline:")
print(f"  Sum: {pops_final.sum():.6f}")
print(f"  Values: {pops_final.numpy()}")

# Now check actual forward
print("\n" + "="*70)
print("ACTUAL FORWARD RESULT")
print("="*70)

pops_actual = kinetic_model().detach()
print(f"\nActual populations at t={timepoints[-1]:.2e}:")
print(f"  Sum: {pops_actual[-1, :].sum():.6f}")
print(f"  Values: {pops_actual[-1, :].numpy()}")

print("\n" + "="*70)
if abs(pops_actual[-1, :].sum() - 1.0) > 0.01:
    print("❌ MISMATCH - forward() not rescaling correctly!")
else:
    print("✓ OK - populations sum to 1")
