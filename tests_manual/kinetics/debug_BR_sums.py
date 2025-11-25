#!/usr/bin/env python
"""
Debug the BR-like kinetics to see what's wrong.
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

timepoints = torch.logspace(-9,-2,10000) 

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
    light_activated=True
)

print("\nStates:", kinetic_model.states)
print("Transitions:", kinetic_model.transitions)

# Compute populations
pops = kinetic_model().detach().numpy()

print(f"\nPopulation statistics:")
print(f"  Shape: {pops.shape}")
print(f"  Min: {pops.min():.6e}")
print(f"  Max: {pops.max():.6e}")

# Check sums at different times
t = timepoints.numpy()
pop_sums = pops.sum(axis=1)

print(f"\nPopulation sums:")
print(f"  At t={t[0]:.2e}: sum = {pop_sums[0]:.10f}")
print(f"  At t={t[len(t)//4]:.2e}: sum = {pop_sums[len(t)//4]:.10f}")
print(f"  At t={t[len(t)//2]:.2e}: sum = {pop_sums[len(t)//2]:.10f}")
print(f"  At t={t[-1]:.2e}: sum = {pop_sums[-1]:.10f}")

print(f"\n  Min sum: {pop_sums.min():.10f}")
print(f"  Max sum: {pop_sums.max():.10f}")

# Check final populations
print(f"\nFinal populations (t={t[-1]:.2e}):")
for i, state in enumerate(kinetic_model.states):
    print(f"  {state}: {pops[-1, i]:.6f}")

print(f"\nTotal: {pops[-1, :].sum():.6f}")

# Check if rescaling is working
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

if pop_sums[-1] > 1.01:
    print("❌ Populations NOT summing to 1 at the end!")
    print("   The rescaling is not working correctly.")
elif pop_sums.max() > 1.5:
    print("❌ Populations exceed 1.5 somewhere!")
    idx_max = pop_sums.argmax()
    print(f"   Max at t={t[idx_max]:.2e}: sum={pop_sums[idx_max]:.6f}")
    print(f"   Populations: {pops[idx_max, :]}")
else:
    print("✓ Populations appear reasonable")

# Check individual state maxima
print("\nIndividual state maxima:")
for i, state in enumerate(kinetic_model.states):
    max_pop = pops[:, i].max()
    if max_pop > 1.0:
        print(f"  ❌ {state}: {max_pop:.6f} (exceeds 1.0!)")
    else:
        print(f"  ✓ {state}: {max_pop:.6f}")
