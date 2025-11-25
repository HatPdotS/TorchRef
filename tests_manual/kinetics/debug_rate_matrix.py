#!/usr/bin/env python
"""
Check the rate matrix structure.
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

timepoints = torch.logspace(-9,-2,10)  # Just a few points

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
    verbose=1
)

print("\n" + "="*70)
print("RATE MATRIX INSPECTION")
print("="*70)

# Build rate matrix
rate_constants = torch.exp(kinetic_model.log_rate_constants)
efficiencies = torch.sigmoid(kinetic_model.logit_efficiencies)
K = kinetic_model._build_rate_matrix(rate_constants, efficiencies)

print(f"\nRate matrix shape: {K.shape}")
print(f"States: {kinetic_model.states}")
print(f"Number of states: {kinetic_model.n_states}")

print(f"\nRate matrix K:")
K_np = K.detach().numpy()
# Print with state labels
print("        ", end="")
for state in kinetic_model.states:
    print(f"{state:>12}", end="")
print()

for i, state in enumerate(kinetic_model.states):
    print(f"{state:>6}: ", end="")
    for j in range(len(kinetic_model.states)):
        print(f"{K_np[i, j]:12.3e}", end="")
    print()

# Check conservation
print(f"\nColumn sums (should be ~0 for conservation):")
col_sums = K.sum(dim=0).detach().numpy()
for i, state in enumerate(kinetic_model.states):
    print(f"  {state}: {col_sums[i]:.6e}")

# Check if A* has any outgoing transitions
A_star_idx = kinetic_model.state_to_idx['A*']
print(f"\nA* (index {A_star_idx}) row (incoming transitions):")
print(K_np[A_star_idx, :])
print(f"\nA* (index {A_star_idx}) column (outgoing transitions):")
print(K_np[:, A_star_idx])

print(f"\nA* diagonal (total outgoing): {K_np[A_star_idx, A_star_idx]:.6e}")
if abs(K_np[A_star_idx, A_star_idx]) < 1e-10:
    print("✓ A* has no outgoing transitions (sink state)")
else:
    print("❌ A* has outgoing transitions!")

# Test forward evolution at one timepoint
print("\n" + "="*70)
print("TEST SINGLE TIMEPOINT EVOLUTION")
print("="*70)

t_test = 1e-8  # Very early time
exp_Kt = torch.matrix_exp(K * t_test)
P0 = kinetic_model.initial_populations
P_t = exp_Kt @ P0

print(f"\nAt t = {t_test:.2e}:")
print(f"  Initial: {P0.numpy()}")
print(f"  Final: {P_t.detach().numpy()}")
print(f"  Sum: {P_t.sum():.10f}")

# Test at long time
t_long = 1e-2
exp_Kt_long = torch.matrix_exp(K * t_long)
P_long = exp_Kt_long @ P0

print(f"\nAt t = {t_long:.2e}:")
print(f"  Populations: {P_long.detach().numpy()}")
print(f"  Sum: {P_long.sum():.10f}")

if P_long.sum() > 1.01:
    print("❌ POPULATION NOT CONSERVED!")
else:
    print("✓ Population conserved")
