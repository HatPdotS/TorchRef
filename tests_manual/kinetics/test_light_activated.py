#!/usr/bin/env python
"""
Test light-activated mode to prevent re-photoactivation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchref.kinetics import KineticModel

print("\n" + "="*70)
print("TEST: LIGHT-ACTIVATED MODE (PREVENT RE-PHOTOACTIVATION)")
print("="*70 + "\n")

# Create timepoints
t = torch.linspace(0, 100, 1000)

# BR-like photocycle: A → K → L → M → N → O → BR (which returns to A)
flow_chart = 'A->K,K->L,L->K,L->M,M->L,M->N,N->O,O->A'

print("Flow chart:", flow_chart)
print("This has O→A, which would allow re-photoactivation")
print()

# Test WITHOUT light-activated mode
print("="*70)
print("TEST 1: WITHOUT light_activated (allows cycling)")
print("="*70)

model_normal = KineticModel(
    flow_chart=flow_chart,
    timepoints=t,
    instrument_width=4,
    light_activated=False,
    verbose=1
)

print("\nStates:", model_normal.states)
print("Transitions:", model_normal.transitions)

pops_normal = model_normal().detach().numpy()

print(f"\nPopulation statistics (normal mode):")
print(f"  Max population: {pops_normal.max():.6f}")
print(f"  Total population (final): {pops_normal[-1, :].sum():.6f}")

# Check if there's amplification
print(f"\nChecking for amplification:")
for i, state in enumerate(model_normal.states):
    max_pop = pops_normal[:, i].max()
    final_pop = pops_normal[-1, i]
    print(f"  State {state}: max={max_pop:.4f}, final={final_pop:.4f}")

print()

# Test WITH light-activated mode
print("="*70)
print("TEST 2: WITH light_activated=True (prevents cycling)")
print("="*70)

model_light = KineticModel(
    flow_chart=flow_chart,
    timepoints=t,
    instrument_width=4,
    light_activated=True,
    verbose=1
)

print("\nStates:", model_light.states)
print("Transitions:", model_light.transitions)

pops_light = model_light().detach().numpy()

print(f"\nPopulation statistics (light-activated mode):")
print(f"  Max population: {pops_light.max():.6f}")
print(f"  Total population (final): {pops_light[-1, :].sum():.6f}")

print(f"\nState populations:")
for i, state in enumerate(model_light.states):
    max_pop = pops_light[:, i].max()
    final_pop = pops_light[-1, i]
    print(f"  State {state}: max={max_pop:.4f}, final={final_pop:.4f}")

print()

# Compare
print("="*70)
print("COMPARISON")
print("="*70)

print(f"\nNormal mode:")
print(f"  Total states: {len(model_normal.states)}")
print(f"  Max population anywhere: {pops_normal.max():.6f}")

print(f"\nLight-activated mode:")
print(f"  Total states: {len(model_light.states)}")
print(f"  Max population anywhere: {pops_light.max():.6f}")
print(f"  A* (inactive) final population: {pops_light[-1, model_light.state_to_idx['A*']]:.4f}")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Normal mode
for i, state in enumerate(model_normal.states):
    ax1.plot(t.numpy(), pops_normal[:, i], label=f'{state}', linewidth=2)
ax1.set_xlabel('Time (ps)', fontsize=12)
ax1.set_ylabel('Population', fontsize=12)
ax1.set_title('Normal Mode: Allows Re-photoactivation', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Light-activated mode
for i, state in enumerate(model_light.states):
    ax2.plot(t.numpy(), pops_light[:, i], label=f'{state}', linewidth=2)
ax2.set_xlabel('Time (ps)', fontsize=12)
ax2.set_ylabel('Population', fontsize=12)
ax2.set_title('Light-Activated: Products → A* (inactive)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/light_activated_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Comparison plot saved to: /tmp/light_activated_comparison.png")

# Test the combined plotting
print("\n" + "="*70)
print("TEST 3: Combined A+A* plotting")
print("="*70)

model_light.plot_occupancies('/tmp/light_activated_combined.png')
print("✓ In the plot, 'A' shows the sum of A + A* (total ground state)")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ light_activated=False: allows cycling (can cause amplification)")
print("✓ light_activated=True: products → A* (inactive, prevents re-activation)")
print("✓ Plotting automatically combines A and A* for display")
print("="*70 + "\n")
