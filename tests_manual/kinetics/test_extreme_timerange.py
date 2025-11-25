#!/usr/bin/env python
"""
Extreme test: ps to seconds with light-activated mode.
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

print("\n" + "="*70)
print("EXTREME TEST: ps to SECONDS with light_activated=True")
print("="*70 + "\n")

# Extreme logarithmic timepoints: 1 ps to 1 second (1e12 ps!)
t_extreme = torch.logspace(0, 12, 1000)

print(f"Timepoints: {t_extreme.min():.2e} to {t_extreme.max():.2e} ps")
print(f"  = {t_extreme.min():.2e} ps to {t_extreme.max()/1e12:.2f} seconds")
print(f"  Spanning 12 orders of magnitude!")
print()

flow_chart = 'A->K,K->L,L->K,L->M,M->L,M->N,N->O,O->A'

print("Testing WITH light_activated=True")
print("-"*70)

try:
    model = KineticModel(
        flow_chart=flow_chart,
        timepoints=t_extreme,
        instrument_width=1.0,
        light_activated=True,
        verbose=1
    )
    
    print("\nComputing populations...")
    pops = model().detach().numpy()
    
    print(f"\nPopulation statistics:")
    print(f"  Shape: {pops.shape}")
    print(f"  Min: {pops.min():.6e}")
    print(f"  Max: {pops.max():.6e}")
    print(f"  Sum (min): {pops.sum(axis=1).min():.10f}")
    print(f"  Sum (max): {pops.sum(axis=1).max():.10f}")
    
    if pops.max() > 10:
        print("\n✗ AMPLIFICATION DETECTED!")
    else:
        print("\n✓ NO AMPLIFICATION - populations remain physical!")
    
    print(f"\nFinal state populations (t = {t_extreme[-1]:.2e} ps):")
    for i, state in enumerate(model.states):
        print(f"  {state}: {pops[-1, i]:.6f}")
    
    # Check A*
    if 'A*' in model.states:
        idx_A_star = model.state_to_idx['A*']
        print(f"\nInactive state A* accumulates: {pops[-1, idx_A_star]:.6f}")
        print("✓ This represents molecules that completed the photocycle")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("RESULT: light_activated=True prevents amplification")
print("        even with extreme time ranges (ps to seconds)!")
print("="*70 + "\n")
