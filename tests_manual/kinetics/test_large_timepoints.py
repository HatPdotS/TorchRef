#!/usr/bin/env python
"""
Test to reproduce the large timepoint issue.
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

print("\n" + "="*70)
print("TEST: LARGE LOGARITHMIC TIMEPOINTS (ps to ms regime)")
print("="*70 + "\n")

# Bacteriorhodopsin-like kinetics: ps to ms regime
t_ps = np.logspace(-1, 6, 100)  # 0.1 ps to 1 ms (1e6 ps)

print(f"Timepoints range: {t_ps.min():.2e} to {t_ps.max():.2e} ps")
print(f"Time span: {t_ps.max()/1e6:.2f} ms")
print()

try:
    model = KineticModel(
        flow_chart="A->B,B->C,C->D",
        timepoints=t_ps,
        instrument_width=0.5,
        verbose=1
    )
    
    print("\nComputing populations...")
    populations = model().detach().numpy()
    
    print(f"\nPopulation statistics:")
    print(f"  Min: {populations.min():.6e}")
    print(f"  Max: {populations.max():.6e}")
    print(f"  Sum (min): {populations.sum(axis=1).min():.6f}")
    print(f"  Sum (max): {populations.sum(axis=1).max():.6f}")
    
    if populations.max() > 1e10:
        print("\n✗ NUMERICAL INSTABILITY DETECTED!")
        print("  Populations exceed physically meaningful range")
        
        # Check where it breaks
        for i in range(len(t_ps)):
            pop_sum = populations[i, :].sum()
            if pop_sum > 10 or pop_sum < 0.1:
                print(f"\n  First problematic timepoint:")
                print(f"    t = {t_ps[i]:.6e} ps")
                print(f"    Populations: {populations[i, :]}")
                print(f"    Sum: {pop_sum:.6e}")
                break
    else:
        print("\n✓ Populations within reasonable range")
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
