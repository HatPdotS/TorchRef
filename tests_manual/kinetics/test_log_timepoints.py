#!/usr/bin/env python
"""
Test with LOGARITHMIC timepoints as mentioned by user.
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

print("\n" + "="*70)
print("TEST: LOGARITHMIC TIMEPOINTS (BR-LIKE)")
print("="*70 + "\n")

# Try logarithmic timepoints from ps to ms
t_log = torch.logspace(-1, 6, 1000)  # 0.1 ps to 1 ms

print(f"Using LOGARITHMIC timepoints:")
print(f"  Range: {t_log.min():.2e} to {t_log.max():.2e}")
print(f"  Number of points: {len(t_log)}")
print()

flow_chart = 'A->K,K->L,L->K,L->M,M->L,M->N,N->O,O->BR'

print(f"Flow chart: {flow_chart}")
print()

try:
    kinetic_model = KineticModel(
        flow_chart=flow_chart,
        timepoints=t_log,
        instrument_width=4,
        verbose=1
    )
    
    print("\nSetting baseline to 0 for state A...")
    kinetic_model.set_baseline(state='A', occupancy=0)
    
    print("\nComputing populations...")
    pops = kinetic_model().detach().numpy()
    
    print(f"\nPopulation statistics:")
    print(f"  Min: {pops.min():.6e}")
    print(f"  Max: {pops.max():.6e}")
    print(f"  Sum (min): {pops.sum(axis=1).min():.10f}")
    print(f"  Sum (max): {pops.sum(axis=1).max():.10f}")
    
    if pops.max() > 1e10:
        print("\n✗ NUMERICAL INSTABILITY - POPULATIONS EXPLODE!")
        # Find where
        for i in range(len(t_log)):
            if pops[i, :].max() > 10:
                print(f"\nFirst problematic timepoint:")
                print(f"  Index: {i}")
                print(f"  Time: {t_log[i]:.6e} ps")
                print(f"  Populations: {pops[i, :]}")
                print(f"  Sum: {pops[i, :].sum():.6e}")
                if i > 0:
                    print(f"\nPrevious timepoint (OK):")
                    print(f"  Time: {t_log[i-1]:.6e} ps")
                    print(f"  Populations: {pops[i-1, :]}")
                break
    else:
        print("\n✓ Populations OK")
        
        # Plot
        print("\nPlotting...")
        kinetic_model.plot_occupancies('/tmp/br_like_logtime.png')
        kinetic_model.plot_occupancies('/tmp/br_like_logtime_log.png', log=True)
        print("✓ Plots saved")
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
