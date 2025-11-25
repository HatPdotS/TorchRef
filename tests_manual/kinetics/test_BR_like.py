#!/usr/bin/env python
"""
Bacteriorhodopsin-like test with realistic kinetics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchref.kinetics import KineticModel

print("\n" + "="*70)
print("BACTERIORHODOPSIN-LIKE KINETICS TEST")
print("="*70 + "\n")

# BR photocycle: K → L → M → N → O → BR
# Timescales: ps (K), ns (L), μs (M), ms (N, O)

# Create logarithmic timepoints from ps to ms
t_min_ps = 0.1       # 100 fs
t_max_ps = 1e6       # 1 ms
n_points = 200

t = np.logspace(np.log10(t_min_ps), np.log10(t_max_ps), n_points)

print(f"Time range: {t.min():.2e} to {t.max():.2e} ps")
print(f"  = {t.min()/1e3:.2e} to {t.max()/1e6:.2f} ms")
print(f"Number of points: {n_points}")
print()

# BR-like kinetics with realistic timescales
# K (τ~3 ps) → L (τ~10 μs) → M (τ~2 ms) → BR
print("Creating model with BR-like kinetics...")
print("-"*70)

# Rate constants (1/ps)
k_K_to_L = 1.0 / 3.0      # τ = 3 ps
k_L_to_M = 1.0 / 10000.0  # τ = 10 μs = 10,000 ps
k_M_to_BR = 1.0 / 2e6     # τ = 2 ms = 2,000,000 ps

print(f"K → L: k = {k_K_to_L:.6e} (1/ps), τ = {1/k_K_to_L:.2f} ps")
print(f"L → M: k = {k_L_to_M:.6e} (1/ps), τ = {1/k_L_to_M:.0f} ps = {1/k_L_to_M/1e3:.1f} ns")
print(f"M → BR: k = {k_M_to_BR:.6e} (1/ps), τ = {1/k_M_to_BR:.0f} ps = {1/k_M_to_BR/1e6:.2f} ms")
print()

try:
    model = KineticModel(
        flow_chart="K->L,L->M,M->BR",
        timepoints=t,
        rate_constants={
            'K->L': k_K_to_L,
            'L->M': k_L_to_M,
            'M->BR': k_M_to_BR
        },
        instrument_width=0.2,  # 200 fs instrument response
        verbose=1
    )
    
    print("\nComputing populations...")
    populations = model().detach().numpy()
    
    print(f"\nPopulation statistics:")
    print(f"  Shape: {populations.shape}")
    print(f"  Min: {populations.min():.6e}")
    print(f"  Max: {populations.max():.6e}")
    print(f"  Mean: {populations.mean():.6e}")
    
    pop_sums = populations.sum(axis=1)
    print(f"\nPopulation sums:")
    print(f"  Min: {pop_sums.min():.10f}")
    print(f"  Max: {pop_sums.max():.10f}")
    print(f"  Std: {pop_sums.std():.10e}")
    
    if populations.max() > 1e10:
        print("\n✗ NUMERICAL INSTABILITY DETECTED!")
        print("  Finding problematic timepoints...")
        for i in range(len(t)):
            if populations[i, :].max() > 10:
                print(f"  t = {t[i]:.6e} ps:")
                print(f"    Populations: {populations[i, :]}")
                print(f"    Sum: {pop_sums[i]:.6e}")
                if i > 0:
                    print(f"  Previous t = {t[i-1]:.6e} ps:")
                    print(f"    Populations: {populations[i-1, :]}")
                break
    else:
        print("\n✓ Populations within physically reasonable range")
    
    # Plot
    print("\nCreating visualization...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Linear time scale
    for i, state in enumerate(model.states):
        ax1.plot(t, populations[:, i], label=f'State {state}', linewidth=2)
    ax1.set_xscale('log')
    ax1.set_xlabel('Time (ps)', fontsize=12)
    ax1.set_ylabel('Population', fontsize=12)
    ax1.set_title('BR-like Photocycle: Log Time Scale', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Check for reasonableness
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    # Population sums
    ax2.plot(t, pop_sums, 'k-', linewidth=2)
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Expected sum = 1')
    ax2.set_xscale('log')
    ax2.set_xlabel('Time (ps)', fontsize=12)
    ax2.set_ylabel('Total Population', fontsize=12)
    ax2.set_title('Population Conservation Check', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('/tmp/br_like_kinetics.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved to: /tmp/br_like_kinetics.png")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
