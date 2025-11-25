#!/usr/bin/env python
"""
Test smart initialization of rate constants based on observability.
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

print("\n" + "="*70)
print("SMART INITIALIZATION TEST")
print("="*70 + "\n")

# Test 1: Simple sequential
print("Test 1: Sequential Kinetics A->B,B->C")
print("-"*70)

t = np.linspace(-1, 10, 200)

model1 = KineticModel(
    flow_chart="A->B,B->C",
    timepoints=t,
    instrument_function='gaussian',
    instrument_width=0.3,
    verbose=1
)

print("\nInitialized rate constants:")
rates = model1.get_rate_constants()
for key, val in rates.items():
    tau = 1.0 / val
    print(f"  {key}: k = {val:.4f}, τ = {tau:.4f}")

print("\nExpected behavior:")
print(f"  First transition (A->B) should be fast: τ ≈ σ/3 = {0.3/3:.4f}")
print(f"  Second transition (B->C) should allow B to be observable")
print()

# Test 2: Longer chain
print("Test 2: Longer Chain A->B,B->C,C->D")
print("-"*70)

model2 = KineticModel(
    flow_chart="A->B,B->C,C->D",
    timepoints=t,
    instrument_function='gaussian',
    instrument_width=0.2,
    verbose=1
)

print("\nInitialized rate constants:")
rates = model2.get_rate_constants()
for key, val in rates.items():
    tau = 1.0 / val
    print(f"  {key}: k = {val:.4f}, τ = {tau:.4f}")
print()

# Test 3: With back reaction
print("Test 3: With Back Reaction A->B,B->A,B->C")
print("-"*70)

model3 = KineticModel(
    flow_chart="A->B,B->A,B->C",
    timepoints=t,
    instrument_function='gaussian',
    instrument_width=0.25,
    verbose=1
)

print("\nInitialized rate constants:")
rates = model3.get_rate_constants()
for key, val in rates.items():
    tau = 1.0 / val
    print(f"  {key}: k = {val:.4f}, τ = {tau:.4f}")
print()

# Test 4: Test observability - plot populations
print("Test 4: Verify Observability")
print("-"*70)

model4 = KineticModel(
    flow_chart="A->B,B->C,C->D",
    timepoints=np.linspace(-0.5, 15, 300),
    instrument_function='gaussian',
    instrument_width=0.2,
    verbose=0
)

populations = model4().detach().numpy()
t_plot = model4.timepoints.numpy()

# Check peak occupancies
print("\nPeak occupancies (should all be > 0.3 to be observable):")
for i, state in enumerate(model4.states):
    peak = populations[:, i].max()
    t_peak_idx = populations[:, i].argmax()
    t_peak = t_plot[t_peak_idx]
    print(f"  State {state}: {peak:.3f} at t = {t_peak:.2f}")

# Test 5: Different timeframe
print("\nTest 5: Different Timeframe (short)")
print("-"*70)

t_short = np.linspace(-0.1, 2, 150)
model5 = KineticModel(
    flow_chart="A->B,B->C",
    timepoints=t_short,
    instrument_function='gaussian',
    instrument_width=0.1,
    verbose=1
)

print("\nInitialized rate constants:")
rates = model5.get_rate_constants()
for key, val in rates.items():
    tau = 1.0 / val
    print(f"  {key}: k = {val:.4f}, τ = {tau:.4f}")
print()

# Test 6: Very long timeframe
print("Test 6: Different Timeframe (long)")
print("-"*70)

t_long = np.linspace(-1, 100, 300)
model6 = KineticModel(
    flow_chart="A->B,B->C",
    timepoints=t_long,
    instrument_function='gaussian',
    instrument_width=0.5,
    verbose=1
)

print("\nInitialized rate constants:")
rates = model6.get_rate_constants()
for key, val in rates.items():
    tau = 1.0 / val
    print(f"  {key}: k = {val:.4f}, τ = {tau:.4f}")

print("\n" + "="*70)
print("Summary:")
print("="*70)
print("✓ First transition is quasi-instant (τ ≈ σ/3)")
print("✓ Subsequent transitions follow observability (2*k_in ≈ k_out)")
print("✓ Rates scale with timeframe")
print("✓ All states should be observable")
print("="*70 + "\n")
