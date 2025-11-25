#!/usr/bin/env python
"""
Demonstration of the updated KineticModel with new features:
- Comma-based relational syntax
- Rate constants (k) and efficiencies (η) for each transition
- Refinable instrument function
- Visualization capabilities
- Easy access to all parameters
"""

import torch
import numpy as np
from torchref.kinetics import KineticModel

print("\n" + "="*70)
print("UPDATED KINETIC MODEL DEMONSTRATION")
print("="*70 + "\n")

# Define timepoints
t = np.linspace(-1, 10, 200)

# ============================================================================
# Example 1: Simple Sequential with Rate Constants and Efficiencies
# ============================================================================
print("Example 1: Sequential Kinetics with Efficiencies")
print("-" * 70)

# Create model with NEW COMMA SYNTAX
model1 = KineticModel(
    flow_chart="A->B,B->C",  # NEW: Comma-separated transitions
    timepoints=t,
    rate_constants={"A->B": 2.0, "B->C": 0.5},  # NEW: Initialize rates
    efficiencies={"A->B": 0.9, "B->C": 0.8},    # NEW: Reaction efficiencies
    instrument_function='gaussian',
    instrument_width=0.2,
    verbose=0
)

# Print parameters
model1.print_parameters()

# NEW: Get all tensors for optimization
all_tensors = model1.get_all_tensors()
print(f"Number of flexible parameters: {len(all_tensors)}")
print(f"  - log_rate_constants: {all_tensors[0].shape}")
print(f"  - logit_efficiencies: {all_tensors[1].shape}")
print(f"  - log_instrument_width: {all_tensors[2].shape}\n")

# NEW: Visualize occupancies
model1.plot_occupancies('/tmp/demo_occupancies_linear.png', log=False)
print()

# ============================================================================
# Example 2: Complex Cyclic Scheme
# ============================================================================
print("Example 2: Cyclic Kinetics (A → B → C → A)")
print("-" * 70)

model2 = KineticModel(
    flow_chart="A->B,B->C,C->A",  # Cyclic scheme
    timepoints=np.linspace(-0.5, 15, 300),
    rate_constants=[1.0, 0.8, 0.5],  # NEW: Can use list format too
    efficiencies=[1.0, 0.9, 0.7],    # Different efficiencies
    instrument_function='gaussian',
    instrument_width=0.1,
    verbose=0
)

model2.print_parameters()

# Get effective rates
eff_rates = model2.get_effective_rates()
print("Effective rates (k*η):")
for key, val in eff_rates.items():
    print(f"  {key}: {val:.4f}")
print()

# Visualize with log scale
model2.plot_occupancies('/tmp/demo_occupancies_log.png', log=True, 
                        title="Cyclic Kinetics (Log Scale)")
print()

# ============================================================================
# Example 3: Fitting with New Features
# ============================================================================
print("Example 3: Fitting with Rate Constants and Efficiencies")
print("-" * 70)

# Generate synthetic data
t_fit = np.linspace(-0.5, 8, 150)
true_model = KineticModel(
    flow_chart="A->B,B->C",
    timepoints=t_fit,
    rate_constants={"A->B": 1.5, "B->C": 0.6},
    efficiencies={"A->B": 0.85, "B->C": 0.75},
    instrument_function='gaussian',
    instrument_width=0.15,
    verbose=0
)

with torch.no_grad():
    true_populations = true_model()
noisy_data = true_populations + torch.randn_like(true_populations) * 0.02

print("True parameters:")
print(f"  A->B: k={true_model.get_rate_constants()['A->B']:.3f}, "
      f"η={true_model.get_efficiencies()['A->B']:.3f}")
print(f"  B->C: k={true_model.get_rate_constants()['B->C']:.3f}, "
      f"η={true_model.get_efficiencies()['B->C']:.3f}")
print(f"  σ_IRF={torch.exp(true_model.log_instrument_width).item():.3f}")
print()

# Create fitting model
fit_model = KineticModel(
    flow_chart="A->B,B->C",
    timepoints=t_fit,
    instrument_function='gaussian',
    instrument_width=0.2,  # Start with wrong value
    verbose=0
)

print("Initial (random) parameters:")
print(f"  A->B: k={fit_model.get_rate_constants()['A->B']:.3f}, "
      f"η={fit_model.get_efficiencies()['A->B']:.3f}")
print(f"  B->C: k={fit_model.get_rate_constants()['B->C']:.3f}, "
      f"η={fit_model.get_efficiencies()['B->C']:.3f}")
print(f"  σ_IRF={torch.exp(fit_model.log_instrument_width).item():.3f}")
print()

# Optimize
optimizer = torch.optim.Adam(fit_model.parameters(), lr=0.01)
print("Fitting (1000 epochs)...")
for epoch in range(1000):
    optimizer.zero_grad()
    pred_populations = fit_model()
    loss = torch.mean((pred_populations - noisy_data) ** 2)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 200 == 0:
        print(f"  Epoch {epoch+1}: Loss = {loss.item():.6f}")

print("\nFitted parameters:")
print(f"  A->B: k={fit_model.get_rate_constants()['A->B']:.3f}, "
      f"η={fit_model.get_efficiencies()['A->B']:.3f}")
print(f"  B->C: k={fit_model.get_rate_constants()['B->C']:.3f}, "
      f"η={fit_model.get_efficiencies()['B->C']:.3f}")
print(f"  σ_IRF={torch.exp(fit_model.log_instrument_width).item():.3f}")
print()

# Compare with true
print("Comparison with true values:")
print(f"  A->B rate error: {abs(fit_model.get_rate_constants()['A->B'] - 1.5):.4f}")
print(f"  A->B efficiency error: {abs(fit_model.get_efficiencies()['A->B'] - 0.85):.4f}")
print(f"  IRF width error: {abs(torch.exp(fit_model.log_instrument_width).item() - 0.15):.4f}")
print()

# Visualize fitting result
fit_model.visualize('/tmp/demo_fitted_result.png', 
                   title="Fitted Kinetics")

# ============================================================================
# Example 4: Parallel Pathways
# ============================================================================
print("Example 4: Parallel Pathways")
print("-" * 70)

model4 = KineticModel(
    flow_chart="A->B,A->C,B->D,C->D",  # Two parallel pathways to D
    timepoints=t,
    rate_constants={"A->B": 2.0, "A->C": 1.5, "B->D": 0.8, "C->D": 0.6},
    efficiencies={"A->B": 0.9, "A->C": 0.85, "B->D": 0.95, "C->D": 0.90},
    instrument_function='gaussian',
    instrument_width=0.12,
    verbose=0
)

model4.print_parameters()
model4.plot_occupancies('/tmp/demo_parallel_pathways.png',
                        title="Parallel Pathways: A → B/C → D")

# ============================================================================
# Summary
# ============================================================================
print("="*70)
print("NEW FEATURES DEMONSTRATED:")
print("="*70)
print("✓ Comma-based relational syntax: 'A->B,B->C,C->D,C->A'")
print("✓ Rate constants (k) for each transition")
print("✓ Reaction efficiencies (η) for each transition (0-1)")
print("✓ Effective rates = k * η")
print("✓ Refinable instrument function width")
print("✓ get_all_tensors() for easy access to all parameters")
print("✓ plot_occupancies() for visualization (linear or log scale)")
print("✓ Enhanced print_parameters() showing k, η, and k*η")
print("✓ Flexible initialization via dicts or lists")
print("="*70)
print("\nAll demonstration plots saved to /tmp/")
print("  - demo_occupancies_linear.png")
print("  - demo_occupancies_log.png")
print("  - demo_fitted_result.png")
print("  - demo_parallel_pathways.png")
print("\n" + "="*70 + "\n")
