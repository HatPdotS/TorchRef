#!/usr/bin/env python3
"""
Quick test to verify the new occupancy manager classes work correctly.
Tests both uniform and per-atom occupancy managers.
"""
import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
import numpy as np
from torchref.Model import uniform_occupancy_manager, per_atom_occupancy_manager

print("Testing uniform_occupancy_manager...")
# Test uniform occupancy manager
uniform_occ = uniform_occupancy_manager(torch.tensor(0.8))
occ_value = uniform_occ.get_occupancy()
print(f"  Uniform occupancy value: {occ_value}")
print(f"  Shape: {occ_value.shape}")
print(f"  Grad enabled: {occ_value.requires_grad}")

# Test expansion
expanded = occ_value.unsqueeze(0).expand(5)
print(f"  Expanded to 5 atoms: {expanded}")
print(f"  Expanded shape: {expanded.shape}")
print(f"  Expanded grad enabled: {expanded.requires_grad}")

print("\nTesting per_atom_occupancy_manager...")
# Test per-atom occupancy manager
occ_values = torch.tensor([1.0, 1.0, 1.0, 0.84, 0.84, 0.84, 0.16, 0.16, 0.16])
per_atom_occ = per_atom_occupancy_manager(occ_values)
per_atom_values = per_atom_occ.get_occupancy()
print(f"  Per-atom occupancy values: {per_atom_values}")
print(f"  Shape: {per_atom_values.shape}")
print(f"  Grad enabled: {per_atom_values.requires_grad}")

print("\nTesting gradient flow...")
# Test gradient flow for uniform
uniform_occ = uniform_occupancy_manager(torch.tensor(0.5))
occ = uniform_occ.get_occupancy()
expanded = occ.unsqueeze(0).expand(3)
loss = (expanded * torch.tensor([1.0, 2.0, 3.0])).sum()
loss.backward()
print(f"  Uniform manager gradient: {uniform_occ.hidden.grad}")

# Test gradient flow for per-atom
per_atom_occ = per_atom_occupancy_manager(torch.tensor([0.5, 0.6, 0.7]))
occ = per_atom_occ.get_occupancy()
loss = (occ * torch.tensor([1.0, 2.0, 3.0])).sum()
loss.backward()
print(f"  Per-atom manager gradient: {per_atom_occ.hidden.grad}")

print("\n✓ All tests passed!")
