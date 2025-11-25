#!/usr/bin/env python
"""
Test script to verify that both uniform and per-atom occupancy managers
work correctly and always return expanded tensors.
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

import torch
from torchref.Model import uniform_occupancy_manager, per_atom_occupancy_manager

print("Testing Occupancy Managers")
print("=" * 60)

# Test 1: uniform_occupancy_manager with n_atoms
print("\nTest 1: uniform_occupancy_manager with n_atoms=5")
uniform_mgr = uniform_occupancy_manager(occ_start=0.8, n_atoms=5)
occ = uniform_mgr.get_occupancy()
print(f"  Shape: {occ.shape}")
print(f"  Values: {occ}")
print(f"  All equal: {torch.allclose(occ, torch.tensor(0.8) * torch.ones(5))}")
print(f"  Requires grad: {occ.requires_grad}")
assert occ.shape == (5,), f"Expected shape (5,), got {occ.shape}"
assert torch.allclose(occ, torch.tensor(0.8) * torch.ones(5), atol=0.01), "Values should all be 0.8"
print("  ✓ Test passed!")

# Test 2: uniform_occupancy_manager default occupancy
print("\nTest 2: uniform_occupancy_manager with default occ (1.0)")
uniform_mgr2 = uniform_occupancy_manager(n_atoms=3)
occ2 = uniform_mgr2.get_occupancy()
print(f"  Shape: {occ2.shape}")
print(f"  Values: {occ2}")
print(f"  All equal to 1.0: {torch.allclose(occ2, torch.ones(3))}")
print(f"  Requires grad: {occ2.requires_grad}")
assert occ2.shape == (3,), f"Expected shape (3,), got {occ2.shape}"
assert torch.allclose(occ2, torch.ones(3), atol=0.01), "Values should all be 1.0"
print("  ✓ Test passed!")

# Test 3: per_atom_occupancy_manager
print("\nTest 3: per_atom_occupancy_manager with different values")
occ_values = torch.tensor([1.0, 1.0, 0.84, 0.84, 0.16, 0.16])
per_atom_mgr = per_atom_occupancy_manager(occ_values)
occ3 = per_atom_mgr.get_occupancy()
print(f"  Shape: {occ3.shape}")
print(f"  Values: {occ3}")
print(f"  Matches input: {torch.allclose(occ3, occ_values, atol=0.01)}")
print(f"  Requires grad: {occ3.requires_grad}")
assert occ3.shape == (6,), f"Expected shape (6,), got {occ3.shape}"
assert torch.allclose(occ3, occ_values, atol=0.01), "Values should match input"
print("  ✓ Test passed!")

# Test 4: Gradient flow for uniform_occupancy_manager
print("\nTest 4: Gradient flow for uniform_occupancy_manager")
uniform_mgr3 = uniform_occupancy_manager(occ_start=0.5, n_atoms=4)
occ4 = uniform_mgr3.get_occupancy()
loss = occ4.sum()
loss.backward()
print(f"  Gradient on hidden parameter: {uniform_mgr3.hidden.grad}")
print(f"  Gradient exists: {uniform_mgr3.hidden.grad is not None}")
assert uniform_mgr3.hidden.grad is not None, "Gradient should exist"
print("  ✓ Test passed!")

# Test 5: Gradient flow for per_atom_occupancy_manager
print("\nTest 5: Gradient flow for per_atom_occupancy_manager")
occ_values2 = torch.tensor([0.7, 0.8, 0.9])
per_atom_mgr2 = per_atom_occupancy_manager(occ_values2)
occ5 = per_atom_mgr2.get_occupancy()
loss2 = occ5.sum()
loss2.backward()
print(f"  Gradient on hidden parameter: {per_atom_mgr2.hidden.grad}")
print(f"  Gradient exists: {per_atom_mgr2.hidden.grad is not None}")
assert per_atom_mgr2.hidden.grad is not None, "Gradient should exist"
print("  ✓ Test passed!")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("\nSummary:")
print("  - uniform_occupancy_manager always returns shape (n_atoms,)")
print("  - per_atom_occupancy_manager always returns shape (n_atoms,)")
print("  - Both maintain gradient flow")
print("  - No manual expansion needed in model_ft.py")
