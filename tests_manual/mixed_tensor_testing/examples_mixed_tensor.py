"""
Example usage of MixedTensor for partial refinement.

This demonstrates how to use MixedTensor to refine only part of a tensor
while keeping other parts fixed during optimization.
"""

import torch
import torch.nn as nn
from torchref.model import MixedTensor


def example_basic_usage():
    """Basic example: refine only a subset of parameters."""
    print("=" * 60)
    print("Example 1: Basic Usage - Refining subset of parameters")
    print("=" * 60)
    
    # Create initial tensor with 100 elements
    initial = torch.randn(100, dtype=torch.float64)
    
    # Create mask: only refine elements 20-40
    mask = torch.zeros(100, dtype=torch.bool)
    mask[20:40] = True
    
    # Create MixedTensor
    mixed = MixedTensor(initial, refinable_mask=mask, requires_grad=True)
    
    print(f"Total elements: {mixed.shape[0]}")
    print(f"Refinable elements: {mixed.get_refinable_count()}")
    print(f"Fixed elements: {mixed.get_fixed_count()}")
    print(f"\n{mixed}")
    
    # Use in optimization
    target = torch.ones(100, dtype=torch.float64)
    optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.1)
    
    print("\nOptimizing...")
    for i in range(100):
        optimizer.zero_grad()
        current = mixed()
        loss = (current - target).pow(2).sum()
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Step {i:3d}: Loss = {loss.item():.6f}")
    
    # Verify that only refinable elements changed
    final = mixed()
    initial_fixed = initial[~mask]
    final_fixed = final[~mask]
    
    print(f"\nFixed elements changed: {not torch.allclose(initial_fixed, final_fixed)}")
    print(f"Refinable elements changed: {not torch.allclose(initial[mask], final[mask])}")
    print(f"Fixed elements match initial: {torch.allclose(initial_fixed, final_fixed)}")


def example_3d_coordinates():
    """Example: refining atomic coordinates where some atoms are fixed."""
    print("\n" + "=" * 60)
    print("Example 2: Refining 3D Coordinates (some atoms fixed)")
    print("=" * 60)
    
    # Simulate 1000 atoms with x,y,z coordinates
    n_atoms = 1000
    coords = torch.randn(n_atoms, 3, dtype=torch.float64)
    
    # Fix first 100 atoms (e.g., reference structure)
    # Refine atoms 100-1000
    mask = torch.ones(n_atoms, 3, dtype=torch.bool)
    mask[:100, :] = False  # Fix first 100 atoms
    
    mixed_coords = MixedTensor(coords, refinable_mask=mask, requires_grad=True)
    
    print(f"Total atoms: {n_atoms}")
    print(f"Fixed atoms: {100}")
    print(f"Refinable atoms: {n_atoms - 100}")
    print(f"Total parameters: {mixed_coords.shape[0] * mixed_coords.shape[1]}")
    print(f"Refinable parameters: {mixed_coords.get_refinable_count()}")
    print(f"{mixed_coords}")
    
    # Simulate refinement toward a target structure
    target_coords = torch.randn(n_atoms, 3, dtype=torch.float64)
    optimizer = torch.optim.Adam([mixed_coords.refinable_params], lr=0.01)
    
    print("\nRefining coordinates...")
    for i in range(50):
        optimizer.zero_grad()
        current = mixed_coords()
        # Only optimize refinable atoms
        loss = (current[mask] - target_coords[mask]).pow(2).sum()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            rmsd = torch.sqrt((current - coords).pow(2).sum(dim=1).mean())
            print(f"Step {i:3d}: Loss = {loss.item():.6f}, RMSD = {rmsd.item():.6f}")
    
    # Verify first 100 atoms didn't move
    final = mixed_coords()
    print(f"\nFixed atoms unchanged: {torch.allclose(coords[:100], final[:100])}")
    print(f"Refinable atoms changed: {not torch.allclose(coords[100:], final[100:])}")


def example_in_neural_network():
    """Example: using MixedTensor as part of a neural network module."""
    print("\n" + "=" * 60)
    print("Example 3: Using MixedTensor in a Neural Network")
    print("=" * 60)
    
    class RefinementModel(nn.Module):
        def __init__(self, n_params):
            super().__init__()
            # Some parameters are fixed, others refinable
            initial = torch.randn(n_params, dtype=torch.float64)
            mask = torch.rand(n_params) > 0.5  # Randomly select ~50% to refine
            
            self.mixed_params = MixedTensor(
                initial, 
                refinable_mask=mask, 
                requires_grad=True
            )
            
            # Other regular parameters
            self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        
        def forward(self, x):
            params = self.mixed_params()
            return x @ params * self.scale
    
    # Create model
    model = RefinementModel(n_params=50)
    
    print(f"Model: {model}")
    print(f"\nMixed parameters: {model.mixed_params}")
    
    # Setup optimizer - note we pass the model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Dummy data
    X = torch.randn(10, 50, dtype=torch.float64)
    y = torch.randn(10, dtype=torch.float64)
    
    print("\nTraining model...")
    for i in range(50):
        optimizer.zero_grad()
        pred = model(X)
        loss = (pred - y).pow(2).mean()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i:3d}: Loss = {loss.item():.6f}, Scale = {model.scale.item():.4f}")
    
    print(f"\nFinal refinable count: {model.mixed_params.get_refinable_count()}")
    print(f"Final fixed count: {model.mixed_params.get_fixed_count()}")


def example_dynamic_mask_update():
    """Example: dynamically changing which parameters are refinable."""
    print("\n" + "=" * 60)
    print("Example 4: Dynamically Updating Refinable Mask")
    print("=" * 60)
    
    # Start with all parameters refinable
    initial = torch.randn(100, dtype=torch.float64)
    mask = torch.ones(100, dtype=torch.bool)
    
    mixed = MixedTensor(initial, refinable_mask=mask, requires_grad=True)
    print(f"Initial state: {mixed}")
    
    # Phase 1: Refine all parameters
    target = torch.ones(100, dtype=torch.float64)
    optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.1)
    
    print("\nPhase 1: Refining all parameters")
    for i in range(30):
        optimizer.zero_grad()
        loss = (mixed() - target).pow(2).sum()
        loss.backward()
        optimizer.step()
    
    phase1_result = mixed().detach().clone()
    print(f"After phase 1 loss: {(mixed() - target).pow(2).sum().item():.6f}")
    
    # Phase 2: Fix first 50 parameters, continue refining last 50
    new_mask = torch.zeros(100, dtype=torch.bool)
    new_mask[50:] = True
    mixed.update_refinable_mask(new_mask, reset_refinable=False)
    
    # Need to recreate optimizer with new parameters
    optimizer = torch.optim.Adam([mixed.refinable_params], lr=0.1)
    
    print(f"\nPhase 2: Only refining last 50 parameters")
    print(f"Updated state: {mixed}")
    
    for i in range(30):
        optimizer.zero_grad()
        loss = (mixed() - target).pow(2).sum()
        loss.backward()
        optimizer.step()
    
    final_result = mixed()
    print(f"After phase 2 loss: {(final_result - target).pow(2).sum().item():.6f}")
    
    # Verify first 50 stayed the same as phase 1
    print(f"\nFirst 50 unchanged: {torch.allclose(phase1_result[:50], final_result[:50])}")
    print(f"Last 50 changed: {not torch.allclose(phase1_result[50:], final_result[50:])}")


def example_integration_with_existing_code():
    """Example showing how to integrate with existing refinement code."""
    print("\n" + "=" * 60)
    print("Example 5: Integration Pattern for Existing Code")
    print("=" * 60)
    
    # Simulate existing refinement setup
    n_residues = 100
    n_params_per_residue = 7  # e.g., x, y, z, occ, B, etc.
    
    # Create parameters for all residues
    all_params = torch.randn(n_residues, n_params_per_residue, dtype=torch.float64)
    
    # Define selection: refine only residues 20-80
    selection_start = 20
    selection_end = 80
    
    # Create refinable mask
    mask = torch.zeros(n_residues, n_params_per_residue, dtype=torch.bool)
    mask[selection_start:selection_end, :] = True
    
    # You could also selectively refine only certain parameters (e.g., not occupancy)
    # mask[:, 3] = False  # Don't refine occupancy (column 3)
    
    mixed_params = MixedTensor(all_params, refinable_mask=mask, requires_grad=True)
    
    print(f"Total residues: {n_residues}")
    print(f"Parameters per residue: {n_params_per_residue}")
    print(f"Total parameters: {n_residues * n_params_per_residue}")
    print(f"Refinable parameters: {mixed_params.get_refinable_count()}")
    print(f"\n{mixed_params}")
    
    # In your refinement loop, replace tensor with mixed_params()
    def compute_structure_factor(params):
        """Dummy structure factor calculation."""
        return params.sum(dim=1)  # Simplified
    
    target_sf = torch.randn(n_residues, dtype=torch.float64)
    optimizer = torch.optim.Adam([mixed_params.refinable_params], lr=0.01)
    
    print("\nSimulating refinement...")
    for i in range(50):
        optimizer.zero_grad()
        
        # Get current parameters (automatically combines fixed + refinable)
        current_params = mixed_params()
        
        # Use in structure factor calculation
        sf = compute_structure_factor(current_params)
        
        # Compute loss
        loss = (sf - target_sf).pow(2).sum()
        
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Step {i:3d}: Loss = {loss.item():.6f}")
    
    # Extract final parameters
    final_params = mixed_params.detach()
    print(f"\nRefinement complete!")
    print(f"Parameters outside selection unchanged: "
          f"{torch.allclose(all_params[:selection_start], final_params[:selection_start])}")


if __name__ == "__main__":
    example_basic_usage()
    example_3d_coordinates()
    example_in_neural_network()
    example_dynamic_mask_update()
    example_integration_with_existing_code()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
