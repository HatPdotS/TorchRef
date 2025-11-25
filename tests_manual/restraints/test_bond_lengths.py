"""
Test suite for bond length restraints functionality.

This module tests that bond length restraints are correctly extracted from the CIF
dictionary and that the computed bond lengths match expected values.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from torchref.model import Model
from torchref.restraints import Restraints


def compute_bond_lengths(xyz, bond_indices):
    """
    Compute bond lengths from coordinates and indices.
    
    Args:
        xyz: Tensor of shape (N_atoms, 3) with atomic coordinates
        bond_indices: Tensor of shape (N_bonds, 2) with atom pair indices
        
    Returns:
        Tensor of shape (N_bonds,) with bond lengths
    """
    xyz1 = xyz[bond_indices[:, 0]]
    xyz2 = xyz[bond_indices[:, 1]]
    distances = torch.sqrt(torch.sum((xyz1 - xyz2) ** 2, dim=1))
    return distances


def test_bond_length_computation():
    """Test that bond lengths can be computed from restraints."""
    print("\n" + "="*80)
    print("Test: Bond Length Computation")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.bond_indices is None:
        print("Warning: No bond restraints found, skipping test")
        return
    
    # Get coordinates
    xyz = model.xyz()
    
    # Compute bond lengths
    bond_lengths = compute_bond_lengths(xyz, restraints.bond_indices)
    
    print(f"\nComputed {len(bond_lengths)} bond lengths")
    print(f"  Min: {bond_lengths.min():.3f} Å")
    print(f"  Max: {bond_lengths.max():.3f} Å")
    print(f"  Mean: {bond_lengths.mean():.3f} Å")
    
    # Check that bond lengths are reasonable (typically 1.0 - 2.0 Å for most bonds)
    assert bond_lengths.min() > 0.5, "Bond lengths should be > 0.5 Å"
    assert bond_lengths.max() < 3.0, "Bond lengths should be < 3.0 Å"
    
    print(f"\n✓ Test passed: Bond lengths are in reasonable range")


def test_bond_length_deviations():
    """Test computation of bond length deviations from ideal values."""
    print("\n" + "="*80)
    print("Test: Bond Length Deviations")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.bond_indices is None:
        print("Warning: No bond restraints found, skipping test")
        return
    
    # Get coordinates
    xyz = model.xyz()
    
    # Compute bond lengths
    bond_lengths = compute_bond_lengths(xyz, restraints.bond_indices)
    
    # Compute deviations from ideal values
    deviations = bond_lengths - restraints.bond_references
    
    print(f"\nBond length deviations:")
    print(f"  Min deviation: {deviations.min():.3f} Å")
    print(f"  Max deviation: {deviations.max():.3f} Å")
    print(f"  Mean deviation: {deviations.mean():.3f} Å")
    print(f"  RMS deviation: {torch.sqrt((deviations**2).mean()):.3f} Å")
    
    # Compute normalized deviations (in units of sigma)
    normalized_deviations = deviations / restraints.bond_sigmas
    
    print(f"\nNormalized deviations (in σ units):")
    print(f"  Min: {normalized_deviations.min():.2f} σ")
    print(f"  Max: {normalized_deviations.max():.2f} σ")
    print(f"  Mean: {normalized_deviations.mean():.2f} σ")
    print(f"  RMS: {torch.sqrt((normalized_deviations**2).mean()):.2f} σ")
    
    # Count bonds with deviations > 3 sigma
    outliers = (torch.abs(normalized_deviations) > 3).sum()
    print(f"\nBonds with deviations > 3σ: {outliers} ({100*outliers/len(bond_lengths):.1f}%)")
    
    print(f"\n✓ Test passed: Bond length deviations computed successfully")


def test_bond_indices_validity():
    """Test that all bond indices are valid."""
    print("\n" + "="*80)
    print("Test: Bond Indices Validity")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.bond_indices is None:
        print("Warning: No bond restraints found, skipping test")
        return
    
    n_atoms = len(model.pdb)
    
    # Check that all indices are within bounds
    max_idx = restraints.bond_indices.max().item()
    min_idx = restraints.bond_indices.min().item()
    
    print(f"\nNumber of atoms: {n_atoms}")
    print(f"Bond indices range: {min_idx} to {max_idx}")
    
    assert min_idx >= 0, "All indices should be >= 0"
    assert max_idx < n_atoms, f"All indices should be < {n_atoms}"
    
    # Check that there are no self-bonds
    self_bonds = (restraints.bond_indices[:, 0] == restraints.bond_indices[:, 1]).sum()
    print(f"Self-bonds (same atom bonded to itself): {self_bonds}")
    assert self_bonds == 0, "There should be no self-bonds"
    
    print(f"\n✓ Test passed: All bond indices are valid")


def test_bond_reference_values():
    """Test that bond reference values are reasonable."""
    print("\n" + "="*80)
    print("Test: Bond Reference Values")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.bond_indices is None:
        print("Warning: No bond restraints found, skipping test")
        return
    
    print(f"\nBond reference values:")
    print(f"  Min: {restraints.bond_references.min():.3f} Å")
    print(f"  Max: {restraints.bond_references.max():.3f} Å")
    print(f"  Mean: {restraints.bond_references.mean():.3f} Å")
    
    # Check that reference values are reasonable
    assert restraints.bond_references.min() > 0.8, "Bond references should be > 0.8 Å"
    assert restraints.bond_references.max() < 3.0, "Bond references should be < 3.0 Å"
    
    print(f"\nBond sigma values:")
    print(f"  Min: {restraints.bond_sigmas.min():.4f} Å")
    print(f"  Max: {restraints.bond_sigmas.max():.4f} Å")
    print(f"  Mean: {restraints.bond_sigmas.mean():.4f} Å")
    
    # Check that sigmas are positive and reasonable
    assert restraints.bond_sigmas.min() > 0, "Bond sigmas should be positive"
    assert restraints.bond_sigmas.max() < 0.5, "Bond sigmas should be < 0.5 Å"
    
    print(f"\n✓ Test passed: Bond reference values are reasonable")


def test_bond_gradient():
    """Test that gradients can be computed through bond length calculations."""
    print("\n" + "="*80)
    print("Test: Bond Length Gradients")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.bond_indices is None:
        print("Warning: No bond restraints found, skipping test")
        return
    
    # Get coordinates with gradient tracking
    xyz = model.xyz().detach().clone()
    xyz.requires_grad = True
    
    # Compute bond lengths
    bond_lengths = compute_bond_lengths(xyz, restraints.bond_indices)
    
    # Compute a simple loss (sum of squared deviations)
    deviations = (bond_lengths - restraints.bond_references) / restraints.bond_sigmas
    loss = (deviations ** 2).sum()
    
    print(f"\nInitial loss: {loss.item():.3f}")
    
    # Compute gradients
    loss.backward()
    
    assert xyz.grad is not None, "Gradients should be computed"
    
    grad_norm = torch.norm(xyz.grad)
    print(f"Gradient norm: {grad_norm:.3f}")
    
    assert grad_norm > 0, "Gradient norm should be positive"
    
    print(f"\n✓ Test passed: Gradients computed successfully")


if __name__ == '__main__':
    print("\n" + "#"*80)
    print("# Running Bond Length Restraints Tests")
    print("#"*80)
    
    try:
        # Run tests
        test_bond_length_computation()
        test_bond_length_deviations()
        test_bond_indices_validity()
        test_bond_reference_values()
        test_bond_gradient()
        
        print("\n" + "#"*80)
        print("# All tests passed! ✓")
        print("#"*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
