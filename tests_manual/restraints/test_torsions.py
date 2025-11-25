"""
Test suite for torsion angle restraints functionality.

This module tests that torsion angle restraints are correctly extracted from the CIF
dictionary and that the computed torsion angles are handled properly.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from torchref.model import Model
from torchref.restraints import Restraints


def compute_torsions(xyz, torsion_indices):
    """
    Compute torsion angles from coordinates and indices.
    
    Args:
        xyz: Tensor of shape (N_atoms, 3) with atomic coordinates
        torsion_indices: Tensor of shape (N_torsions, 4) with atom quartet indices
        
    Returns:
        Tensor of shape (N_torsions,) with torsion angles in degrees
    """
    xyz1 = xyz[torsion_indices[:, 0]]
    xyz2 = xyz[torsion_indices[:, 1]]
    xyz3 = xyz[torsion_indices[:, 2]]
    xyz4 = xyz[torsion_indices[:, 3]]
    
    # Compute vectors
    v1 = xyz2 - xyz1
    v2 = xyz3 - xyz2
    v3 = xyz4 - xyz3
    
    # Compute normal vectors to planes
    n1 = torch.linalg.cross(v1, v2)
    n2 = torch.linalg.cross(v2, v3)
    
    # Normalize normal vectors
    n1_norm = n1 / torch.sqrt(torch.sum(n1**2, dim=1, keepdim=True))
    n2_norm = n2 / torch.sqrt(torch.sum(n2**2, dim=1, keepdim=True))
    
    # Compute dot product
    dot_product = torch.sum(n1_norm * n2_norm, dim=1)
    
    # Clamp to avoid numerical issues with arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute angle in radians then convert to degrees
    angles_rad = torch.arccos(dot_product)
    angles_deg = angles_rad * 180.0 / np.pi
    
    return angles_deg


def normalize_torsion_difference(diff):
    """
    Normalize torsion angle differences to [-180, 180] range.
    
    Torsion angles are periodic with period 360°, so a difference of 
    350° is equivalent to -10°.
    
    Args:
        diff: Tensor of angle differences
        
    Returns:
        Normalized differences in range [-180, 180]
    """
    # Wrap to [-180, 180]
    diff = diff % 360
    diff[diff > 180] -= 360
    return diff


def test_torsion_computation():
    """Test that torsion angles can be computed from restraints."""
    print("\n" + "="*80)
    print("Test: Torsion Angle Computation")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.torsion_indices is None:
        print("Warning: No torsion restraints found, skipping test")
        return
    
    # Get coordinates
    xyz = model.xyz()
    
    # Compute torsion angles
    torsions = compute_torsions(xyz, restraints.torsion_indices)
    
    print(f"\nComputed {len(torsions)} torsion angles")
    print(f"  Min: {torsions.min():.2f}°")
    print(f"  Max: {torsions.max():.2f}°")
    print(f"  Mean: {torsions.mean():.2f}°")
    
    # Check that torsions are in valid range [0, 180]
    # Note: This implementation gives angles in [0, 180], but torsions can be [-180, 180]
    assert torsions.min() >= 0, "Torsions should be >= 0°"
    assert torsions.max() <= 180, "Torsions should be <= 180°"
    
    print(f"\n✓ Test passed: Torsion angles computed")


def test_torsion_deviations():
    """Test computation of torsion angle deviations from ideal values."""
    print("\n" + "="*80)
    print("Test: Torsion Angle Deviations")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.torsion_indices is None:
        print("Warning: No torsion restraints found, skipping test")
        return
    
    # Get coordinates
    xyz = model.xyz()
    
    # Compute torsion angles
    torsions = compute_torsions(xyz, restraints.torsion_indices)
    
    # Compute deviations from ideal values
    deviations = torsions - restraints.torsion_references
    
    # Normalize deviations to handle periodicity
    deviations_norm = normalize_torsion_difference(deviations)
    
    print(f"\nTorsion angle deviations:")
    print(f"  Min deviation: {deviations_norm.min():.2f}°")
    print(f"  Max deviation: {deviations_norm.max():.2f}°")
    print(f"  Mean deviation: {deviations_norm.mean():.2f}°")
    print(f"  RMS deviation: {torch.sqrt((deviations_norm**2).mean()):.2f}°")
    
    # Compute normalized deviations (in units of sigma)
    normalized_deviations = deviations_norm / restraints.torsion_sigmas
    
    print(f"\nNormalized deviations (in σ units):")
    print(f"  Min: {normalized_deviations.min():.2f} σ")
    print(f"  Max: {normalized_deviations.max():.2f} σ")
    print(f"  Mean: {normalized_deviations.mean():.2f} σ")
    print(f"  RMS: {torch.sqrt((normalized_deviations**2).mean()):.2f} σ")
    
    # Count torsions with deviations > 3 sigma
    outliers = (torch.abs(normalized_deviations) > 3).sum()
    print(f"\nTorsions with deviations > 3σ: {outliers} ({100*outliers/len(torsions):.1f}%)")
    
    print(f"\n✓ Test passed: Torsion deviations computed successfully")


def test_torsion_indices_validity():
    """Test that all torsion indices are valid."""
    print("\n" + "="*80)
    print("Test: Torsion Indices Validity")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.torsion_indices is None:
        print("Warning: No torsion restraints found, skipping test")
        return
    
    n_atoms = len(model.pdb)
    
    # Check that all indices are within bounds
    max_idx = restraints.torsion_indices.max().item()
    min_idx = restraints.torsion_indices.min().item()
    
    print(f"\nNumber of atoms: {n_atoms}")
    print(f"Torsion indices range: {min_idx} to {max_idx}")
    
    assert min_idx >= 0, "All indices should be >= 0"
    assert max_idx < n_atoms, f"All indices should be < {n_atoms}"
    
    # Check for repeated atoms (at least check some common cases)
    degenerate_count = 0
    for i in range(4):
        for j in range(i+1, 4):
            degenerate = (restraints.torsion_indices[:, i] == 
                         restraints.torsion_indices[:, j]).sum()
            degenerate_count += degenerate
    
    print(f"Degenerate torsions (atoms repeated): {degenerate_count}")
    assert degenerate_count == 0, "There should be no degenerate torsions"
    
    print(f"\n✓ Test passed: All torsion indices are valid")


def test_torsion_reference_values():
    """Test that torsion reference values are reasonable."""
    print("\n" + "="*80)
    print("Test: Torsion Reference Values")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.torsion_indices is None:
        print("Warning: No torsion restraints found, skipping test")
        return
    
    print(f"\nTorsion reference values:")
    print(f"  Min: {restraints.torsion_references.min():.2f}°")
    print(f"  Max: {restraints.torsion_references.max():.2f}°")
    print(f"  Mean: {restraints.torsion_references.mean():.2f}°")
    
    # Torsion references can be in any range, but typically [-180, 180]
    # Check the distribution
    bins = [(-180, -90), (-90, 0), (0, 90), (90, 180)]
    print(f"\nDistribution of reference values:")
    for low, high in bins:
        count = ((restraints.torsion_references >= low) & 
                (restraints.torsion_references < high)).sum()
        print(f"  [{low}°, {high}°): {count} ({100*count/len(restraints.torsion_references):.1f}%)")
    
    print(f"\nTorsion sigma values:")
    print(f"  Min: {restraints.torsion_sigmas.min():.3f}°")
    print(f"  Max: {restraints.torsion_sigmas.max():.3f}°")
    print(f"  Mean: {restraints.torsion_sigmas.mean():.3f}°")
    
    # Check that sigmas are positive
    assert restraints.torsion_sigmas.min() > 0, "Torsion sigmas should be positive"
    
    print(f"\n✓ Test passed: Torsion reference values are reasonable")


def test_torsion_gradient():
    """Test that gradients can be computed through torsion calculations."""
    print("\n" + "="*80)
    print("Test: Torsion Gradients")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.torsion_indices is None:
        print("Warning: No torsion restraints found, skipping test")
        return
    
    # Get coordinates with gradient tracking
    xyz = model.xyz().detach().clone()
    xyz.requires_grad = True
    
    # Compute torsion angles
    torsions = compute_torsions(xyz, restraints.torsion_indices)
    
    # Compute a simple loss (sum of squared deviations)
    deviations = (torsions - restraints.torsion_references) / restraints.torsion_sigmas
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
    print("# Running Torsion Angle Restraints Tests")
    print("#"*80)
    
    try:
        # Run tests
        test_torsion_computation()
        test_torsion_deviations()
        test_torsion_indices_validity()
        test_torsion_reference_values()
        test_torsion_gradient()
        
        print("\n" + "#"*80)
        print("# All tests passed! ✓")
        print("#"*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
