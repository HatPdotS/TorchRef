"""
Test suite for angle restraints functionality.

This module tests that angle restraints are correctly extracted from the CIF
dictionary and that the computed angles match expected values.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from torchref.model import Model
from torchref.restraints import Restraints


def compute_angles(xyz, angle_indices):
    """
    Compute angles from coordinates and indices.
    
    Args:
        xyz: Tensor of shape (N_atoms, 3) with atomic coordinates
        angle_indices: Tensor of shape (N_angles, 3) with atom triplet indices
        
    Returns:
        Tensor of shape (N_angles,) with angles in degrees
    """
    xyz1 = xyz[angle_indices[:, 0]]
    xyz2 = xyz[angle_indices[:, 1]]  # Vertex atom
    xyz3 = xyz[angle_indices[:, 2]]
    
    # Compute vectors from vertex to other atoms
    v1 = xyz1 - xyz2
    v2 = xyz3 - xyz2
    
    # Normalize vectors
    v1_norm = v1 / torch.sqrt(torch.sum(v1**2, dim=1, keepdim=True))
    v2_norm = v2 / torch.sqrt(torch.sum(v2**2, dim=1, keepdim=True))
    
    # Compute dot product
    dot_product = torch.sum(v1_norm * v2_norm, dim=1)
    
    # Clamp to avoid numerical issues with arccos
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    
    # Compute angle in radians then convert to degrees
    angles_rad = torch.arccos(dot_product)
    angles_deg = angles_rad * 180.0 / np.pi
    
    return angles_deg


def test_angle_computation():
    """Test that angles can be computed from restraints."""
    print("\n" + "="*80)
    print("Test: Angle Computation")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.angle_indices is None:
        print("Warning: No angle restraints found, skipping test")
        return
    
    # Get coordinates
    xyz = model.xyz()
    
    # Compute angles
    angles = compute_angles(xyz, restraints.angle_indices)
    
    print(f"\nComputed {len(angles)} angles")
    print(f"  Min: {angles.min():.2f}°")
    print(f"  Max: {angles.max():.2f}°")
    print(f"  Mean: {angles.mean():.2f}°")
    
    # Check that angles are in valid range [0, 180]
    assert angles.min() >= 0, "Angles should be >= 0°"
    assert angles.max() <= 180, "Angles should be <= 180°"
    
    print(f"\n✓ Test passed: Angles are in valid range")


def test_angle_deviations():
    """Test computation of angle deviations from ideal values."""
    print("\n" + "="*80)
    print("Test: Angle Deviations")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.angle_indices is None:
        print("Warning: No angle restraints found, skipping test")
        return
    
    # Get coordinates
    xyz = model.xyz()
    
    # Compute angles
    angles = compute_angles(xyz, restraints.angle_indices)
    
    # Compute deviations from ideal values
    deviations = angles - restraints.angle_references
    
    print(f"\nAngle deviations:")
    print(f"  Min deviation: {deviations.min():.2f}°")
    print(f"  Max deviation: {deviations.max():.2f}°")
    print(f"  Mean deviation: {deviations.mean():.2f}°")
    print(f"  RMS deviation: {torch.sqrt((deviations**2).mean()):.2f}°")
    
    # Compute normalized deviations (in units of sigma)
    normalized_deviations = deviations / restraints.angle_sigmas
    
    print(f"\nNormalized deviations (in σ units):")
    print(f"  Min: {normalized_deviations.min():.2f} σ")
    print(f"  Max: {normalized_deviations.max():.2f} σ")
    print(f"  Mean: {normalized_deviations.mean():.2f} σ")
    print(f"  RMS: {torch.sqrt((normalized_deviations**2).mean()):.2f} σ")
    
    # Count angles with deviations > 3 sigma
    outliers = (torch.abs(normalized_deviations) > 3).sum()
    print(f"\nAngles with deviations > 3σ: {outliers} ({100*outliers/len(angles):.1f}%)")
    
    print(f"\n✓ Test passed: Angle deviations computed successfully")


def test_angle_indices_validity():
    """Test that all angle indices are valid."""
    print("\n" + "="*80)
    print("Test: Angle Indices Validity")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.angle_indices is None:
        print("Warning: No angle restraints found, skipping test")
        return
    
    n_atoms = len(model.pdb)
    
    # Check that all indices are within bounds
    max_idx = restraints.angle_indices.max().item()
    min_idx = restraints.angle_indices.min().item()
    
    print(f"\nNumber of atoms: {n_atoms}")
    print(f"Angle indices range: {min_idx} to {max_idx}")
    
    assert min_idx >= 0, "All indices should be >= 0"
    assert max_idx < n_atoms, f"All indices should be < {n_atoms}"
    
    # Check that there are no degenerate angles (same atom repeated)
    degenerate1 = (restraints.angle_indices[:, 0] == restraints.angle_indices[:, 1]).sum()
    degenerate2 = (restraints.angle_indices[:, 1] == restraints.angle_indices[:, 2]).sum()
    degenerate3 = (restraints.angle_indices[:, 0] == restraints.angle_indices[:, 2]).sum()
    
    total_degenerate = degenerate1 + degenerate2 + degenerate3
    print(f"Degenerate angles (atoms repeated): {total_degenerate}")
    assert total_degenerate == 0, "There should be no degenerate angles"
    
    print(f"\n✓ Test passed: All angle indices are valid")


def test_angle_reference_values():
    """Test that angle reference values are reasonable."""
    print("\n" + "="*80)
    print("Test: Angle Reference Values")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.angle_indices is None:
        print("Warning: No angle restraints found, skipping test")
        return
    
    print(f"\nAngle reference values:")
    print(f"  Min: {restraints.angle_references.min():.2f}°")
    print(f"  Max: {restraints.angle_references.max():.2f}°")
    print(f"  Mean: {restraints.angle_references.mean():.2f}°")
    
    # Check that reference values are in valid range
    assert restraints.angle_references.min() >= 0, "Angle references should be >= 0°"
    assert restraints.angle_references.max() <= 180, "Angle references should be <= 180°"
    
    # Most angles should be between 90 and 180 degrees
    typical_range = ((restraints.angle_references >= 90) & 
                     (restraints.angle_references <= 180)).float().mean()
    print(f"  Fraction in typical range [90°, 180°]: {typical_range:.1%}")
    
    print(f"\nAngle sigma values:")
    print(f"  Min: {restraints.angle_sigmas.min():.3f}°")
    print(f"  Max: {restraints.angle_sigmas.max():.3f}°")
    print(f"  Mean: {restraints.angle_sigmas.mean():.3f}°")
    
    # Check that sigmas are positive and reasonable
    assert restraints.angle_sigmas.min() > 0, "Angle sigmas should be positive"
    assert restraints.angle_sigmas.max() < 20, "Angle sigmas should be < 20°"
    
    print(f"\n✓ Test passed: Angle reference values are reasonable")


def test_angle_gradient():
    """Test that gradients can be computed through angle calculations."""
    print("\n" + "="*80)
    print("Test: Angle Gradients")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    if restraints.angle_indices is None:
        print("Warning: No angle restraints found, skipping test")
        return
    
    # Get coordinates with gradient tracking
    xyz = model.xyz().detach().clone()
    xyz.requires_grad = True
    
    # Compute angles
    angles = compute_angles(xyz, restraints.angle_indices)
    
    # Compute a simple loss (sum of squared deviations)
    deviations = (angles - restraints.angle_references) / restraints.angle_sigmas
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
    print("# Running Angle Restraints Tests")
    print("#"*80)
    
    try:
        # Run tests
        test_angle_computation()
        test_angle_deviations()
        test_angle_indices_validity()
        test_angle_reference_values()
        test_angle_gradient()
        
        print("\n" + "#"*80)
        print("# All tests passed! ✓")
        print("#"*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
