"""
Test suite for the Restraints class creation and initialization.

This module tests the basic functionality of creating a Restraints instance,
loading CIF files, and verifying that restraints are properly parsed and stored.
"""

import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from torchref.model import Model
from torchref.restraints import Restraints


def test_restraints_creation():
    """Test basic creation of Restraints instance."""
    print("\n" + "="*80)
    print("Test: Restraints Creation")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    print(f"Loaded model with {len(model.pdb)} atoms")
    print(f"Model coordinates shape: {model.xyz().shape}")
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    print(f"\nRestraints created successfully!")
    print(restraints)
    
    # Verify that restraints were created
    assert restraints.bond_indices is not None, "Bond indices should not be None"
    assert restraints.angle_indices is not None, "Angle indices should not be None"
    
    print(f"\n✓ Test passed: Restraints created successfully")
    
    return restraints


def test_restraints_structure():
    """Test that restraints have the correct structure."""
    print("\n" + "="*80)
    print("Test: Restraints Structure")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    # Test bond restraints structure
    if restraints.bond_indices is not None:
        print(f"\nBond restraints:")
        print(f"  Indices shape: {restraints.bond_indices.shape}")
        print(f"  Expected shape: (N, 2)")
        assert restraints.bond_indices.dim() == 2, "Bond indices should be 2D"
        assert restraints.bond_indices.shape[1] == 2, "Bond indices should have 2 columns"
        
        print(f"  References shape: {restraints.bond_references.shape}")
        assert restraints.bond_references.dim() == 1, "Bond references should be 1D"
        assert restraints.bond_references.shape[0] == restraints.bond_indices.shape[0], \
            "References length should match indices"
        
        print(f"  Sigmas shape: {restraints.bond_sigmas.shape}")
        assert restraints.bond_sigmas.dim() == 1, "Bond sigmas should be 1D"
        assert restraints.bond_sigmas.shape[0] == restraints.bond_indices.shape[0], \
            "Sigmas length should match indices"
        
        print(f"  ✓ Bond restraints structure correct")
    
    # Test angle restraints structure
    if restraints.angle_indices is not None:
        print(f"\nAngle restraints:")
        print(f"  Indices shape: {restraints.angle_indices.shape}")
        print(f"  Expected shape: (N, 3)")
        assert restraints.angle_indices.dim() == 2, "Angle indices should be 2D"
        assert restraints.angle_indices.shape[1] == 3, "Angle indices should have 3 columns"
        
        print(f"  References shape: {restraints.angle_references.shape}")
        assert restraints.angle_references.dim() == 1, "Angle references should be 1D"
        assert restraints.angle_references.shape[0] == restraints.angle_indices.shape[0], \
            "References length should match indices"
        
        print(f"  Sigmas shape: {restraints.angle_sigmas.shape}")
        assert restraints.angle_sigmas.dim() == 1, "Angle sigmas should be 1D"
        assert restraints.angle_sigmas.shape[0] == restraints.angle_indices.shape[0], \
            "Sigmas length should match indices"
        
        print(f"  ✓ Angle restraints structure correct")
    
    # Test torsion restraints structure
    if restraints.torsion_indices is not None:
        print(f"\nTorsion restraints:")
        print(f"  Indices shape: {restraints.torsion_indices.shape}")
        print(f"  Expected shape: (N, 4)")
        assert restraints.torsion_indices.dim() == 2, "Torsion indices should be 2D"
        assert restraints.torsion_indices.shape[1] == 4, "Torsion indices should have 4 columns"
        
        print(f"  References shape: {restraints.torsion_references.shape}")
        assert restraints.torsion_references.dim() == 1, "Torsion references should be 1D"
        assert restraints.torsion_references.shape[0] == restraints.torsion_indices.shape[0], \
            "References length should match indices"
        
        print(f"  Sigmas shape: {restraints.torsion_sigmas.shape}")
        assert restraints.torsion_sigmas.dim() == 1, "Torsion sigmas should be 1D"
        assert restraints.torsion_sigmas.shape[0] == restraints.torsion_indices.shape[0], \
            "Sigmas length should match indices"
        
        print(f"  ✓ Torsion restraints structure correct")
    
    print(f"\n✓ Test passed: All restraint structures are correct")


def test_restraints_device():
    """Test that restraints can be moved between devices."""
    print("\n" + "="*80)
    print("Test: Device Management")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    # Check initial device
    if restraints.bond_indices is not None:
        initial_device = restraints.bond_indices.device
        print(f"Initial device: {initial_device}")
        assert str(initial_device) == 'cpu', "Initial device should be CPU"
    
    # Test CUDA if available
    if torch.cuda.is_available():
        print(f"\nMoving to CUDA...")
        restraints.cuda()
        
        if restraints.bond_indices is not None:
            assert restraints.bond_indices.device.type == 'cuda', "Should be on CUDA"
            assert restraints.bond_references.device.type == 'cuda', "Should be on CUDA"
            assert restraints.bond_sigmas.device.type == 'cuda', "Should be on CUDA"
        
        print(f"✓ Successfully moved to CUDA")
        
        # Move back to CPU
        print(f"\nMoving back to CPU...")
        restraints.cpu()
        
        if restraints.bond_indices is not None:
            assert restraints.bond_indices.device.type == 'cpu', "Should be on CPU"
            assert restraints.bond_references.device.type == 'cpu', "Should be on CPU"
            assert restraints.bond_sigmas.device.type == 'cpu', "Should be on CPU"
        
        print(f"✓ Successfully moved back to CPU")
    else:
        print(f"\nCUDA not available, skipping CUDA test")
    
    print(f"\n✓ Test passed: Device management works correctly")


def test_restraints_summary():
    """Test the summary method."""
    print("\n" + "="*80)
    print("Test: Summary Method")
    print("="*80)
    
    # Load test model
    model = Model()
    test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/test_all.pdb'
    model.load_pdb_from_file(test_pdb)
    
    # Create restraints
    cif_path = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/Merged_restraints_all_opened.cif'
    restraints = Restraints(model, cif_path)
    
    # Call summary method
    restraints.summary()
    
    print(f"\n✓ Test passed: Summary method works correctly")


if __name__ == '__main__':
    print("\n" + "#"*80)
    print("# Running Restraints Creation Tests")
    print("#"*80)
    
    try:
        # Run tests
        restraints = test_restraints_creation()
        test_restraints_structure()
        test_restraints_device()
        test_restraints_summary()
        
        print("\n" + "#"*80)
        print("# All tests passed! ✓")
        print("#"*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
