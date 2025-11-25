#!/usr/bin/env python3
"""
Test the integration of OccupancyTensor with the Model class.

This test verifies:
1. Model loads with OccupancyTensor correctly
2. Residue-level sharing groups are created
3. Only partial occupancies are refined
4. Collapsed storage is used
"""

import torch
import sys
import os
import tempfile

sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import Model, OccupancyTensor

def create_test_pdb():
    """Create a minimal test PDB file."""
    pdb_content = """CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  1.00 20.00           C
ATOM      3  C   ALA A   1      12.000  12.000  12.000  1.00 20.00           C
ATOM      4  O   ALA A   1      13.000  13.000  13.000  1.00 20.00           O
ATOM      5  CB  ALA A   1      14.000  14.000  14.000  1.00 20.00           C
ATOM      6  N   GLY A   2      15.000  15.000  15.000  0.80 25.00           N
ATOM      7  CA  GLY A   2      16.000  16.000  16.000  0.80 25.00           C
ATOM      8  C   GLY A   2      17.000  17.000  17.000  0.80 25.00           C
ATOM      9  O   GLY A   2      18.000  18.000  18.000  0.80 25.00           O
ATOM     10  N  AVAL A   3      19.000  19.000  19.000  0.60 30.00           N
ATOM     11  CA AVAL A   3      20.000  20.000  20.000  0.60 30.00           C
ATOM     12  C  AVAL A   3      21.000  21.000  21.000  0.60 30.00           C
ATOM     13  O  AVAL A   3      22.000  22.000  22.000  0.60 30.00           O
ATOM     14  CB AVAL A   3      23.000  23.000  23.000  0.60 30.00           C
ATOM     15  N  BVAL A   3      19.000  19.000  19.000  0.40 30.00           N
ATOM     16  CA BVAL A   3      20.000  20.000  20.000  0.40 30.00           C
ATOM     17  C  BVAL A   3      21.000  21.000  21.000  0.40 30.00           C
ATOM     18  O  BVAL A   3      22.000  22.000  22.000  0.40 30.00           C
ATOM     19  CB BVAL A   3      23.000  23.000  23.000  0.40 30.00           C
END
"""
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        return f.name

def test_model_integration():
    """Test Model class integration with OccupancyTensor."""
    print("\n" + "="*70)
    print("TEST: Model Integration with OccupancyTensor")
    print("="*70)
    
    # Create test PDB
    pdb_file = create_test_pdb()
    
    try:
        # Load model
        print("\nLoading model...")
        model = Model(verbose=1)
        model.load_pdb_from_file(pdb_file)
        
        # Check that occupancy is OccupancyTensor
        print(f"\nOccupancy type: {type(model.occupancy)}")
        assert isinstance(model.occupancy, OccupancyTensor), "Occupancy should be OccupancyTensor"
        
        # Check shapes
        print(f"Full shape: {model.occupancy.shape}")
        print(f"Collapsed shape: {model.occupancy.collapsed_shape}")
        print(f"Number of atoms: {len(model.pdb)}")
        
        assert model.occupancy.shape[0] == len(model.pdb), "Full shape should match atom count"
        
        # Check that collapsed storage is used
        n_atoms = model.occupancy.shape[0]
        n_collapsed = model.occupancy.collapsed_shape[0]
        print(f"\nCompression: {n_atoms} atoms → {n_collapsed} parameters")
        
        # Should have compression since we have residues
        assert n_collapsed < n_atoms, "Should have compression with residue grouping"
        
        # Check refinable count
        n_refinable = model.occupancy.get_refinable_count()
        print(f"Refinable parameters: {n_refinable}")
        
        # Should only refine partial occupancies (residue 2 and 3)
        # Residue 1 has occ=1.0 (should be fixed)
        # Residue 2 has occ=0.8 (should be refinable)
        # Residue 3 has altlocs with occ=0.6/0.4 (should be refinable)
        assert n_refinable > 0, "Should have some refinable parameters"
        assert n_refinable < n_collapsed, "Not all parameters should be refinable"
        
        # Check occupancy values
        occ_values = model.occupancy()
        print(f"\nOccupancy values shape: {occ_values.shape}")
        print(f"Occupancy range: [{occ_values.min():.3f}, {occ_values.max():.3f}]")
        
        # Check that values are in [0, 1]
        assert torch.all(occ_values >= 0) and torch.all(occ_values <= 1), "Occupancies should be in [0, 1]"
        
        # Check residue grouping
        # Residue 1 (atoms 0-4): all should have same occupancy
        res1_occ = occ_values[0:5]
        print(f"\nResidue 1 occupancies: {res1_occ}")
        assert torch.allclose(res1_occ, res1_occ[0].expand(5), atol=1e-5), "Residue 1 should have uniform occupancy"
        
        # Residue 2 (atoms 5-8): all should have same occupancy
        res2_occ = occ_values[5:9]
        print(f"Residue 2 occupancies: {res2_occ}")
        assert torch.allclose(res2_occ, res2_occ[0].expand(4), atol=1e-5), "Residue 2 should have uniform occupancy"
        
        # Residue 3 conf A (atoms 9-13): all should have same occupancy
        res3a_occ = occ_values[9:14]
        print(f"Residue 3A occupancies: {res3a_occ}")
        assert torch.allclose(res3a_occ, res3a_occ[0].expand(5), atol=1e-5), "Residue 3A should have uniform occupancy"
        
        # Residue 3 conf B (atoms 14-18): all should have same occupancy
        res3b_occ = occ_values[14:19]
        print(f"Residue 3B occupancies: {res3b_occ}")
        assert torch.allclose(res3b_occ, res3b_occ[0].expand(5), atol=1e-5), "Residue 3B should have uniform occupancy"
        
        # Test optimization
        print("\nTesting optimization...")
        optimizer = torch.optim.Adam([model.occupancy.refinable_params], lr=0.01)
        
        initial_occ = model.occupancy().clone()
        
        # Run a few optimization steps
        for i in range(10):
            optimizer.zero_grad()
            occ = model.occupancy()
            # Dummy loss: push refinable occupancies toward 0.5
            loss = ((occ[5:] - 0.5) ** 2).sum()  # Only for residues 2 and 3
            loss.backward()
            optimizer.step()
        
        final_occ = model.occupancy()
        
        # Check that residue 1 (fixed) didn't change
        print(f"\nResidue 1 initial: {initial_occ[0].item():.4f}, final: {final_occ[0].item():.4f}")
        assert torch.allclose(initial_occ[0:5], final_occ[0:5], atol=1e-4), "Fixed residue 1 should not change"
        
        # Check that residue 2 (refinable) changed
        print(f"Residue 2 initial: {initial_occ[5].item():.4f}, final: {final_occ[5].item():.4f}")
        assert not torch.allclose(initial_occ[5], final_occ[5], atol=1e-2), "Refinable residue 2 should change"
        
        # Check sharing is maintained after optimization
        assert torch.allclose(final_occ[0:5], final_occ[0].expand(5), atol=1e-5), "Residue 1 sharing maintained"
        assert torch.allclose(final_occ[5:9], final_occ[5].expand(4), atol=1e-5), "Residue 2 sharing maintained"
        
        print("\n" + "="*70)
        print("✓ ALL INTEGRATION TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print(f"  ✓ OccupancyTensor correctly integrated")
        print(f"  ✓ Collapsed storage working ({n_atoms} → {n_collapsed})")
        print(f"  ✓ Residue-level sharing enforced")
        print(f"  ✓ Only partial occupancies refined")
        print(f"  ✓ Fixed parameters stay fixed")
        print(f"  ✓ Optimization works correctly")
        
        return True
        
    finally:
        # Clean up temp file
        if os.path.exists(pdb_file):
            os.remove(pdb_file)

if __name__ == '__main__':
    try:
        success = test_model_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
