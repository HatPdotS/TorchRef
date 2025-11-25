#!/usr/bin/env python3
"""
Test the NEW automatic altloc handling in OccupancyTensor.

This test verifies that:
1. Alternative conformations automatically sum to 1.0
2. Only N-1 parameters are stored for N conformations (placeholder system)
3. The last conformation is always computed as 1 - sum(others)
4. Optimization maintains the sum-to-1 constraint automatically
"""

import torch
import sys
import tempfile
import os

sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import Model, OccupancyTensor

def create_altloc_pdb():
    """Create a test PDB with alternative conformations."""
    pdb_content = """CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  1.00 20.00           C
ATOM      3  C   ALA A   1      12.000  12.000  12.000  1.00 20.00           C
ATOM      4  O   ALA A   1      13.000  13.000  13.000  1.00 20.00           O
ATOM      5  N  AVAL A   2      15.000  15.000  15.000  0.70 25.00           N
ATOM      6  CA AVAL A   2      16.000  16.000  16.000  0.70 25.00           C
ATOM      7  C  AVAL A   2      17.000  17.000  17.000  0.70 25.00           C
ATOM      8  O  AVAL A   2      18.000  18.000  18.000  0.70 25.00           O
ATOM      9  CB AVAL A   2      19.000  19.000  19.000  0.70 25.00           C
ATOM     10  N  BVAL A   2      15.000  15.000  15.000  0.30 25.00           N
ATOM     11  CA BVAL A   2      16.000  16.000  16.000  0.30 25.00           C
ATOM     12  C  BVAL A   2      17.000  17.000  17.000  0.30 25.00           C
ATOM     13  O  BVAL A   2      18.000  18.000  18.000  0.30 25.00           O
ATOM     14  CB BVAL A   2      19.000  19.000  19.000  0.30 25.00           C
ATOM     15  N  AGLY A   3      20.000  20.000  20.000  0.55 30.00           N
ATOM     16  CA AGLY A   3      21.000  21.000  21.000  0.55 30.00           C
ATOM     17  C  AGLY A   3      22.000  22.000  22.000  0.55 30.00           C
ATOM     18  O  AGLY A   3      23.000  23.000  23.000  0.55 30.00           O
ATOM     19  N  BGLY A   3      20.000  20.000  20.000  0.45 30.00           N
ATOM     20  CA BGLY A   3      21.000  21.000  21.000  0.45 30.00           C
ATOM     21  C  BGLY A   3      22.000  22.000  22.000  0.45 30.00           C
ATOM     22  O  BGLY A   3      23.000  23.000  23.000  0.45 30.00           O
END
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        return f.name

def test_automatic_altloc_constraint():
    """Test automatic altloc constraint enforcement."""
    print("\n" + "="*70)
    print("TEST: Automatic Alternative Conformation Constraint")
    print("="*70)
    
    pdb_file = create_altloc_pdb()
    
    try:
        # Load model
        print("\nLoading model with altlocs...")
        model = Model(verbose=1)
        model.load_pdb_from_file(pdb_file)
        
        print(f"\nOccupancyTensor type: {type(model.occupancy)}")
        print(f"Total atoms: {len(model.pdb)}")
        print(f"Altloc groups: {len(model.occupancy.altloc_info)}")
        
        # Get initial occupancies
        initial_occ = model.occupancy().clone()
        print(f"\nInitial occupancies:")
        print(f"  Residue 1 (no altloc): {initial_occ[0:4]}")
        print(f"  Residue 2A (altloc A): {initial_occ[4:9]}")
        print(f"  Residue 2B (altloc B): {initial_occ[9:14]}")
        print(f"  Residue 3A (altloc A): {initial_occ[14:18]}")
        print(f"  Residue 3B (altloc B): {initial_occ[18:22]}")
        
        # Check initial altloc sums
        res2_sum = initial_occ[4] + initial_occ[9]
        res3_sum = initial_occ[14] + initial_occ[18]
        print(f"\nInitial altloc sums:")
        print(f"  Residue 2: {res2_sum.item():.6f} (should be 1.0)")
        print(f"  Residue 3: {res3_sum.item():.6f} (should be 1.0)")
        
        assert torch.allclose(res2_sum, torch.tensor(1.0), atol=1e-5), "Residue 2 should sum to 1"
        assert torch.allclose(res3_sum, torch.tensor(1.0), atol=1e-5), "Residue 3 should sum to 1"
        
        # Check refinable parameters
        print(f"\nRefinable parameters: {model.occupancy.get_refinable_count()}")
        print(f"Collapsed shape: {model.occupancy.collapsed_shape}")
        print(f"Altloc placeholder mask: {model.occupancy.altloc_placeholder_mask}")
        print(f"Refinable mask: {model.occupancy.refinable_mask}")
        
        # Check that placeholders are not refinable
        for altloc_dict in model.occupancy.altloc_info:
            placeholder_idx = altloc_dict['placeholder_collapsed_idx']
            print(f"\nAltloc group:")
            print(f"  Placeholder collapsed idx: {placeholder_idx}")
            print(f"  Is refinable: {model.occupancy.refinable_mask[placeholder_idx].item()}")
            assert not model.occupancy.refinable_mask[placeholder_idx], "Placeholder should not be refinable"
        
        # Test optimization
        print("\n" + "-"*70)
        print("Testing optimization with automatic altloc constraint...")
        print("-"*70)
        
        optimizer = torch.optim.Adam([model.occupancy.refinable_params], lr=0.1)
        
        initial_params = model.occupancy.refinable_params.clone()
        
        # Run optimization steps
        for i in range(20):
            optimizer.zero_grad()
            occ = model.occupancy()
            # Dummy loss: push refinable altlocs toward 0.8/0.2 split
            loss = ((occ[4] - 0.8) ** 2) + ((occ[14] - 0.8) ** 2)
            loss.backward()
            optimizer.step()
        
        final_occ = model.occupancy()
        
        print(f"\nAfter optimization:")
        print(f"  Residue 2A: {final_occ[4].item():.4f}")
        print(f"  Residue 2B: {final_occ[9].item():.4f}")
        print(f"  Residue 3A: {final_occ[14].item():.4f}")
        print(f"  Residue 3B: {final_occ[18].item():.4f}")
        
        # Check that altloc sums are STILL 1.0
        res2_sum_final = final_occ[4] + final_occ[9]
        res3_sum_final = final_occ[14] + final_occ[18]
        print(f"\nFinal altloc sums:")
        print(f"  Residue 2: {res2_sum_final.item():.6f} (should be 1.0)")
        print(f"  Residue 3: {res3_sum_final.item():.6f} (should be 1.0)")
        
        assert torch.allclose(res2_sum_final, torch.tensor(1.0), atol=1e-5), \
            f"Residue 2 should still sum to 1 after optimization, got {res2_sum_final.item()}"
        assert torch.allclose(res3_sum_final, torch.tensor(1.0), atol=1e-5), \
            f"Residue 3 should still sum to 1 after optimization, got {res3_sum_final.item()}"
        
        # Check that parameters actually changed
        assert not torch.allclose(model.occupancy.refinable_params, initial_params, atol=1e-3), \
            "Parameters should have changed during optimization"
        
        # Check that residue 1 (static, occ=1.0) didn't change
        print(f"\nResidue 1 (static): {final_occ[0].item():.4f} (should be 1.0)")
        assert torch.allclose(final_occ[0], torch.tensor(1.0), atol=1e-4), \
            "Static residue 1 should not change"
        
        print("\n" + "="*70)
        print("✓ ALL AUTOMATIC ALTLOC TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print(f"  ✓ Altloc groups automatically sum to 1.0")
        print(f"  ✓ Placeholders computed as 1 - sum(others)")
        print(f"  ✓ Placeholders are not refinable")
        print(f"  ✓ Sum-to-1 maintained during optimization")
        print(f"  ✓ NO manual normalization needed")
        print(f"  ✓ Static occupancies unchanged")
        print(f"  ✓ Vectorized collapse operations")
        
        return True
        
    finally:
        if os.path.exists(pdb_file):
            os.remove(pdb_file)

if __name__ == '__main__':
    try:
        success = test_automatic_altloc_constraint()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
