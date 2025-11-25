#!/usr/bin/env python3
"""
Test alternative conformation (altloc) handling with OccupancyTensor.

This test verifies that:
1. Alternative conformations sum to 1.0
2. enforce_occ_alternative_conformations works with collapsed storage
3. Altlocs maintain sharing within each conformation
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

def test_altloc_handling():
    """Test alternative conformation handling."""
    print("\n" + "="*70)
    print("TEST: Alternative Conformation Handling")
    print("="*70)
    
    pdb_file = create_altloc_pdb()
    
    try:
        # Load model
        print("\nLoading model with altlocs...")
        model = Model(verbose=1)
        model.load_pdb_from_file(pdb_file)
        
        print(f"\nOccupancy type: {type(model.occupancy)}")
        print(f"Total atoms: {len(model.pdb)}")
        
        # Get initial occupancies
        initial_occ = model.occupancy().clone()
        print(f"\nInitial occupancies:")
        print(f"  Residue 1 (no altloc): {initial_occ[0:4]}")
        print(f"  Residue 2A (altloc A): {initial_occ[4:9]}")
        print(f"  Residue 2B (altloc B): {initial_occ[9:14]}")
        print(f"  Residue 3A (altloc A): {initial_occ[14:18]}")
        print(f"  Residue 3B (altloc B): {initial_occ[18:22]}")
        
        # Verify initial state
        # Residue 2: 0.7 + 0.3 = 1.0
        res2_sum = initial_occ[4] + initial_occ[9]
        print(f"\nResidue 2 altloc sum: {res2_sum.item():.4f} (should be ~1.0)")
        assert torch.allclose(res2_sum, torch.tensor(1.0), atol=0.01), "Residue 2 altlocs should sum to 1"
        
        # Residue 3: 0.55 + 0.45 = 1.0
        res3_sum = initial_occ[14] + initial_occ[18]
        print(f"Residue 3 altloc sum: {res3_sum.item():.4f} (should be ~1.0)")
        assert torch.allclose(res3_sum, torch.tensor(1.0), atol=0.01), "Residue 3 altlocs should sum to 1"
        
        # Manually change occupancies to violate sum=1 constraint
        print("\n" + "-"*70)
        print("Testing enforce_occ_alternative_conformations...")
        print("-"*70)
        
        # We'll modify the refinable parameters directly to create inconsistent values
        # This simulates what happens during optimization when altlocs drift apart
        with torch.no_grad():
            # Get the refinable parameter indices for each residue
            res2a_collapsed_idx = model.occupancy.expansion_mask[4].item()
            res2b_collapsed_idx = model.occupancy.expansion_mask[9].item()
            res3a_collapsed_idx = model.occupancy.expansion_mask[14].item()
            res3b_collapsed_idx = model.occupancy.expansion_mask[18].item()
            
            print(f"\nCollapsed indices: 2A={res2a_collapsed_idx}, 2B={res2b_collapsed_idx}, 3A={res3a_collapsed_idx}, 3B={res3b_collapsed_idx}")
            print(f"Refinable mask: {model.occupancy.refinable_mask}")
            
            # Create new values in probability space
            new_values = torch.zeros(model.occupancy.collapsed_shape[0])
            new_values[:] = model.occupancy.fixed_values  # Start with current values
            
            # Modify specific residues: 2A=0.8, 2B=0.3, 3A=0.6, 3B=0.2
            target_values = torch.zeros_like(new_values)
            target_values[:] = torch.sigmoid(new_values)  # Convert current logits to probabilities
            target_values[res2a_collapsed_idx] = 0.8
            target_values[res2b_collapsed_idx] = 0.3
            target_values[res3a_collapsed_idx] = 0.6
            target_values[res3b_collapsed_idx] = 0.2
            
            # Convert back to logits
            clamped = torch.clamp(target_values, min=1e-6, max=1-1e-6)
            new_logits = torch.logit(clamped)
            
            # Update the parameters
            model.occupancy.fixed_values = new_logits.clone()
            if model.occupancy.refinable_mask.any():
                model.occupancy.refinable_params.data = new_logits[model.occupancy.refinable_mask].clone()
        
        # Check values before normalization
        current_occ = model.occupancy()
        print(f"\nBefore normalization:")
        print(f"  Residue 2A: {current_occ[4].item():.4f}, 2B: {current_occ[9].item():.4f}, sum: {(current_occ[4] + current_occ[9]).item():.4f}")
        print(f"  Residue 3A: {current_occ[14].item():.4f}, 3B: {current_occ[18].item():.4f}, sum: {(current_occ[14] + current_occ[18]).item():.4f}")
        
        # Enforce altloc constraints (this should normalize to sum=1)
        model.enforce_occ_alternative_conformations()
        
        # Check normalized occupancies
        normalized_occ = model.occupancy()
        print(f"\nAfter normalization:")
        print(f"  Residue 2A: {normalized_occ[4].item():.4f}, 2B: {normalized_occ[9].item():.4f}, sum: {(normalized_occ[4] + normalized_occ[9]).item():.4f}")
        print(f"  Residue 3A: {normalized_occ[14].item():.4f}, 3B: {normalized_occ[18].item():.4f}, sum: {(normalized_occ[14] + normalized_occ[18]).item():.4f}")
        
        # Verify sums are now 1.0
        res2_sum_after = normalized_occ[4] + normalized_occ[9]
        res3_sum_after = normalized_occ[14] + normalized_occ[18]
        
        assert torch.allclose(res2_sum_after, torch.tensor(1.0), atol=1e-4), f"Residue 2 should sum to 1, got {res2_sum_after.item()}"
        assert torch.allclose(res3_sum_after, torch.tensor(1.0), atol=1e-4), f"Residue 3 should sum to 1, got {res3_sum_after.item()}"
        
        # Verify ratios are preserved
        # Residue 2: 0.8 / (0.8+0.3) = 0.727, 0.3 / 1.1 = 0.273
        expected_2a = 0.8 / 1.1
        expected_2b = 0.3 / 1.1
        print(f"\nResidue 2 ratios:")
        print(f"  Expected A: {expected_2a:.4f}, got: {normalized_occ[4].item():.4f}")
        print(f"  Expected B: {expected_2b:.4f}, got: {normalized_occ[9].item():.4f}")
        assert torch.allclose(normalized_occ[4], torch.tensor(expected_2a), atol=1e-3), "Residue 2A ratio incorrect"
        assert torch.allclose(normalized_occ[9], torch.tensor(expected_2b), atol=1e-3), "Residue 2B ratio incorrect"
        
        # Residue 3: 0.6 / 0.8 = 0.75, 0.2 / 0.8 = 0.25
        expected_3a = 0.6 / 0.8
        expected_3b = 0.2 / 0.8
        print(f"\nResidue 3 ratios:")
        print(f"  Expected A: {expected_3a:.4f}, got: {normalized_occ[14].item():.4f}")
        print(f"  Expected B: {expected_3b:.4f}, got: {normalized_occ[18].item():.4f}")
        assert torch.allclose(normalized_occ[14], torch.tensor(expected_3a), atol=1e-3), "Residue 3A ratio incorrect"
        assert torch.allclose(normalized_occ[18], torch.tensor(expected_3b), atol=1e-3), "Residue 3B ratio incorrect"
        
        # Verify sharing is maintained within each conformation
        print(f"\nVerifying sharing within conformations:")
        assert torch.allclose(normalized_occ[4:9], normalized_occ[4].expand(5), atol=1e-5), "Residue 2A sharing broken"
        print(f"  ✓ Residue 2A: all atoms have same occupancy")
        assert torch.allclose(normalized_occ[9:14], normalized_occ[9].expand(5), atol=1e-5), "Residue 2B sharing broken"
        print(f"  ✓ Residue 2B: all atoms have same occupancy")
        assert torch.allclose(normalized_occ[14:18], normalized_occ[14].expand(4), atol=1e-5), "Residue 3A sharing broken"
        print(f"  ✓ Residue 3A: all atoms have same occupancy")
        assert torch.allclose(normalized_occ[18:22], normalized_occ[18].expand(4), atol=1e-5), "Residue 3B sharing broken"
        print(f"  ✓ Residue 3B: all atoms have same occupancy")
        
        print("\n" + "="*70)
        print("✓ ALL ALTLOC TESTS PASSED!")
        print("="*70)
        print("\nSummary:")
        print(f"  ✓ Alternative conformations properly normalized")
        print(f"  ✓ Altloc sum-to-1 constraint enforced")
        print(f"  ✓ Relative ratios preserved")
        print(f"  ✓ Sharing maintained within conformations")
        print(f"  ✓ Works correctly with collapsed storage")
        
        return True
        
    finally:
        if os.path.exists(pdb_file):
            os.remove(pdb_file)

if __name__ == '__main__':
    try:
        success = test_altloc_handling()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
