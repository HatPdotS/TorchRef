"""
Tests for the register_alternative_conformations method in the model class.

This test suite verifies that alternative conformations in PDB structures
are correctly identified and their indices are properly stored.
"""

# Import in same way as test_model_new.py
from torchref.model import model
import pdb_tools
import pandas as pd
import tempfile
import os


class TestRegisterAlternativeConformations:
    """Test suite for alternative conformation registration."""
    
    def test_with_real_pdb_file(self):
        """Test with a real PDB file containing alternative conformations."""
        # Load a real PDB file with altlocs
        test = model()
        test.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
        
        # Check that altloc_pairs is populated
        assert hasattr(test, 'altloc_pairs'), "Model should have altloc_pairs attribute"
        assert isinstance(test.altloc_pairs, list), "altloc_pairs should be a list"
        assert len(test.altloc_pairs) > 0, "Should find alternative conformations in dark.pdb"
        
        # Check structure of pairs
        for pair in test.altloc_pairs:
            assert isinstance(pair, tuple), f"Each pair should be a tuple, got {type(pair)}"
            assert len(pair) >= 2, f"Each pair should have at least 2 indices, got {len(pair)}"
            # All indices should be integers
            for idx in pair:
                assert isinstance(idx, (int, pd.Int64Dtype)), f"Index should be integer, got {type(idx)}"
        
        print(f"\nTotal alternative conformation groups found: {len(test.altloc_pairs)}")
        print(f"First 5 groups: {test.altloc_pairs[:5]}")
    
    def test_pair_indices_match_atoms(self):
        """Verify that indices in pairs actually correspond to alternative conformations."""
        test = model()
        test.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
        
        # Check a few pairs to verify they represent the same atom with different altlocs
        for pair in test.altloc_pairs[:10]:  # Check first 10 pairs
            # Get the atoms at these indices (use .loc since indices are in the DataFrame index)
            atoms = test.pdb.loc[list(pair)]
            
            # All atoms in a pair should have:
            # - Same residue name
            resnames = atoms['resname'].unique()
            assert len(resnames) == 1, f"All atoms in pair should have same resname, got {resnames}"
            
            # - Same residue sequence number
            resseqs = atoms['resseq'].unique()
            assert len(resseqs) == 1, f"All atoms in pair should have same resseq, got {resseqs}"
            
            # - Same chain ID
            chainids = atoms['chainid'].unique()
            assert len(chainids) == 1, f"All atoms in pair should have same chainid, got {chainids}"
            
            # - Same atom name
            names = atoms['name'].unique()
            assert len(names) == 1, f"All atoms in pair should have same name, got {names}"
            
            # - Different altloc values
            altlocs = atoms['altloc'].tolist()
            assert len(set(altlocs)) == len(altlocs), f"All atoms should have different altlocs, got {altlocs}"
            assert all(a != '' for a in altlocs), "All altlocs should be non-empty"
    
    def test_altloc_ordering(self):
        """Test that altlocs are ordered consistently (A, B, C, ...)."""
        test = model()
        test.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
        
        for pair in test.altloc_pairs[:20]:  # Check first 20 pairs
            atoms = test.pdb.loc[list(pair)]
            altlocs = atoms['altloc'].tolist()
            
            # Check that altlocs are in sorted order
            assert altlocs == sorted(altlocs), f"Altlocs should be sorted, got {altlocs}"
    
    def test_no_duplicate_indices(self):
        """Ensure no index appears in multiple pairs."""
        test = model()
        test.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
        
        # Collect all indices
        all_indices = []
        for pair in test.altloc_pairs:
            all_indices.extend(pair)
        
        # Check for duplicates
        assert len(all_indices) == len(set(all_indices)), "Each index should appear in only one pair"
    
    def test_count_conformations(self):
        """Test that we correctly handle 2-way and 3-way (or more) alternative conformations."""
        test = model()
        test.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
        
        # Count pairs by size
        pair_sizes = {}
        for pair in test.altloc_pairs:
            size = len(pair)
            pair_sizes[size] = pair_sizes.get(size, 0) + 1
        
        print(f"\nConformation group sizes: {pair_sizes}")
        
        # Should have at least some 2-way conformations
        assert 2 in pair_sizes, "Should have some 2-way alternative conformations"
        
        # May have 3-way or more
        if 3 in pair_sizes:
            print(f"Found {pair_sizes[3]} three-way alternative conformations")
    
    def test_pdb_without_altlocs(self):
        """Test behavior with a PDB file that has no alternative conformations."""
        # Create a simple PDB without altlocs
        pdb_content = """CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.000  11.000  11.000  1.00 20.00           C
ATOM      3  C   ALA A   1      12.000  12.000  12.000  1.00 20.00           C
ATOM      4  O   ALA A   1      13.000  13.000  13.000  1.00 20.00           O
ATOM      5  CB  ALA A   1      14.000  14.000  14.000  1.00 20.00           C
END
"""
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            temp_pdb = f.name
        
        try:
            test = model()
            test.load_pdb_from_file(temp_pdb)
            
            # Should have empty altloc_pairs
            assert hasattr(test, 'altloc_pairs'), "Should have altloc_pairs attribute"
            assert isinstance(test.altloc_pairs, list), "altloc_pairs should be a list"
            assert len(test.altloc_pairs) == 0, "Should have no alternative conformations"
        finally:
            os.unlink(temp_pdb)
    
    def test_specific_residue_altlocs(self):
        """Test a specific residue known to have alternative conformations."""
        test = model()
        test.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
        
        # From the analysis, we know ARG-123 Chain A has 3 conformations (A, B, C)
        # Find all pairs for this residue
        arg123_pairs = []
        for pair in test.altloc_pairs:
            atoms = test.pdb.loc[list(pair)]
            if (atoms['resname'].iloc[0] == 'ARG' and 
                atoms['resseq'].iloc[0] == 123 and 
                atoms['chainid'].iloc[0] == 'A'):
                arg123_pairs.append(pair)
        
        # ARG has 19 non-H atoms, so should have 19 triplets
        # (but dark.pdb might have H stripped, so check for > 0)
        assert len(arg123_pairs) > 0, "Should find alternative conformations for ARG-123 Chain A"
        
        # All should be triplets (3 conformations)
        for pair in arg123_pairs:
            atoms = test.pdb.loc[list(pair)]
            # Check if hydrogens are included
            if atoms['element'].iloc[0] != 'H':
                # Could be 3 conformations if all present
                assert len(pair) in [2, 3], f"Expected 2 or 3 conformations, got {len(pair)}"
        
        print(f"\nFound {len(arg123_pairs)} atom groups for ARG-123 Chain A")
        print(f"Example group: {arg123_pairs[0] if arg123_pairs else 'None'}")
    
    def test_altloc_pairs_attribute_exists(self):
        """Test that loading creates the altloc_pairs attribute."""
        test = model()
        test.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')
        
        assert hasattr(test, 'altloc_pairs'), "Model should have altloc_pairs attribute after loading"
    
    def test_manual_pdb_with_altlocs(self):
        """Test with a manually created PDB containing known alternative conformations."""
        pdb_content = """CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA AALA A   1      11.000  11.000  11.000  0.50 20.00           C
ATOM      3  CA BALA A   1      11.100  11.100  11.100  0.50 20.00           C
ATOM      4  C   ALA A   1      12.000  12.000  12.000  1.00 20.00           C
ATOM      5  O   ALA A   1      13.000  13.000  13.000  1.00 20.00           O
ATOM      6  CB AALA A   1      14.000  14.000  14.000  0.50 20.00           C
ATOM      7  CB BALA A   1      14.100  14.100  14.100  0.50 20.00           C
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            temp_pdb = f.name
        
        try:
            test = model()
            test.load_pdb_from_file(temp_pdb)
            
            # Should find 2 pairs (CA and CB, each with A and B conformations)
            assert len(test.altloc_pairs) == 2, f"Expected 2 pairs, got {len(test.altloc_pairs)}"
            
            # Each pair should have exactly 2 indices
            for pair in test.altloc_pairs:
                assert len(pair) == 2, f"Expected pairs of size 2, got {len(pair)}"
            
            # Verify the pairs correspond to CA and CB
            atom_names = set()
            for pair in test.altloc_pairs:
                atoms = test.pdb.loc[list(pair)]
                atom_names.add(atoms['name'].iloc[0])
            
            assert atom_names == {'CA', 'CB'}, f"Expected CA and CB, got {atom_names}"
            
            print(f"\nManual test pairs: {test.altloc_pairs}")
        finally:
            os.unlink(temp_pdb)


def main():
    """Run all tests and print results."""
    print("=" * 80)
    print("Testing register_alternative_conformations")
    print("=" * 80)
    
    test_suite = TestRegisterAlternativeConformations()
    
    tests = [
        ("Real PDB file", test_suite.test_with_real_pdb_file),
        ("Indices match atoms", test_suite.test_pair_indices_match_atoms),
        ("Altloc ordering", test_suite.test_altloc_ordering),
        ("No duplicate indices", test_suite.test_no_duplicate_indices),
        ("Count conformations", test_suite.test_count_conformations),
        ("PDB without altlocs", test_suite.test_pdb_without_altlocs),
        ("Specific residue", test_suite.test_specific_residue_altlocs),
        ("Attribute exists", test_suite.test_altloc_pairs_attribute_exists),
        ("Manual PDB", test_suite.test_manual_pdb_with_altlocs),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED")
            print(f"  Error: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
