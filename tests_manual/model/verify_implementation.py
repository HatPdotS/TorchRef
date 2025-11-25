from torchref.model import model

# Load the test PDB
test = model()
test.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

# Verify that altloc_pairs was created
print("✓ Model loaded successfully")
print(f"✓ altloc_pairs attribute exists: {hasattr(test, 'altloc_pairs')}")
print(f"✓ Number of alternative conformation groups: {len(test.altloc_pairs)}")
print(f"✓ First 3 groups: {test.altloc_pairs[:3]}")

# Verify data integrity
if len(test.altloc_pairs) > 0:
    first_group = test.altloc_pairs[0]
    atoms = test.pdb.loc[list(first_group)]
    print(f"\n✓ First group details:")
    print(f"  - Atom name: {atoms['name'].iloc[0]}")
    print(f"  - Residue: {atoms['resname'].iloc[0]}-{atoms['resseq'].iloc[0]}")
    print(f"  - Chain: {atoms['chainid'].iloc[0]}")
    print(f"  - Altlocs: {atoms['altloc'].tolist()}")
    print(f"  - Indices: {first_group}")

print("\n✓ All checks passed!")
