from torchref.model import model

test = model()
test.load_pdb_from_file('/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb')

print('Total pairs:', len(test.altloc_pairs))
print('First 3 pairs:', test.altloc_pairs[:3])

# Check pair that's failing in test
for i, pair in enumerate(test.altloc_pairs[:20]):
    atoms = test.pdb.iloc[list(pair)]
    resseqs = atoms['resseq'].unique()
    names = atoms['name'].unique()
    altlocs = atoms['altloc'].tolist()
    
    if len(resseqs) != 1 or len(names) != 1:
        print(f"\nProblem at pair {i}: {pair}")
        print(f"  Resseqs: {resseqs}")
        print(f"  Names: {names}")
        print(f"  Altlocs: {altlocs}")
        print("  Atom details:")
        print(atoms[['serial', 'name', 'altloc', 'resname', 'chainid', 'resseq', 'index']])
        break
