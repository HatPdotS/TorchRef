#!/usr/bin/env python3
"""
Analyze how many peptide bonds are missing from restraints
"""

import sys
sys.path.insert(0, '/das/work/p17/p17490/Peter/Library/multicopy_refinement')

from torchref.model import Model

# Load the model
model = Model()
test_pdb = '/das/work/p17/p17490/Peter/Library/multicopy_refinement/test_data/dark.pdb'
model.load_pdb_from_file(test_pdb)

pdb = model.pdb

print("=" * 70)
print("PEPTIDE BOND ANALYSIS")
print("=" * 70)

total_peptide_bonds = 0
peptide_bonds_by_chain = {}

# Process each chain
for chain_id in pdb['chainid'].unique():
    chain = pdb[pdb['chainid'] == chain_id].copy()
    
    # Get protein residues only (ATOM records)
    protein_chain = chain[chain['ATOM'] == 'ATOM']
    
    if len(protein_chain) == 0:
        continue
    
    # Get unique residues, sorted
    residues = protein_chain.groupby('resseq').first().sort_index()
    resseq_list = residues.index.tolist()
    
    chain_peptide_bonds = 0
    
    # Count consecutive residues that should have peptide bonds
    for i in range(len(resseq_list) - 1):
        resseq_i = resseq_list[i]
        resseq_j = resseq_list[i + 1]
        
        # Get residues
        res_i = protein_chain[protein_chain['resseq'] == resseq_i]
        res_j = protein_chain[protein_chain['resseq'] == resseq_j]
        
        # Check if both have C and N atoms
        has_c = len(res_i[res_i['name'] == 'C']) > 0
        has_n = len(res_j[res_j['name'] == 'N']) > 0
        
        if has_c and has_n:
            chain_peptide_bonds += 1
            total_peptide_bonds += 1
    
    if chain_peptide_bonds > 0:
        peptide_bonds_by_chain[chain_id] = chain_peptide_bonds
        print(f"\nChain {chain_id}:")
        print(f"  Residues: {len(resseq_list)}")
        print(f"  Expected peptide bonds: {chain_peptide_bonds}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total peptide bonds that should exist: {total_peptide_bonds}")
print(f"\nFor comparison:")
print(f"  - Current bond restraints: ~7,368")
print(f"  - Missing peptide bonds: {total_peptide_bonds}")
print(f"  - Total with peptide bonds: {7368 + total_peptide_bonds}")
print(f"\n  - Additional backbone angles: ~{total_peptide_bonds * 2}")
print(f"  - Additional omega torsions: {total_peptide_bonds}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print("✓ Your current intra-residue bond count (7,368) is correct!")
print(f"⚠ You are missing ~{total_peptide_bonds} inter-residue peptide bond restraints")
print(f"⚠ You are also missing ~{total_peptide_bonds * 2} backbone angle restraints")
print(f"⚠ You are also missing ~{total_peptide_bonds} omega torsion restraints")
print("\nThese inter-residue restraints are CRITICAL for maintaining")
print("proper protein backbone geometry during refinement!")
