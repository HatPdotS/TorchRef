#!/usr/bin/env python3
"""Check for broken chains and coordinate issues."""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torchref.model import Model

def check_structure_issues():
    """Check for structural problems causing high bond NLL."""
    
    pdb_file = Path(__file__).parent.parent.parent / "test_data" / "dark.pdb"
    
    print(f"Loading structure: {pdb_file}")
    model = Model()
    model.load_pdb_from_file(str(pdb_file))
    
    pdb = model.pdb
    xyz = model.xyz()
    
    print(f"\n{'='*80}")
    print(f"CHECKING PROBLEM RESIDUES")
    print(f"{'='*80}")
    
    # Check B:278 SER
    print(f"\n🔍 Residue B:278 SER:")
    res_278 = pdb[(pdb['chainid'] == 'B') & (pdb['resseq'] == 278)]
    print(f"   Atoms found: {len(res_278)}")
    if len(res_278) > 0:
        print(f"   Atom names: {', '.join(res_278['name'].values)}")
        for _, atom in res_278.iterrows():
            coord = xyz[atom['index']].detach().cpu().numpy()
            print(f"   {atom['name']:4s}: ({coord[0]:7.3f}, {coord[1]:7.3f}, {coord[2]:7.3f})")
    
    # Check B:284 (should be next after 278 for peptide bond)
    print(f"\n🔍 Residue B:284 (next in sequence?):")
    res_284 = pdb[(pdb['chainid'] == 'B') & (pdb['resseq'] == 284)]
    print(f"   Atoms found: {len(res_284)}")
    if len(res_284) > 0:
        print(f"   Residue: {res_284['resname'].values[0]}")
        print(f"   Atom names: {', '.join(res_284['name'].values)}")
        # Check N atom
        n_atoms = res_284[res_284['name'] == 'N']
        if len(n_atoms) > 0:
            n_atom = n_atoms.iloc[0]
            n_coord = xyz[n_atom['index']].detach().cpu().numpy()
            print(f"   N coord: ({n_coord[0]:7.3f}, {n_coord[1]:7.3f}, {n_coord[2]:7.3f})")
    
    # Check what's between 278 and 284
    print(f"\n🔍 Residues between 278 and 284 in chain B:")
    between = pdb[(pdb['chainid'] == 'B') & (pdb['resseq'] > 278) & (pdb['resseq'] < 284)]
    print(f"   Residues found: {between['resseq'].unique()}")
    
    # Check chain B continuity
    print(f"\n🔍 Chain B sequence:")
    chain_b = pdb[pdb['chainid'] == 'B']
    resseqs = sorted(chain_b['resseq'].unique())
    print(f"   Total residues: {len(resseqs)}")
    print(f"   Residue range: {resseqs[0]} to {resseqs[-1]}")
    
    # Find gaps
    gaps = []
    for i in range(len(resseqs) - 1):
        if resseqs[i+1] - resseqs[i] > 1:
            gaps.append((resseqs[i], resseqs[i+1]))
    
    if gaps:
        print(f"   ⚠️  Found {len(gaps)} gaps in numbering:")
        for gap_start, gap_end in gaps:
            print(f"      Gap: {gap_start} → {gap_end} (missing {gap_end - gap_start - 1} residues)")
    else:
        print(f"   ✓ No gaps in numbering")
    
    # Check AZO ligand
    print(f"\n{'='*80}")
    print(f"CHECKING AZO LIGAND")
    print(f"{'='*80}")
    
    print(f"\n🔍 Residue B:702 AZO:")
    azo = pdb[(pdb['chainid'] == 'B') & (pdb['resseq'] == 702)]
    print(f"   Atoms found: {len(azo)}")
    if len(azo) > 0:
        print(f"   HETATM: {azo['ATOM'].values[0]}")
        print(f"   Atoms: {', '.join(azo['name'].values)}")
        
        # Check some key bonds
        print(f"\n   Checking inter-atomic distances:")
        for i in range(min(5, len(azo)-1)):
            atom1 = azo.iloc[i]
            atom2 = azo.iloc[i+1]
            coord1 = xyz[atom1['index']].detach().cpu().numpy()
            coord2 = xyz[atom2['index']].detach().cpu().numpy()
            dist = ((coord1 - coord2)**2).sum()**0.5
            print(f"      {atom1['name']}-{atom2['name']}: {dist:.3f} Å")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"DIAGNOSIS")
    print(f"{'='*80}")
    
    if len(gaps) > 0 and (278, 284) in gaps:
        print(f"\n❌ CHAIN BREAK DETECTED:")
        print(f"   Residue 278 and 284 are not consecutive")
        print(f"   The C-N 'bond' is actually across a chain break")
        print(f"   This should NOT be treated as a bond restraint!")
    
    print(f"\n💡 SOLUTION:")
    print(f"   Need to filter out 'bonds' between non-consecutive residues")
    print(f"   Peptide bonds should only be between resseq and resseq+1")

if __name__ == '__main__':
    check_structure_issues()
